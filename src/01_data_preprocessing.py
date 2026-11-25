"""
==============================================================================
 NASA C-MAPSS Turbofan Engine RUL Prediction
 Module 1: Data Loading and Preprocessing Pipeline
==============================================================================

Dataset: NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)
Subsets: FD001, FD002, FD003, FD004

Each subset contains:
  - train_FDxxx.txt  : Multi-engine operational sensor readings (run-to-failure)
  - test_FDxxx.txt   : Truncated engine readings (stops before failure)
  - RUL_FDxxx.txt    : Ground-truth Remaining Useful Life for each test engine

Sensor layout per row (26 space-delimited columns):
  Col 0  : Engine unit number
  Col 1  : Operational cycle number
  Col 2-4: Operational settings (setting_1, setting_2, setting_3)
  Col 5-25: 21 sensor measurements (s_1 through s_21)

Reference:
  Saxena, A., Goebel, K., Simon, D., Eklund, N. (2008). Damage Propagation
  Modeling for Aircraft Engine Run-to-Failure Simulation. PHM Conference.
==============================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# =============================================================================
# Column Definitions
# =============================================================================

INDEX_COLS = ["unit_nr", "time_cycles"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]   # s_1 through s_21
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS  # 26 columns total
FEATURE_COLS = SETTING_COLS + SENSOR_COLS           # 24 input features (3 settings + 21 sensors)

# =============================================================================
# Hyperparameters
# =============================================================================

WINDOW_SIZE = 50       # Sliding temporal window length (50 operational cycles)
MAX_RUL_CLIP = 125     # Piecewise-linear RUL cap — standard for C-MAPSS benchmark


# =============================================================================
# 1. Data Loading
# =============================================================================

def load_cmapss_raw(data_dir: str, subset: str, split: str = "train") -> pd.DataFrame:
    """
    Load a raw NASA C-MAPSS text file into a pandas DataFrame.

    Parameters
    ----------
    data_dir : str
        Directory path containing the CMAPSS .txt files.
    subset : str
        One of "FD001", "FD002", "FD003", "FD004".
    split : str
        One of "train", "test", or "RUL".

    Returns
    -------
    pd.DataFrame
        For "train"/"test": columns = ALL_COLS (26 columns).
        For "RUL": single column "rul" (one entry per test engine).

    Example
    -------
    >>> train_df = load_cmapss_raw("data", "FD001", "train")
    >>> train_df.shape
    (20631, 26)
    """
    if split == "RUL":
        filename = f"RUL_{subset}.txt"
        filepath = os.path.join(data_dir, filename)
        return pd.read_csv(filepath, sep=r"\s+", header=None, names=["rul"])

    filename = f"{split}_{subset}.txt"
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath, sep=r"\s+", header=None)
    df = df.iloc[:, :26]        # Keep only the first 26 columns (some files have trailing NaN cols)
    df.columns = ALL_COLS
    return df


# =============================================================================
# 2. RUL Target Generation (Piecewise Linear)
# =============================================================================

def add_remaining_useful_life(df: pd.DataFrame, max_clip: int = MAX_RUL_CLIP) -> pd.DataFrame:
    """
    Compute the Remaining Useful Life (RUL) for each row in a training DataFrame.

    Strategy: Piecewise-linear RUL (PLR)
      - True RUL at cycle t for engine i = max_cycle_i - t
      - Clipped at MAX_RUL_CLIP (125): early health states treated as equivalent
        because we cannot distinguish "healthy" from "very healthy" in practice.

    This is the standard benchmark approach used in the PHM 2008 Data Challenge
    and most published C-MAPSS literature (Heimes 2008, Babu et al. 2016).

    Parameters
    ----------
    df : pd.DataFrame
        Raw training dataframe with columns "unit_nr" and "time_cycles".
    max_clip : int
        Upper bound for RUL clipping (default: 125 cycles).

    Returns
    -------
    pd.DataFrame
        Input dataframe with two new columns:
          - "RUL"         : Raw (unclipped) RUL
          - "RUL_clipped" : Piecewise-linear clipped RUL (used as training target)
    """
    df = df.copy()
    max_cycle = df.groupby("unit_nr")["time_cycles"].max().reset_index()
    max_cycle.columns = ["unit_nr", "max_cycles"]
    df = df.merge(max_cycle, on="unit_nr", how="left")
    df["RUL"] = df["max_cycles"] - df["time_cycles"]
    df.drop(columns=["max_cycles"], inplace=True)
    if max_clip is not None:
        df["RUL_clipped"] = df["RUL"].clip(upper=max_clip)
    else:
        df["RUL_clipped"] = df["RUL"]
    return df


# =============================================================================
# 3. Feature Normalization (Zero Data Leakage)
# =============================================================================

def get_scaler(train_df: pd.DataFrame, feature_cols: list = None) -> StandardScaler:
    """
    Fit a StandardScaler exclusively on the training partition.

    IMPORTANT: The scaler is NEVER fit on test data — this is a strict requirement
    to prevent data leakage. The same fitted scaler is later applied to the test set.

    StandardScaler transforms each feature to zero mean and unit variance:
        x_scaled = (x - mean_train) / std_train

    We use StandardScaler (not MinMaxScaler) because sensor readings are approximately
    Gaussian and several sensors have constant or near-constant values across certain
    operating regimes — MinMaxScaler is sensitive to outliers in such cases.

    Parameters
    ----------
    train_df : pd.DataFrame
        Full training dataframe (all engines, all cycles).
    feature_cols : list, optional
        Columns to scale. Defaults to FEATURE_COLS (3 settings + 21 sensors).

    Returns
    -------
    sklearn.preprocessing.StandardScaler
        Fitted scaler ready for transform().
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def transform_data(df: pd.DataFrame, scaler: StandardScaler, feature_cols: list = None) -> pd.DataFrame:
    """
    Apply a pre-fitted scaler to a dataframe's feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to transform (train or test).
    scaler : StandardScaler
        A scaler that has already been fit on training data.
    feature_cols : list, optional
        Columns to transform. Defaults to FEATURE_COLS.

    Returns
    -------
    pd.DataFrame
        Copy of input with feature columns replaced by scaled values.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


# =============================================================================
# 4. Sequence Generation for Deep Learning Models (CNN, CNN-LSTM, Attention)
# =============================================================================

def generate_sequences(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    feature_cols: list = None,
    target_col: str = "RUL_clipped"
):
    """
    Generate sliding temporal windows from training data for time-series deep learning.

    For each engine, we slide a window of length `window_size` (50 cycles) across the
    full operational trajectory. The label for each window is the RUL at the LAST
    cycle in that window.

    Visual example (window_size=4, engine has 7 cycles):
      Cycles: [1, 2, 3, 4, 5, 6, 7]
      Window 1: [c1, c2, c3, c4] → label = RUL at cycle 4
      Window 2: [c2, c3, c4, c5] → label = RUL at cycle 5
      Window 3: [c3, c4, c5, c6] → label = RUL at cycle 6
      Window 4: [c4, c5, c6, c7] → label = RUL at cycle 7

    Parameters
    ----------
    df : pd.DataFrame
        Scaled training dataframe (must contain "unit_nr" and RUL_clipped columns).
    window_size : int
        Number of cycles per input window (default: 50).
    feature_cols : list, optional
        Sensor/setting columns to include. Defaults to FEATURE_COLS (24 features).
    target_col : str
        Column name to use as regression target.

    Returns
    -------
    X : np.ndarray of shape (N_windows, window_size, n_features)
        3D input tensor for Conv1D/LSTM layers.
    y : np.ndarray of shape (N_windows,)
        RUL targets corresponding to each window.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    sequences = []
    targets = []

    for unit_id in df["unit_nr"].unique():
        unit_df = df[df["unit_nr"] == unit_id]
        data_arr = unit_df[feature_cols].values
        target_arr = unit_df[target_col].values if target_col in unit_df.columns else None

        num_cycles = len(unit_df)
        if num_cycles >= window_size:
            for start_idx in range(0, num_cycles - window_size + 1):
                end_idx = start_idx + window_size
                sequences.append(data_arr[start_idx:end_idx])
                if target_arr is not None:
                    targets.append(target_arr[end_idx - 1])

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32) if len(targets) > 0 else None
    return X, y


def generate_test_last_windows(
    test_df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    feature_cols: list = None
):
    """
    Extract the last operational window for each test engine (inference mode).

    For test engines, we only have access to the FINAL operational state before
    the end of recording. We extract the last `window_size` cycles.

    If an engine has fewer than `window_size` total cycles recorded, we pad the
    beginning with the FIRST observed cycle (front-padding), which assumes the
    engine was healthy at the start of recording.

    Parameters
    ----------
    test_df : pd.DataFrame
        Scaled test dataframe.
    window_size : int
        Number of cycles per input window (default: 50).
    feature_cols : list, optional
        Feature columns to include. Defaults to FEATURE_COLS.

    Returns
    -------
    X : np.ndarray of shape (N_engines, window_size, n_features)
        One window per engine, ready for model.predict().
    unit_ids : np.ndarray of shape (N_engines,)
        Engine unit numbers corresponding to each row in X.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    windows = []
    unit_ids = []

    for unit_id in test_df["unit_nr"].unique():
        unit_df = test_df[test_df["unit_nr"] == unit_id]
        data_arr = unit_df[feature_cols].values
        num_cycles = len(unit_df)

        if num_cycles >= window_size:
            window = data_arr[-window_size:]
        else:
            pad_len = window_size - num_cycles
            pad = np.tile(data_arr[0], (pad_len, 1))   # Repeat first row to fill gap
            window = np.vstack([pad, data_arr])

        windows.append(window)
        unit_ids.append(unit_id)

    return np.array(windows, dtype=np.float32), np.array(unit_ids)


# =============================================================================
# 5. Tabular Feature Engineering for Classical ML Models
# =============================================================================

def engineer_tabular_features(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    feature_cols: list = None,
    is_train: bool = True
):
    """
    Derive statistical summary features from rolling windows for XGBoost / Random Forest.

    Deep learning models ingest raw 3D sequences (shape: N x T x F).
    Classical models require fixed-length 2D feature vectors. We produce these by
    computing 5 aggregate statistics (mean, std, min, max, last) over each feature
    across the 50-cycle window, yielding:
        5 statistics × 24 features = 120 tabular features per sample.

    Training mode (is_train=True):
      - Slides a step-3 window across each engine trajectory (dense sampling)
      - Returns all windows and their RUL labels for model fitting

    Inference mode (is_train=False):
      - Uses only the LAST window per engine (same logic as generate_test_last_windows)
      - Returns one row per test engine

    Parameters
    ----------
    df : pd.DataFrame
        Scaled dataframe.
    window_size : int
        Window size in cycles (default: 50).
    feature_cols : list, optional
        Feature columns to aggregate. Defaults to FEATURE_COLS.
    is_train : bool
        True → sliding-window training mode; False → last-window test mode.

    Returns
    -------
    X : np.ndarray of shape (N_samples, 120)
        Flattened statistical feature matrix.
    y or unit_ids : np.ndarray
        If is_train=True  → RUL targets (float32)
        If is_train=False → unit IDs (for alignment with ground-truth)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    rows = []
    targets = []
    unit_list = []

    grouped = df.groupby("unit_nr")

    for unit_id, unit_df in grouped:
        data_vals = unit_df[feature_cols].values
        target_vals = unit_df["RUL_clipped"].values if "RUL_clipped" in unit_df.columns else None
        num_cycles = len(unit_df)

        if is_train:
            step = 3  # Sample every 3rd window for training efficiency
            if num_cycles >= window_size:
                for start_idx in range(0, num_cycles - window_size + 1, step):
                    end_idx = start_idx + window_size
                    window = data_vals[start_idx:end_idx]
                    mean_f = np.mean(window, axis=0)
                    std_f  = np.std(window, axis=0)
                    min_f  = np.min(window, axis=0)
                    max_f  = np.max(window, axis=0)
                    last_f = window[-1]
                    row = np.concatenate([mean_f, std_f, min_f, max_f, last_f])  # 5 × 24 = 120 features
                    rows.append(row)
                    if target_vals is not None:
                        targets.append(target_vals[end_idx - 1])
        else:
            # Test: extract last window per engine
            if num_cycles >= window_size:
                window = data_vals[-window_size:]
            else:
                pad_len = window_size - num_cycles
                pad = np.tile(data_vals[0], (pad_len, 1))
                window = np.vstack([pad, data_vals])

            mean_f = np.mean(window, axis=0)
            std_f  = np.std(window, axis=0)
            min_f  = np.min(window, axis=0)
            max_f  = np.max(window, axis=0)
            last_f = window[-1]
            row = np.concatenate([mean_f, std_f, min_f, max_f, last_f])
            rows.append(row)
            unit_list.append(unit_id)

    X = np.array(rows, dtype=np.float32)
    if is_train:
        y = np.array(targets, dtype=np.float32)
        return X, y
    else:
        return X, np.array(unit_list)


# =============================================================================
# 6. NASA PHM Scoring Function
# =============================================================================

def nasa_phm_score(y_true, y_pred) -> float:
    """
    Official NASA PHM 2008 Challenge asymmetric scoring function.

    This score penalizes LATE predictions (predicting more RUL than remaining)
    more heavily than EARLY predictions. This is intentional: an engine failing
    earlier than predicted is catastrophic; predicting early enables conservative
    maintenance scheduling.

    Formula:
        d_i = y_pred_i - y_true_i   (prediction error)
        s_i = exp(-d_i / 13) - 1    if d_i < 0  (early prediction, light penalty)
        s_i = exp(d_i  / 10) - 1    if d_i >= 0 (late prediction, heavy penalty)
        Score = sum(s_i)

    Lower is better. Perfect prediction → Score = 0.

    Parameters
    ----------
    y_true : array-like
        Ground-truth RUL values (from RUL_FDxxx.txt).
    y_pred : array-like
        Model-predicted RUL values.

    Returns
    -------
    float
        Scalar NASA PHM asymmetric score (lower is better).
    """
    d = np.array(y_pred) - np.array(y_true)
    early = np.exp(-d[d < 0] / 13.0) - 1.0   # Underprediction penalty
    late  = np.exp(d[d >= 0] / 10.0) - 1.0   # Overprediction penalty (steeper)
    return float(np.sum(early) + np.sum(late))


# =============================================================================
# Quick Demo (run as script)
# =============================================================================

if __name__ == "__main__":
    DATA_DIR = os.path.join("..", "data")
    SUBSET = "FD001"

    print("Loading FD001 training data...")
    train_raw = load_cmapss_raw(DATA_DIR, SUBSET, "train")
    print(f"  Train shape: {train_raw.shape}")

    print("Computing piecewise-linear RUL targets...")
    train_df = add_remaining_useful_life(train_raw)
    print(f"  RUL range: [{train_df['RUL_clipped'].min():.0f}, {train_df['RUL_clipped'].max():.0f}]")

    print("Fitting StandardScaler on training set only (no data leakage)...")
    scaler = get_scaler(train_df)
    train_scaled = transform_data(train_df, scaler)

    print("Generating sliding-window sequences for DL models...")
    X_train, y_train = generate_sequences(train_scaled)
    print(f"  X_train shape: {X_train.shape}  (N_windows × 50 cycles × 24 features)")
    print(f"  y_train shape: {y_train.shape}")

    print("Engineering tabular features for XGBoost / Random Forest...")
    X_tab, y_tab = engineer_tabular_features(train_scaled, is_train=True)
    print(f"  X_tabular shape: {X_tab.shape}  (N_windows × 120 statistical features)")

    print("Done. Preprocessing pipeline verified.")
