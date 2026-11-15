"""
CMAPSS Data Preprocessing and Feature Pipeline
NASA Turbofan Engine Degradation Simulation Data Set
Standardized for FD001, FD002, FD003, FD004 with zero data leakage.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

INDEX_COLS = ["unit_nr", "time_cycles"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS
FEATURE_COLS = SETTING_COLS + SENSOR_COLS

WINDOW_SIZE = 50
MAX_RUL_CLIP = 125  # Piece-wise linear upper bound standard for C-MAPSS


def load_cmapss_raw(data_dir: str, subset: str, split: str = "train"):
    """
    Load raw space-delimited CMAPSS text files.
    """
    if split == "RUL":
        filename = f"RUL_{subset}.txt"
        filepath = os.path.join(data_dir, filename)
        return pd.read_csv(filepath, sep=r"\s+", header=None, names=["rul"])

    filename = f"{split}_{subset}.txt"
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath, sep=r"\s+", header=None)
    df = df.iloc[:, :26]
    df.columns = ALL_COLS
    return df


def add_remaining_useful_life(df: pd.DataFrame, max_clip: int = MAX_RUL_CLIP) -> pd.DataFrame:
    """
    Compute piece-wise linear RUL for training sets.
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


def get_scaler(train_df: pd.DataFrame, feature_cols: list = None) -> StandardScaler:
    """
    Fit StandardScaler on training partition only to guarantee zero data leakage.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def transform_data(df: pd.DataFrame, scaler: StandardScaler, feature_cols: list = None) -> pd.DataFrame:
    """
    Scale feature columns using pre-fitted scaler.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


def generate_sequences(df: pd.DataFrame, window_size: int = WINDOW_SIZE, feature_cols: list = None, target_col: str = "RUL_clipped"):
    """
    Generate sliding temporal windows of shape (N, window_size, n_features) and target RULs.
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


def generate_test_last_windows(test_df: pd.DataFrame, window_size: int = WINDOW_SIZE, feature_cols: list = None):
    """
    Extract the final operational window for each engine in the test set.
    Pads engines shorter than window_size with initial cycle state.
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
            pad = np.tile(data_arr[0], (pad_len, 1))
            window = np.vstack([pad, data_arr])

        windows.append(window)
        unit_ids.append(unit_id)

    return np.array(windows, dtype=np.float32), np.array(unit_ids)


def engineer_tabular_features(df: pd.DataFrame, window_size: int = WINDOW_SIZE, feature_cols: list = None, is_train: bool = True):
    """
    Extract summary statistics (mean, std, min, max, last) across windows
    for classical ML models (XGBoost, Random Forest).
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
            step = 3  # High density sampling across engine trajectory
            if num_cycles >= window_size:
                for start_idx in range(0, num_cycles - window_size + 1, step):
                    end_idx = start_idx + window_size
                    window = data_vals[start_idx:end_idx]
                    mean_f = np.mean(window, axis=0)
                    std_f = np.std(window, axis=0)
                    min_f = np.min(window, axis=0)
                    max_f = np.max(window, axis=0)
                    last_f = window[-1]
                    row = np.concatenate([mean_f, std_f, min_f, max_f, last_f])
                    rows.append(row)
                    if target_vals is not None:
                        targets.append(target_vals[end_idx - 1])
        else:
            if num_cycles >= window_size:
                window = data_vals[-window_size:]
            else:
                pad_len = window_size - num_cycles
                pad = np.tile(data_vals[0], (pad_len, 1))
                window = np.vstack([pad, data_vals])

            mean_f = np.mean(window, axis=0)
            std_f = np.std(window, axis=0)
            min_f = np.min(window, axis=0)
            max_f = np.max(window, axis=0)
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


def nasa_phm_score(y_true, y_pred):
    """
    Official NASA C-MAPSS / PHM asymmetric scoring function:
    d = y_pred - y_true
    s_i = exp(-d / 13) - 1 for d < 0 (early prediction)
    s_i = exp(d / 10) - 1 for d >= 0 (late prediction)
    """
    d = np.array(y_pred) - np.array(y_true)
    early = np.exp(-d[d < 0] / 13.0) - 1.0
    late = np.exp(d[d >= 0] / 10.0) - 1.0
    return float(np.sum(early) + np.sum(late))
