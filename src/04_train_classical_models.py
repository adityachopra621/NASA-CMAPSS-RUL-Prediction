"""
==============================================================================
 NASA C-MAPSS Turbofan Engine RUL Prediction
 Module 4: Classical ML Model Training (XGBoost & Random Forest)
==============================================================================

Trains two classical machine learning models for RUL regression:
  1. XGBoost   (Gradient Boosted Decision Trees)
  2. Random Forest (Ensemble of Decision Trees)

Unlike deep learning models that ingest raw 3D sequences (N × 50 × 24),
classical models require 2D tabular input. We engineer a fixed-length
feature vector of 120 statistical features from each 50-cycle window
(5 statistics × 24 features = 120 features per sample).

Feature Engineering Summary (from preprocessing module):
  For each 50-cycle window and each of the 24 features:
    1. mean(feature) over 50 cycles
    2. std(feature)  over 50 cycles
    3. min(feature)  over 50 cycles
    4. max(feature)  over 50 cycles
    5. last(feature) = value at the final cycle in the window

  Total: 5 × 24 = 120 features per sample.

Hyperparameters
---------------
XGBoost:
  n_estimators  = 150
  max_depth     = 5
  learning_rate = 0.05
  random_state  = 42
  n_jobs        = -1   (use all CPU cores)

Random Forest:
  n_estimators  = 150
  max_depth     = 12
  random_state  = 42
  n_jobs        = -1

Usage
-----
    python 04_train_classical_models.py            # Trains on FD001
    python 04_train_classical_models.py FD003      # Trains on FD003

Outputs (saved to ../models/):
    xgboost_{subset}.joblib
    random_forest_{subset}.joblib
==============================================================================
"""

import os
import sys
import time

import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Allow running from project root or from src/ directory
# Allow importing sibling src/ files by their actual filenames (Python can't
# import filenames starting with a digit via normal "import" statement,
# so we use importlib to load 01_data_preprocessing.py by path.)
import importlib.util as _ilu
import os as _os

def _load_module(filepath, module_name):
    spec = _ilu.spec_from_file_location(module_name, filepath)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_src_dir = _os.path.dirname(_os.path.abspath(__file__))
_prep    = _load_module(_os.path.join(_src_dir, "01_data_preprocessing.py"), "data_preprocessing")

load_cmapss_raw           = _prep.load_cmapss_raw
add_remaining_useful_life = _prep.add_remaining_useful_life
get_scaler                = _prep.get_scaler
transform_data            = _prep.transform_data
engineer_tabular_features = _prep.engineer_tabular_features
nasa_phm_score            = _prep.nasa_phm_score


def train_classical_models(
    data_dir: str = "data",
    models_dir: str = "models",
    subset: str = "FD001",
):
    """
    Full training pipeline for XGBoost and Random Forest on a CMAPSS subset.

    Steps
    -----
    1. Load raw CMAPSS files.
    2. Compute piecewise-linear RUL targets.
    3. Fit StandardScaler on training set only.
    4. Engineer 120-dimensional tabular features from sliding windows.
    5. Train XGBoost and Random Forest.
    6. Save models as .joblib artifacts.
    7. Evaluate on official NASA test set.

    Parameters
    ----------
    data_dir : str
        Directory containing CMAPSS text files.
    models_dir : str
        Directory to save trained model files.
    subset : str
        One of "FD001", "FD002", "FD003", "FD004".
    """
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Training Classical ML Models — NASA C-MAPSS {subset}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Step 1 & 2: Load and prepare data
    # ------------------------------------------------------------------
    print("\n[1/4] Loading raw data...")
    train_raw = load_cmapss_raw(data_dir, subset, "train")
    test_raw  = load_cmapss_raw(data_dir, subset, "test")
    rul_raw   = load_cmapss_raw(data_dir, subset, "RUL")
    y_test    = rul_raw["rul"].values

    train_df  = add_remaining_useful_life(train_raw)

    print(f"       Training engines: {train_raw['unit_nr'].nunique()}")
    print(f"       Test engines    : {test_raw['unit_nr'].nunique()}")

    # ------------------------------------------------------------------
    # Step 3: Normalize (training statistics only — no data leakage)
    # ------------------------------------------------------------------
    print("\n[2/4] Fitting StandardScaler on training set...")
    scaler       = get_scaler(train_df)
    train_scaled = transform_data(train_df, scaler)
    test_scaled  = transform_data(test_raw, scaler)

    # Save scaler (also used by deep learning pipeline)
    scaler_path = os.path.join(models_dir, f"scaler_{subset}.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"       Scaler saved → {scaler_path}")

    # ------------------------------------------------------------------
    # Step 4: Feature engineering (120-dim tabular vectors)
    # ------------------------------------------------------------------
    print("\n[3/4] Engineering tabular features (5 stats × 24 features = 120)...")

    X_train, y_train = engineer_tabular_features(train_scaled, is_train=True)
    X_test, _        = engineer_tabular_features(test_scaled,  is_train=False)

    print(f"       X_train shape: {X_train.shape}  |  y_train shape: {y_train.shape}")
    print(f"       X_test shape : {X_test.shape}")

    # ------------------------------------------------------------------
    # Step 5 & 6: Train models and save
    # ------------------------------------------------------------------
    print("\n[4/4] Training and evaluating classical models...\n")
    results = []

    # ---- XGBoost --------------------------------------------------------
    print("  [XGBoost] Fitting gradient boosted trees...")
    xgb_path = os.path.join(models_dir, f"xgboost_{subset}.joblib")

    t0 = time.time()
    xgb_model = XGBRegressor(
        n_estimators  = 150,
        max_depth     = 5,
        learning_rate = 0.05,
        random_state  = 42,
        n_jobs        = -1,
    )
    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - t0

    joblib.dump(xgb_model, xgb_path)
    print(f"  [XGBoost] Trained in {xgb_time:.1f}s  →  saved to {xgb_path}")

    xgb_preds = np.clip(xgb_model.predict(X_test), 0, 150)
    xgb_rmse  = float(np.sqrt(np.mean((xgb_preds - y_test) ** 2)))
    xgb_mae   = float(np.mean(np.abs(xgb_preds - y_test)))
    xgb_r2    = float(1 - np.sum((y_test - xgb_preds)**2) / np.sum((y_test - np.mean(y_test))**2))
    xgb_phm   = nasa_phm_score(y_test, xgb_preds)

    print(f"  [XGBoost] RMSE={xgb_rmse:.2f}  MAE={xgb_mae:.2f}  R²={xgb_r2:.3f}  PHM={xgb_phm:.1f}\n")
    results.append({"model": "XGBoost", "rmse": xgb_rmse, "mae": xgb_mae, "r2": xgb_r2, "phm": xgb_phm})

    # ---- Random Forest --------------------------------------------------
    print("  [Random Forest] Fitting ensemble of decision trees...")
    rf_path = os.path.join(models_dir, f"random_forest_{subset}.joblib")

    t0 = time.time()
    rf_model = RandomForestRegressor(
        n_estimators = 150,
        max_depth    = 12,
        random_state = 42,
        n_jobs       = -1,
    )
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - t0

    joblib.dump(rf_model, rf_path)
    print(f"  [Random Forest] Trained in {rf_time:.1f}s  →  saved to {rf_path}")

    rf_preds = np.clip(rf_model.predict(X_test), 0, 150)
    rf_rmse  = float(np.sqrt(np.mean((rf_preds - y_test) ** 2)))
    rf_mae   = float(np.mean(np.abs(rf_preds - y_test)))
    rf_r2    = float(1 - np.sum((y_test - rf_preds)**2) / np.sum((y_test - np.mean(y_test))**2))
    rf_phm   = nasa_phm_score(y_test, rf_preds)

    print(f"  [Random Forest] RMSE={rf_rmse:.2f}  MAE={rf_mae:.2f}  R²={rf_r2:.3f}  PHM={rf_phm:.1f}\n")
    results.append({"model": "Random Forest", "rmse": rf_rmse, "mae": rf_mae, "r2": rf_r2, "phm": rf_phm})

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("Summary")
    print("-" * 60)
    print(f"{'Model':<20}  {'RMSE':>6}  {'MAE':>6}  {'R²':>6}  {'PHM Score':>10}")
    print("-" * 60)
    for r in results:
        print(f"  {r['model']:<18}  {r['rmse']:>6.2f}  {r['mae']:>6.2f}  {r['r2']:>6.3f}  {r['phm']:>10.1f}")
    print("-" * 60)


if __name__ == "__main__":
    subset = sys.argv[1] if len(sys.argv) > 1 else "FD001"

    project_root = os.path.join(os.path.dirname(__file__), "..")
    data_dir     = os.path.join(project_root, "data")
    models_dir   = os.path.join(project_root, "models")

    train_classical_models(data_dir=data_dir, models_dir=models_dir, subset=subset)
