"""
==============================================================================
 NASA C-MAPSS Turbofan Engine RUL Prediction
 Module 3: Deep Learning Model Training Pipeline
==============================================================================

Trains three deep learning architectures on NASA C-MAPSS data:
  1. CNN_Temporal      (1D Temporal CNN)
  2. CNN_LSTM          (Convolutional + Recurrent)
  3. CNN_LSTM_Attention (CNN + LSTM + Multi-Head Self-Attention)

Training Protocol
-----------------
- Optimizer : Adam (lr=0.001)
- Loss      : Mean Squared Error (regression)
- Epochs    : 20 (CNN, CNN-LSTM), 25 (Attention)
- Batch size: 64
- Validation: 15% holdout from training set
- Early Stopping: patience=5 (restore best weights)
- Input shape: (50, 24) → 50-cycle window × 24 features

Usage
-----
    python 03_train_deep_learning.py               # Trains on FD001
    python 03_train_deep_learning.py FD002         # Trains on FD002

Outputs (saved to ../models/):
    cnn_{subset}.keras
    cnn_lstm_{subset}.keras
    cnn_lstm_attention_{subset}.keras   (skips if pre-trained model exists)
==============================================================================
"""

import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from tensorflow import keras

# Import from the same src/ directory; adjust path if running from project root
# sys.path setup handled above via importlib

# Allow importing sibling src/ files by their actual filenames (Python can't
# import filenames starting with a digit via normal "import" statement,
# so we use importlib instead, or add sys.path to import from backend/ directly).
# For standalone teaching use, copy functions inline or run via train_models.py.
import importlib.util as _ilu
import os as _os

def _load_module(filepath, module_name):
    spec = _ilu.spec_from_file_location(module_name, filepath)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_src_dir = _os.path.dirname(_os.path.abspath(__file__))
_arch   = _load_module(_os.path.join(_src_dir, "02_model_architectures.py"), "model_architectures")
_prep   = _load_module(_os.path.join(_src_dir, "01_data_preprocessing.py"),  "data_preprocessing")

build_cnn_model                = _arch.build_cnn_model
build_cnn_lstm_model           = _arch.build_cnn_lstm_model
build_cnn_lstm_attention_model = _arch.build_cnn_lstm_attention_model

load_cmapss_raw          = _prep.load_cmapss_raw
add_remaining_useful_life= _prep.add_remaining_useful_life
get_scaler               = _prep.get_scaler
transform_data           = _prep.transform_data
generate_sequences       = _prep.generate_sequences
generate_test_last_windows = _prep.generate_test_last_windows
WINDOW_SIZE              = _prep.WINDOW_SIZE
FEATURE_COLS             = _prep.FEATURE_COLS


def train_deep_learning_models(
    data_dir: str = "data",
    models_dir: str = "models",
    subset: str = "FD001",
):
    """
    Full deep learning training pipeline for a single CMAPSS subset.

    Steps
    -----
    1. Load raw train/test/RUL text files.
    2. Compute piecewise-linear RUL targets for training set.
    3. Fit StandardScaler on training set only (zero data leakage).
    4. Generate 50-cycle sliding windows: X_train (N×50×24), y_train (N,).
    5. Extract last 50-cycle window for each test engine: X_test (100×50×24).
    6. For each architecture:
        a. Build the Keras model.
        b. Train with early stopping (patience=5).
        c. Save trained weights to .keras file.
        d. Run inference on X_test.
        e. Print RMSE, MAE on official test set.

    Parameters
    ----------
    data_dir : str
        Path to directory containing CMAPSS .txt files.
    models_dir : str
        Path to save trained model files.
    subset : str
        One of "FD001", "FD002", "FD003", "FD004".
    """
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Training Deep Learning Models — NASA C-MAPSS {subset}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Step 1 & 2: Load data and compute RUL targets
    # ------------------------------------------------------------------
    print("\n[1/5] Loading raw data...")
    train_raw = load_cmapss_raw(data_dir, subset, "train")
    test_raw  = load_cmapss_raw(data_dir, subset, "test")
    rul_raw   = load_cmapss_raw(data_dir, subset, "RUL")
    y_test    = rul_raw["rul"].values

    train_df = add_remaining_useful_life(train_raw)
    print(f"       Training engines   : {train_raw['unit_nr'].nunique()}")
    print(f"       Total training rows: {len(train_raw)}")
    print(f"       Test engines       : {test_raw['unit_nr'].nunique()}")

    # ------------------------------------------------------------------
    # Step 3: Fit scaler on training data only
    # ------------------------------------------------------------------
    print("\n[2/5] Fitting StandardScaler (training set only, no data leakage)...")
    scaler       = get_scaler(train_df)
    train_scaled = transform_data(train_df, scaler)
    test_scaled  = transform_data(test_raw, scaler)

    # ------------------------------------------------------------------
    # Step 4 & 5: Generate sequence tensors
    # ------------------------------------------------------------------
    print("\n[3/5] Generating sliding-window sequences...")
    X_train, y_train = generate_sequences(train_scaled)
    X_test, _        = generate_test_last_windows(test_scaled)
    print(f"       X_train: {X_train.shape}  (windows × time-steps × features)")
    print(f"       y_train: {y_train.shape}")
    print(f"       X_test : {X_test.shape}   (one window per test engine)")

    # ------------------------------------------------------------------
    # Step 6: Train and evaluate each architecture
    # ------------------------------------------------------------------
    architectures = [
        {
            "name"      : "CNN",
            "build_fn"  : build_cnn_model,
            "epochs"    : 20,
            "save_path" : os.path.join(models_dir, f"cnn_{subset}.keras"),
        },
        {
            "name"      : "CNN-LSTM",
            "build_fn"  : build_cnn_lstm_model,
            "epochs"    : 20,
            "save_path" : os.path.join(models_dir, f"cnn_lstm_{subset}.keras"),
        },
        {
            "name"      : "CNN-LSTM-Attention",
            "build_fn"  : build_cnn_lstm_attention_model,
            "epochs"    : 25,
            "save_path" : os.path.join(models_dir, f"cnn_lstm_attention_{subset}.keras"),
        },
    ]

    print("\n[4/5] Training architectures...\n")
    results = []

    for arch in architectures:
        name       = arch["name"]
        save_path  = arch["save_path"]

        # Skip training if pre-trained artifact already exists
        if os.path.exists(save_path):
            print(f"  [{name}] Pre-trained model found at {save_path} — loading...")
            model = keras.models.load_model(save_path)
        else:
            print(f"  [{name}] Training from scratch...")
            model = arch["build_fn"]((WINDOW_SIZE, len(FEATURE_COLS)))

            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True,
                    verbose=1,
                )
            ]

            t_start = time.time()
            history = model.fit(
                X_train, y_train,
                epochs          = arch["epochs"],
                batch_size      = 64,
                validation_split= 0.15,         # Reserve 15% of training data for validation
                callbacks       = callbacks,
                verbose         = 1,
            )
            elapsed = time.time() - t_start
            print(f"  [{name}] Training complete in {elapsed:.1f}s")
            model.save(save_path)
            print(f"  [{name}] Model saved → {save_path}")

        # Inference on official test set
        preds = model.predict(X_test, verbose=0).flatten()
        preds = np.clip(preds, 0, 150)   # Physical bounds: RUL in [0, 150]

        rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
        mae  = float(np.mean(np.abs(preds - y_test)))
        ss_res = np.sum((y_test - preds) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2   = float(1 - ss_res / ss_tot)

        print(f"  [{name}] Test RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}\n")
        results.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("[5/5] Summary")
    print("-" * 50)
    print(f"{'Model':<30}  {'RMSE':>6}  {'MAE':>6}  {'R²':>6}")
    print("-" * 50)
    for r in results:
        print(f"  {r['model']:<28}  {r['rmse']:>6.2f}  {r['mae']:>6.2f}  {r['r2']:>6.3f}")
    print("-" * 50)
    best = min(results, key=lambda x: x["rmse"])
    print(f"\n  Best model: {best['model']}  (RMSE={best['rmse']:.2f})")


if __name__ == "__main__":
    subset = sys.argv[1] if len(sys.argv) > 1 else "FD001"

    # Paths relative to project root
    project_root = os.path.join(os.path.dirname(__file__), "..")
    data_dir     = os.path.join(project_root, "data")
    models_dir   = os.path.join(project_root, "models")

    train_deep_learning_models(data_dir=data_dir, models_dir=models_dir, subset=subset)
