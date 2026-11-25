"""
==============================================================================
 NASA C-MAPSS Turbofan Engine RUL Prediction
 Module 6: Inference — Load Model and Predict RUL for Any Engine
==============================================================================

Demonstrates how to load a trained model and generate a Remaining Useful Life
(RUL) prediction for a specific engine from the NASA C-MAPSS test dataset.

This module mirrors the prediction logic used by the FastAPI backend (backend/main.py)
and can be used as a standalone inference script without the web server.

Workflow
--------
  1. Load test engine telemetry from the NASA dataset.
  2. Apply the pre-fitted StandardScaler (trained on training data only).
  3. Extract the last 50-cycle operational window.
  4. Feed the window into the loaded model.
  5. Post-process prediction (clip to [0, 150]).
  6. Display RUL prediction and maintenance urgency classification.

Maintenance Urgency Classification
------------------------------------
  RUL <= 20  → CRITICAL  (Immediate grounding & teardown inspection)
  RUL <= 50  → WARNING   (High-priority maintenance within next service window)
  RUL <= 85  → MONITOR   (Accelerated telemetry monitoring)
  RUL > 85   → OPTIMAL   (Nominal flight operational envelope)

Usage
-----
    # Predict RUL for engine #1 in FD001 using CNN-LSTM-Attention model
    python 06_inference.py

    # Predict for engine #42 in FD003
    python 06_inference.py --subset FD003 --engine 42 --model "CNN-LSTM"

Supported Model Names
---------------------
    "CNN-LSTM with Self-Attention"   (default, best model)
    "CNN-LSTM"
    "CNN"
    "XGBoost"
    "Random Forest"
==============================================================================
"""

import os
import sys
import argparse

import numpy as np
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow import keras

# Allow importing sibling src/ files by their actual filenames
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
transform_data            = _prep.transform_data
engineer_tabular_features = _prep.engineer_tabular_features
FEATURE_COLS              = _prep.FEATURE_COLS
WINDOW_SIZE               = _prep.WINDOW_SIZE


# =============================================================================
# Health Status Classification
# =============================================================================

def classify_health_status(rul: float) -> dict:
    """
    Classify engine health urgency based on predicted RUL.

    Parameters
    ----------
    rul : float
        Predicted Remaining Useful Life in operational cycles.

    Returns
    -------
    dict
        status : str   — Urgency level (CRITICAL / WARNING / MONITOR / OPTIMAL)
        action : str   — Recommended maintenance action
    """
    if rul <= 20:
        return {
            "status": "CRITICAL",
            "action": "Immediate grounding & teardown inspection recommended.",
        }
    elif rul <= 50:
        return {
            "status": "WARNING",
            "action": "Schedule high-priority maintenance cycle within next servicing window.",
        }
    elif rul <= 85:
        return {
            "status": "MONITOR",
            "action": "Accelerated telemetry monitoring; nominal degradation detected.",
        }
    else:
        return {
            "status": "OPTIMAL",
            "action": "Nominal flight operational envelope. Fleet airworthiness approved.",
        }


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_name: str, subset: str, models_dir: str):
    """
    Load a trained model artifact from the models directory.

    Parameters
    ----------
    model_name : str
        One of: "CNN", "CNN-LSTM", "CNN-LSTM with Self-Attention",
                "XGBoost", "Random Forest".
    subset : str
        One of: "FD001", "FD002", "FD003", "FD004".
    models_dir : str
        Path to directory containing model artifacts.

    Returns
    -------
    model : keras.Model or sklearn estimator
        Loaded model ready for prediction.
    model_type : str
        "keras" or "sklearn".
    """
    model_map = {
        "CNN"                         : (f"cnn_{subset}.keras",                  "keras"),
        "CNN-LSTM"                    : (f"cnn_lstm_{subset}.keras",              "keras"),
        "CNN-LSTM with Self-Attention": (f"cnn_lstm_attention_{subset}.keras",    "keras"),
        "XGBoost"                     : (f"xgboost_{subset}.joblib",              "sklearn"),
        "Random Forest"               : (f"random_forest_{subset}.joblib",        "sklearn"),
    }

    if model_name not in model_map:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Choose from: {list(model_map.keys())}"
        )

    filename, model_type = model_map[model_name]
    model_path = os.path.join(models_dir, filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}\n"
            f"Train the model first using the training scripts."
        )

    if model_type == "keras":
        model = keras.models.load_model(model_path)
    else:
        model = joblib.load(model_path)

    return model, model_type


# =============================================================================
# Core Inference Function
# =============================================================================

def predict_rul(
    subset: str,
    engine_id: int,
    model_name: str,
    data_dir: str,
    models_dir: str,
) -> dict:
    """
    Generate a RUL prediction for a specific engine.

    This function implements the identical inference pipeline used by the
    FastAPI backend endpoint POST /api/predict-sample.

    Parameters
    ----------
    subset : str
        CMAPSS subset ID (e.g., "FD001").
    engine_id : int
        Engine unit number (1-indexed, as in the dataset).
    model_name : str
        Name of the trained model to use for prediction.
    data_dir : str
        Path to CMAPSS data files.
    models_dir : str
        Path to trained model artifacts.

    Returns
    -------
    dict
        predicted_rul   : float — Model's RUL estimate (cycles)
        actual_rul      : float — Ground-truth RUL from RUL_FDxxx.txt
        residual        : float — predicted - actual (positive = late, negative = early)
        health_status   : str   — CRITICAL / WARNING / MONITOR / OPTIMAL
        action          : str   — Recommended maintenance action
        model           : str   — Model name used
        engine_id       : int
        subset          : str
    """
    # 1. Load test data for the specific engine
    test_df = load_cmapss_raw(data_dir, subset, "test")
    unit_df = test_df[test_df["unit_nr"] == engine_id]

    if unit_df.empty:
        raise ValueError(f"Engine {engine_id} not found in {subset} test set.")

    # 2. Load ground-truth RUL
    rul_df     = load_cmapss_raw(data_dir, subset, "RUL")
    actual_rul = float(rul_df.iloc[engine_id - 1]["rul"])

    # 3. Load pre-fitted scaler and normalize
    scaler_path = os.path.join(models_dir, f"scaler_{subset}.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)

    unit_scaled = transform_data(unit_df, scaler)

    # 4. Extract the last 50-cycle operational window
    data_arr   = unit_scaled[FEATURE_COLS].values
    num_cycles = len(data_arr)

    if num_cycles >= WINDOW_SIZE:
        window = data_arr[-WINDOW_SIZE:]                           # Last 50 cycles
    else:
        # Front-pad with the first observed cycle for short trajectories
        pad_len = WINDOW_SIZE - num_cycles
        pad     = np.tile(data_arr[0], (pad_len, 1))
        window  = np.vstack([pad, data_arr])

    # 5. Load the model and run prediction
    model, model_type = load_model(model_name, subset, models_dir)

    if model_type == "keras":
        # Deep learning: needs 3D input (1, 50, 24)
        x_input      = window[np.newaxis, :, :]        # Shape: (1, 50, 24)
        predicted_rul = float(model.predict(x_input, verbose=0).flatten()[0])
    else:
        # Classical ML: needs 2D tabular features (1, 120)
        # Re-use engineer_tabular_features logic for the single engine
        mean_f = np.mean(window, axis=0)
        std_f  = np.std(window, axis=0)
        min_f  = np.min(window, axis=0)
        max_f  = np.max(window, axis=0)
        last_f = window[-1]
        x_tab  = np.concatenate([mean_f, std_f, min_f, max_f, last_f]).reshape(1, -1)
        predicted_rul = float(model.predict(x_tab)[0])

    # 6. Clip to physical bounds and classify
    predicted_rul = float(np.clip(predicted_rul, 0, 150))
    residual      = round(predicted_rul - actual_rul, 1)
    health_info   = classify_health_status(predicted_rul)

    return {
        "subset"       : subset,
        "engine_id"    : engine_id,
        "model"        : model_name,
        "predicted_rul": round(predicted_rul, 1),
        "actual_rul"   : actual_rul,
        "residual"     : residual,
        "health_status": health_info["status"],
        "action"       : health_info["action"],
        "total_test_cycles_recorded": num_cycles,
    }


# =============================================================================
# Command-Line Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Predict RUL for a NASA C-MAPSS engine using a trained model."
    )
    parser.add_argument("--subset",  default="FD001",
                        choices=["FD001", "FD002", "FD003", "FD004"],
                        help="CMAPSS dataset subset (default: FD001)")
    parser.add_argument("--engine",  type=int, default=1,
                        help="Engine unit number to predict (default: 1)")
    parser.add_argument("--model",   default="CNN-LSTM with Self-Attention",
                        help='Model name (default: "CNN-LSTM with Self-Attention")')
    args = parser.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), "..")
    data_dir     = os.path.join(project_root, "data")
    models_dir   = os.path.join(project_root, "models")

    print(f"\n  Predicting RUL for Engine #{args.engine} in {args.subset} using {args.model}...")

    result = predict_rul(
        subset     = args.subset,
        engine_id  = args.engine,
        model_name = args.model,
        data_dir   = data_dir,
        models_dir = models_dir,
    )

    print(f"\n{'='*55}")
    print(f"  NASA C-MAPSS RUL Prediction Report")
    print(f"{'='*55}")
    print(f"  Subset               : {result['subset']}")
    print(f"  Engine ID            : #{result['engine_id']}")
    print(f"  Model                : {result['model']}")
    print(f"  Recorded Cycles      : {result['total_test_cycles_recorded']}")
    print(f"  Predicted RUL        : {result['predicted_rul']:.1f} cycles")
    print(f"  Actual RUL (ground)  : {result['actual_rul']:.0f} cycles")
    print(f"  Residual (pred-true) : {result['residual']:+.1f} cycles")
    print(f"  Health Status        : {result['health_status']}")
    print(f"  Recommended Action   : {result['action']}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
