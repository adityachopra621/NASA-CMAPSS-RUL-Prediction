# NASA C-MAPSS Turbofan Engine Remaining Useful Life (RUL) Prediction

A machine learning benchmarking platform that predicts the **Remaining Useful Life (RUL)** of turbofan aircraft engines using the NASA CMAPSS dataset. Five models are trained and evaluated: three deep learning architectures (1D-CNN, CNN-LSTM, CNN-LSTM with Attention) and two classical ML models (XGBoost, Random Forest).

---

## Project Source Code

The complete, readable Python source code is in the [`src/`](src/) directory:

| File | Description |
|------|-------------|
| [`src/01_data_preprocessing.py`](src/01_data_preprocessing.py) | Dataset loading, RUL target generation, StandardScaler normalization, sliding window sequence generation, tabular feature engineering, NASA PHM scoring function |
| [`src/02_model_architectures.py`](src/02_model_architectures.py) | Keras model definitions: 1D-CNN, CNN-LSTM, CNN-LSTM with Multi-Head Self-Attention |
| [`src/03_train_deep_learning.py`](src/03_train_deep_learning.py) | Deep learning training pipeline with early stopping and model saving |
| [`src/04_train_classical_models.py`](src/04_train_classical_models.py) | XGBoost and Random Forest training with 120-feature window statistics |
| [`src/05_evaluate_models.py`](src/05_evaluate_models.py) | Model evaluation: RMSE, MAE, R², NASA PHM Score benchmark table |
| [`src/06_inference.py`](src/06_inference.py) | Load a trained model and predict RUL for any test engine |

The backend API and frontend source code:

| File | Description |
|------|-------------|
| [`backend/preprocessing.py`](backend/preprocessing.py) | Core preprocessing functions used by the live API |
| [`backend/train_models.py`](backend/train_models.py) | Full 5-model training and benchmarking pipeline |
| [`backend/main.py`](backend/main.py) | FastAPI REST API server |
| [`frontend/src/App.jsx`](frontend/src/App.jsx) | React aerospace dashboard UI |

---

## Dataset: NASA C-MAPSS

**Reference:** Saxena, A., Goebel, K., Simon, D., Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* Proceedings of the PHM Conference.

Four subsets with varying fault modes and operating conditions:

| Subset | Train Engines | Test Engines | Operating Conditions | Fault Modes |
|--------|:---:|:---:|---|---|
| FD001  | 100 | 100 | 1 | 1 (HPC degradation) |
| FD002  | 260 | 259 | 6 | 1 |
| FD003  | 100 | 100 | 1 | 2 (HPC + Fan) |
| FD004  | 249 | 248 | 6 | 2 |

Each row contains **26 columns**: engine ID, cycle number, 3 operational settings, and 21 sensor measurements (s_1 through s_21).

---

## Model Architectures

### Architecture 3: CNN-LSTM with Multi-Head Self-Attention *(Best Model)*

```
Input (50 cycles × 24 features)
    ↓
Conv1D(64 filters, kernel=3, ReLU)      ← Local temporal feature extraction
    ↓
LSTM(64 units, return_sequences=True)   ← Sequential context (all 50 steps)
    ↓
MultiHeadAttention(4 heads, key_dim=16) ← Self-attention over time-steps
    ↓
Add + LayerNormalization                ← Residual connection (Transformer-style)
    ↓
GlobalAveragePooling1D                  ← Aggregate over time
    ↓
Dense(32, ReLU) → Dense(1, linear)      ← RUL prediction
```

### Architecture 2: CNN-LSTM
```
Input (50 × 24) → Conv1D(64, k=3) → BatchNorm → LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)
```

### Architecture 1: 1D Temporal CNN
```
Input (50 × 24) → Conv1D(64, k=5) → BN → Conv1D(64, k=3) → BN → Conv1D(32, k=3)
               → GlobalAvgPool → Dense(64) → Dropout(0.2) → Dense(1)
```

---

## Benchmark Results (FD001 Test Set — 100 Engines)

| Model | RMSE | MAE | R² | NASA PHM Score |
|-------|:----:|:---:|:--:|:--------------:|
| **CNN-LSTM + Attention** | **15.46** | **11.59** | **0.862** | **485.6** |
| CNN-LSTM | 17.30 | 12.98 | 0.827 | 541.1 |
| XGBoost | 18.41 | 13.36 | 0.804 | 742.3 |
| Random Forest | 19.08 | 13.81 | 0.789 | 908.2 |
| CNN (Temporal) | 23.78 | 18.72 | 0.672 | 1782.0 |

> **NASA PHM Score** is an asymmetric metric that penalizes late predictions (overestimating RUL) more severely than early predictions — lower is better.

---

## Preprocessing Pipeline

1. **Load** space-delimited CMAPSS text files into pandas DataFrames
2. **RUL Target Generation**: Piecewise-linear RUL clipped at 125 cycles (standard C-MAPSS benchmark approach)
3. **Normalization**: `StandardScaler` fitted on training set only — zero data leakage
4. **Sequence Generation** (DL models): 50-cycle sliding windows → shape `(N, 50, 24)`
5. **Feature Engineering** (classical ML): 5 statistics × 24 features = **120 tabular features** per window
   - Statistics: mean, std, min, max, last value over 50-cycle window

---

## Project Structure

```
RULproject/
├── src/                          # Readable Python source code (teacher-facing)
│   ├── 01_data_preprocessing.py  # Data loading, RUL generation, feature engineering
│   ├── 02_model_architectures.py # CNN, CNN-LSTM, CNN-LSTM-Attention definitions
│   ├── 03_train_deep_learning.py # Deep learning training pipeline
│   ├── 04_train_classical_models.py  # XGBoost & Random Forest training
│   ├── 05_evaluate_models.py     # Benchmark evaluation metrics
│   └── 06_inference.py           # RUL prediction for any test engine
│
├── backend/                      # FastAPI server
│   ├── main.py                   # REST API endpoints
│   ├── preprocessing.py          # Core preprocessing (used by API)
│   ├── train_models.py           # Full training benchmark script
│   └── requirements.txt          # Python dependencies
│
├── frontend/src/
│   └── App.jsx                   # React aerospace dashboard
│
├── models/                       # Trained model artifacts
│   ├── cnn_FD001.keras           # Trained CNN model
│   ├── cnn_lstm_FD001.keras      # Trained CNN-LSTM model
│   ├── cnn_lstm_attention_FD001.keras  # Best model (attention)
│   ├── xgboost_FD001.joblib      # Trained XGBoost
│   ├── random_forest_FD001.joblib
│   ├── scaler_FD001.joblib       # Fitted StandardScaler
│   └── benchmark_results_FD001.json   # Actual evaluation results
│
├── data/                         # NASA C-MAPSS dataset files
│   ├── train_FD001.txt ... train_FD004.txt
│   ├── test_FD001.txt  ... test_FD004.txt
│   └── RUL_FD001.txt   ... RUL_FD004.txt
│
└── start_app.bat                 # One-click launcher (Windows)
```

---

## Running the Source Code

### Prerequisites
```bash
pip install tensorflow xgboost scikit-learn pandas numpy joblib fastapi uvicorn
```

### Run Preprocessing Demo
```bash
cd src
python 01_data_preprocessing.py
```

### View Model Architectures
```bash
cd src
python 02_model_architectures.py
```

### Train Deep Learning Models
```bash
cd src
python 03_train_deep_learning.py FD001
```

### Train Classical ML Models
```bash
cd src
python 04_train_classical_models.py FD001
```

### Evaluate All Models
```bash
cd src
python 05_evaluate_models.py FD001
```

### Predict RUL for a Specific Engine
```bash
cd src
python 06_inference.py --subset FD001 --engine 1 --model "CNN-LSTM with Self-Attention"
```

### Start the Full Web Application
```bash
start_app.bat
```
Then open http://localhost:5173 in your browser.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow / Keras |
| Classical ML | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Backend API | FastAPI + Uvicorn |
| Frontend | React + Vite |
| Model Serialization | Keras `.keras`, Joblib `.joblib` |

---

## References

1. Saxena, A., Goebel, K., Simon, D., Eklund, N. (2008). Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation. *1st International Conference on Prognostics and Health Management (PHM08)*.
2. Heimes, F. O. (2008). Recurrent neural networks for remaining useful life estimation. *PHM Conference*.
3. Babu, G. S., Zhao, P., & Li, X. L. (2016). Deep convolutional neural network based regression approach for estimation of remaining useful life. *DASFAA*.
4. Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
