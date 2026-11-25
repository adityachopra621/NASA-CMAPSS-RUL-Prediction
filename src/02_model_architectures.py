"""
==============================================================================
 NASA C-MAPSS Turbofan Engine RUL Prediction
 Module 2: Deep Learning Model Architectures
==============================================================================

Three architectures are implemented and benchmarked:

  1. CNN_Temporal      — Pure 1D Temporal Convolutional Network
  2. CNN_LSTM          — Convolutional feature extractor + LSTM sequence encoder
  3. CNN_LSTM_Attention — CNN-LSTM with Multi-Head Self-Attention (best performer)

All models:
  - Input shape: (50, 24) → 50-cycle window, 24 features (3 settings + 21 sensors)
  - Output: single scalar (predicted RUL in cycles)
  - Loss: Mean Squared Error (MSE)
  - Optimizer: Adam (lr=0.001)

Architecture Design Rationale
------------------------------
1D CNNs are effective at learning local temporal patterns (short-term degradation trends).
LSTMs capture long-range sequential dependencies across the operational trajectory.
Multi-Head Self-Attention allows the model to focus on the most informative
time-steps within the 50-cycle window, regardless of their position.

The attention mechanism here is a residual self-attention block applied ON TOP of
the LSTM outputs (transformer-style), forming a hybrid architecture that
outperforms pure CNN or pure LSTM baselines on all four CMAPSS subsets.
==============================================================================
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =============================================================================
# Architecture 1: 1D Temporal CNN
# =============================================================================

def build_cnn_model(input_shape: tuple) -> keras.Model:
    """
    1D Temporal Convolutional Neural Network for RUL regression.

    Architecture:
        Input (50, 24)
            ↓
        Conv1D(64 filters, kernel=5, padding='same', ReLU)
            ↓
        BatchNormalization
            ↓
        Conv1D(64 filters, kernel=3, padding='same', ReLU)
            ↓
        BatchNormalization
            ↓
        Conv1D(32 filters, kernel=3, padding='same', ReLU)
            ↓
        GlobalAveragePooling1D       ← Collapses temporal dimension
            ↓
        Dense(64, ReLU)
            ↓
        Dropout(0.2)
            ↓
        Dense(1, linear)             ← RUL output

    Design notes:
      - Three Conv1D layers learn local degradation patterns at different scales
        (kernel sizes 5 and 3 capture multi-scale temporal correlations).
      - BatchNormalization stabilizes training and allows higher learning rates.
      - GlobalAveragePooling1D provides a compact summary across time, reducing
        overfitting compared to Flatten() for time-series data.
      - Dropout(0.2) regularizes the final dense layer.

    Parameters
    ----------
    input_shape : tuple
        Shape of one training sample, e.g. (50, 24).

    Returns
    -------
    keras.Model
        Compiled Keras model ready for training.
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1: Wide kernel for coarse temporal patterns
    x = layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    # Block 2: Narrower kernel for fine-grained patterns
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Block 3: Feature compression
    x = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(x)

    # Aggregate over time
    x = layers.GlobalAveragePooling1D()(x)

    # Regression head
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_Temporal")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


# =============================================================================
# Architecture 2: CNN-LSTM Hybrid
# =============================================================================

def build_cnn_lstm_model(input_shape: tuple) -> keras.Model:
    """
    Hybrid CNN-LSTM architecture for RUL regression.

    Architecture:
        Input (50, 24)
            ↓
        Conv1D(64 filters, kernel=3, padding='same', ReLU)
            ↓
        BatchNormalization
            ↓
        LSTM(64 units, return_sequences=False)   ← LSTM reads CNN features sequentially
            ↓
        Dropout(0.2)
            ↓
        Dense(32, ReLU)
            ↓
        Dense(1, linear)                          ← RUL output

    Design notes:
      - Conv1D first extracts spatial/local features from the sensor time-series.
      - LSTM then models the sequential evolution of those features.
      - return_sequences=False: LSTM outputs only the final hidden state,
        which encodes the entire 50-cycle trajectory into a compact vector.
      - This two-stage approach typically outperforms pure CNNs or pure LSTMs
        for prognostics tasks (Babu et al., 2016; Li et al., 2018).

    Parameters
    ----------
    input_shape : tuple
        Shape of one training sample, e.g. (50, 24).

    Returns
    -------
    keras.Model
        Compiled Keras model ready for training.
    """
    inputs = layers.Input(shape=input_shape)

    # CNN feature extractor
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    # LSTM sequence encoder
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)

    # Regression head
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


# =============================================================================
# Architecture 3: CNN-LSTM with Multi-Head Self-Attention (Best Model)
# =============================================================================

def build_cnn_lstm_attention_model(input_shape: tuple) -> keras.Model:
    """
    CNN-LSTM with Residual Multi-Head Self-Attention for RUL regression.

    This is the best-performing architecture in our benchmark, achieving:
      FD001: RMSE=15.46, MAE=11.59, R²=0.862, NASA PHM Score=485.6

    Architecture:
        Input (50, 24)
            ↓
        Conv1D(64 filters, kernel=3, padding='same', ReLU)   [local pattern extraction]
            ↓
        LSTM(64 units, return_sequences=True)                 [sequential context, all steps]
            ↓
        MultiHeadAttention(num_heads=4, key_dim=16)          [attend over all 50 time-steps]
            ↓
        Add([LSTM_output, Attention_output])                  [residual connection]
            ↓
        LayerNormalization                                     [stabilize residual]
            ↓
        GlobalAveragePooling1D                                 [aggregate over time]
            ↓
        Dense(32, ReLU)
            ↓
        Dense(1, linear)                                      ← RUL output

    Design notes:
      - Unlike Architecture 2, LSTM uses return_sequences=True here, outputting
        a hidden state for EVERY time-step (shape: 50 × 64).
      - MultiHeadAttention performs self-attention: each time-step attends to
        all other time-steps, allowing the model to identify WHICH cycles are
        most informative for predicting failure proximity.
      - The residual connection (Add) and LayerNormalization follow the
        standard Transformer block pattern (Vaswani et al., 2017).
      - 4 attention heads with key_dim=16 (total attention dim = 64, matching LSTM).

    Hyperparameters
    ---------------
    num_heads  : 4   (parallel attention heads)
    key_dim    : 16  (dimension per head)
    LSTM units : 64
    CNN filters: 64

    Parameters
    ----------
    input_shape : tuple
        Shape of one training sample, e.g. (50, 24).

    Returns
    -------
    keras.Model
        Compiled Keras model ready for training.
    """
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # Stage 1: Local feature extraction via 1D CNN
    x = layers.Conv1D(
        filters=64, kernel_size=3, padding="same", activation="relu", name="conv1d"
    )(inputs)

    # Stage 2: Sequential modeling via LSTM (return all 50 hidden states)
    lstm_out = layers.LSTM(64, return_sequences=True, name="lstm")(x)

    # Stage 3: Multi-Head Self-Attention over the LSTM sequence
    # Query = Key = Value = lstm_out (self-attention)
    attn_out = layers.MultiHeadAttention(
        num_heads=4, key_dim=16, name="multi_head_attention"
    )(lstm_out, lstm_out)

    # Stage 4: Residual connection + Layer Normalization (Transformer-style block)
    add_out  = layers.Add(name="add")([lstm_out, attn_out])
    norm_out = layers.LayerNormalization(name="layer_normalization")(add_out)

    # Stage 5: Aggregate over time dimension
    gap_out = layers.GlobalAveragePooling1D(name="global_average_pooling1d")(norm_out)

    # Stage 6: Regression head
    dense_out = layers.Dense(32, activation="relu", name="dense")(gap_out)
    outputs   = layers.Dense(1, activation="linear", name="dense_1")(dense_out)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_Attention")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


# =============================================================================
# Architecture Comparison Summary
# =============================================================================

def print_all_summaries():
    """Print model summaries for all three architectures."""
    input_shape = (50, 24)   # 50-cycle window, 24 features

    print("=" * 60)
    print("Architecture 1: CNN_Temporal")
    print("=" * 60)
    build_cnn_model(input_shape).summary()

    print("\n" + "=" * 60)
    print("Architecture 2: CNN_LSTM")
    print("=" * 60)
    build_cnn_lstm_model(input_shape).summary()

    print("\n" + "=" * 60)
    print("Architecture 3: CNN_LSTM_Attention (Best Model)")
    print("=" * 60)
    build_cnn_lstm_attention_model(input_shape).summary()


if __name__ == "__main__":
    print_all_summaries()
