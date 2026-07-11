"""
Training loop and callback configurations.
"""

import os
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from src.utils import setup_logger, fetch_stock_data, add_technical_indicators
from src.preprocessing import DataPreprocessor
from src.model import build_lstm_model
from src.config import (
    FEATURE_COLUMNS, TARGET_COLUMN, MODELS_DIR, 
    DEFAULT_SEQ_LEN, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, 
    DEFAULT_LEARNING_RATE, DEFAULT_DROPOUT, DEFAULT_LSTM_UNITS
)

logger = setup_logger(__name__)

def train_model(ticker: str, 
                start_date: str, 
                end_date: str = None, 
                seq_len: int = DEFAULT_SEQ_LEN,
                epochs: int = DEFAULT_EPOCHS,
                batch_size: int = DEFAULT_BATCH_SIZE,
                learning_rate: float = DEFAULT_LEARNING_RATE,
                dropout: float = DEFAULT_DROPOUT,
                lstm_units: list = DEFAULT_LSTM_UNITS):
    """
    End-to-end pipeline to fetch data, preprocess, build, and train the model.
    """
    logger.info(f"--- Starting Training Pipeline for {ticker} ---")
    
    # 1. Fetch Data
    df = fetch_stock_data(ticker, start_date, end_date)
    
    # 2. Add Technical Indicators
    df = add_technical_indicators(df)
    
    # 3. Preprocess Data
    preprocessor = DataPreprocessor(FEATURE_COLUMNS, TARGET_COLUMN)
    scaled_data = preprocessor.fit_transform(df, ticker)
    
    # Target column index
    target_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)
    
    # Create Sequences
    X, y = preprocessor.create_sequences(scaled_data, seq_len, target_col_idx=target_idx)
    
    # Train-Test Split (Sequential, not random for time series)
    # We will use 80% for training and 20% for validation
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
    
    # 4. Build Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(
        input_shape=input_shape,
        units=lstm_units,
        dropout=dropout,
        learning_rate=learning_rate
    )
    
    # 5. Callbacks
    model_save_path = os.path.join(MODELS_DIR, f"{ticker}_best_model.keras")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    # 6. Train Model
    logger.info("Starting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info(f"Training completed. Best model saved to {model_save_path}")
    return history, model
