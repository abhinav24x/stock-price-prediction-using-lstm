"""
Inference and Evaluation module.
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from src.utils import setup_logger, ModelLoadError
from src.config import MODELS_DIR, FEATURE_COLUMNS, TARGET_COLUMN, DEFAULT_SEQ_LEN
from src.preprocessing import DataPreprocessor

logger = setup_logger(__name__)

def load_trained_model_and_scalers(ticker: str):
    """Loads the Keras model and the fitted scalers for a given ticker."""
    model_path = os.path.join(MODELS_DIR, f"{ticker}_best_model.keras")
    scaler_X_path = os.path.join(MODELS_DIR, f"{ticker}_feature_scaler.pkl")
    scaler_y_path = os.path.join(MODELS_DIR, f"{ticker}_target_scaler.pkl")
    
    if not all(os.path.exists(p) for p in [model_path, scaler_X_path, scaler_y_path]):
        raise ModelLoadError(f"Model or scalers for {ticker} not found. Please train the model first.")
        
    model = load_model(model_path)
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    
    return model, scaler_X, scaler_y


def evaluate_model(ticker: str, df: pd.DataFrame, seq_len: int = DEFAULT_SEQ_LEN):
    """
    Evaluates the model on a given dataset (e.g., validation set or recent data)
    and returns evaluation metrics.
    """
    logger.info(f"Evaluating model for {ticker}...")
    model, scaler_X, scaler_y = load_trained_model_and_scalers(ticker)
    
    preprocessor = DataPreprocessor(FEATURE_COLUMNS, TARGET_COLUMN)
    # Mocking the loaded scaler to the preprocessor
    preprocessor.feature_scaler = scaler_X
    preprocessor.target_scaler = scaler_y
    
    scaled_data = preprocessor.transform(df, ticker)
    target_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)
    
    X, y_true_scaled = preprocessor.create_sequences(scaled_data, seq_len, target_col_idx=target_idx)
    
    y_pred_scaled = model.predict(X)
    
    # Inverse transform
    y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1))
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    logger.info(f"Evaluation Metrics: {metrics}")
    
    # Return metrics and data for plotting
    return metrics, y_true, y_pred


def predict_future(ticker: str, df: pd.DataFrame, days_to_predict: int = 1, seq_len: int = DEFAULT_SEQ_LEN):
    """
    Predicts future stock prices based on the most recent data.
    Note: For days_to_predict > 1, this uses its own predictions iteratively.
    For more complex feature sets (like TA), this simple iterative approach 
    keeps other features constant or requires predicting them too.
    For simplicity and robustness in this demo, we will iteratively predict the 'Close' 
    and keep other features at their last known value, which is a standard approximation.
    """
    logger.info(f"Predicting next {days_to_predict} days for {ticker}...")
    model, scaler_X, scaler_y = load_trained_model_and_scalers(ticker)
    
    preprocessor = DataPreprocessor(FEATURE_COLUMNS, TARGET_COLUMN)
    preprocessor.feature_scaler = scaler_X
    preprocessor.target_scaler = scaler_y
    
    # Transform all data to get the last sequence
    scaled_data = preprocessor.transform(df, ticker)
    
    # Take the last `seq_len` days to predict the next day
    last_sequence = scaled_data[-seq_len:]
    target_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)
    
    predictions_scaled = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_to_predict):
        # Predict the next step
        # Reshape to (1, seq_len, num_features)
        current_sequence_reshaped = current_sequence.reshape(1, seq_len, len(FEATURE_COLUMNS))
        
        next_pred_scaled = model.predict(current_sequence_reshaped)[0][0]
        predictions_scaled.append(next_pred_scaled)
        
        # We need to append the new prediction to the sequence and pop the oldest
        # We duplicate the last day's features, but update the 'Close' price
        new_step = current_sequence[-1].copy()
        new_step[target_idx] = next_pred_scaled
        
        # Shift sequence
        current_sequence = np.append(current_sequence[1:], [new_step], axis=0)
        
    # Inverse transform predictions
    predictions_scaled_arr = np.array(predictions_scaled).reshape(-1, 1)
    predictions = scaler_y.inverse_transform(predictions_scaled_arr).flatten()
    
    # Generate future dates (approximate, ignoring weekends/holidays for simplicity here)
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict, freq='B')
    
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': predictions
    })
    
    return pred_df
