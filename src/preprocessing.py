"""
Data preprocessing module for scaling and sequence generation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from typing import Tuple, List

from src.utils import setup_logger
from src.config import MODELS_DIR

logger = setup_logger(__name__)

class DataPreprocessor:
    """Handles data scaling, missing values, and sequence creation for LSTM."""
    
    def __init__(self, feature_columns: List[str], target_column: str):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        
    def fit_transform(self, df: pd.DataFrame, ticker: str) -> np.ndarray:
        """
        Cleans data, fits the scalers, transforms the data, and saves scalers.
        """
        logger.info(f"Preprocessing data for {ticker}...")
        
        # Ensure we only use the selected features
        df_selected = df[self.feature_columns].copy()
        
        # Handle missing values by forward filling then backward filling
        df_selected.fillna(method='ffill', inplace=True)
        df_selected.fillna(method='bfill', inplace=True)
        
        # We need to scale features and target separately to inverse_transform the target later easily
        features_data = df_selected.values
        target_data = df_selected[[self.target_column]].values
        
        scaled_features = self.feature_scaler.fit_transform(features_data)
        self.target_scaler.fit_transform(target_data) # Fit target scaler
        
        # Save the scalers for future predictions
        scaler_path_X = os.path.join(MODELS_DIR, f"{ticker}_feature_scaler.pkl")
        scaler_path_y = os.path.join(MODELS_DIR, f"{ticker}_target_scaler.pkl")
        
        joblib.dump(self.feature_scaler, scaler_path_X)
        joblib.dump(self.target_scaler, scaler_path_y)
        logger.info(f"Scalers saved to {MODELS_DIR}")
        
        return scaled_features

    def transform(self, df: pd.DataFrame, ticker: str) -> np.ndarray:
        """
        Transforms new data using previously fitted and saved scalers.
        """
        scaler_path_X = os.path.join(MODELS_DIR, f"{ticker}_feature_scaler.pkl")
        if not os.path.exists(scaler_path_X):
            raise FileNotFoundError(f"Scaler for {ticker} not found. Train the model first.")
            
        self.feature_scaler = joblib.load(scaler_path_X)
        
        df_selected = df[self.feature_columns].copy()
        df_selected.fillna(method='ffill', inplace=True)
        df_selected.fillna(method='bfill', inplace=True)
        
        scaled_features = self.feature_scaler.transform(df_selected.values)
        return scaled_features

    def create_sequences(self, data: np.ndarray, seq_len: int, target_col_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates sequences of length `seq_len` for LSTM input.
        
        Args:
            data (np.ndarray): Scaled feature data.
            seq_len (int): Number of time steps to look back.
            target_col_idx (int): Index of the target column in the data array.
            
        Returns:
            X, y arrays.
        """
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i-seq_len:i])
            y.append(data[i, target_col_idx]) # The target value for the next day
            
        X, y = np.array(X), np.array(y)
        logger.info(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
