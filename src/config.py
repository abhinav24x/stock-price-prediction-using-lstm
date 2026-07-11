"""
Configuration settings for the Stock Price Prediction System.
"""

import os

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Default Hyperparameters
DEFAULT_TICKER = "AAPL"
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_SEQ_LEN = 60         # Number of previous days to use for prediction
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT = 0.2
DEFAULT_LSTM_UNITS = [100, 50]  # Deep LSTM architecture (two layers)

# Features to use for training (Target variable MUST be the last one, or we handle it explicitly. 
# Here, we will use these features and predict 'Close')
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'RSI', 'MACD', 'EMA_20', 'SMA_50', 'BB_High', 'BB_Low', 'ATR'
]

TARGET_COLUMN = 'Close'
