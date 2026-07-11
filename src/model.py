"""
Deep LSTM Model Architecture.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from typing import List, Tuple

from src.utils import setup_logger

logger = setup_logger(__name__)

def build_lstm_model(input_shape: Tuple[int, int], 
                     units: List[int] = [100, 50], 
                     dropout: float = 0.2, 
                     learning_rate: float = 0.001) -> Sequential:
    """
    Builds and compiles a Deep LSTM neural network.
    
    Args:
        input_shape (Tuple[int, int]): (sequence_length, number_of_features)
        units (List[int]): Number of units in each LSTM layer.
        dropout (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for Adam optimizer.
        
    Returns:
        Sequential: Compiled Keras model.
    """
    logger.info(f"Building LSTM Model with {len(units)} layers, units={units}, dropout={dropout}")
    
    model = Sequential()
    
    # First LSTM Layer
    model.add(LSTM(units=units[0], return_sequences=(len(units) > 1), input_shape=input_shape))
    model.add(Dropout(dropout))
    
    # Additional LSTM Layers
    for i in range(1, len(units)):
        return_seq = (i < len(units) - 1)
        model.add(LSTM(units=units[i], return_sequences=return_seq))
        model.add(Dropout(dropout))
        
    # Dense Output Layers
    model.add(Dense(units=25, activation='relu'))
    model.add(Dense(units=1)) # Predict next step (Close price)
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    logger.info("Model compiled successfully.")
    # model.summary(print_fn=logger.info) # Optional: logs model summary
    
    return model
