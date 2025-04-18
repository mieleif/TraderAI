import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from config import TrainingConfig
import numpy as np

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def build_model(state_size, action_size):
    """
    Builds and returns an optimized DQN model with deterministic behavior
    """
    # Configure for deterministic operations
    tf.config.experimental.enable_op_determinism()
    
    model = Sequential([
        Input(shape=(state_size,)),
        Dense(32, activation='relu',
              kernel_initializer=tf.keras.initializers.HeNormal(seed=42)),
        Dense(16, activation='relu',
              kernel_initializer=tf.keras.initializers.HeNormal(seed=42)),
        Dense(action_size, activation='linear',
              kernel_initializer=tf.keras.initializers.HeNormal(seed=42))
    ])
    
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=TrainingConfig.LEARNING_RATE)
    )
    
    return model
