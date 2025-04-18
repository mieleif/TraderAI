import numpy as np
import tensorflow as tf
import random
from collections import deque
from models.network import build_model
from config import TrainingConfig
import logging

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

logger = logging.getLogger(__name__)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=TrainingConfig.MEMORY_SIZE)
        self.gamma = TrainingConfig.GAMMA
        self.epsilon = TrainingConfig.EPSILON
        self.epsilon_min = TrainingConfig.EPSILON_MIN
        self.epsilon_decay = TrainingConfig.EPSILON_DECAY
        self.model = build_model(state_size, action_size)
        self.rng = np.random.default_rng(42)  # Use NumPy's newer random generator

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if self.rng.random() <= self.epsilon:
            return self.rng.integers(self.action_size)
        # Add batch prediction for better performance
        act_values = self.model.predict(state.reshape(1, -1), verbose=0, batch_size=1)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """Memory replay method"""
        # Add debug logging
        logger.debug(f"Replay called with memory size: {len(self.memory)}")
        
        if len(self.memory) < batch_size:
            logger.debug("Skipping replay: not enough memories")
            return
        
        # Use numpy arrays directly
        minibatch = self.rng.choice(len(self.memory), batch_size, replace=False)
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        actions = np.zeros(batch_size, dtype=int)
        rewards = np.zeros(batch_size)
        dones = np.zeros(batch_size)
        
        # Fill arrays
        for i, idx in enumerate(minibatch):
            states[i] = self.memory[idx][0]
            actions[i] = self.memory[idx][1]
            rewards[i] = self.memory[idx][2]
            next_states[i] = self.memory[idx][3]
            dones[i] = self.memory[idx][4]
        
        # Predict all values at once
        targets = self.model.predict(states, batch_size=batch_size, verbose=0)
        next_state_values = self.model.predict(next_states, batch_size=batch_size, verbose=0)
        
        # Update targets
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_state_values[i])
        
        # Train in one batch
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath):
        """Save the model to disk"""
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load the model from disk"""
        self.model = tf.keras.models.load_model(filepath)
