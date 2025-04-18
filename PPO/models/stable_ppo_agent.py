import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from collections import deque
import os
import json

class StablePPOAgent:
    def __init__(
        self, 
        state_size, 
        clip_ratio=0.2,
        policy_lr=0.0003,
        value_lr=0.001,
        entropy_beta=0.02,
        gamma=0.95,
        lam=0.95,
        train_epochs=8,
        batch_size=64,  # Increased batch size
        action_bounds=[0.0, 1.0],
        target_kl=0.01  # Early stopping based on KL divergence
    ):
        # Environment parameters
        self.state_size = state_size
        
        # PPO hyperparameters
        self.clip_ratio = clip_ratio
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.entropy_beta = entropy_beta
        self.gamma = gamma
        self.lam = lam
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
        
        # Action space parameters
        self.action_bounds = action_bounds
        self.action_dim = 1
        
        # Memory with larger capacity
        self.memory = deque(maxlen=4000)
        
        # Build actor (policy) and critic (value) networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # Set up logging of training metrics
        self.policy_loss_history = []
        self.value_loss_history = []
        self.entropy_history = []
        self.kl_divergence_history = []
        
        # Use standard optimizers with gradient clipping
        self.actor_optimizer = Adam(learning_rate=self.policy_lr)
        self.critic_optimizer = Adam(learning_rate=self.value_lr)
    
    def remember(self, state, action, action_info, reward, next_state, done):
        """Store trajectory in memory"""
        self.memory.append((state, action, action_info, reward, next_state, done))
    
    def _build_actor(self):
        """Build policy network with proper initialization"""
        # Input layer
        state_input = Input(shape=(self.state_size,))
        
        # Normalize inputs for stability
        x = BatchNormalization()(state_input)
        
        # Hidden layers with better initialization
        x = Dense(64, activation='relu', 
                 kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        
        x = Dense(32, activation='relu',
                 kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        
        # Action mean (mu)
        mu = Dense(self.action_dim, activation='tanh',
                  kernel_initializer=tf.keras.initializers.RandomUniform(
                      minval=-0.003, maxval=0.003))(x)
        
        # Scale mu to action bounds using a more stable approach
        mu = Lambda(lambda x: (x + 1.0) / 2.0 * 
                   (self.action_bounds[1] - self.action_bounds[0]) + 
                   self.action_bounds[0])(mu)
        
        # Separate network branch for log_std with fixed bounds
        log_std = Dense(self.action_dim, activation=None,
                      kernel_initializer=tf.keras.initializers.constant(-1.0))(x)
        
        # Constrain log_std to prevent extreme values
        log_std = Lambda(lambda x: tf.clip_by_value(x, -5.0, 0.0))(log_std)
        
        # Build model
        model = Model(inputs=state_input, outputs=[mu, log_std])
        return model
    
    def _build_critic(self):
        """Build value network with proper initialization"""
        state_input = Input(shape=(self.state_size,))
        
        # Normalize inputs for stability
        x = BatchNormalization()(state_input)
        
        # Hidden layers with better initialization
        x = Dense(128, activation='relu',
                 kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        
        x = Dense(64, activation='relu',
                 kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        
        # Value output
        value = Dense(1, activation=None,
                     kernel_initializer=tf.keras.initializers.RandomUniform(
                         minval=-0.003, maxval=0.003))(x)
        
        # Build model
        model = Model(inputs=state_input, outputs=value)
        return model
    
    def get_action(self, state, training=True):
        """Get action with improved numerical stability"""
        state = np.reshape(state, [1, self.state_size])
        mu, log_std = self.actor.predict(state, verbose=0)
        
        # Debug NaN values
        if np.isnan(mu).any() or np.isnan(log_std).any():
            print(f"WARNING: NaN detected in network output - mu: {mu}, log_std: {log_std}")
            # Fallback to safe values
            mu = np.array([[0.5]])
            log_std = np.array([[-2.0]])
        
        # More stable std calculation
        log_std = np.clip(log_std, -5.0, 0.0)
        std = np.exp(log_std)
        
        if training:
            # Sample from truncated normal distribution for better stability
            z = np.random.normal(0, 1, size=mu.shape)
            action = mu + z * std
            
            # Determine direction more systematically
            direction = 1 if np.random.random() > 0.5 else -1
        else:
            # For evaluation, use deterministic action
            action = mu
            direction = 1 if state[0][0] > 0 else -1
        
        # Clip action to bounds
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        
        # Calculate log probability with improved stability
        log_prob = self._log_prob(action, mu, log_std)
        
        # Create action dictionary
        combined_action = {
            'direction': direction,
            'size': float(action[0][0])
        }
        
        return combined_action, {'mu': mu, 'log_std': log_std, 'log_prob': log_prob}
    
    def _log_prob(self, action, mu, log_std):
        """Numerically stable log probability calculation"""
        # Clip log_std to prevent extreme values
        log_std = np.clip(log_std, -5.0, 0.0)
        std = np.exp(log_std)
        
        # Add epsilon to prevent division by zero
        var = np.square(std) + 1e-8
        
        # More stable log probability calculation
        log_prob = -0.5 * (
            np.log(2.0 * np.pi) + 
            2.0 * log_std + 
            np.square(action - mu) / var
        )
        
        return np.sum(log_prob, axis=-1)
    
    def train(self):
        """Train with improved stability and early stopping - optimized for speed"""
        if len(self.memory) < self.batch_size:
            return
        
        # Get trajectories from memory
        states, actions, old_mus, old_log_stds, old_log_probs, rewards, next_states, dones = self._prepare_training_data()
        
        # Convert to proper format
        states = np.array(states, dtype=np.float32)  # Use float32 for speed
        actions = np.array(actions, dtype=np.float32).reshape(-1, 1)
        
        # Compute returns and advantages more efficiently
        values = self.critic.predict(states, batch_size=128, verbose=0).flatten()
        next_values = self.critic.predict(np.array(next_states, dtype=np.float32), batch_size=128, verbose=0).flatten()
        
        # Initialize returns and advantages
        returns = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        # GAE calculation
        gae = 0
        for t in reversed(range(len(rewards))):
            next_val = 0 if dones[t] else next_values[t]
            delta = rewards[t] + self.gamma * next_val - values[t]
            gae = delta + self.gamma * self.lam * (0 if dones[t] else gae)
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        old_log_probs = np.array(old_log_probs, dtype=np.float32)

        # Training loop with early stopping based on KL divergence
        for epoch in range(self.train_epochs):
            # Generate random indices
            indices = np.random.permutation(len(states))
            
            # Initialize epoch metrics
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy = 0
            epoch_kl = 0
            batch_count = 0
            
            # Use fewer mini-batches for speed
            max_batches = min(10, len(indices) // self.batch_size)
            
            # Mini-batch training
            for batch_idx in range(max_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, len(indices))
                idx = indices[start_idx:end_idx]
                
                # Skip if batch is too small
                if len(idx) < 4:  # Minimum batch size for stable training
                    continue
                
                # Get mini-batch data
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                mb_old_log_probs = old_log_probs[idx]
                
                # Train actor and calculate KL divergence
                policy_loss, entropy, kl = self._train_actor_step(
                    mb_states, mb_actions, mb_advantages, mb_old_log_probs)
                
                # Train critic
                value_loss = self._train_critic_step(mb_states, mb_returns)
                
                # Accumulate metrics
                epoch_policy_loss += policy_loss
                epoch_value_loss += value_loss
                epoch_entropy += entropy
                epoch_kl += kl
                batch_count += 1
            
            # Calculate average metrics (avoid division by zero)
            if batch_count > 0:
                epoch_policy_loss /= batch_count
                epoch_value_loss /= batch_count
                epoch_entropy /= batch_count
                epoch_kl /= batch_count
            
            # Store metrics
            self.policy_loss_history.append(float(epoch_policy_loss))
            self.value_loss_history.append(float(epoch_value_loss))
            self.entropy_history.append(float(epoch_entropy))
            self.kl_divergence_history.append(float(epoch_kl))
            
            # Early stopping based on KL divergence for speed
            if epoch_kl > self.target_kl:
                break
        
        # Clear memory partially to speed up training
        while len(self.memory) > 1000:
            self.memory.popleft()
        
    def _train_actor_step(self, states, actions, advantages, old_log_probs):
        """Single training step for actor with gradient clipping"""
        with tf.GradientTape() as tape:
            # Forward pass
            mus, log_stds = self.actor(tf.cast(states, tf.float32))
            
            # Constrain log_std
            log_stds = tf.clip_by_value(log_stds, -5.0, 0.0)
            stds = tf.exp(log_stds)
            
            # Calculate log probabilities
            vars = tf.square(stds) + 1e-8
            log_probs = -0.5 * (
                tf.math.log(2 * np.pi) + 
                2.0 * log_stds + 
                tf.square(actions - mus) / vars
            )
            log_probs = tf.reduce_sum(log_probs, axis=1)
            
            # Calculate ratios and surrogate objectives
            ratios = tf.exp(log_probs - old_log_probs)
            
            # Clip ratios to prevent extreme values
            ratios = tf.clip_by_value(ratios, 0.0, 10.0)
            
            # Surrogate objectives
            surrogate1 = ratios * advantages
            surrogate2 = tf.clip_by_value(
                ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            
            # Policy loss
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            
            # Entropy bonus for exploration
            entropy = tf.reduce_mean(log_stds + 0.5 * tf.math.log(2 * np.pi * np.e))
            entropy_loss = -self.entropy_beta * entropy
            
            # Total loss
            total_loss = policy_loss + entropy_loss
        
        # Calculate KL divergence for early stopping
        kl = tf.reduce_mean(log_probs - old_log_probs)
        
        # Compute gradients and apply with clipping
        grads = tape.gradient(total_loss, self.actor.trainable_variables)
        
        # Clip gradients to prevent exploding gradients
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        
        # Check for NaN gradients before applying
        if not any(tf.reduce_any(tf.math.is_nan(g)) for g in grads if g is not None):
            self.actor_optimizer.apply_gradients(
                zip(grads, self.actor.trainable_variables))
        
        return float(policy_loss), float(entropy), float(kl)
    
    def _train_critic_step(self, states, returns):
        """Single training step for critic with gradient clipping"""
        with tf.GradientTape() as tape:
            values = self.critic(tf.cast(states, tf.float32))
            values = tf.squeeze(values)
            
            # MSE loss
            critic_loss = tf.reduce_mean(tf.square(returns - values))
        
        # Compute gradients and apply with clipping
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        
        # Clip gradients to prevent exploding gradients
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        
        # Check for NaN gradients before applying
        if not any(tf.reduce_any(tf.math.is_nan(g)) for g in grads if g is not None):
            self.critic_optimizer.apply_gradients(
                zip(grads, self.critic.trainable_variables))
        
        return float(critic_loss)
    
    def _prepare_training_data(self):
        """Extract and prepare data from memory"""
        # Get trajectories from memory
        states = []
        actions = []
        old_mus = []
        old_log_stds = []
        old_log_probs = []
        rewards = []
        next_states = []
        dones = []
        
        # Limit to a reasonable number of samples
        for i in range(min(len(self.memory), 2000)):
            sample = self.memory[i]
            states.append(sample[0])
            actions.append(sample[1]['size'])
            old_mus.append(sample[2]['mu'][0][0])
            old_log_stds.append(sample[2]['log_std'][0][0])
            old_log_probs.append(sample[2]['log_prob'])
            rewards.append(sample[3])
            next_states.append(sample[4])
            dones.append(sample[5])
        
        return states, actions, old_mus, old_log_stds, old_log_probs, rewards, next_states, dones
    
    def _compute_advantages(self, states, rewards, next_states, dones):
        """Compute advantages with GAE"""
        # Predict values for states and next states
        values = self.critic.predict(np.array(states), verbose=0).flatten()
        next_values = self.critic.predict(np.array(next_states), verbose=0).flatten()
        
        # Initialize returns and advantages
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        # GAE calculation
        gae = 0
        for t in reversed(range(len(rewards))):
            next_val = 0 if dones[t] else next_values[t]
            delta = rewards[t] + self.gamma * next_val - values[t]
            gae = delta + self.gamma * self.lam * (0 if dones[t] else gae)
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return returns, advantages
        
    def save_model(self, actor_path='ppo_actor_continuous.h5', critic_path='ppo_critic_continuous.h5'):
        """Save models to disk with error handling for complex models"""
        try:
            # Ensure path ends with .weights.h5 as required by TensorFlow
            if not actor_path.endswith('.weights.h5'):
                actor_path = actor_path.replace('.h5', '.weights.h5')
            if not critic_path.endswith('.weights.h5'):
                critic_path = critic_path.replace('.h5', '.weights.h5')
            
            # Make sure directory exists
            os.makedirs(os.path.dirname(actor_path) if os.path.dirname(actor_path) else '.', exist_ok=True)
            os.makedirs(os.path.dirname(critic_path) if os.path.dirname(critic_path) else '.', exist_ok=True)
            
            # Save weights only instead of full model
            self.actor.save_weights(actor_path)
            self.critic.save_weights(critic_path)
            
            print(f"Model weights saved to {actor_path} and {critic_path}")
            return True
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            
            # Fallback: Try to save just the weights in NumPy format
            try:
                # Create directories if they don't exist
                os.makedirs('models/weights', exist_ok=True)
                
                # Save weights as NumPy arrays - handle each weight tensor separately
                actor_weights = []
                for i, w in enumerate(self.actor.weights):
                    weight_path = f'models/weights/actor_weight_{i}.npy'
                    np.save(weight_path, w.numpy())
                    actor_weights.append(weight_path)
                
                critic_weights = []
                for i, w in enumerate(self.critic.weights):
                    weight_path = f'models/weights/critic_weight_{i}.npy'
                    np.save(weight_path, w.numpy())
                    critic_weights.append(weight_path)
                
                # Save weight paths for loading
                with open('models/weights/weight_paths.json', 'w') as f:
                    json.dump({
                        'actor_weights': actor_weights,
                        'critic_weights': critic_weights
                    }, f)
                
                # Save network architecture information
                with open('models/weights/model_info.txt', 'w') as f:
                    f.write(f"Actor layers: {[layer.name for layer in self.actor.layers]}\n")
                    f.write(f"Critic layers: {[layer.name for layer in self.critic.layers]}\n")
                
                print("Saved model weights as NumPy arrays in models/weights/")
                return True
            except Exception as e2:
                print(f"Failed to save weights: {str(e2)}")
                return False

    def load_model(self, actor_path='ppo_actor_continuous.h5', critic_path='ppo_critic_continuous.h5'):
        """Load models from disk with error handling"""
        try:
            # Check for weights.h5 extension format
            if not actor_path.endswith('.weights.h5') and actor_path.endswith('.h5'):
                actor_weights_path = actor_path.replace('.h5', '.weights.h5')
                if os.path.exists(actor_weights_path):
                    actor_path = actor_weights_path
                    
            if not critic_path.endswith('.weights.h5') and critic_path.endswith('.h5'):
                critic_weights_path = critic_path.replace('.h5', '.weights.h5')
                if os.path.exists(critic_weights_path):
                    critic_path = critic_weights_path
            
            # Try to load weights
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                self.actor.load_weights(actor_path)
                self.critic.load_weights(critic_path)
                print(f"Model weights loaded from {actor_path} and {critic_path}")
                return True
                
            # Fallback to new NumPy individual weights format
            elif os.path.exists('models/weights/weight_paths.json'):
                try:
                    # Load paths to weight files
                    with open('models/weights/weight_paths.json', 'r') as f:
                        weight_paths = json.load(f)
                    
                    # Load actor weights
                    for i, path in enumerate(weight_paths['actor_weights']):
                        if i < len(self.actor.weights):
                            weight = np.load(path, allow_pickle=True)
                            self.actor.weights[i].assign(weight)
                    
                    # Load critic weights
                    for i, path in enumerate(weight_paths['critic_weights']):
                        if i < len(self.critic.weights):
                            weight = np.load(path, allow_pickle=True)
                            self.critic.weights[i].assign(weight)
                    
                    print("Loaded model weights from individual NumPy arrays")
                    return True
                except Exception as e:
                    print(f"Error loading weights from JSON paths: {str(e)}")
                    
            # Try the old-style weights format as last resort        
            elif os.path.exists('models/weights/actor_weights.npy'):
                try:
                    # Load weights from NumPy arrays
                    actor_weights = np.load('models/weights/actor_weights.npy', allow_pickle=True)
                    critic_weights = np.load('models/weights/critic_weights.npy', allow_pickle=True)
                    
                    # Set weights directly
                    for i, w in enumerate(actor_weights):
                        if i < len(self.actor.weights):
                            self.actor.weights[i].assign(w)
                    
                    for i, w in enumerate(critic_weights):
                        if i < len(self.critic.weights):
                            self.critic.weights[i].assign(w)
                    
                    print("Loaded model weights from legacy NumPy arrays")
                    return True
                except Exception as e:
                    print(f"Error loading weights from legacy NumPy format: {str(e)}")
            
            print("Model files not found. Using initialized models.")
            return False
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            print("Using initialized models.")
            return False