import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import time
import os
from datetime import datetime
import json

# Set TensorFlow to only grow GPU memory as needed
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Setup logging with more details
def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/trading_rl_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

class EnhancedTrainingPipeline:
    def __init__(
        self,
        env_class,
        agent_class,
        train_df,
        val_df=None,
        config=None,
        logger=None,
        save_dir="models",
        experiment_name=None
    ):
        """
        Initialize the training pipeline
        
        Parameters:
        env_class: Class for the trading environment
        agent_class: Class for the RL agent
        train_df: Training data DataFrame
        val_df: Validation data DataFrame (optional)
        config: Configuration dict
        logger: Logger object
        save_dir: Directory to save models and results
        experiment_name: Name for this experiment
        """
        self.env_class = env_class
        self.agent_class = agent_class
        self.train_df = train_df
        self.val_df = val_df if val_df is not None else train_df.iloc[int(len(train_df) * 0.8):]
        self.logger = logger if logger else setup_logger()
        
        # Default configuration
        self.config = {
            # Environment parameters
            "holding_period": 60,
            "risk_pct": 0.02,
            "position_value": 1000,
            "time_frame": "4h",
            "reward_scaling": 1.0,
            "reward_shaping": True,
            "use_drawdown_penalty": True,
            
            # Training parameters
            "episodes": 200,
            "early_stop_patience": 20,
            "eval_interval": 5,
            "batch_size": 64,
            "target_kl": 0.01,
            "use_curriculum": True,
            
            # Agent parameters
            "state_size": len(train_df.columns),
            "clip_ratio": 0.2,
            "policy_lr": 0.0003,
            "value_lr": 0.001,
            "entropy_beta": 0.01,
            "gamma": 0.95,
            "lam": 0.95,
            "train_epochs": 10,
            "action_bounds": [0.1, 1.0]
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Setup directories
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name if experiment_name else f"experiment_{self.timestamp}"
        self.save_dir = os.path.join(save_dir, self.experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)
        
        # Initialize environments
        self.train_env = self._create_env(self.train_df)
        self.val_env = self._create_env(self.val_df)
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Initialize metrics tracking
        self.metrics = {
            "train_rewards": [],
            "val_rewards": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "kl_divergences": [],
            "win_rates": [],
            "position_sizes": [],
            "trade_counts": [],
            "pnl_history": []
        }
        
        # For early stopping
        self.best_val_reward = -np.inf
        self.patience_counter = 0
        
        # Create visualization figure
        self.fig = plt.figure(figsize=(15, 10))
    
    def _create_env(self, df):
        """Create a trading environment with the current config"""
        return self.env_class(
            df=df,
            holding_period=self.config["holding_period"],
            risk_pct=self.config["risk_pct"],
            position_value=self.config["position_value"],
            timeframe=self.config["time_frame"],
            reward_scaling=self.config["reward_scaling"],
            reward_shaping=self.config["reward_shaping"],
            use_drawdown_penalty=self.config["use_drawdown_penalty"]
        )
    
    def _create_agent(self):
        """Create an agent with the current config"""
        return self.agent_class(
            state_size=self.config["state_size"],
            clip_ratio=self.config["clip_ratio"],
            policy_lr=self.config["policy_lr"],
            value_lr=self.config["value_lr"],
            entropy_beta=self.config["entropy_beta"],
            gamma=self.config["gamma"],
            lam=self.config["lam"],
            train_epochs=self.config["train_epochs"],
            batch_size=self.config["batch_size"],
            action_bounds=self.config["action_bounds"],
            target_kl=self.config["target_kl"]
        )
    
    def train(self):
        """Main training loop with curriculum learning and early stopping"""
        self.logger.info(f"Starting training for {self.config['episodes']} episodes")
        self.logger.info(f"Training data shape: {self.train_df.shape}")
        
        # Implement curriculum learning if enabled
        if self.config["use_curriculum"]:
            # Start with shorter holding periods and gradually increase
            holding_periods = [
                max(10, self.config["holding_period"] // 4),
                max(20, self.config["holding_period"] // 2),
                self.config["holding_period"]
            ]
            
            # Start with higher entropy for more exploration
            entropy_betas = [
                self.config["entropy_beta"] * 3,
                self.config["entropy_beta"] * 2,
                self.config["entropy_beta"]
            ]
            
            # Divide episodes among curriculum stages
            episodes_per_stage = self.config["episodes"] // len(holding_periods)
            
            self.logger.info(f"Using curriculum learning with {len(holding_periods)} stages")
            self.logger.info(f"Holding periods: {holding_periods}")
            self.logger.info(f"Entropy coefficients: {entropy_betas}")
        else:
            # No curriculum - use configured values
            holding_periods = [self.config["holding_period"]]
            entropy_betas = [self.config["entropy_beta"]]
            episodes_per_stage = self.config["episodes"]
        
        # Track overall episode count
        total_episodes = 0
        
        # Run each curriculum stage
        for stage, (holding_period, entropy_beta) in enumerate(zip(holding_periods, entropy_betas)):
            self.logger.info(f"\nStarting curriculum stage {stage + 1}")
            self.logger.info(f"Holding period: {holding_period}, Entropy beta: {entropy_beta}")
            
            # Update environment and agent for this stage
            self.train_env.holding_period = holding_period
            self.val_env.holding_period = holding_period
            self.agent.entropy_beta = entropy_beta
            
            # Episodes for this stage
            stage_episodes = min(episodes_per_stage, self.config["episodes"] - total_episodes)
            
            # Progress bar for this stage
            pbar = tqdm(range(stage_episodes), desc=f"Stage {stage + 1}")
            
            for episode in pbar:
                # Current overall episode
                current_episode = total_episodes + episode
                
                # Run one training episode
                train_metrics = self._run_episode(self.train_env, training=True)
                
                # Store metrics
                self.metrics["train_rewards"].append(train_metrics["total_reward"])
                self.metrics["trade_counts"].append(train_metrics["trades"])
                self.metrics["position_sizes"].append(train_metrics["avg_position_size"])
                if "win_rate" in train_metrics:
                    self.metrics["win_rates"].append(train_metrics["win_rate"])
                if 'episode_pnl' in train_metrics:
                    self.metrics["pnl_history"].append(train_metrics["episode_pnl"])
                else:
                    self.metrics["pnl_history"].append(0.0)
                
                # Store agent metrics if available
                if hasattr(self.agent, "policy_loss_history") and self.agent.policy_loss_history:
                    self.metrics["policy_losses"].extend(self.agent.policy_loss_history)
                    self.agent.policy_loss_history = []
                
                if hasattr(self.agent, "value_loss_history") and self.agent.value_loss_history:
                    self.metrics["value_losses"].extend(self.agent.value_loss_history)
                    self.agent.value_loss_history = []
                
                if hasattr(self.agent, "entropy_history") and self.agent.entropy_history:
                    self.metrics["entropies"].extend(self.agent.entropy_history)
                    self.agent.entropy_history = []
                
                if hasattr(self.agent, "kl_divergence_history") and self.agent.kl_divergence_history:
                    self.metrics["kl_divergences"].extend(self.agent.kl_divergence_history)
                    self.agent.kl_divergence_history = []
                
                # Update progress bar
                pbar.set_postfix({
                    'Reward': f'{train_metrics["total_reward"]:.2f}',
                    'Trades': train_metrics["trades"],
                    'AvgPos': f'{train_metrics["avg_position_size"]:.2f}'
                })
                
                # Run evaluation on validation set
                if (current_episode + 1) % self.config["eval_interval"] == 0:
                    val_metrics = self._evaluate()
                    
                    # Log evaluation results
                    self.logger.info(f"Episode {current_episode + 1} validation: " +
                                     f"Reward = {val_metrics['mean_reward']:.2f}, " +
                                     f"Trades = {val_metrics['mean_trades']:.1f}, " +
                                     f"Win rate = {val_metrics['win_rate'] * 100:.1f}%")
                    
                    # Store validation reward
                    self.metrics["val_rewards"].append(val_metrics["mean_reward"])
                    
                    # Update plots
                    self._update_plots()
                    
                    # Check for early stopping
                    if val_metrics["mean_reward"] > self.best_val_reward:
                        self.best_val_reward = val_metrics["mean_reward"]
                        self.patience_counter = 0
                        
                        # Save best model
                        self._save_model("best_model")
                        self.logger.info(f"New best model with validation reward: {self.best_val_reward:.2f}")
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.config["early_stop_patience"]:
                            self.logger.info(f"Early stopping triggered after {self.patience_counter} evaluations without improvement")
                            break
            
            # Update total episodes
            total_episodes += stage_episodes
            
            # Save stage model
            self._save_model(f"stage_{stage + 1}_model")
            
            # Check if we should stop early
            if self.patience_counter >= self.config["early_stop_patience"]:
                self.logger.info("Early stopping - ending training")
                break
        
        # Save final model
        self._save_model("final_model")
        
        # Save all metrics
        self._save_metrics()
        
        return self.metrics, self.agent
    
    def _run_episode(self, env, training=True):
        """Run a single episode and collect metrics"""
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        trades = 0
        position_sizes = []
        episode_pnl = 0
        
        while not done:
            # Get action from agent
            action, action_info = self.agent.get_action(state, training=training)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience if training
            if training:
                self.agent.remember(state, action, action_info, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            steps += 1
            
            # Track trades and position sizes
            if info["trade_taken"]:
                trades += 1
                position_sizes.append(info["position_size"])
            
            # Train agent if enough samples
            if training and len(self.agent.memory) >= self.config["batch_size"] * 4:
                self.agent.train()
        
        # Calculate metrics
        metrics = {
            "total_reward": total_reward,
            "steps": steps,
            "trades": trades,
            "avg_position_size": np.mean(position_sizes) if position_sizes else 0.0,
            "episode_pnl": episode_pnl
        }
        
        # Add win rate if available
        if "win_rate" in info:
            metrics["win_rate"] = info["win_rate"]

        if 'episode_pnl' in info:
            metrics['episode_pnl'] = info['episode_pnl']
        else:
            metrics['episode_pnl'] = 0.0

        return metrics
    
    def _evaluate(self, episodes=5):
        """Evaluate the agent on the validation environment"""
        rewards = []
        trade_counts = []
        position_sizes = []
        win_rates = []
        
        for _ in range(episodes):
            metrics = self._run_episode(self.val_env, training=False)
            rewards.append(metrics["total_reward"])
            trade_counts.append(metrics["trades"])
            position_sizes.append(metrics["avg_position_size"])
            if "win_rate" in metrics:
                win_rates.append(metrics["win_rate"])
        
        # Calculate mean metrics
        eval_metrics = {
            "mean_reward": np.mean(rewards),
            "mean_trades": np.mean(trade_counts),
            "mean_position_size": np.mean(position_sizes) if position_sizes else 0.0,
            "win_rate": np.mean(win_rates) if win_rates else 0.0
        }
        
        return eval_metrics
    
    def _update_plots(self):
        """Update training plots - simplified version that just saves image files"""
        # Create a new figure each time (discard the old one)
        plt.figure(figsize=(12, 20))
        
        # Determine number of plots
        n_plots = 5
        if len(self.metrics["val_rewards"]) > 0:
            n_plots += 1
        
        # Create subplots
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
        
        # Plot training rewards
        ax1 = axes[0]
        ax1.plot(self.metrics["train_rewards"], label='Training')
        if len(self.metrics["train_rewards"]) > 10:
            # Plot moving average
            window = min(10, len(self.metrics["train_rewards"]) // 5)
            ax1.plot(pd.Series(self.metrics["train_rewards"]).rolling(window=window).mean(), 
                    label=f'MA({window})')
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        
        # Plot validation rewards if available
        plot_idx = 1
        if len(self.metrics["val_rewards"]) > 0:
            ax2 = axes[plot_idx]
            eval_indices = np.arange(0, len(self.metrics["train_rewards"]), self.config["eval_interval"])[:len(self.metrics["val_rewards"])]
            ax2.plot(eval_indices, self.metrics["val_rewards"], label='Validation')
            ax2.set_title('Validation Rewards')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Reward')
            ax2.axhline(y=self.best_val_reward, color='r', linestyle='--', label=f'Best: {self.best_val_reward:.2f}')
            ax2.legend()
            plot_idx += 1
        
            # Add new PnL plot
        if len(self.metrics["pnl_history"]) > 0:
            ax_pnl = axes[plot_idx]
            ax_pnl.plot(self.metrics["pnl_history"], label='Cumulative PnL', color='green')
            
            # Add moving average for PnL
            if len(self.metrics["pnl_history"]) > 10:
                window = min(10, len(self.metrics["pnl_history"]) // 5)
                ax_pnl.plot(pd.Series(self.metrics["pnl_history"]).rolling(window=window).mean(), 
                        label=f'MA({window})', color='darkgreen', linestyle='--')
            
            ax_pnl.set_title('Cumulative Profit and Loss (PnL)')
            ax_pnl.set_xlabel('Episode')
            ax_pnl.set_ylabel('PnL Value')
            ax_pnl.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Zero line
            ax_pnl.legend()
            plot_idx += 1

        # Plot policy loss
        if self.metrics["policy_losses"]:
            ax = axes[plot_idx]
            ax.plot(self.metrics["policy_losses"])
            ax.set_title('Policy Loss')
            ax.set_xlabel('Update')
            ax.set_ylabel('Loss')
            plot_idx += 1
        
        # Plot trade counts and position sizes
        ax = axes[plot_idx]
        ax.plot(self.metrics["trade_counts"], label='Trades')
        ax.set_title('Trade Count per Episode')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Count')
        plot_idx += 1
        
        # Plot position sizes
        ax = axes[plot_idx]
        ax.plot(self.metrics["position_sizes"], label='Position Size')
        ax.set_title('Average Position Size')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Size')
        
        # Adjust layout and save
        plt.tight_layout()

        """
        # Save with timestamp to see progress over time
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save two versions - one with timestamp and one that gets overwritten
        plt.savefig(os.path.join(self.save_dir, f"training_progress_{timestamp}.png"))"""

        plt.savefig(os.path.join(self.save_dir, "latest_progress.png"))
        
        # Close the plot to free memory
        plt.close()
        
    def _save_model(self, name):
        """Save the agent model with error handling"""
        if hasattr(self.agent, 'save_model'):
            try:
                # Create the directory if it doesn't exist
                os.makedirs(self.save_dir, exist_ok=True)
                
                # Set file paths - use the .weights.h5 extension directly
                actor_path = os.path.join(self.save_dir, f"{name}_actor.weights.h5")
                critic_path = os.path.join(self.save_dir, f"{name}_critic.weights.h5")
                
                # Try to save the model
                success = self.agent.save_model(actor_path, critic_path)
                
                if success:
                    self.logger.info(f"Model saved: {name}")
                else:
                    self.logger.warning(f"Failed to save model: {name}")
                    
                # Save training metrics as a backup
                self._save_metrics()
                    
            except Exception as e:
                self.logger.error(f"Error saving model: {str(e)}")
                # Continue without crashing - this way training can continue even if saving fails
        else:
            self.logger.warning("Agent does not have save_model method")
    
    def _save_metrics(self):
        """Save all training metrics"""
        metrics_path = os.path.join(self.save_dir, "metrics.json")
        
        # Convert numpy arrays to lists for JSON serialization
        json_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                json_metrics[key] = [v.tolist() for v in value]
            else:
                json_metrics[key] = value
        
        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=4)
        
        self.logger.info(f"Metrics saved to: {metrics_path}")

# Example usage
if __name__ == "__main__":
    # Import your actual classes
    from environments.enhanced_trading_env import EnhancedTradingEnv
    from models.stable_ppo_agent import StablePPOAgent
    
    # Load data
    df = pd.read_csv('data/ETHUSDT_4_hours_data.csv')
    
    # Create training pipeline
    pipeline = EnhancedTrainingPipeline(
        env_class=EnhancedTradingEnv,
        agent_class=StablePPOAgent,
        train_df=df,
        experiment_name="eth_trading_ppo"
    )
    
    # Train the agent
    metrics, trained_agent = pipeline.train()