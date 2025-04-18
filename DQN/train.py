import pandas as pd
import logging
from environments.trading_env import TradingEnv
from models.dqn_agent import DQNAgent
from config import TrainingConfig, DataConfig
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(filepath='data/ETHUSDT_4h_data.csv'):
    """
    Load and prepare data from CSV file
    """
    logger.info(f"Loading data from {filepath}")
    
    # Read CSV file
    df = pd.read_csv(filepath)
    
    # Convert datetime column to pandas datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    # Clean symbol column (remove 'BINANCE:' prefix)
    df['symbol'] = df['symbol'].str.replace('BINANCE:', '')
    
    # Drop any NaN values
    df = df.dropna()
    
    logger.info(f"Loaded data shape: {df.shape}")
    return df

def plot_training_results(episodes, rewards, epsilons, window=10):
    """Plot training metrics without blocking execution"""
    clear_output(wait=True)  # Clear previous plots
    plt.figure(figsize=(15, 5))
    
    # Plot rewards
    plt.subplot(131)
    plt.plot(rewards, label='Reward')
    plt.plot(pd.Series(rewards).rolling(window).mean(), label='Moving Avg')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Plot moving average of rewards
    plt.subplot(132)
    plt.plot(pd.Series(rewards).rolling(window).mean())
    plt.title(f'Moving Average of Rewards (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    # Plot epsilon
    plt.subplot(133)
    plt.plot(epsilons)
    plt.title('Epsilon Value')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    if episodes == TrainingConfig.EPISODES:
        plt.show()
    plt.close('all')  # Close the figure to free memory

def train(df):
    """
    Main training function
    """
    try:
        logger.info(f"Training data shape: {df.shape}")
        logger.info("Initializing environment and agent...")
        
        # Initialize environment
        env = TradingEnv(
            df=df,
            holding_period=TrainingConfig.HOLDING_PERIOD,
            risk_pct=TrainingConfig.RISK_PCT,
            position_value=TrainingConfig.POSITION_VALUE
        )
        
        # Initialize agent
        agent = DQNAgent(
            state_size=TrainingConfig.STATE_SIZE,
            action_size=TrainingConfig.ACTION_SIZE
        )
        
        logger.info("Starting training loop...")
        
        # Lists to store metrics
        rewards_history = []
        epsilon_history = []
        trades_history = []
        
        pbar = tqdm(range(TrainingConfig.EPISODES), desc='Training Progress')
        for episode in pbar:
            logger.info(f"\nStarting Episode {episode + 1}")
            state = env.reset()
            total_reward = 0
            done = False
            trades_this_episode = 0
            steps_this_episode = 0
            
            # Episode loop
            while not done:
                steps_this_episode += 1
                
                # Debug every 100 steps
                if steps_this_episode % 100 == 0:
                    logger.info(f"Episode {episode + 1}, Step {steps_this_episode}")
                
                # Get action
                action = agent.act(state)
                logger.debug(f"Selected action: {action}")
                
                # Take step
                try:
                    next_state, reward, done, _ = env.step(action)
                    logger.debug(f"Step result - Reward: {reward}, Done: {done}")
                except Exception as e:
                    logger.error(f"Error in environment step: {str(e)}")
                    raise
                
                # Store in memory
                agent.remember(state, action, reward, next_state, done)
                
                # Update state and reward
                state = next_state
                total_reward += reward
                
                if action == 1:
                    trades_this_episode += 1
                
                # Training step
                if len(agent.memory) > TrainingConfig.BATCH_SIZE:
                    logger.debug(f"Episode {episode + 1}, Step {steps_this_episode}: Calling replay")
                    try:
                        agent.replay(TrainingConfig.BATCH_SIZE)
                    except Exception as e:
                        logger.error(f"Error in replay: {str(e)}")
                        raise
            
            # Episode complete
            logger.info(f"Episode {episode + 1} completed:")
            logger.info(f"Steps: {steps_this_episode}")
            logger.info(f"Total Reward: {total_reward:.2f}")
            logger.info(f"Trades Made: {trades_this_episode}")
            logger.info(f"Epsilon: {agent.epsilon:.4f}")
            
            # Store metrics
            rewards_history.append(total_reward)
            epsilon_history.append(agent.epsilon)
            trades_history.append(trades_this_episode)
            
            # Update progress bar
            pbar.set_postfix({
                'Reward': f'{total_reward:.2f}',
                'Epsilon': f'{agent.epsilon:.2f}',
                'Trades': trades_this_episode
            })
            
            # Update plot every 5 episodes (you can adjust this number)
            if episode % 5 == 0:
                plot_training_results(
                    range(len(rewards_history)), 
                    rewards_history, 
                    epsilon_history
                )
        
        return agent, rewards_history, epsilon_history, trades_history
    
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def evaluate_model(env, agent, episodes=10):
    """
    Evaluate the trained model
    """
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        trades = []
        
        while not done:
            action = agent.act(state)  # During evaluation, this will use the trained policy
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            if action == 1:  # If trade was made
                trades.append({
                    'entry_price': env.trade_entry_price,
                    'reward': reward
                })
            state = next_state
            
        total_rewards.append(episode_reward)
        
        logger.info(f"Evaluation Episode {episode + 1}: Reward = {episode_reward:.2f}, Trades = {len(trades)}")
    
    return np.mean(total_rewards), trades

if __name__ == "__main__":
    # Load data from CSV
    logger.info("Loading and preparing data...")
    df = load_data()
    
    logger.info("Starting training...")
    agent, rewards, epsilons, trades = train(df)
    
    # Final plots
    plot_training_results(range(len(rewards)), rewards, epsilons)
    
    # Save the trained model
    agent.save_model('trained_model.h5')
    logger.info("Training completed and model saved!")
