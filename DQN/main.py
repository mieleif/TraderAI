import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Import the enhanced models and environment
from environments.enhanced_trading_env import EnhancedTradingEnv
from models.stable_ppo_agent import StablePPOAgent
from enhanced_training_pipeline import EnhancedTrainingPipeline
from config import TrainingConfig, DataConfig

# Setup logging
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

def load_data(filepath='data/ETHUSDT_4h_data.csv'):
    """
    Load and prepare data from CSV file
    """
    logger = setup_logger()
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Check if the dataframe has the expected columns
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    return df

def prepare_data(df):
    """
    Prepare the DataFrame for use in the trading environment
    by ensuring all columns used for state representation are numeric
    """
    logger = setup_logger()
    
    # 1. Convert datetime to index if it exists as a column
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    
    # 2. Clean symbol column if exists
    if 'symbol' in df.columns:
        if df['symbol'].dtype == object:  # Only if it's a string
            df['symbol'] = df['symbol'].str.replace('BINANCE:', '')
    
    # 3. Drop any non-numeric columns that aren't needed for features
    # Keep only required columns and numeric ones
    required_cols = ['open', 'high', 'low', 'close']
    if 'fundingRate' in df.columns:
        required_cols.append('fundingRate')
    
    # Add Ichimoku indicator columns if they exist
    for col in df.columns:
        if any(x in str(col).lower() for x in ['sen', 'span', 'tenkan', 'kijun']):
            required_cols.append(col)
    
    # Find numeric columns
    feature_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    # Get the intersection of required columns and what's in the dataframe, plus any numeric columns
    keep_cols = list(set(required_cols).intersection(set(df.columns))) + feature_cols
    
    # Remove duplicates
    keep_cols = list(dict.fromkeys(keep_cols))
    
    logger.info(f"Using columns: {keep_cols}")
    
    # Use only the columns we need
    df_clean = df[keep_cols].copy()
    
    # 4. Fill any remaining NaN values with forward-fill then backward-fill
    df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
    
    # 5. Verify that all columns are now numeric
    non_numeric = [col for col in df_clean.columns if not pd.api.types.is_numeric_dtype(df_clean[col])]
    if non_numeric:
        logger.warning(f"The following columns are still non-numeric: {non_numeric}")
        # Convert any remaining non-numeric columns to numeric if possible
        for col in non_numeric:
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            except:
                logger.warning(f"Could not convert column {col} to numeric, dropping it.")
                df_clean.drop(columns=[col], inplace=True)
    
    # Log information about the prepared dataframe
    logger.info(f"Prepared data shape: {df_clean.shape}")
    logger.info(f"Final columns: {df_clean.columns.tolist()}")
    
    # Check for any remaining NaN values
    nan_cols = df_clean.columns[df_clean.isna().any()].tolist()
    if nan_cols:
        logger.warning(f"Columns with NaN values after preparation: {nan_cols}")
        # Fill any remaining NaNs with column mean
        for col in nan_cols:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    return df_clean

def create_config(df):
    """Create a configuration dictionary based on the dataframe columns"""
    config = {
        # Environment parameters
        "holding_period": TrainingConfig.HOLDING_PERIOD,
        "risk_pct": TrainingConfig.RISK_PCT,
        "position_value": TrainingConfig.POSITION_VALUE,
        "time_frame": TrainingConfig.TIME_FRAME,
        "reward_scaling": 1.0,  # Default value
        "reward_shaping": True,
        "use_drawdown_penalty": True,
        
        # Training parameters
        "episodes": TrainingConfig.EPISODES,
        "early_stop_patience": 20,
        "eval_interval": 5,
        "batch_size": TrainingConfig.BATCH_SIZE,
        "target_kl": 0.01,
        "use_curriculum": True,
        
        # Agent parameters
        "state_size": len(df.columns),
        "clip_ratio": TrainingConfig.PPO_CLIP_RATIO,
        "policy_lr": TrainingConfig.PPO_POLICY_LR,
        "value_lr": TrainingConfig.PPO_VALUE_LR,
        "entropy_beta": TrainingConfig.PPO_ENTROPY_BETA,
        "gamma": TrainingConfig.GAMMA,
        "lam": TrainingConfig.PPO_LAMBDA,
        "train_epochs": TrainingConfig.PPO_EPOCHS,
        "action_bounds": [TrainingConfig.PPO_MIN_POSITION_SIZE, TrainingConfig.PPO_MAX_POSITION_SIZE]
    }
    return config

# Add these modifications to main.py to create a faster configuration

def create_optimized_config(df):
    """Create a faster training configuration for testing"""
    config = {
        # Environment parameters
        "holding_period": 15,  # Shorter holding period
        "risk_pct": 0.02,
        "position_value": 1000,
        "time_frame": TrainingConfig.TIME_FRAME,
        "reward_scaling": 1.0,
        "reward_shaping": True,
        "use_drawdown_penalty": True,
        
        # Training parameters - drastically reduced for speed
        "episodes": 20,  # Fewer episodes for testing
        "early_stop_patience": 5,
        "eval_interval": 2,  # More frequent evaluation
        "batch_size": 32,  # Smaller batch for faster updates
        "target_kl": 0.02,  # More lenient KL constraint
        "use_curriculum": False,  # Disable curriculum for testing
        
        # Agent parameters
        "state_size": len(df.columns),
        "clip_ratio": TrainingConfig.PPO_CLIP_RATIO,
        "policy_lr": 0.0005,  # Slightly higher for faster learning
        "value_lr": 0.001,
        "entropy_beta": 0.01,
        "gamma": 0.95,
        "lam": 0.95,
        "train_epochs": 5,  # Fewer training epochs
        "action_bounds": [TrainingConfig.PPO_MIN_POSITION_SIZE, TrainingConfig.PPO_MAX_POSITION_SIZE]
    }
    return config

# In main() function, replace the config creation with:
# config = create_optimized_config(df)  # Use optimized config for faster training

# Also add this line after creating the pipeline:
print("Using optimized training settings for faster execution. Episodes will take 1-2 minutes each.")

def main():
    # Setup logger
    logger = setup_logger()
    logger.info("Starting the cryptocurrency trading RL training system")
    
    # Load and prepare data
    raw_df = load_data()
    df = prepare_data(raw_df)

    # Use a subset of data for faster training
    if len(df) > 1000:
        print(f"Reducing dataset from {len(df)} to 1000 rows for faster training")
        df = df.iloc[-1000:].copy()  # Use most recent 1000 rows
    
    # Create configuration
    config = create_optimized_config(df)
    
    # Create training pipeline
    pipeline = EnhancedTrainingPipeline(
        env_class=EnhancedTradingEnv,
        agent_class=StablePPOAgent,
        train_df=df,
        config=config,
        logger=logger,
        experiment_name="crypto_trading_ppo"
    )
    
    # Train the agent
    logger.info("Starting training process")
    try:
        metrics, trained_agent = pipeline.train()
        logger.info("Training completed successfully")
        
        # Save the final model explicitly
        if hasattr(trained_agent, 'save_model'):
            trained_agent.save_model('models/final_ppo_actor.h5', 'models/final_ppo_critic.h5')
            logger.info("Final model saved")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()