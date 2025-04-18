import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

# Import the enhanced models and environment
from environments.enhanced_trading_env import EnhancedTradingEnv
from models.stable_ppo_agent import StablePPOAgent
from enhanced_training_pipeline import EnhancedTrainingPipeline
from config import TrainingConfig, DataConfig

# Import new strategy components
from strategies.ichimoku_signals import IchimokuSignals
from strategies.feature_engineering import FeatureEngineering
from strategies.hybrid_decision import HybridDecisionSystem
from strategies.backtest_engine import BacktestEngine
from strategies.visualization import TradingVisualization

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
        "price_target_pct": 0.05,  # Maximum price target of 5%
        "risk_reward_ratio": 5.0,  # 5:1 risk/reward ratio
        "ichimoku_warmup_period": 52,  # Wait for Ichimoku indicators to be calculated
        "inactivity_threshold": 50,  # Encourage trading after 50 periods of inactivity
        
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

def calculate_ichimoku_signals(df):
    """
    Calculate Ichimoku Cloud signals and components
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        
    Returns:
        pd.DataFrame: DataFrame with Ichimoku components and signals
    """
    logger = setup_logger()
    logger.info("Calculating Ichimoku signals")
    
    # Initialize Ichimoku signals class
    ichimoku = IchimokuSignals()
    
    # Calculate Ichimoku components
    ichimoku_df = ichimoku.calculate_ichimoku(df)
    
    # Calculate overall signal score
    ichimoku_df['ichimoku_signal'] = ichimoku.calculate_ichimoku_signal_score(ichimoku_df)
    
    # Generate trading decisions
    result_df = ichimoku.generate_trading_decisions(
        ichimoku_df, 
        threshold=0.3
    )
    
    logger.info("Ichimoku signals calculation completed")
    return result_df

def run_ai_trading(df, trained_agent, plot=True):
    """
    Run the AI trading model on the provided data
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC and feature data
        trained_agent: Trained RL agent
        plot (bool): Whether to generate plots
        
    Returns:
        pd.DataFrame: DataFrame with AI trading signals and positions
    """
    logger = setup_logger()
    logger.info("Running AI model predictions")
    
    # Create trading environment for prediction
    env = EnhancedTradingEnv(
        df=df,
        holding_period=15,
        risk_pct=0.02,
        position_value=1000,
        timeframe="4h",
        price_target_pct=0.05,
        risk_reward_ratio=5.0,
        ichimoku_warmup_period=52,
        inactivity_threshold=50
    )
    
    # Reset environment to start prediction
    state = env.reset()
    done = False
    
    # Store predictions
    ai_signals = []
    positions = []
    
    # Run predictions
    while not done:
        # Get action from trained agent
        action, _ = trained_agent.get_action(state, training=False)
        
        # Step environment
        next_state, reward, done, info = env.step(action)
        
        # Store prediction
        ai_signals.append(action['direction'])  # Direction component of action
        positions.append(info.get('position_size', 0))
        
        # Update state
        state = next_state
    
    # Add predictions to dataframe
    result_df = df.copy()
    result_df['ai_signal'] = pd.Series(ai_signals, index=result_df.index[:len(ai_signals)])
    result_df['ai_position'] = pd.Series(positions, index=result_df.index[:len(positions)])
    
    # Fill remaining values (if any)
    result_df['ai_signal'].fillna(0, inplace=True)
    result_df['ai_position'].fillna(0, inplace=True)
    
    logger.info("AI model predictions completed")
    
    # Plot results if requested
    if plot:
        plt.figure(figsize=(14, 7))
        plt.plot(result_df.index, result_df['close'], label='Close Price')
        plt.scatter(result_df[result_df['ai_position'] > 0].index, 
                   result_df[result_df['ai_position'] > 0]['close'], 
                   marker='^', color='green', s=100, label='Buy')
        plt.scatter(result_df[result_df['ai_position'] < 0].index, 
                   result_df[result_df['ai_position'] < 0]['close'], 
                   marker='v', color='red', s=100, label='Sell')
        plt.title('AI Trading Model Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Create output directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/ai_trading_signals.png')
    
    return result_df

def run_hybrid_trading(df, ai_signals_df):
    """
    Run hybrid trading strategy combining Ichimoku signals and AI predictions
    
    Args:
        df (pd.DataFrame): Original price data
        ai_signals_df (pd.DataFrame): DataFrame with AI signals
        
    Returns:
        pd.DataFrame: DataFrame with hybrid trading decisions
    """
    logger = setup_logger()
    logger.info("Running hybrid trading strategy")
    
    # Calculate Ichimoku signals
    ichimoku_df = calculate_ichimoku_signals(df)
    
    # Initialize hybrid decision system
    hybrid_system = HybridDecisionSystem(ai_weight=0.6, ichimoku_weight=0.4)
    
    # Process dataframe
    hybrid_df = hybrid_system.process_dataframe(
        df=ichimoku_df,
        ai_signals=ai_signals_df['ai_signal'],
        ichimoku_signals=ichimoku_df['ichimoku_signal'],
        initial_capital=10000
    )
    
    # Analyze performance
    performance = hybrid_system.analyze_performance(hybrid_df)
    
    # Log performance metrics
    logger.info(f"Hybrid strategy performance: ")
    logger.info(f"Total return: {performance['total_return']:.2%}")
    logger.info(f"Max drawdown: {performance['max_drawdown']:.2%}")
    logger.info(f"Sharpe ratio: {performance['sharpe_ratio']:.2f}")
    logger.info(f"Win rate: {performance['win_rate']:.2%}")
    
    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save results
    hybrid_df.to_csv('results/hybrid_trading_results.csv')
    
    return hybrid_df, performance

def run_backtest(df, strategy_type="hybrid"):
    """
    Run a backtest on the specified strategy
    
    Args:
        df (pd.DataFrame): DataFrame with trading signals
        strategy_type (str): Strategy type ('ichimoku', 'ai', 'hybrid')
        
    Returns:
        tuple: (results DataFrame, trades DataFrame, performance metrics)
    """
    logger = setup_logger()
    logger.info(f"Running backtest for {strategy_type} strategy")
    
    # Initialize backtest engine
    backtest = BacktestEngine(initial_capital=10000, commission=0.001, slippage=0.0005)
    
    # Determine position column based on strategy type
    position_col = f'{strategy_type}_position' if strategy_type != 'hybrid' else 'position'
    
    # Prepare positions series
    if position_col in df.columns:
        positions = df[position_col]
    else:
        logger.error(f"Position column '{position_col}' not found in DataFrame")
        return None, None, None
    
    # Prepare confidence series if available
    confidence = df['confidence'] if 'confidence' in df.columns else None
    
    # Run backtest
    results, trades = backtest.run_backtest(df, positions, confidence)
    
    # Calculate performance metrics
    metrics = backtest.calculate_performance_metrics(results)
    
    # Log performance
    logger.info(f"Backtest results for {strategy_type} strategy:")
    logger.info(f"Total return: {metrics['total_return']:.2%}")
    logger.info(f"Annualized return: {metrics['annualized_return']:.2%}")
    logger.info(f"Max drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Win rate: {metrics['win_rate']:.2%}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results.to_csv(f'results/{strategy_type}_backtest_results.csv')
    
    if not trades.empty:
        trades.to_csv(f'results/{strategy_type}_trades.csv')
    
    return results, trades, metrics

def visualize_results(df, backtest_results, trades, metrics, strategy_type="hybrid"):
    """
    Create visualizations for backtest results
    
    Args:
        df (pd.DataFrame): Original DataFrame with signals
        backtest_results (pd.DataFrame): DataFrame with backtest results
        trades (pd.DataFrame): DataFrame with trade information
        metrics (dict): Performance metrics
        strategy_type (str): Strategy type ('ichimoku', 'ai', 'hybrid')
    """
    logger = setup_logger()
    logger.info(f"Creating visualizations for {strategy_type} strategy")
    
    # Initialize visualization class
    viz = TradingVisualization()
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    # 1. Plot Ichimoku cloud chart if Ichimoku components exist
    ichimoku_cols = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
    if all(col in df.columns for col in ichimoku_cols):
        # Plot Ichimoku cloud with trading signals
        ich_fig = viz.plot_ichimoku_cloud(
            df, 
            show_volume=True, 
            show_signals=True,
            output_file=f'plots/{strategy_type}_ichimoku_cloud.png'
        )
        
        # Plot detailed Ichimoku components
        comp_fig = viz.create_ichimoku_components_plot(
            df,
            output_file=f'plots/{strategy_type}_ichimoku_components.png'
        )
    
    # 2. Plot signal comparison if signals exist
    if all(col in df.columns for col in ['ai_signal', 'ichimoku_signal', 'combined_signal']):
        sig_fig = viz.plot_signal_comparison(
            df,
            output_file=f'plots/{strategy_type}_signal_comparison.png'
        )
    
    # 3. Plot performance dashboard
    perf_fig = viz.plot_performance_dashboard(
        backtest_results,
        trades,
        metrics,
        output_file=f'plots/{strategy_type}_performance.png'
    )
    
    # 4. Plot equity curve and drawdowns from backtest
    # Create backtest engine for plotting
    from strategies.backtest_engine import BacktestEngine
    backtest_engine = BacktestEngine()
    
    if 'equity' in backtest_results.columns:
        try:
            equity_fig = backtest_engine.plot_equity_curve(backtest_results)
            if equity_fig:
                equity_fig.savefig(f'plots/{strategy_type}_equity_curve.png', dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.error(f"Error plotting equity curve: {str(e)}")
    
    # 5. Plot trade analysis
    if not trades.empty and 'pnl' in trades.columns:
        try:
            trade_fig = backtest_engine.plot_trade_analysis(trades)
            if trade_fig:
                trade_fig.savefig(f'plots/{strategy_type}_trade_analysis.png', dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.error(f"Error plotting trade analysis: {str(e)}")
    
    # 6. Plot monthly returns
    if 'daily_return' in backtest_results.columns:
        try:
            monthly_fig = backtest_engine.plot_monthly_returns(backtest_results)
            if monthly_fig:
                monthly_fig.savefig(f'plots/{strategy_type}_monthly_returns.png', dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.error(f"Error plotting monthly returns: {str(e)}")
    
    logger.info(f"Visualizations created for {strategy_type} strategy")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Crypto Trading AI with Ichimoku Cloud')
    
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'backtest', 'hybrid'],
                        help='Operation mode: train, backtest, or hybrid')
    
    parser.add_argument('--data', type=str, default='data/ETHUSDT_4h_data.csv',
                        help='Path to the data file')
    
    parser.add_argument('--model', type=str, default=None,
                        help='Path to a pretrained model (for backtest and hybrid modes)')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Number of data points to use (use 0 for all)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger()
    logger.info("Starting the cryptocurrency trading system")
    logger.info(f"Running in {args.mode} mode")
    
    # Load and prepare data
    raw_df = load_data(args.data)
    df = prepare_data(raw_df)

    # Use a subset of data if specified
    if args.sample_size > 0 and len(df) > args.sample_size:
        print(f"Reducing dataset from {len(df)} to {args.sample_size} rows")
        df = df.iloc[-args.sample_size:].copy()  # Use most recent rows
    
    if args.mode == 'train':
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
                trained_agent.save_model('models/final_ppo_actor.weights.h5', 'models/final_ppo_critic.weights.h5')
                logger.info("Final model saved")
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    elif args.mode == 'backtest':
        # Load the model
        model_path = args.model if args.model else 'models/final_ppo_actor.weights.h5'
        critic_path = model_path.replace('actor', 'critic')
        
        # Initialize agent
        agent = StablePPOAgent(
            state_size=len(df.columns),
            action_bounds=[0.1, 1.0],
            clip_ratio=0.2,
            policy_lr=0.0005,
            value_lr=0.001,
            entropy_beta=0.01
        )
        
        # Load model weights
        try:
            agent.load_model(model_path, critic_path)
            logger.info(f"Model loaded from {model_path} and {critic_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return
        
        # Calculate Ichimoku signals
        ichimoku_df = calculate_ichimoku_signals(df)
        
        # Run AI predictions
        ai_results = run_ai_trading(df, agent, plot=args.visualize)
        
        # Run backtest for Ichimoku strategy
        ich_results, ich_trades, ich_metrics = run_backtest(ichimoku_df, "ichimoku")
        
        # Run backtest for AI strategy
        ai_backtest, ai_trades, ai_metrics = run_backtest(ai_results, "ai")
        
        # Visualize results if requested
        if args.visualize:
            # Visualize Ichimoku strategy
            backtest = BacktestEngine()  # Create an instance for plotting
            visualize_results(ichimoku_df, ich_results, ich_trades, ich_metrics, "ichimoku")
            
            # Visualize AI strategy
            visualize_results(ai_results, ai_backtest, ai_trades, ai_metrics, "ai")
    
    elif args.mode == 'hybrid':
        # Load the model
        model_path = args.model if args.model else 'models/final_ppo_actor.weights.h5'
        critic_path = model_path.replace('actor', 'critic')
        
        # Initialize agent
        agent = StablePPOAgent(
            state_size=len(df.columns),
            action_bounds=[0.1, 1.0],
            clip_ratio=0.2,
            policy_lr=0.0005,
            value_lr=0.001,
            entropy_beta=0.01
        )
        
        # Load model weights
        try:
            agent.load_model(model_path, critic_path)
            logger.info(f"Model loaded from {model_path} and {critic_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return
        
        # Calculate Ichimoku signals
        ichimoku_df = calculate_ichimoku_signals(df)
        
        # Run AI predictions
        ai_results = run_ai_trading(df, agent, plot=False)
        
        # Run hybrid strategy
        hybrid_results, hybrid_performance = run_hybrid_trading(df, ai_results)
        
        # Run backtest on hybrid strategy
        hybrid_backtest, hybrid_trades, hybrid_metrics = run_backtest(hybrid_results, "hybrid")
        
        # Visualize results if requested
        if args.visualize:
            backtest = BacktestEngine()  # Create an instance for plotting
            visualize_results(hybrid_results, hybrid_backtest, hybrid_trades, hybrid_metrics, "hybrid")
            
            # Compare strategies
            plt.figure(figsize=(15, 8))
            
            # Load Ichimoku and AI results if they exist
            try:
                ich_results = pd.read_csv('results/ichimoku_backtest_results.csv', index_col=0, parse_dates=True)
                plt.plot(ich_results.index, ich_results['equity'], label='Ichimoku Strategy')
            except:
                logger.warning("Ichimoku backtest results not found for comparison")
                
            try:
                ai_backtest = pd.read_csv('results/ai_backtest_results.csv', index_col=0, parse_dates=True)
                plt.plot(ai_backtest.index, ai_backtest['equity'], label='AI Strategy')
            except:
                logger.warning("AI backtest results not found for comparison")
            
            # Plot hybrid strategy
            plt.plot(hybrid_backtest.index, hybrid_backtest['equity'], label='Hybrid Strategy')
            
            plt.title('Strategy Comparison')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.legend()
            plt.grid(True)
            plt.savefig('plots/strategy_comparison.png', dpi=300, bbox_inches='tight')
    
    logger.info("Process completed successfully")

if __name__ == "__main__":
    main()