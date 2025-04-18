import numpy as np
import pandas as pd
from gym import spaces

class EnhancedTradingEnv:
    """
    An enhanced trading environment with better reward shaping and stability
    """
    def __init__(
        self, 
        df, 
        holding_period=60, 
        risk_pct=0.02, 
        position_value=1000, 
        timeframe="4h",
        reward_scaling=1.0,
        reward_shaping=True,
        use_drawdown_penalty=True,
        additional_metrics=True,
        price_target_pct=0.05,
        risk_reward_ratio=5.0,
        ichimoku_warmup_period=52,
        inactivity_threshold=50
    ):
        """
        Initialize the environment with improved parameters
        
        Parameters:
        df (pandas.DataFrame): Historical price and indicator data
        holding_period (int): How many bars to hold a position
        risk_pct (float): Percentage of position value at risk
        position_value (float): Value of each trade
        timeframe (str): Data timeframe (e.g., "1h", "4h", "1d")
        reward_scaling (float): Scale factor for rewards
        reward_shaping (bool): Whether to use advanced reward shaping
        use_drawdown_penalty (bool): Whether to penalize large drawdowns
        additional_metrics (bool): Whether to track additional metrics
        price_target_pct (float): Maximum price target as percentage of entry price (5% = 0.05)
        risk_reward_ratio (float): Ratio of reward to risk (5.0 means potential reward is 5x the risk)
        ichimoku_warmup_period (int): Minimum periods to wait for Ichimoku indicators to be fully calculated
        inactivity_threshold (int): Number of periods of inactivity before encouraging trading
        """
        self.df = df
        self.holding_period = holding_period
        self.risk_pct = risk_pct
        self.position_value = position_value
        self.timeframe = timeframe
        self.reward_scaling = reward_scaling
        self.reward_shaping = reward_shaping
        self.use_drawdown_penalty = use_drawdown_penalty
        self.additional_metrics = additional_metrics
        self.price_target_pct = price_target_pct
        self.risk_reward_ratio = risk_reward_ratio
        self.ichimoku_warmup_period = ichimoku_warmup_period
        self.inactivity_threshold = inactivity_threshold
        self.periods_since_last_trade = 0
        
        # Calculate funding rate multiplier based on timeframe
        self.funding_rate_multiplier = self._calculate_funding_multiplier(timeframe)

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1, 0.0]),  
            high=np.array([1, 1.0]),
            shape=(2,),
            dtype=np.float32
        )  # [direction, size]
        
        # Normalize state space for stability
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, 
            shape=(len(self.df.columns),), 
            dtype=np.float32
        )
        
        # Enhanced statistics for better normalization
        self._compute_statistics()
        
        # Initialize state variables
        self.reset()
        
        # Tracking variables for reward shaping
        self.portfolio_value = 1000.0  # Initial portfolio value
        self.max_portfolio_value = 1000.0
        self.consecutive_losses = 0
        self.trade_outcomes = []  # Store outcomes for adaptive shaping
        
    def _compute_statistics(self):
        """Compute statistics for better normalization, only for numeric columns"""
        # Compute rolling statistics for more robust normalization
        # Use rolling mean and std over a window to reduce outlier impact
        window = min(500, len(self.df) // 4)
        
        # Rolling means and stds
        self.feature_means = {}
        self.feature_stds = {}
        self.feature_mins = {}
        self.feature_maxs = {}
        
        for col in self.df.columns:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(self.df[col]):
                try:
                    self.feature_means[col] = self.df[col].mean()
                    self.feature_stds[col] = self.df[col].std()
                    self.feature_mins[col] = self.df[col].quantile(0.01)  # 1st percentile
                    self.feature_maxs[col] = self.df[col].quantile(0.99)  # 99th percentile
                    
                    # Ensure non-zero std
                    if np.isclose(self.feature_stds[col], 0) or np.isnan(self.feature_stds[col]):
                        self.feature_stds[col] = 1.0
                except Exception as e:
                    print(f"Warning: Could not compute statistics for column {col}: {e}")
                    self.feature_means[col] = 0.0
                    self.feature_stds[col] = 1.0
                    self.feature_mins[col] = 0.0
                    self.feature_maxs[col] = 1.0
            else:
                # For non-numeric columns, set default values
                print(f"Skipping non-numeric column for statistics: {col}")
                self.feature_means[col] = 0.0
                self.feature_stds[col] = 1.0
                self.feature_mins[col] = 0.0
                self.feature_maxs[col] = 1.0
    
    def normalize_state(self, features):
        """Normalize state with robust scaling"""
        normalized = []
        
        for i, col in enumerate(self.df.columns):
            value = features[i]
            mean = self.feature_means.get(col, 0)
            std = self.feature_stds.get(col, 1)
            
            # Z-score normalization with clipping
            norm_val = (value - mean) / std
            norm_val = np.clip(norm_val, -3.0, 3.0)  # Clip to reasonable range
            normalized.append(norm_val)
            
        return np.array(normalized, dtype=np.float32)
    
    def denormalize_state(self, normalized_features):
        """Denormalize state for debugging"""
        original = []
        
        for i, col in enumerate(self.df.columns):
            norm_val = normalized_features[i]
            mean = self.feature_means.get(col, 0)
            std = self.feature_stds.get(col, 1)
            
            # Inverse of Z-score normalization
            value = norm_val * std + mean
            original.append(value)
            
        return np.array(original)
    
    def reset(self):
        """Reset the environment with bounds checking to prevent out-of-bounds errors"""
        # Make sure we have enough data to work with
        if len(self.df) <= self.holding_period * 3:
            raise ValueError(f"Dataframe too small ({len(self.df)} rows) for holding period of {self.holding_period}")
        
        # Ensure we have enough history and future data
        # Start at a position that leaves enough room for the holding period and Ichimoku indicators
        min_start = max(self.ichimoku_warmup_period, self.holding_period // 2)  # At least ichimoku_warmup_period rows in
        
        # Make sure we have enough room after the start position
        max_start = len(self.df) - self.holding_period - 10  # Leave room for holding period + buffer
        
        # Safeguard against invalid ranges
        if max_start <= min_start:
            print(f"Warning: Adjusting start range due to small dataframe. min_start={min_start}, max_start={max_start}")
            max_start = min_start + 1  # Ensure valid range
        
        # Set start position with safety check
        try:
            self.current_step = np.random.randint(min_start, max_start)
        except ValueError as e:
            print(f"Error with random start: {e}. Using min_start={min_start}")
            self.current_step = min_start
        
        print(f"Reset environment, starting at step {self.current_step} of {len(self.df)} rows")
        
        # Reset trade tracking
        self.trade_entry_price = None
        self.trade_entry_step = None
        self.trade_direction = None
        self.trade_size = None
        self.trade_price_target = None
        self.trade_stop_loss = None
        self.total_trades = 0
        self.successful_trades = 0
        self.trades_info = []
        self.episode_pnl = 0.0
        self.active_trade = False
        self.periods_since_last_trade = 0

        # Reset portfolio tracking
        self.portfolio_value = 1000.0
        self.max_portfolio_value = 1000.0
        self.consecutive_losses = 0
        
        # Get initial state
        return self._get_state()

    
    def _get_state(self):
        """Get normalized state representation with improved error handling"""
        current_idx = self.current_step
        
        # Extract raw features from numeric columns only
        features = []
        
        for col in self.df.columns:
            if col in self.feature_means and col in self.feature_stds:
                try:
                    # Get value safely
                    value = float(self.df.iloc[current_idx][col])
                    
                    # Check for NaN or inf
                    if np.isnan(value) or np.isinf(value):
                        print(f"Warning: NaN or inf value in column {col} at step {current_idx}")
                        value = self.feature_means[col]  # Use mean as fallback
                    
                    # Normalize using pre-computed statistics
                    mean = self.feature_means[col]
                    std = self.feature_stds[col]
                    
                    # Z-score normalization with clipping
                    norm_val = (value - mean) / (std + 1e-8)
                    norm_val = np.clip(norm_val, -3.0, 3.0)  # Clip to reasonable range
                    
                    features.append(norm_val)
                except Exception as e:
                    print(f"Error processing column {col}: {e}")
                    # Use 0 as a fallback (this should be rare if prepare_data is used)
                    features.append(0.0)
        
        # Ensure we have at least one feature
        if not features:
            raise ValueError("No valid features found in the dataframe")
        
        return np.array(features, dtype=np.float32)
    

    def step(self, action):
        """
        Take a step with improved reward calculation and tracking
        
        Parameters:
        action: Dictionary with 'direction' (-1: sell, 1: buy) and 'size' (0.0-1.0)
        """
        # Handle different action formats
        if isinstance(action, dict):
            direction = action['direction']
            position_size = action['size']
        elif isinstance(action, np.ndarray) and len(action) == 2:
            direction = 1 if action[0] > 0 else -1
            position_size = action[1]
        else:
            # Fallback for backward compatibility
            direction = 1
            position_size = float(action) if isinstance(action, (int, float)) else 0.5
        
        # Ensure position size is within bounds
        position_size = np.clip(position_size, 0.1, 1.0)
        
        # Check if we've reached the end of data
        if self.current_step >= len(self.df) - 1:
            print(f"Reached end of data at step {self.current_step}")
            done = True
            
            # If we have an active trade, close it at the last price
            if self.active_trade:
                last_price = float(self.df.iloc[-1]['close'])
                reward, trade_info = self._close_trade(last_price)
                
                # Update info with trade results
                info = {
                    'trade_taken': True,
                    'position_size': self.trade_size,
                    'total_trades': self.total_trades,
                    'win_rate': self.successful_trades / max(1, self.total_trades),
                    'episode_pnl': self.episode_pnl + trade_info['pnl'],
                    'periods_since_last_trade': self.periods_since_last_trade
                }
                info.update(trade_info)
                
                # Scale final reward
                reward *= self.reward_scaling
                
                # Return final state
                next_state = self._get_state()
                return next_state, reward, done, info
            
            # No active trades, return default info
            next_state = self._get_state()  # Get final state
            return next_state, 0.0, done, {
                'trade_taken': False,
                'position_size': 0.0,
                'total_trades': self.total_trades,
                'win_rate': self.successful_trades / max(1, self.total_trades),
                'periods_since_last_trade': self.periods_since_last_trade
            }
        else:
            done = False
        
        # Default reward
        reward = 0.0
        info = {
            'trade_taken': False,
            'position_size': 0.0,
            'total_trades': self.total_trades,
            'win_rate': 0.0,
            'episode_pnl': 0.0,
            'periods_since_last_trade': self.periods_since_last_trade
        }
        
        # Check if we're in the warmup period - no trading allowed during warmup
        in_warmup = self.current_step < self.ichimoku_warmup_period
        if in_warmup:
            info['in_warmup'] = True
            self.current_step += 1
            return self._get_state(), -0.01, done, info
        
        # First check if we have an active trade that needs to be managed
        if self.active_trade:
            # Check if any exit conditions have been met
            exit_result = self._check_exit_conditions()
            
            if exit_result:
                # Trade has been closed
                reward, trade_info = exit_result
                
                # Update info
                info.update(trade_info)
                info['trade_taken'] = True
                info['position_size'] = 0  # Position is now closed
                
                # Update portfolio value based on reward
                self.portfolio_value += reward
                self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
                self.episode_pnl += trade_info['pnl'] if 'pnl' in trade_info else 0
                info['episode_pnl'] = self.episode_pnl
                
                # Advance step after closing trade
                self.current_step += 1
                self.periods_since_last_trade = 0  # Reset inactivity counter when a trade is closed
            else:
                # Trade is still active, just advance the step
                self.current_step += 1
        else:
            # No active trade, determine if a new trade should be taken
            take_trade = abs(direction) > 0 and position_size >= 0.1
            
            # Add incentive for trading after inactivity period
            inactivity_bonus = 0.0
            force_trade = False
            
            # If we've been inactive for too long, add incentive to trade
            if self.periods_since_last_trade >= self.inactivity_threshold:
                # Provide progressively stronger incentives to trade as inactivity increases
                inactivity_bonus = min(1.0, (self.periods_since_last_trade - self.inactivity_threshold) / 20.0)
                info['inactivity_bonus'] = inactivity_bonus
                
                # Force a trade if we've been inactive for too long (50 periods beyond threshold)
                if self.periods_since_last_trade >= (self.inactivity_threshold + 50) and abs(direction) > 0:
                    force_trade = True
                    info['forced_trade'] = True
            
            # Check if we can execute the trade (enough data remaining)
            if take_trade or force_trade:
                # Check if we have enough data remaining
                if self.current_step + self.holding_period >= len(self.df):
                    print(f"Not enough data for max holding period at step {self.current_step}, skipping trade")
                    take_trade = False
                    reward = -0.01  # Small penalty
                    self.current_step += 1  # Advance step
                    self.periods_since_last_trade += 1  # Increment inactivity counter
                else:
                    # Adjust position size if this is a forced trade
                    if force_trade:
                        # Use smaller position size for forced trades as a safety measure
                        position_size = max(0.1, position_size * 0.5)
                    
                    # Execute trade - this will internally advance the step
                    reward, trade_info = self._execute_trade(direction, position_size)
                    
                    # Add inactivity bonus to reward if applicable
                    if inactivity_bonus > 0:
                        reward += inactivity_bonus
                    
                    # Update info
                    info.update(trade_info)
                    info['trade_taken'] = True
                    info['position_size'] = position_size
                    
                    # Update portfolio value based on any immediate reward
                    if reward != 0:
                        self.portfolio_value += reward
                        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
                        if 'pnl' in trade_info:
                            self.episode_pnl += trade_info['pnl']
                            info['episode_pnl'] = self.episode_pnl
                    
                    # Reset inactivity counter when a new trade is taken
                    self.periods_since_last_trade = 0
            else:
                # No trade taken, advance step
                self.current_step += 1
                self.periods_since_last_trade += 1  # Increment inactivity counter
                # Small penalty for not trading to encourage action
                reward = -0.01
        
        # Calculate win rate
        if self.total_trades > 0:
            info['win_rate'] = self.successful_trades / self.total_trades
        
        # Apply drawdown penalty if enabled
        if self.use_drawdown_penalty and self.max_portfolio_value > 0:
            drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            if drawdown > 0.1:  # Penalize drawdowns over 10%
                drawdown_penalty = -0.2 * drawdown
                reward += drawdown_penalty
                info['drawdown_penalty'] = drawdown_penalty
        
        # Scale final reward
        reward *= self.reward_scaling
        
        # Get next state
        next_state = self._get_state()
        
        return next_state, reward, done, info

    def _execute_trade(self, direction, position_size):
        """Execute a trade and calculate reward with safety checks for boundaries"""
        # Safety check - ensure we're within dataframe bounds
        if self.current_step >= len(self.df):
            print(f"Warning: Attempted to execute trade at step {self.current_step} but dataframe only has {len(self.df)} rows")
            # Return minimal data to avoid crashing
            return 0.0, {'entry_price': 0.0, 'exit_price': 0.0, 'pnl': 0.0, 'pnl_pct': 0.0, 
                    'funding_penalty': 0.0, 'mae_pct': 0.0, 'direction': direction, 
                    'holding_periods': 0}
        
        # Check if we already have an active trade
        if self.active_trade:
            # Check if we've hit price target or stop loss
            exit_result = self._check_exit_conditions()
            if exit_result:
                return exit_result
        
        # This is a new trade
        self.total_trades += 1
        self.active_trade = True
        
        # Calculate scaled position value
        scaled_position = self.position_value * position_size
        
        # Entry price and step
        entry_price = float(self.df.iloc[self.current_step]['close'])
        self.trade_entry_step = self.current_step
        self.trade_entry_price = entry_price
        self.trade_direction = direction
        self.trade_size = position_size
        
        # Calculate price target and stop loss based on risk/reward ratio
        # For long trades: target is higher, stop loss is lower
        # For short trades: target is lower, stop loss is higher
        if direction > 0:  # Long
            # Calculate maximum target (capped at price_target_pct)
            max_target_pct = min(self.price_target_pct, 1.0)  # Ensure it's at most 100%
            target_pct = max_target_pct
            
            # Calculate stop loss based on risk/reward ratio
            stop_loss_pct = target_pct / self.risk_reward_ratio
            
            # Calculate actual prices
            self.trade_price_target = entry_price * (1 + target_pct)
            self.trade_stop_loss = entry_price * (1 - stop_loss_pct)
            
        else:  # Short
            # Calculate maximum target (capped at price_target_pct)
            max_target_pct = min(self.price_target_pct, 1.0)  # Ensure it's at most 100%
            target_pct = max_target_pct
            
            # Calculate stop loss based on risk/reward ratio
            stop_loss_pct = target_pct / self.risk_reward_ratio
            
            # Calculate actual prices
            self.trade_price_target = entry_price * (1 - target_pct)
            self.trade_stop_loss = entry_price * (1 + stop_loss_pct)
        
        # Advance one step to avoid instant exit
        self.current_step += 1
        
        # Check if we've reached the end of data
        if self.current_step >= len(self.df) - 1:
            # Force exit at the end of data
            result = self._close_trade(float(self.df.iloc[-1]['close']))
            return result
        
        # Return partial info but no reward yet since trade is still open
        info = {
            'entry_price': entry_price,
            'price_target': self.trade_price_target,
            'stop_loss': self.trade_stop_loss,
            'direction': direction,
            'trade_opened': True
        }
        
        # No reward yet as the trade is still active
        return 0.0, info
    
    def _check_exit_conditions(self):
        """Check if price target or stop loss has been hit"""
        if not self.active_trade:
            return None
            
        # Get current OHLC data for this step
        if self.current_step >= len(self.df):
            # We've reached the end of data, force close at the last price
            return self._close_trade(float(self.df.iloc[-1]['close']))
            
        current_data = self.df.iloc[self.current_step]
        has_high = 'high' in self.df.columns
        has_low = 'low' in self.df.columns
        
        # Default to close price if high/low not available
        high_price = float(current_data['high']) if has_high else float(current_data['close'])
        low_price = float(current_data['low']) if has_low else float(current_data['close'])
        close_price = float(current_data['close'])
        
        # Check exit conditions based on direction
        if self.trade_direction > 0:  # Long
            # Check if high price hit target
            if high_price >= self.trade_price_target:
                return self._close_trade(self.trade_price_target)
                
            # Check if low price hit stop loss
            if low_price <= self.trade_stop_loss:
                return self._close_trade(self.trade_stop_loss)
                
        else:  # Short
            # Check if low price hit target
            if low_price <= self.trade_price_target:
                return self._close_trade(self.trade_price_target)
                
            # Check if high price hit stop loss
            if high_price >= self.trade_stop_loss:
                return self._close_trade(self.trade_stop_loss)
        
        # Emergency exit if we've held for too long (max holding period)
        holding_time = self.current_step - self.trade_entry_step
        if holding_time >= self.holding_period:
            return self._close_trade(close_price)
            
        # No exit conditions met
        return None
        
    def _close_trade(self, exit_price):
        """Close the trade and calculate performance metrics"""
        if not self.active_trade:
            return 0.0, {'trade_closed': False}
            
        # Calculate P&L based on direction
        entry_price = self.trade_entry_price
        
        if self.trade_direction > 0:  # Long
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # Short
            pnl_pct = (entry_price - exit_price) / entry_price
        
        # Calculate scaled position value
        scaled_position = self.position_value * self.trade_size
        
        # Calculate P&L
        pnl = pnl_pct * scaled_position
        
        # Calculate holding period
        entry_step = self.trade_entry_step
        holding_periods = self.current_step - entry_step
        
        # Apply funding rate penalty if applicable
        funding_penalty = self._calculate_funding_penalty(
            entry_step, 
            self.current_step, 
            self.trade_direction, 
            self.trade_size
        )
        
        # Calculate risk amount
        risk_amount = self.risk_pct * scaled_position
        
        # Base reward on risk-adjusted return
        if risk_amount > 0:
            reward = pnl / risk_amount
        else:
            reward = pnl
            
        # Apply funding penalty
        reward += funding_penalty
        
        # Track if this was a successful trade
        if pnl > 0:
            self.successful_trades += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        # Reward shaping if enabled
        if self.reward_shaping:
            # Reward consistency: bonus for lower variance outcomes
            if len(self.trade_outcomes) > 5:
                consistency_bonus = 0.05 * (1.0 - np.std(self.trade_outcomes[-5:]) / (np.abs(np.mean(self.trade_outcomes[-5:])) + 1e-8))
                consistency_bonus = np.clip(consistency_bonus, 0, 0.2)
                reward += consistency_bonus
            
            # Risk management bonus: Reward for proper sizing
            # Larger positions should be taken when confidence is higher (historical win rate)
            if self.total_trades > 10:
                win_rate = self.successful_trades / self.total_trades
                optimal_size = win_rate  # Simple linear relationship
                sizing_error = abs(self.trade_size - optimal_size)
                sizing_penalty = -0.1 * sizing_error
                reward += sizing_penalty
        
        # Store trade info
        self.trade_outcomes.append(reward)
        
        # Calculate MAE (Maximum Adverse Excursion)
        # Find the range of steps to analyze
        start_step = entry_step
        end_step = self.current_step
        
        # Safety check - ensure valid range
        if start_step >= end_step:
            print(f"Warning: Invalid step range {start_step}:{end_step}")
            start_step = max(0, end_step - 1)
        
        # Ensure 'high' and 'low' columns exist, otherwise use 'close'
        has_low = 'low' in self.df.columns
        has_high = 'high' in self.df.columns
        
        if self.trade_direction > 0:  # Long
            # Find lowest price during the holding period
            if has_low:
                mae_price = self.df.iloc[start_step:end_step+1]['low'].min()
            else:
                mae_price = self.df.iloc[start_step:end_step+1]['close'].min()
            mae_pct = (entry_price - mae_price) / entry_price
        else:  # Short
            # Find highest price during the holding period
            if has_high:
                mae_price = self.df.iloc[start_step:end_step+1]['high'].max()
            else:
                mae_price = self.df.iloc[start_step:end_step+1]['close'].max()
            mae_pct = (mae_price - entry_price) / entry_price
        
        # Reset trade tracking
        self.active_trade = False
        self.trade_entry_price = None
        self.trade_entry_step = None
        self.trade_direction = None
        self.trade_size = None
        self.trade_price_target = None
        self.trade_stop_loss = None
        
        # Return reward and trade info
        info = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct * 100,
            'funding_penalty': funding_penalty,
            'mae_pct': mae_pct * 100,
            'direction': self.trade_direction,
            'holding_periods': holding_periods,
            'trade_closed': True
        }
        
        return reward, info
        
    def _calculate_funding_penalty(self, entry_step, exit_step, direction, position_size):
        """Calculate funding rate penalty with realistic weighting and boundary checks"""
        funding_penalty = 0.0
        
        # Safety check - ensure valid range
        if entry_step >= len(self.df) or exit_step >= len(self.df):
            print(f"Warning: Invalid step range {entry_step}:{exit_step} for funding calculation")
            return 0.0
        
        # Define safe range
        start_step = min(entry_step, len(self.df) - 1)
        end_step = min(exit_step, len(self.df) - 1)
        
        # Only calculate if we have fundingRate column
        if 'fundingRate' in self.df.columns:
            for step in range(start_step, end_step + 1):
                if step < len(self.df):  # Safety check
                    try:
                        funding_rate = float(self.df.iloc[step]['fundingRate']) * self.funding_rate_multiplier
                        
                        # Apply realistic funding impact:
                        # - For long positions: negative funding rate is bad
                        # - For short positions: positive funding rate is bad
                        impact = -funding_rate * direction
                        
                        # Scale by position size and duration
                        funding_penalty += impact * position_size * 0.1  # Reduced factor for realism
                    except Exception as e:
                        print(f"Error calculating funding at step {step}: {e}")
        else:
            print("Warning: 'fundingRate' column not found, funding penalty set to 0")
        
        return funding_penalty
    
    def _calculate_funding_multiplier(self, timeframe):
        """Calculate funding rate multiplier based on data timeframe"""
        # Standard funding period is 8 hours for most exchanges
        FUNDING_PERIOD_HOURS = 8
        
        timeframe_multipliers = {
            "1h": 1 / FUNDING_PERIOD_HOURS,
            "4h": 1 / 2,
            "8h": 1,
            "1d": FUNDING_PERIOD_HOURS / 24 * 3,
            "15m": 1 / (FUNDING_PERIOD_HOURS * 4)
        }
        
        return timeframe_multipliers.get(timeframe, 1 / 2)  # Default to 4h