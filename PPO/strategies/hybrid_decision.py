import numpy as np
import pandas as pd

class HybridDecisionSystem:
    """
    A hybrid decision system that combines AI model predictions with 
    traditional Ichimoku trading signals.
    """
    
    def __init__(self, ai_weight=0.6, ichimoku_weight=0.4, risk_manager=None):
        """
        Initialize the hybrid decision system.
        
        Args:
            ai_weight (float): Weight given to AI model predictions (0-1)
            ichimoku_weight (float): Weight given to Ichimoku signals (0-1)
            risk_manager (object): Risk manager instance for position sizing
        """
        # Ensure weights sum to 1
        total_weight = ai_weight + ichimoku_weight
        self.ai_weight = ai_weight / total_weight
        self.ichimoku_weight = ichimoku_weight / total_weight
        self.risk_manager = risk_manager
        
        # Decision thresholds
        self.entry_threshold = 0.2  # Minimum combined score to enter a position
        self.exit_threshold = 0.1   # Threshold below which to exit a position
    
    def set_thresholds(self, entry_threshold=0.2, exit_threshold=0.1):
        """
        Set decision thresholds.
        
        Args:
            entry_threshold (float): Threshold to enter a position
            exit_threshold (float): Threshold to exit a position
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
    
    def calculate_combined_signal(self, ai_signal, ichimoku_signal):
        """
        Calculate the combined signal from AI and Ichimoku inputs.
        
        Args:
            ai_signal (float): Signal from AI model (-1 to 1)
            ichimoku_signal (float): Signal from Ichimoku analysis (-1 to 1)
            
        Returns:
            float: Combined signal value (-1 to 1)
        """
        # Basic weighted average
        combined_signal = (
            self.ai_weight * ai_signal + 
            self.ichimoku_weight * ichimoku_signal
        )
        
        # Enhance signal when both agree strongly (synergy effect)
        if ai_signal * ichimoku_signal > 0:  # Same direction
            agreement_strength = min(abs(ai_signal), abs(ichimoku_signal))
            synergy_boost = agreement_strength * 0.2  # Boost by up to 20%
            combined_signal *= (1 + synergy_boost)
        
        # Clip to ensure in range [-1, 1]
        combined_signal = np.clip(combined_signal, -1, 1)
        
        return combined_signal
    
    def calculate_confidence(self, ai_signal, ichimoku_signal):
        """
        Calculate confidence level based on agreement between signals.
        
        Args:
            ai_signal (float): Signal from AI model (-1 to 1)
            ichimoku_signal (float): Signal from Ichimoku analysis (-1 to 1)
            
        Returns:
            float: Confidence level (0 to 1)
        """
        # Base confidence on signal strength
        base_confidence = max(abs(ai_signal), abs(ichimoku_signal))
        
        # Agreement factor (1 when perfectly aligned, 0 when opposite)
        if abs(ai_signal) < 0.1 or abs(ichimoku_signal) < 0.1:
            # If either signal is very weak, we consider them neutral
            agreement = 0.5
        else:
            # Dot product normalized to [0,1] range
            agreement = (np.sign(ai_signal) * np.sign(ichimoku_signal) + 1) / 2
        
        # Final confidence is a blend of strength and agreement
        confidence = 0.7 * base_confidence + 0.3 * agreement
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def determine_position_size(self, combined_signal, confidence, current_price, available_capital):
        """
        Determine position size based on signal, confidence and available capital.
        
        Args:
            combined_signal (float): The combined trading signal (-1 to 1)
            confidence (float): Confidence level (0 to 1)
            current_price (float): Current asset price
            available_capital (float): Available capital for trading
            
        Returns:
            float: Position size in units of the asset
        """
        if self.risk_manager is not None:
            # Use the risk manager if available
            position_size = self.risk_manager.calculate_position_size(
                signal_strength=abs(combined_signal),
                confidence=confidence,
                price=current_price,
                capital=available_capital
            )
        else:
            # Basic position sizing based on signal strength and confidence
            # Start with base size of 1% of capital
            base_size = 0.01 * available_capital / current_price
            
            # Scale by signal strength and confidence
            signal_factor = abs(combined_signal)
            confidence_factor = confidence
            
            # Calculate position size
            position_size = base_size * signal_factor * confidence_factor
            
            # Cap at 10% of available capital
            max_size = 0.1 * available_capital / current_price
            position_size = min(position_size, max_size)
        
        return position_size
    
    def make_decision(self, ai_signal, ichimoku_signal, current_price, available_capital, current_position=0):
        """
        Make a trading decision based on AI and Ichimoku signals.
        
        Args:
            ai_signal (float): Signal from AI model (-1 to 1)
            ichimoku_signal (float): Signal from Ichimoku analysis (-1 to 1)
            current_price (float): Current asset price
            available_capital (float): Available capital for trading
            current_position (float): Current position size (negative for short)
            
        Returns:
            dict: Decision including action, position size, and confidence
        """
        # Calculate combined signal
        combined_signal = self.calculate_combined_signal(ai_signal, ichimoku_signal)
        
        # Calculate confidence
        confidence = self.calculate_confidence(ai_signal, ichimoku_signal)
        
        # Initialize decision dictionary
        decision = {
            'combined_signal': combined_signal,
            'confidence': confidence,
            'action': 'hold',
            'position_size': 0,
            'entry_price': current_price
        }
        
        # Determine action based on signal and current position
        signal_strength = abs(combined_signal)
        
        if current_position == 0:  # No current position
            if signal_strength > self.entry_threshold:
                if combined_signal > 0:
                    decision['action'] = 'buy'
                else:
                    decision['action'] = 'sell'
                
                # Calculate position size
                decision['position_size'] = self.determine_position_size(
                    combined_signal, confidence, current_price, available_capital
                )
                
                # Adjust sign for short positions
                if combined_signal < 0:
                    decision['position_size'] = -decision['position_size']
                    
        else:  # Existing position
            current_position_sign = np.sign(current_position)
            signal_sign = np.sign(combined_signal)
            
            # Check if we should exit or reverse position
            if signal_strength < self.exit_threshold or current_position_sign != signal_sign:
                if current_position > 0:
                    decision['action'] = 'sell'
                else:
                    decision['action'] = 'buy'
                
                # Close the current position
                decision['position_size'] = -current_position
                
                # If signal is strong in opposite direction, add a new position
                if signal_strength > self.entry_threshold and current_position_sign != signal_sign:
                    new_position_size = self.determine_position_size(
                        combined_signal, confidence, current_price, available_capital
                    )
                    
                    if combined_signal < 0:
                        new_position_size = -new_position_size
                        
                    decision['position_size'] += new_position_size
                    decision['action'] = 'reverse'
            
            # Check if we should add to position
            elif signal_strength > self.entry_threshold and current_position_sign == signal_sign:
                additional_size = self.determine_position_size(
                    combined_signal, confidence, current_price, available_capital
                )
                
                if combined_signal < 0:
                    additional_size = -additional_size
                
                # Only add if it would increase position in the same direction
                if (additional_size > 0 and current_position > 0) or (additional_size < 0 and current_position < 0):
                    decision['position_size'] = additional_size
                    decision['action'] = 'add'
        
        return decision
    
    def process_dataframe(self, df, ai_signals, ichimoku_signals, initial_capital=10000):
        """
        Process a dataframe of data to generate trading decisions.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            ai_signals (pd.Series): Series of AI-generated signals (-1 to 1)
            ichimoku_signals (pd.Series): Series of Ichimoku signals (-1 to 1)
            initial_capital (float): Initial trading capital
            
        Returns:
            pd.DataFrame: DataFrame with trading decisions added
        """
        # Initialize result dataframe
        result = df.copy()
        result['ai_signal'] = ai_signals
        result['ichimoku_signal'] = ichimoku_signals
        result['combined_signal'] = 0.0
        result['confidence'] = 0.0
        result['action'] = 'hold'
        result['position'] = 0.0
        result['position_value'] = 0.0
        result['capital'] = initial_capital
        
        # Current state
        position = 0
        available_capital = initial_capital
        
        # Process each row
        for i in range(len(result)):
            # Get current price and signals
            current_price = result['close'].iloc[i]
            ai_signal = result['ai_signal'].iloc[i]
            ichimoku_signal = result['ichimoku_signal'].iloc[i]
            
            # Make decision
            decision = self.make_decision(
                ai_signal, ichimoku_signal, current_price, 
                available_capital, position
            )
            
            # Update dataframe with decision
            result.loc[result.index[i], 'combined_signal'] = decision['combined_signal']
            result.loc[result.index[i], 'confidence'] = decision['confidence']
            result.loc[result.index[i], 'action'] = decision['action']
            
            # Update position based on decision
            if decision['action'] != 'hold':
                position += decision['position_size']
            
            # Record current position
            result.loc[result.index[i], 'position'] = position
            position_value = position * current_price
            result.loc[result.index[i], 'position_value'] = position_value
            
            # Update capital (simplified, assuming no transaction costs)
            available_capital = initial_capital - position_value
            result.loc[result.index[i], 'capital'] = available_capital
        
        return result
    
    def analyze_performance(self, result_df):
        """
        Analyze the performance of the hybrid strategy.
        
        Args:
            result_df (pd.DataFrame): DataFrame with trading decisions and results
            
        Returns:
            dict: Dictionary of performance metrics
        """
        # Calculate daily returns
        result_df['daily_return'] = result_df['close'].pct_change()
        
        # Calculate strategy returns based on position
        result_df['strategy_return'] = result_df['position'].shift(1) * result_df['daily_return']
        
        # Calculate cumulative returns
        result_df['cumulative_return'] = (1 + result_df['strategy_return']).cumprod() - 1
        
        # Count trades
        trades = result_df['action'].replace('hold', np.nan).dropna()
        total_trades = len(trades)
        buy_trades = len(trades[trades == 'buy'])
        sell_trades = len(trades[trades == 'sell'])
        
        # Calculate win rate (simplified)
        result_df['trade_profit'] = result_df['strategy_return'].copy()
        result_df.loc[result_df['action'] == 'hold', 'trade_profit'] = np.nan
        winning_trades = len(result_df[result_df['trade_profit'] > 0]['trade_profit'].dropna())
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate various metrics
        total_return = result_df['cumulative_return'].iloc[-1]
        max_drawdown = (result_df['cumulative_return'].cummax() - result_df['cumulative_return']).max()
        sharpe_ratio = (result_df['strategy_return'].mean() / result_df['strategy_return'].std()) * np.sqrt(252)
        
        # Create metrics dictionary
        metrics = {
            'total_return': total_return,
            'annualized_return': total_return / len(result_df) * 252,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'win_rate': win_rate
        }
        
        return metrics