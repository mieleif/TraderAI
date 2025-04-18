import numpy as np
import pandas as pd

class IchimokuSignals:
    """
    A class to generate trading signals based on Ichimoku Cloud indicators.
    Implements various Ichimoku-based signals with confidence scoring.
    """
    
    def __init__(self):
        """Initialize the IchimokuSignals class."""
        pass
    
    def calculate_ichimoku(self, df, tenkan_period=9, kijun_period=26, senkou_span_b_period=52, displacement=26):
        """
        Calculate Ichimoku Cloud components.
        
        Args:
            df (pd.DataFrame): Price data with 'high', 'low', 'close' columns
            tenkan_period (int): Period for Tenkan-sen calculation
            kijun_period (int): Period for Kijun-sen calculation
            senkou_span_b_period (int): Period for Senkou Span B calculation
            displacement (int): Displacement period for Senkou spans
            
        Returns:
            pd.DataFrame: DataFrame with Ichimoku components added
        """
        # Make a copy of the dataframe to avoid modifying the original
        ichimoku_df = df.copy()
        
        # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for the past 9 periods
        ichimoku_df['tenkan_sen'] = (
            ichimoku_df['high'].rolling(window=tenkan_period).max() + 
            ichimoku_df['low'].rolling(window=tenkan_period).min()
        ) / 2
        
        # Kijun-sen (Base Line): (highest high + lowest low)/2 for the past 26 periods
        ichimoku_df['kijun_sen'] = (
            ichimoku_df['high'].rolling(window=kijun_period).max() + 
            ichimoku_df['low'].rolling(window=kijun_period).min()
        ) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 displaced forward 26 periods
        ichimoku_df['senkou_span_a'] = ((ichimoku_df['tenkan_sen'] + ichimoku_df['kijun_sen']) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for the past 52 periods, displaced forward 26 periods
        ichimoku_df['senkou_span_b'] = ((
            ichimoku_df['high'].rolling(window=senkou_span_b_period).max() + 
            ichimoku_df['low'].rolling(window=senkou_span_b_period).min()
        ) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span): Close price displaced backward 26 periods
        ichimoku_df['chikou_span'] = ichimoku_df['close'].shift(-displacement)
        
        return ichimoku_df
    
    def detect_tk_cross(self, df, window=3):
        """
        Detect Tenkan-Kijun crosses (TK cross).
        
        Args:
            df (pd.DataFrame): DataFrame with Ichimoku components
            window (int): Window to check for crosses
            
        Returns:
            pd.Series: Signal series (1 for bullish cross, -1 for bearish cross, 0 for no cross)
        """
        signals = pd.Series(0, index=df.index)
        
        # Bullish TK Cross: Tenkan crosses above Kijun
        bullish_cross = (
            (df['tenkan_sen'] > df['kijun_sen']) & 
            (df['tenkan_sen'].shift(1) <= df['kijun_sen'].shift(1))
        )
        
        # Bearish TK Cross: Tenkan crosses below Kijun
        bearish_cross = (
            (df['tenkan_sen'] < df['kijun_sen']) & 
            (df['tenkan_sen'].shift(1) >= df['kijun_sen'].shift(1))
        )
        
        signals[bullish_cross] = 1
        signals[bearish_cross] = -1
        
        # Calculate strength of the cross based on distance and angle
        cross_strength = pd.Series(0.0, index=df.index)
        for i in range(len(df) - 1):
            idx = df.index[i]
            if signals.loc[idx] != 0:
                # Calculate the distance between the lines
                distance = abs(df['tenkan_sen'].iloc[i] - df['kijun_sen'].iloc[i])
                
                # Calculate the angle of the cross (using the change in the ratio over the last few periods)
                tk_ratio_change = abs(
                    (df['tenkan_sen'].iloc[i] / df['kijun_sen'].iloc[i]) - 
                    (df['tenkan_sen'].iloc[i-1] / df['kijun_sen'].iloc[i-1])
                )
                
                # Normalize and combine
                cross_strength.loc[idx] = min(distance * tk_ratio_change * 10, 1.0)
        
        # Combine signal direction with strength
        signals = signals * (0.5 + cross_strength/2)
        
        return signals
    
    def detect_kumo_breakout(self, df, window=1):
        """
        Detect price breaking above or below the Kumo (cloud).
        
        Args:
            df (pd.DataFrame): DataFrame with Ichimoku components
            window (int): Window to confirm the breakout
            
        Returns:
            pd.Series: Signal series (1 for bullish breakout, -1 for bearish breakout, 0 for no breakout)
        """
        signals = pd.Series(0, index=df.index)
        
        # Upper edge of the cloud
        kumo_upper = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        
        # Lower edge of the cloud
        kumo_lower = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
        
        # Bullish breakout: Price closes above the cloud after being below/in the cloud
        bullish_breakout = (
            (df['close'] > kumo_upper) & 
            (df['close'].shift(window) <= kumo_upper.shift(window))
        )
        
        # Bearish breakout: Price closes below the cloud after being above/in the cloud
        bearish_breakout = (
            (df['close'] < kumo_lower) & 
            (df['close'].shift(window) >= kumo_lower.shift(window))
        )
        
        signals[bullish_breakout] = 1
        signals[bearish_breakout] = -1
        
        # Calculate breakout strength based on distance from cloud
        breakout_strength = pd.Series(0.0, index=df.index)
        for i in range(len(df)):
            idx = df.index[i]
            if signals.loc[idx] == 1:  # Bullish breakout
                # Distance from price to upper edge of cloud
                distance = (df['close'].iloc[i] - kumo_upper.iloc[i]) / df['close'].iloc[i]
                breakout_strength.loc[idx] = min(distance * 20, 1.0)  # Normalize
            elif signals.loc[idx] == -1:  # Bearish breakout
                # Distance from price to lower edge of cloud
                distance = (kumo_lower.iloc[i] - df['close'].iloc[i]) / df['close'].iloc[i]
                breakout_strength.loc[idx] = min(distance * 20, 1.0)  # Normalize
        
        # Combine signal direction with strength
        signals = signals * (0.5 + breakout_strength/2)
        
        return signals
    
    def detect_kumo_twist(self, df, forecast_period=26):
        """
        Detect Kumo twists (when Senkou Span A crosses Senkou Span B).
        
        Args:
            df (pd.DataFrame): DataFrame with Ichimoku components
            forecast_period (int): How many periods ahead to look for twists
            
        Returns:
            pd.Series: Signal series (1 for bullish twist, -1 for bearish twist, 0 for no twist)
        """
        signals = pd.Series(0, index=df.index)
        
        # Future bullish twist: Senkou Span A crosses above Senkou Span B in the future cloud
        bullish_twist = (
            (df['senkou_span_a'] > df['senkou_span_b']) & 
            (df['senkou_span_a'].shift(1) <= df['senkou_span_b'].shift(1))
        )
        
        # Future bearish twist: Senkou Span A crosses below Senkou Span B in the future cloud
        bearish_twist = (
            (df['senkou_span_a'] < df['senkou_span_b']) & 
            (df['senkou_span_a'].shift(1) >= df['senkou_span_b'].shift(1))
        )
        
        # Adjust the signals to match when the twist will become visible to traders
        signals[bullish_twist] = 1
        signals[bearish_twist] = -1
        
        # Calculate twist strength based on distance between lines
        twist_strength = pd.Series(0.0, index=df.index)
        for i in range(len(df) - 1):
            idx = df.index[i]
            if signals.loc[idx] != 0:
                # Calculate the distance between the lines at the twist
                distance = abs(df['senkou_span_a'].iloc[i] - df['senkou_span_b'].iloc[i])
                
                # Normalize
                twist_strength.loc[idx] = min(distance / df['close'].iloc[i] * 30, 1.0)
        
        # Combine signal direction with strength
        signals = signals * (0.5 + twist_strength/2)
        
        return signals
    
    def detect_price_kijun_relationship(self, df):
        """
        Analyze the relationship between price and Kijun-sen (Base Line).
        
        Args:
            df (pd.DataFrame): DataFrame with Ichimoku components
            
        Returns:
            pd.Series: Signal series (positive for bullish, negative for bearish)
        """
        signals = pd.Series(0.0, index=df.index)
        
        # Calculate distance between price and Kijun-sen
        distance = (df['close'] - df['kijun_sen']) / df['close']
        
        # Calculate a smoothed slope of the Kijun-sen
        kijun_slope = (df['kijun_sen'] - df['kijun_sen'].shift(3)) / df['kijun_sen'].shift(3)
        
        # Combine distance and slope for signaling
        for i in range(len(df) - 3):
            idx = df.index[i]
            # If price is above Kijun
            if df['close'].iloc[i] > df['kijun_sen'].iloc[i]:
                # Calculate bullish signal strength based on distance and kijun slope
                signal_strength = distance.iloc[i] * 5  # Scale factor for distance
                if kijun_slope.iloc[i] > 0:  # Upward sloping Kijun strengthens bullish signal
                    signal_strength *= (1 + kijun_slope.iloc[i] * 10)
                signals.loc[idx] = min(signal_strength, 1.0)  # Cap at 1.0
                
            # If price is below Kijun
            elif df['close'].iloc[i] < df['kijun_sen'].iloc[i]:
                # Calculate bearish signal strength based on distance and kijun slope
                signal_strength = distance.iloc[i] * 5  # Already negative due to distance calculation
                if kijun_slope.iloc[i] < 0:  # Downward sloping Kijun strengthens bearish signal
                    signal_strength *= (1 + abs(kijun_slope.iloc[i]) * 10)
                signals.loc[idx] = max(signal_strength, -1.0)  # Cap at -1.0
        
        return signals
    
    def detect_chikou_span_confirmation(self, df, confirmation_period=26):
        """
        Check if Chikou Span confirms the trend by comparing it with price.
        
        Args:
            df (pd.DataFrame): DataFrame with Ichimoku components
            confirmation_period (int): Period to look back for confirmation
            
        Returns:
            pd.Series: Signal series (positive for bullish, negative for bearish)
        """
        signals = pd.Series(0.0, index=df.index)
        
        # Only calculate for valid data points
        valid_range = df.index[confirmation_period:]
        
        for i in range(confirmation_period, len(df)):
            current_idx = df.index[i]
            
            # Get chikou span (current close price shifted back 26 periods)
            chikou = df['chikou_span'].iloc[i-confirmation_period]
            
            # Get historical price at the same point chikou is referring to
            if i >= confirmation_period * 2:
                hist_price = df['close'].iloc[i-confirmation_period*2]
                hist_high = df['high'].iloc[i-confirmation_period*2]
                hist_low = df['low'].iloc[i-confirmation_period*2]
                
                # Chikou above historical price is bullish
                if chikou > hist_price:
                    # Calculate strength based on how far above the price range chikou is
                    strength = (chikou - hist_price) / (hist_high - hist_low) if (hist_high - hist_low) > 0 else 0.5
                    signals.loc[current_idx] = min(strength, 1.0)
                
                # Chikou below historical price is bearish
                elif chikou < hist_price:
                    # Calculate strength based on how far below the price range chikou is
                    strength = (hist_price - chikou) / (hist_high - hist_low) if (hist_high - hist_low) > 0 else 0.5
                    signals.loc[current_idx] = max(-strength, -1.0)
        
        return signals
    
    def calculate_ichimoku_signal_score(self, df, weights=None):
        """
        Calculate an overall Ichimoku signal score by combining multiple signals.
        
        Args:
            df (pd.DataFrame): DataFrame with price and Ichimoku data
            weights (dict): Dictionary of weights for each signal type
            
        Returns:
            pd.Series: Overall signal score (-1 to 1, where positive values are bullish)
        """
        # Default weights if none provided
        if weights is None:
            weights = {
                'tk_cross': 0.25,
                'kumo_breakout': 0.25,
                'kumo_twist': 0.15,
                'price_kijun': 0.15,
                'chikou_confirmation': 0.20
            }
            
        # Ensure dataframe has all required Ichimoku components
        if not all(component in df.columns for component in 
                  ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']):
            # Calculate Ichimoku if not already in dataframe
            df = self.calculate_ichimoku(df)
            
        # Calculate individual signals
        tk_cross_signals = self.detect_tk_cross(df)
        kumo_breakout_signals = self.detect_kumo_breakout(df)
        kumo_twist_signals = self.detect_kumo_twist(df)
        price_kijun_signals = self.detect_price_kijun_relationship(df)
        chikou_signals = self.detect_chikou_span_confirmation(df)
        
        # Calculate weighted average of signals
        combined_signal = (
            tk_cross_signals * weights['tk_cross'] +
            kumo_breakout_signals * weights['kumo_breakout'] +
            kumo_twist_signals * weights['kumo_twist'] +
            price_kijun_signals * weights['price_kijun'] +
            chikou_signals * weights['chikou_confirmation']
        )
        
        # Clip the combined signal to the range [-1, 1]
        combined_signal = combined_signal.clip(-1, 1)
        
        return combined_signal
    
    def generate_trading_decisions(self, df, threshold=0.3, ichimoku_params=None, signal_weights=None):
        """
        Generate trading decisions based on Ichimoku signals.
        
        Args:
            df (pd.DataFrame): Price dataframe with OHLC data
            threshold (float): Signal threshold to trigger trades
            ichimoku_params (dict): Parameters for Ichimoku calculation
            signal_weights (dict): Weights for different Ichimoku signals
            
        Returns:
            pd.DataFrame: DataFrame with original data plus signals and trading decisions
        """
        # Deep copy to avoid modifying original
        result_df = df.copy()
        
        # Use default Ichimoku parameters if none provided
        if ichimoku_params is None:
            ichimoku_params = {
                'tenkan_period': 9,
                'kijun_period': 26,
                'senkou_span_b_period': 52,
                'displacement': 26
            }
        
        # Calculate Ichimoku components
        ichimoku_df = self.calculate_ichimoku(
            result_df,
            tenkan_period=ichimoku_params['tenkan_period'],
            kijun_period=ichimoku_params['kijun_period'],
            senkou_span_b_period=ichimoku_params['senkou_span_b_period'],
            displacement=ichimoku_params['displacement']
        )
        
        # Calculate overall signal score
        ichimoku_df['ichimoku_signal'] = self.calculate_ichimoku_signal_score(
            ichimoku_df, weights=signal_weights
        )
        
        # Generate trading decisions based on signal and threshold
        ichimoku_df['ichimoku_position'] = 0  # Initialize with no position
        
        # Long position when signal exceeds positive threshold
        ichimoku_df.loc[ichimoku_df['ichimoku_signal'] > threshold, 'ichimoku_position'] = 1
        
        # Short position when signal exceeds negative threshold
        ichimoku_df.loc[ichimoku_df['ichimoku_signal'] < -threshold, 'ichimoku_position'] = -1
        
        # Calculate signal strength (absolute value of signal, normalized to [0,1])
        ichimoku_df['signal_strength'] = ichimoku_df['ichimoku_signal'].abs()
        
        return ichimoku_df