import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class FeatureEngineering:
    """
    Feature engineering class for creating machine learning-friendly features
    from Ichimoku Cloud indicators and price data.
    """
    
    def __init__(self, scale_method='standard'):
        """
        Initialize the FeatureEngineering class.
        
        Args:
            scale_method (str): Scaling method to use ('standard', 'minmax', or 'none')
        """
        self.scale_method = scale_method
        if scale_method == 'standard':
            self.scaler = StandardScaler()
        elif scale_method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.scaler = None
        
        self.fitted = False
    
    def extract_ichimoku_features(self, df):
        """
        Extract features from Ichimoku components.
        
        Args:
            df (pd.DataFrame): DataFrame with Ichimoku components
            
        Returns:
            pd.DataFrame: DataFrame with extracted features
        """
        # Create a new dataframe for features
        features = pd.DataFrame(index=df.index)
        
        # Check if all Ichimoku components exist
        required_cols = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("DataFrame is missing required Ichimoku components")
        
        # 1. Distance features - normalized by price
        features['tenkan_kijun_distance'] = (df['tenkan_sen'] - df['kijun_sen']) / df['close']
        features['price_tenkan_distance'] = (df['close'] - df['tenkan_sen']) / df['close']
        features['price_kijun_distance'] = (df['close'] - df['kijun_sen']) / df['close']
        
        # Kumo (cloud) features
        features['kumo_thickness'] = (df['senkou_span_a'] - df['senkou_span_b']).abs() / df['close']
        
        # Future kumo thickness (26 periods ahead) - useful for predicting trend strength
        features['future_kumo_thickness'] = features['kumo_thickness'].shift(-26)
        
        # Kumo direction (positive when Senkou A > Senkou B, negative otherwise)
        features['kumo_direction'] = np.sign(df['senkou_span_a'] - df['senkou_span_b'])
        
        # 2. Position relative to cloud
        # Calculate upper and lower edges of the cloud
        kumo_upper = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        kumo_lower = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
        
        # Normalized distance from price to cloud
        features['price_above_cloud'] = ((df['close'] > kumo_upper) * 
                                         (df['close'] - kumo_upper) / df['close']).clip(0, 1)
        features['price_below_cloud'] = ((df['close'] < kumo_lower) * 
                                         (kumo_lower - df['close']) / df['close']).clip(0, 1)
        features['price_in_cloud'] = ((df['close'] >= kumo_lower) & 
                                      (df['close'] <= kumo_upper)).astype(float)
        
        # 3. Moving features (momentum)
        # Tenkan-sen momentum (rate of change)
        features['tenkan_momentum'] = (df['tenkan_sen'] - df['tenkan_sen'].shift(3)) / df['tenkan_sen'].shift(3)
        # Kijun-sen momentum
        features['kijun_momentum'] = (df['kijun_sen'] - df['kijun_sen'].shift(3)) / df['kijun_sen'].shift(3)
        
        # 4. Cross features
        # TK Cross indicator: positive when Tenkan crosses above Kijun, negative when crossing below
        tk_cross = (
            (df['tenkan_sen'] > df['kijun_sen']).astype(int) - 
            (df['tenkan_sen'].shift(1) > df['kijun_sen'].shift(1)).astype(int)
        )
        features['tk_cross'] = tk_cross
        
        # 5. Chikou span features
        # Relationship between Chikou span and historical price
        valid_indices = df.index[26:]  # Chikou is shifted by 26 periods
        
        # Initialize with zeros
        features['chikou_vs_price'] = 0.0
        
        for i in range(26, len(df)):
            current_idx = df.index[i]
            chikou_idx = df.index[i-26] if i-26 >= 0 else None
            
            if chikou_idx is not None:
                # Current Chikou value (current close shifted back 26 periods)
                chikou = df['close'].iloc[i-26]
                
                # Historical price at the point Chikou is referring to
                if i >= 52:  # Need at least 26*2 periods
                    hist_price = df['close'].iloc[i-52]
                    # Normalized difference
                    features.loc[current_idx, 'chikou_vs_price'] = (chikou - hist_price) / hist_price
        
        # 6. Categorical features as one-hot encoding
        # Market phases based on cloud configuration
        features['bullish_market'] = ((df['close'] > kumo_upper) & 
                                     (df['senkou_span_a'] > df['senkou_span_b'])).astype(float)
        features['bearish_market'] = ((df['close'] < kumo_lower) & 
                                     (df['senkou_span_a'] < df['senkou_span_b'])).astype(float)
        features['neutral_market'] = (~(features['bullish_market'] | features['bearish_market'])).astype(float)
        
        # 7. Volatility features
        # Price volatility relative to Ichimoku components
        features['price_volatility'] = df['close'].rolling(window=10).std() / df['close']
        
        # Clean up inf and NaN values
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.fillna(0, inplace=True)
        
        return features
    
    def calculate_technical_features(self, df):
        """
        Calculate additional technical indicators beyond Ichimoku.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            
        Returns:
            pd.DataFrame: DataFrame with technical features
        """
        # Create a new dataframe for features
        tech_features = pd.DataFrame(index=df.index)
        
        # 1. Moving averages
        tech_features['sma_7'] = df['close'].rolling(window=7).mean() / df['close']
        tech_features['sma_25'] = df['close'].rolling(window=25).mean() / df['close']
        tech_features['sma_99'] = df['close'].rolling(window=99).mean() / df['close']
        
        # 2. Price momentum indicators
        tech_features['roc_5'] = df['close'].pct_change(periods=5)
        tech_features['roc_10'] = df['close'].pct_change(periods=10)
        tech_features['roc_20'] = df['close'].pct_change(periods=20)
        
        # 3. Volume indicators (if volume is available)
        if 'volume' in df.columns:
            # Normalize volume
            tech_features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
            
            # On-balance volume normalized
            obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            tech_features['obv_sma_ratio'] = obv / obv.rolling(window=20).mean()
        
        # 4. Volatility indicators
        # ATR (Average True Range) normalized by close price
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()
        tech_features['atr_ratio'] = atr / df['close']
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        
        upper_band = sma_20 + (2 * std_20)
        lower_band = sma_20 - (2 * std_20)
        
        tech_features['bb_width'] = (upper_band - lower_band) / sma_20
        tech_features['bb_position'] = (df['close'] - lower_band) / (upper_band - lower_band)
        
        # 5. Trend indicators
        # ADX (Average Directional Index) - simplified calculation
        up_move = df['high'].diff()
        down_move = df['low'].diff().abs()
        plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
        
        tr = true_range  # Using previously calculated True Range
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        tech_features['adx'] = dx.rolling(window=14).mean() / 100  # Normalize to [0,1]
        
        # Clean up inf and NaN values
        tech_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        tech_features.fillna(0, inplace=True)
        
        return tech_features
    
    def combine_features(self, df, ichimoku_df=None):
        """
        Combine all features from price data and Ichimoku components.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            ichimoku_df (pd.DataFrame): DataFrame with Ichimoku components
            
        Returns:
            pd.DataFrame: DataFrame with all features
        """
        # If ichimoku_df is not provided, assume Ichimoku components are in df
        if ichimoku_df is None:
            ichimoku_df = df
            
        # Extract Ichimoku features
        ichimoku_features = self.extract_ichimoku_features(ichimoku_df)
        
        # Calculate technical features
        tech_features = self.calculate_technical_features(df)
        
        # Combine all features
        combined_features = pd.concat([ichimoku_features, tech_features], axis=1)
        
        # Remove features with too many missing values
        threshold = 0.8  # Keep features with at least 80% non-missing values
        missing_ratio = combined_features.isnull().sum() / len(combined_features)
        valid_columns = missing_ratio[missing_ratio < (1 - threshold)].index
        combined_features = combined_features[valid_columns]
        
        # Forward fill any remaining missing values
        combined_features.fillna(method='ffill', inplace=True)
        # Fill any remaining NaNs with zeros
        combined_features.fillna(0, inplace=True)
        
        return combined_features
    
    def scale_features(self, features, fit=True):
        """
        Scale features using the selected scaling method.
        
        Args:
            features (pd.DataFrame): DataFrame with features to scale
            fit (bool): Whether to fit the scaler or use pre-fitted scaler
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        if self.scale_method == 'none' or self.scaler is None:
            return features
            
        # Create a copy of the dataframe
        scaled_features = features.copy()
        feature_names = scaled_features.columns
        
        # Convert to numpy array for scaling
        feature_array = scaled_features.values
        
        if fit:
            # Fit the scaler and transform
            scaled_array = self.scaler.fit_transform(feature_array)
            self.fitted = True
        else:
            # Use the pre-fitted scaler
            if not self.fitted:
                raise ValueError("Scaler has not been fitted yet")
            scaled_array = self.scaler.transform(feature_array)
        
        # Convert back to dataframe
        scaled_features = pd.DataFrame(scaled_array, index=features.index, columns=feature_names)
        
        return scaled_features
    
    def create_ml_features(self, df, ichimoku_df=None, scale=True):
        """
        Create machine learning features from price data and Ichimoku components.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            ichimoku_df (pd.DataFrame): DataFrame with Ichimoku components
            scale (bool): Whether to scale the features
            
        Returns:
            pd.DataFrame: DataFrame with ML-ready features
        """
        # Combine features
        features = self.combine_features(df, ichimoku_df)
        
        # Scale features if requested
        if scale and self.scale_method != 'none':
            features = self.scale_features(features)
        
        return features
    
    def time_series_features(self, features, lookback_periods=[1, 2, 3, 5, 10]):
        """
        Create time series features by adding lagged values.
        
        Args:
            features (pd.DataFrame): DataFrame with features
            lookback_periods (list): List of periods to look back
            
        Returns:
            pd.DataFrame: DataFrame with time series features
        """
        # Create a new dataframe for the time series features
        ts_features = features.copy()
        
        # Add lagged features
        for period in lookback_periods:
            period_features = features.shift(period)
            period_features.columns = [f'{col}_lag_{period}' for col in features.columns]
            ts_features = pd.concat([ts_features, period_features], axis=1)
        
        # Fill NaN values with 0
        ts_features.fillna(0, inplace=True)
        
        return ts_features
    
    def create_feature_differences(self, features, periods=[1, 3, 5]):
        """
        Create features based on differences between current and past values.
        
        Args:
            features (pd.DataFrame): DataFrame with features
            periods (list): List of periods for differencing
            
        Returns:
            pd.DataFrame: DataFrame with difference features
        """
        diff_features = features.copy()
        
        # Add differenced features
        for period in periods:
            period_diff = features.diff(period)
            period_diff.columns = [f'{col}_diff_{period}' for col in features.columns]
            diff_features = pd.concat([diff_features, period_diff], axis=1)
        
        # Fill NaN values with 0
        diff_features.fillna(0, inplace=True)
        
        return diff_features