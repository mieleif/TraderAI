from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import requests
import datetime as dt

def fetch_and_prepare_data(symbol='ETHUSDT', timeframe='4_hours', n_bars=5000):
    """
    Fetches data from TradingView and prepares it for the model
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'ETHUSDT', 'BTCUSDT')
        timeframe (str): Timeframe for the candles ('4_hours', '1_hour', 'daily', etc.)
        n_bars (int): Number of candles to fetch
        
    Returns:
        pd.DataFrame: Prepared DataFrame with all indicators
    """
    tv = TvDatafeed()
    
    # Map timeframe string to Interval enum
    interval_map = {
        '1_minute': Interval.in_1_minute,
        '3_minutes': Interval.in_3_minute,
        '5_minutes': Interval.in_5_minute,
        '15_minutes': Interval.in_15_minute,
        '30_minutes': Interval.in_30_minute,
        '45_minutes': Interval.in_45_minute,
        '1_hour': Interval.in_1_hour,
        '2_hours': Interval.in_2_hour,
        '3_hours': Interval.in_3_hour,
        '4_hours': Interval.in_4_hour,
        'daily': Interval.in_daily,
        'weekly': Interval.in_weekly,
        'monthly': Interval.in_monthly
    }
    
    selected_interval = interval_map.get(timeframe)
    if selected_interval is None:
        raise ValueError(f"Invalid timeframe. Choose from: {list(interval_map.keys())}")
    
    # Fetch data
    df = tv.get_hist(
        symbol=symbol,
        exchange='BINANCE',
        interval=selected_interval,
        n_bars=n_bars
    )
    
    # Prepare features
    df['prev_close'] = df['close'].shift(1)
    
    # Add the moving average indicators
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['ma_25'] = df['close'].rolling(window=25).mean()
    df['ma_100'] = df['close'].rolling(window=100).mean()
    
    # Calculate Ichimoku indicators
    df = compute_ichimoku(df)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def compute_ichimoku(df):
    """
    Calculates Ichimoku Cloud indicators
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        
    Returns:
        pd.DataFrame: DataFrame with added Ichimoku indicators
    """
    # Calculate Tenkan-sen (Conversion Line)
    period9_high = df['high'].rolling(window=9).max()
    period9_low = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (period9_high + period9_low) / 2

    # Calculate Kijun-sen (Base Line)
    period26_high = df['high'].rolling(window=26).max()
    period26_low = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (period26_high + period26_low) / 2

    # Calculate Senkou Span A (Leading Span A)
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    # Calculate Senkou Span B (Leading Span B)
    period52_high = df['high'].rolling(window=52).max()
    period52_low = df['low'].rolling(window=52).min()
    df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)

    # Calculate Chikou Span (Lagging Span)
    df['chikou_span'] = df['close'].shift(-26)

    return df

def fetch_funding_rates(symbol, df):
    """
    Fetches funding rates for the given symbol within the date range of the DataFrame
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'ETHUSDT')
        df (pd.DataFrame): DataFrame with index as datetime
        
    Returns:
        pd.DataFrame: DataFrame with added funding rate column
    """
    # Get the date range from our data
    start_date = df.index.min()
    end_date = df.index.max()
    
    print(f"Fetching funding rates from {start_date} to {end_date}")
    
    # Initialize empty list to store all funding data
    all_funding_data = []
    
    # Fetch data in chunks of 1000 records
    chunk_size = 1000
    current_start = start_date
    
    while current_start < end_date:
        # Calculate end time for this chunk
        current_end = min(current_start + pd.Timedelta(days=30), end_date)
        
        start_ts = int(current_start.timestamp() * 1000)
        end_ts = int(current_end.timestamp() * 1000)
        
        # Binance Funding Rate API endpoint for USDS-margined futures
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {
            "symbol": symbol,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": chunk_size
        }
        
        print(f"Fetching chunk: {current_start} to {current_end}")
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            chunk_data = response.json()
            if chunk_data:  # Only extend if we got data
                all_funding_data.extend(chunk_data)
        else:
            print(f"Error retrieving funding rates, status code: {response.status_code}")
            print(f"Error response: {response.text}")
        
        # Move to next chunk
        current_start = current_end
    
    # Process the collected funding data
    if all_funding_data:
        funding_df = pd.DataFrame(all_funding_data)
        # Convert fundingTime to datetime and fundingRate to float
        funding_df['fundingTime'] = pd.to_datetime(funding_df['fundingTime'], unit='ms')
        funding_df['fundingRate'] = funding_df['fundingRate'].astype(float)
        
        # Group by date and compute the average funding rate for that day
        funding_df['date'] = funding_df['fundingTime'].dt.date
        daily_funding = funding_df.groupby('date')['fundingRate'].mean().reset_index()
        daily_funding['date'] = pd.to_datetime(daily_funding['date'])
        daily_funding = daily_funding.set_index('date')
        
        # Merge daily funding rate into the main DataFrame
        df['date'] = df.index.date
        df['date'] = pd.to_datetime(df['date'])
        df = df.merge(daily_funding, left_on='date', right_index=True, how='left')
        df = df.drop(columns=['date'])
        
        # Fill missing funding rates with 0
        df['fundingRate'] = df['fundingRate'].fillna(0)
    else:
        print("No funding rate data collected")
        df['fundingRate'] = 0
    
    return df

def main(symbol='ETHUSDT', timeframe='4_hours', n_bars=5000, output_file=None):
    """
    Main function to fetch data, compute indicators, and save to CSV
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'ETHUSDT', 'BTCUSDT')
        timeframe (str): Timeframe for the candles ('4_hours', '1_hour', 'daily', etc.)
        n_bars (int): Number of candles to fetch
        output_file (str, optional): Output file name. If None, will use {symbol}_{timeframe}_data.csv
        
    Returns:
        pd.DataFrame: The processed DataFrame
    """
    # Fetch and prepare data
    df = fetch_and_prepare_data(symbol=symbol, timeframe=timeframe, n_bars=n_bars)
    print(f"Initial data shape: {df.shape}")
    
    # Fetch funding rates
    df = fetch_funding_rates(symbol, df)
    
    # Save to CSV
    if output_file is None:
        output_file = f'{symbol}_{timeframe}_data.csv'
    
    df.to_csv(output_file)
    print(f"Data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Example usage
    df = main(symbol='ETHUSDT', timeframe='4_hours', n_bars=5000)
    print("\nFirst few rows:")
    print(df.head())