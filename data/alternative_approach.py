import pandas as pd
import requests
import datetime as dt

def fetch_binance_data(symbol='ETHUSDT', interval='4h', limit=5000, start_time=None, end_time=None):
    """
    Fetches historical OHLCV data directly from Binance API
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'ETHUSDT', 'BTCUSDT')
        interval (str): Kline/Candlestick interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
        limit (int): Max 1000 records
        start_time (int): Start time in milliseconds
        end_time (int): End time in milliseconds
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    # Map from our timeframe format to Binance's
    interval_map = {
        '1_minute': '1m',
        '3_minutes': '3m',
        '5_minutes': '5m',
        '15_minutes': '15m',
        '30_minutes': '30m',
        '1_hour': '1h',
        '2_hours': '2h',
        '4_hours': '4h',
        '6_hours': '6h',
        '8_hours': '8h',
        '12_hours': '12h',
        'daily': '1d',
        'weekly': '1w',
        'monthly': '1M'
    }
    
    # If interval was provided in our format, convert it
    if interval in interval_map:
        interval = interval_map[interval]
    
    # Binance API endpoint for klines (candlestick data)
    url = "https://api.binance.com/api/v3/klines"
    
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.text}")
    
    # Parse the response
    data = response.json()
    
    # Create DataFrame with appropriate column names
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                       'quote_asset_volume', 'taker_buy_base_asset_volume', 
                       'taker_buy_quote_asset_volume']
    
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column])
    
    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    
    return df

def fetch_full_history(symbol='ETHUSDT', interval='4h', days_back=600):
    """
    Fetches historical data for the specified number of days back
    by making multiple API calls if necessary
    
    Args:
        symbol (str): Trading pair symbol
        interval (str): Candlestick interval
        days_back (int): Number of days to go back
        
    Returns:
        pd.DataFrame: DataFrame with the full history
    """
    # Calculate the start time
    end_time = int(dt.datetime.now().timestamp() * 1000)
    start_time = end_time - (days_back * 24 * 60 * 60 * 1000)
    
    # Initialize an empty list to store dataframes
    all_df = []
    current_start = start_time
    
    # Binance has a limit of 1000 candles per request
    # We need to make multiple requests to get the full history
    while current_start < end_time:
        print(f"Fetching data from {pd.to_datetime(current_start, unit='ms')}")
        
        # Fetch a batch of data
        df_batch = fetch_binance_data(
            symbol=symbol,
            interval=interval,
            limit=1000,
            start_time=current_start,
            end_time=end_time
        )
        
        if df_batch.empty:
            break
        
        all_df.append(df_batch)
        
        # Update the start time for the next batch
        current_start = int(df_batch.index[-1].timestamp() * 1000) + 1
    
    # Combine all batches
    if not all_df:
        raise Exception("No data was fetched")
        
    df = pd.concat(all_df)
    
    # Remove any duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Sort by timestamp
    df = df.sort_index()
    
    return df

def prepare_data(df):
    """
    Adds technical indicators to the dataframe
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added indicators
    """
    # Previous close
    df['prev_close'] = df['close'].shift(1)
    
    # Moving averages
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['ma_25'] = df['close'].rolling(window=25).mean()
    df['ma_100'] = df['close'].rolling(window=100).mean()
    
    # Ichimoku Cloud
    df = compute_ichimoku(df)
    
    # Funding rates
    df = fetch_funding_rates(df.index.min(), df.index.max(), df)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def compute_ichimoku(df):
    """
    Calculates Ichimoku Cloud indicators
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
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

def fetch_funding_rates(start_date, end_date, df):
    """
    Fetches funding rates from Binance Futures API
    
    Args:
        start_date (datetime): Start date
        end_date (datetime): End date
        df (pd.DataFrame): DataFrame to merge funding rates into
        
    Returns:
        pd.DataFrame: DataFrame with added funding rate column
    """
    symbol = 'ETHUSDT'  # Hardcoded for now, could be parametrized
    print(f"Fetching funding rates from {start_date} to {end_date}")
    
    # Initialize empty list to store all funding data
    all_funding_data = []
    
    # Convert datetime to timestamp in milliseconds
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    # Fetch data in chunks of 1000 records
    chunk_size = 1000
    current_start = start_ts
    
    while current_start < end_ts:
        # Calculate end time for this chunk
        current_end = current_start + (30 * 24 * 60 * 60 * 1000)  # 30 days in milliseconds
        if current_end > end_ts:
            current_end = end_ts
        
        # Binance Funding Rate API endpoint for USDS-margined futures
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": current_end,
            "limit": chunk_size
        }
        
        print(f"Fetching funding rates chunk: {pd.to_datetime(current_start, unit='ms')} to {pd.to_datetime(current_end, unit='ms')}")
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            chunk_data = response.json()
            if chunk_data:  # Only extend if we got data
                all_funding_data.extend(chunk_data)
        else:
            print(f"Error retrieving funding rates, status code: {response.status_code}")
            print(f"Error response: {response.text}")
        
        # Move to next chunk
        current_start = current_end + 1
    
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

def main(symbol='ETHUSDT', interval='4h', days_back=1000, output_file=None):
    """
    Main function to fetch data, compute indicators, and save to CSV
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'ETHUSDT', 'BTCUSDT')
        interval (str): Candlestick interval ('4h', '1h', '1d', etc.)
        days_back (int): Number of days to go back
        output_file (str, optional): Output file name. If None, will use {symbol}_{interval}_data.csv
        
    Returns:
        pd.DataFrame: The processed DataFrame
    """
    # Fetch historical data
    print(f"Fetching {days_back} days of historical data for {symbol} on {interval} timeframe...")
    df = fetch_full_history(symbol=symbol, interval=interval, days_back=days_back)
    print(f"Fetched {len(df)} candles.")
    
    # Prepare data with indicators
    print("Computing technical indicators...")
    df = prepare_data(df)
    print(f"Final data shape: {df.shape}")
    
    # Save to CSV
    if output_file is None:
        output_file = f'{symbol}_{interval}_data.csv'
    
    df.to_csv(output_file)
    print(f"Data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Example usage
    df = main(symbol='ETHUSDT', interval='4h', days_back=1000)
    print("\nFirst few rows:")
    print(df.head())