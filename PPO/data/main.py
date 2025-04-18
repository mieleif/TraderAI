from datafetch import main as fetch_data

if __name__ == "__main__":
    # Configure your parameters here
    symbol = 'ETHUSDT'
    timeframe = '4_hours'
    n_bars = 5000
    output_file = f'{symbol}_{timeframe}_data.csv'
    
    # Run the data fetching pipeline
    df = fetch_data(symbol=symbol, timeframe=timeframe, n_bars=n_bars, output_file=output_file)
    
    # Print information about the output
    print(f"Data fetching complete for {symbol} on {timeframe} timeframe")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Print a sample of the data
    print("\nData sample:")
    print(df.tail(-5)[['close', 'ma_7', 'ma_25', 'ma_100', 'tenkan_sen', 'kijun_sen', 'fundingRate']])