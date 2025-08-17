#!/usr/bin/env python3
"""
Debug script to test data loading
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
from pathlib import Path
from crypto_backtesting.data.providers.binance_futures import BinanceFuturesProvider

def debug_data_loading():
    provider = BinanceFuturesProvider('binance_futures_data')
    
    symbol = 'BTCUSDT'
    year = 2024
    months = [4, 5]
    timeframe = '1m'
    
    # Get the symbol path
    symbol_path = provider.klines_path / symbol / timeframe
    print(f"Symbol path: {symbol_path}")
    print(f"Path exists: {symbol_path.exists()}")
    
    if not symbol_path.exists():
        print("Symbol path doesn't exist!")
        return
    
    data_frames = []
    
    # Process each month
    for month in (months or range(1, 13)):
        month_str = f"{month:02d}"
        year_month = f"{year}-{month_str}"
        
        print(f"\nProcessing month: {year_month}")
        
        # Try different file extensions
        for ext in ['.csv', '.zip', '.parquet']:
            filename = f"{symbol}-1m-{year_month}{ext}"
            if ext == '.parquet':
                # For parquet files, use just year-month format
                filename = f"{year_month}.parquet"
            file_path = symbol_path / filename
            
            print(f"  Trying: {file_path}")
            print(f"  Exists: {file_path.exists()}")
            
            if file_path.exists():
                try:
                    if ext == '.zip':
                        # Read from zip file
                        df = pd.read_csv(file_path, compression='zip')
                        print(f"  ✅ Loaded ZIP file, shape: {df.shape}")
                    elif ext == '.parquet':
                        # Read from parquet file
                        df = pd.read_parquet(file_path)
                        print(f"  ✅ Loaded Parquet file, shape: {df.shape}")
                    else:
                        # Read from CSV
                        df = pd.read_csv(file_path)
                        print(f"  ✅ Loaded CSV file, shape: {df.shape}")
                        
                    # Standardize column names
                    df = provider._standardize_klines_columns(df)
                    print(f"  ✅ Standardized columns: {df.columns.tolist()[:5]}...")
                    data_frames.append(df)
                    print(f"  ✅ Added to data_frames list (total: {len(data_frames)})")
                    break
                    
                except Exception as e:
                    print(f"  ❌ Error reading {file_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print(f"\nTotal data frames collected: {len(data_frames)}")
    
    if not data_frames:
        print("❌ No data frames collected!")
        return None
    
    # Combine all months
    print("Combining data frames...")
    try:
        combined_data = pd.concat(data_frames, ignore_index=True)
        print(f"✅ Combined data shape: {combined_data.shape}")
        
        # Sort by timestamp
        combined_data = combined_data.sort_values('timestamp')
        print(f"✅ Sorted by timestamp")
        
        # Remove duplicates
        combined_data = combined_data.drop_duplicates(subset=['timestamp'])
        print(f"✅ Removed duplicates, final shape: {combined_data.shape}")
        
        return combined_data
        
    except Exception as e:
        print(f"❌ Error combining data: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = debug_data_loading()
    if result is not None:
        print(f"\n🎉 Success! Final data shape: {result.shape}")
        print(f"Date range: {result['timestamp'].min()} to {result['timestamp'].max()}")
    else:
        print("\n❌ Failed to load data")
