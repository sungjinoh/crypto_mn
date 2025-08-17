"""
Data resampling utilities for pairs trading
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


def resample_ohlcv_data(df: pd.DataFrame, timeframe: str = "1H") -> pd.DataFrame:
    """
    Resample OHLCV data to specified timeframe

    Args:
        df: DataFrame with OHLCV data (must have datetime index or datetime column)
        timeframe: Target timeframe ('1H', '4H', '1D', etc.)

    Returns:
        Resampled DataFrame
    """
    # Make a copy to avoid modifying original
    data = df.copy()

    # Ensure we have a datetime index
    if "datetime" in data.columns and data.index.name != "datetime":
        data = data.set_index("datetime")
    elif "timestamp" in data.columns:
        data["datetime"] = pd.to_datetime(data["timestamp"], unit="ms")
        data = data.set_index("datetime")

    # Define aggregation rules for OHLCV data
    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Add any other columns that might exist
    for col in data.columns:
        if col not in agg_rules:
            if col in ["timestamp"]:
                agg_rules[col] = "first"
            elif col in ["symbol"]:
                agg_rules[col] = "first"
            else:
                # For other numeric columns, use mean
                if pd.api.types.is_numeric_dtype(data[col]):
                    agg_rules[col] = "mean"
                else:
                    agg_rules[col] = "first"

    # Resample the data
    resampled = data.resample(timeframe).agg(agg_rules).dropna()

    # Add timestamp back if it was in original data
    if "timestamp" in data.columns:
        resampled["timestamp"] = resampled.index.astype(np.int64) // 10**6

    # Reset index to have datetime as column
    resampled = resampled.reset_index()

    return resampled


def load_and_resample_symbol(
    symbol: str,
    timeframe: str = "1H",
    data_dir: str = "../../../binance_futures_data/pickle",
) -> Optional[pd.DataFrame]:
    """
    Load symbol data and resample to specified timeframe

    Args:
        symbol: Symbol to load
        timeframe: Target timeframe ('1H', '4H', '1D', etc.)
        data_dir: Data directory path

    Returns:
        Resampled DataFrame or None if loading fails
    """
    try:
        # Load original 1-minute data
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        filepath = f"{data_dir}/{safe_symbol}_1m_ohlcv.pkl"

        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return None

        df = pd.read_pickle(filepath)
        df["symbol"] = symbol

        print(f"📊 Loaded {len(df)} 1-minute bars for {symbol}")

        # Resample to target timeframe
        resampled_df = resample_ohlcv_data(df, timeframe)

        print(f"📈 Resampled to {len(resampled_df)} {timeframe} bars")

        return resampled_df

    except Exception as e:
        print(f"❌ Error loading/resampling {symbol}: {e}")
        return None


def compare_timeframes_cointegration(
    symbol1: str, symbol2: str, timeframes: list = ["1H", "4H", "1D"]
) -> Dict:
    """
    Compare cointegration results across different timeframes

    Args:
        symbol1: First symbol
        symbol2: Second symbol
        timeframes: List of timeframes to test

    Returns:
        Dictionary with cointegration results for each timeframe
    """
    from backtesting_framework import PairsBacktester

    results = {}
    backtester = PairsBacktester()

    for tf in timeframes:
        print(f"\n🔬 Testing {tf} timeframe...")

        # Load resampled data
        data1 = load_and_resample_symbol(symbol1, tf)
        data2 = load_and_resample_symbol(symbol2, tf)

        if data1 is None or data2 is None:
            print(f"❌ Could not load data for {tf}")
            continue

        # Prepare pair data
        pair_data = backtester.prepare_pair_data(data1, data2, symbol1, symbol2)

        if len(pair_data) < 50:
            print(f"❌ Insufficient data points for {tf}: {len(pair_data)}")
            continue

        # Test cointegration
        coint_result = backtester.check_cointegration(
            pair_data[f"{symbol1}_close"], pair_data[f"{symbol2}_close"]
        )

        # Calculate correlation
        correlation = pair_data[f"{symbol1}_close"].corr(pair_data[f"{symbol2}_close"])

        results[tf] = {
            "data_points": len(pair_data),
            "cointegration": coint_result,
            "correlation": correlation,
        }

        # Print results
        status = (
            "✅ COINTEGRATED"
            if coint_result["is_cointegrated"]
            else "❌ Not cointegrated"
        )
        print(f"  {status}")
        print(f"  Data points: {len(pair_data)}")
        print(f"  P-value: {coint_result['p_value']:.6f}")
        print(f"  Correlation: {correlation:.4f}")
        if coint_result["hedge_ratio"]:
            print(f"  Hedge ratio: {coint_result['hedge_ratio']:.4f}")

    return results


# Example usage function
def example_resampling():
    """Example of how to use resampling for pairs trading"""
    import os

    print("📊 Data Resampling Example for Pairs Trading")
    print("=" * 50)

    # Test symbols
    symbol1 = "BTC/USDT:USDT"
    symbol2 = "ETH/USDT:USDT"

    # Load original 1-minute data
    print(f"\n1️⃣ Loading original 1-minute data...")

    def load_symbol_ohlcv(
        symbol: str, data_dir: str = "../../../binance_futures_data/pickle"
    ):
        try:
            safe_symbol = symbol.replace("/", "_").replace(":", "_")
            filepath = f"{data_dir}/{safe_symbol}_1m_ohlcv.pkl"

            if not os.path.exists(filepath):
                return None

            df = pd.read_pickle(filepath)
            df["symbol"] = symbol
            return df
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            return None

    data1_1m = load_symbol_ohlcv(symbol1)
    data2_1m = load_symbol_ohlcv(symbol2)

    if data1_1m is None or data2_1m is None:
        print("❌ Could not load 1-minute data")
        return

    print(f"✅ {symbol1}: {len(data1_1m)} 1-minute bars")
    print(f"✅ {symbol2}: {len(data2_1m)} 1-minute bars")

    # Resample to different timeframes
    timeframes = ["1H", "4H", "1D"]

    for tf in timeframes:
        print(f"\n2️⃣ Resampling to {tf}...")

        data1_resampled = resample_ohlcv_data(data1_1m, tf)
        data2_resampled = resample_ohlcv_data(data2_1m, tf)

        print(f"✅ {symbol1}: {len(data1_resampled)} {tf} bars")
        print(f"✅ {symbol2}: {len(data2_resampled)} {tf} bars")

        # Show sample of resampled data
        print(f"\n📋 Sample {tf} data for {symbol1}:")
        print(
            data1_resampled[
                ["datetime", "open", "high", "low", "close", "volume"]
            ].head(3)
        )

    # Compare cointegration across timeframes
    print(f"\n3️⃣ Comparing cointegration across timeframes...")
    coint_comparison = compare_timeframes_cointegration(symbol1, symbol2, timeframes)

    # Summary
    print(f"\n📊 COINTEGRATION COMPARISON SUMMARY")
    print("=" * 50)

    for tf, results in coint_comparison.items():
        coint = results["cointegration"]
        print(f"\n{tf} Timeframe:")
        print(f"  Data points: {results['data_points']}")
        print(f"  Cointegrated: {'✅ YES' if coint['is_cointegrated'] else '❌ NO'}")
        print(f"  P-value: {coint['p_value']:.6f}")
        print(f"  Correlation: {results['correlation']:.4f}")

    return coint_comparison


if __name__ == "__main__":
    example_resampling()
