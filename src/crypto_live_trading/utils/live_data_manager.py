"""
Live Data Manager - Fetches real-time data using CCXT
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


class LiveDataManager:
    def __init__(self, timeframe="4h", lookback_periods=200):
        """
        Initialize live data manager with Binance exchange

        Args:
            timeframe: CCXT timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            lookback_periods: Number of periods to fetch for strategy calculation
        """
        self.exchange = ccxt.binance(
            {
                "apiKey": "",  # Add your API key if needed
                "secret": "",  # Add your secret if needed
                "sandbox": True,  # Use testnet for testing
                "enableRateLimit": True,
            }
        )
        self.timeframe = timeframe
        self.lookback_periods = lookback_periods

    def get_latest_pair_data(self, symbol1, symbol2):
        """
        Fetch latest OHLCV data for a pair of symbols

        Args:
            symbol1: First symbol (e.g., 'BTC/USDT')
            symbol2: Second symbol (e.g., 'ETH/USDT')

        Returns:
            df1, df2: DataFrames with OHLCV data
        """
        try:
            # Convert symbols to CCXT format if needed
            symbol1_ccxt = self._format_symbol(symbol1)
            symbol2_ccxt = self._format_symbol(symbol2)

            # Fetch OHLCV data
            df1 = self._fetch_ohlcv(symbol1_ccxt)
            df2 = self._fetch_ohlcv(symbol2_ccxt)

            # Align timestamps and resample if needed
            df1, df2 = self._align_dataframes(df1, df2)

            return df1, df2

        except Exception as e:
            print(f"Error fetching data for {symbol1}-{symbol2}: {e}")
            return None, None

    def _format_symbol(self, symbol):
        """Convert symbol format to CCXT format"""
        if "/" not in symbol:
            # Assume it's a futures symbol like 'BTCUSDT'
            if symbol.endswith("USDT"):
                base = symbol[:-4]
                return f"{base}/USDT"
        return symbol

    def _fetch_ohlcv(self, symbol):
        """Fetch OHLCV data for a single symbol"""
        # Calculate since timestamp for lookback periods
        since = self.exchange.milliseconds() - (
            self.lookback_periods * self._timeframe_to_ms()
        )

        # Fetch OHLCV data
        ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, since=since)

        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        return df

    def _timeframe_to_ms(self):
        """Convert timeframe to milliseconds"""
        timeframe_map = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }
        return timeframe_map.get(self.timeframe, 4 * 60 * 60 * 1000)

    def _align_dataframes(self, df1, df2):
        """Align two dataframes by timestamp"""
        # Find common timestamps
        common_index = df1.index.intersection(df2.index)

        if len(common_index) == 0:
            raise ValueError("No common timestamps found between symbols")

        # Align dataframes
        df1_aligned = df1.loc[common_index].copy()
        df2_aligned = df2.loc[common_index].copy()

        return df1_aligned, df2_aligned
