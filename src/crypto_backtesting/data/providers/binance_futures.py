"""
Binance Futures data provider implementation.

This module provides access to Binance Futures market data including
klines and funding rates stored in the local file system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import logging

from .base import BaseDataProvider


class BinanceFuturesProvider(BaseDataProvider):
    """
    Data provider for Binance Futures market data.
    
    Loads klines and funding rate data from local file system
    in the standard Binance data format.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize Binance Futures data provider.
        
        Args:
            data_path: Path to Binance futures data directory
        """
        super().__init__(data_path)
        self.base_path = Path(data_path)
        self.klines_path = self.base_path / "klines"
        self.funding_path = self.base_path / "fundingRate"
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def get_klines_data(
        self,
        symbol: str,
        year: int,
        months: Optional[List[int]] = None,
        timeframe: str = "1m"
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV klines data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            year: Year of data
            months: List of months (None for all months)
            timeframe: Timeframe for data (currently only supports 1m)
            
        Returns:
            DataFrame with OHLCV data or None if not available
        """
        try:
            # Construct path to symbol data
            symbol_path = self.klines_path / symbol / timeframe
            
            if not symbol_path.exists():
                self.logger.warning(f"No data directory found for symbol: {symbol}")
                return None
                
            # Collect all data files for the year
            data_frames = []
            
            if months is None:
                months = list(range(1, 13))  # All months
                
            for month in months:
                # Binance file naming pattern: SYMBOL-1m-YYYY-MM.zip or .csv
                month_str = f"{month:02d}"
                year_month = f"{year}-{month_str}"
                
                # Try different file extensions
                for ext in ['.csv', '.zip', '.parquet']:
                    filename = f"{symbol}-1m-{year_month}{ext}"
                    if ext == '.parquet':
                        # For parquet files, use just year-month format
                        filename = f"{year_month}.parquet"
                    file_path = symbol_path / filename
                    
                    if file_path.exists():
                        try:
                            if ext == '.zip':
                                # Read from zip file
                                df = pd.read_csv(file_path, compression='zip')
                            elif ext == '.parquet':
                                # Read from parquet file
                                df = pd.read_parquet(file_path)
                            else:
                                # Read from CSV
                                df = pd.read_csv(file_path)
                                
                            # Standardize column names
                            df = self._standardize_klines_columns(df)
                            data_frames.append(df)
                            break
                            
                        except Exception as e:
                            self.logger.warning(f"Error reading {file_path}: {e}")
                            continue
                            
            if not data_frames:
                self.logger.warning(f"No data files found for {symbol} in {year}")
                return None
                
            # Combine all months
            combined_data = pd.concat(data_frames, ignore_index=True)
            
            # Sort by timestamp
            combined_data = combined_data.sort_values('timestamp')
            
            # Remove duplicates
            combined_data = combined_data.drop_duplicates(subset=['timestamp'])
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error loading klines data for {symbol}: {e}")
            return None
            
    def get_funding_rates(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get funding rate data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with funding rate data or None if not available
        """
        try:
            # Construct path to funding rate data
            funding_symbol_path = self.funding_path / symbol
            
            if not funding_symbol_path.exists():
                self.logger.warning(f"No funding rate data found for symbol: {symbol}")
                return None
                
            # Look for funding rate files
            funding_files = list(funding_symbol_path.glob("*.csv"))
            
            if not funding_files:
                self.logger.warning(f"No funding rate files found for {symbol}")
                return None
                
            # Read and combine all funding rate files
            data_frames = []
            
            for file_path in funding_files:
                try:
                    df = pd.read_csv(file_path)
                    df = self._standardize_funding_columns(df)
                    data_frames.append(df)
                except Exception as e:
                    self.logger.warning(f"Error reading funding file {file_path}: {e}")
                    continue
                    
            if not data_frames:
                return None
                
            # Combine all data
            combined_data = pd.concat(data_frames, ignore_index=True)
            
            # Sort by timestamp
            combined_data = combined_data.sort_values('fundingTime')
            
            # Remove duplicates
            combined_data = combined_data.drop_duplicates(subset=['fundingTime'])
            
            # Filter by date range if specified
            if start_date or end_date:
                combined_data = self._filter_by_date_range(
                    combined_data, 'fundingTime', start_date, end_date
                )
                
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error loading funding rates for {symbol}: {e}")
            return None
            
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols.
        
        Returns:
            List of available symbols
        """
        symbols = []
        
        # Check klines directory
        if self.klines_path.exists():
            for symbol_dir in self.klines_path.iterdir():
                if symbol_dir.is_dir():
                    symbols.append(symbol_dir.name)
                    
        # Also check funding rates directory
        if self.funding_path.exists():
            for symbol_dir in self.funding_path.iterdir():
                if symbol_dir.is_dir() and symbol_dir.name not in symbols:
                    symbols.append(symbol_dir.name)
                    
        return sorted(symbols)
        
    def validate_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is valid and has data available.
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        # Check if klines data exists
        klines_exists = (self.klines_path / symbol).exists()
        
        # Check if funding rate data exists
        funding_exists = (self.funding_path / symbol).exists()
        
        return klines_exists or funding_exists
        
    def _standardize_klines_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize klines DataFrame column names and types.
        
        Args:
            df: Raw klines DataFrame
            
        Returns:
            Standardized DataFrame
        """
        # Standard Binance klines columns
        expected_columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        # Handle parquet format with different column names
        if 'open_time' in df.columns:
            # Map parquet columns to standard names
            column_mapping = {
                'open_time': 'timestamp',
                'close_time': 'close_time',
                'quote_volume': 'quote_asset_volume',
                'trades': 'number_of_trades',
                'taker_buy_volume': 'taker_buy_base_asset_volume',
                'taker_buy_quote_volume': 'taker_buy_quote_asset_volume'
            }
            
            # Rename columns that exist
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
        
        # If DataFrame has the right number of columns but no headers (old CSV format)
        elif len(df.columns) == len(expected_columns) and df.columns[0] == 0:
            df.columns = expected_columns
            
        # Ensure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
                
        # Convert data types
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        return df
        
    def _standardize_funding_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize funding rate DataFrame column names and types.
        
        Args:
            df: Raw funding rate DataFrame
            
        Returns:
            Standardized DataFrame
        """
        # Ensure required columns exist
        required_columns = ['symbol', 'fundingTime', 'fundingRate']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required funding rate column: {col}")
                
        # Convert data types
        df['fundingTime'] = pd.to_numeric(df['fundingTime'])
        df['fundingRate'] = pd.to_numeric(df['fundingRate'])
        
        return df
        
    def _filter_by_date_range(
        self,
        df: pd.DataFrame,
        time_column: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """
        Filter DataFrame by date range.
        
        Args:
            df: DataFrame to filter
            time_column: Name of timestamp column
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Filtered DataFrame
        """
        if start_date:
            start_timestamp = int(start_date.timestamp() * 1000)
            df = df[df[time_column] >= start_timestamp]
            
        if end_date:
            end_timestamp = int(end_date.timestamp() * 1000)
            df = df[df[time_column] <= end_timestamp]
            
        return df
