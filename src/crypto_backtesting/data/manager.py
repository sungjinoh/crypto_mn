"""
Data Manager for unified access to cryptocurrency data.

This module provides a unified interface for accessing market data from
various sources, with caching and preprocessing capabilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import logging
from abc import ABC, abstractmethod

from .providers.base import BaseDataProvider
from .providers.binance_futures import BinanceFuturesProvider


class DataManager:
    """
    Unified data management system for cryptocurrency backtesting.
    
    Provides a single interface for accessing klines, funding rates,
    and other market data with transparent caching and preprocessing.
    """
    
    def __init__(
        self,
        data_path: str = "binance_futures_data",
        cache_enabled: bool = True,
        cache_path: Optional[str] = None
    ):
        """
        Initialize DataManager.
        
        Args:
            data_path: Path to market data directory
            cache_enabled: Whether to use data caching
            cache_path: Path for cache files (defaults to data_path/cache)
        """
        self.data_path = Path(data_path)
        self.cache_enabled = cache_enabled
        self.cache_path = Path(cache_path or self.data_path / "cache")
        
        # Initialize cache directory
        if self.cache_enabled:
            self.cache_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize data providers
        self.providers: Dict[str, BaseDataProvider] = {
            "binance_futures": BinanceFuturesProvider(str(self.data_path))
        }
        
        # Data cache
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def get_klines_data(
        self,
        symbol: str,
        year: int,
        months: Optional[List[int]] = None,
        provider: str = "binance_futures",
        timeframe: str = "1m"
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV klines data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            year: Year of data
            months: List of months (None for all months)
            provider: Data provider to use
            timeframe: Timeframe for data
            
        Returns:
            DataFrame with OHLCV data or None if not available
        """
        cache_key = f"klines_{provider}_{symbol}_{year}_{months}_{timeframe}"
        
        # Check cache first
        if self.cache_enabled and cache_key in self._data_cache:
            return self._data_cache[cache_key].copy()
            
        # Get data from provider
        if provider not in self.providers:
            self.logger.error(f"Unknown provider: {provider}")
            return None
            
        try:
            data = self.providers[provider].get_klines_data(
                symbol, year, months, timeframe
            )
            
            if data is not None and self.cache_enabled:
                self._data_cache[cache_key] = data.copy()
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading klines data for {symbol}: {e}")
            return None
            
    def get_funding_rates(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        provider: str = "binance_futures"
    ) -> Optional[pd.DataFrame]:
        """
        Get funding rate data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            provider: Data provider to use
            
        Returns:
            DataFrame with funding rate data or None if not available
        """
        cache_key = f"funding_{provider}_{symbol}_{start_date}_{end_date}"
        
        # Check cache first
        if self.cache_enabled and cache_key in self._data_cache:
            return self._data_cache[cache_key].copy()
            
        # Get data from provider
        if provider not in self.providers:
            self.logger.error(f"Unknown provider: {provider}")
            return None
            
        try:
            data = self.providers[provider].get_funding_rates(
                symbol, start_date, end_date
            )
            
            if data is not None and self.cache_enabled:
                self._data_cache[cache_key] = data.copy()
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading funding rates for {symbol}: {e}")
            return None
            
    def get_pair_data(
        self,
        symbol1: str,
        symbol2: str,
        year: int,
        months: Optional[List[int]] = None,
        provider: str = "binance_futures",
        resample_timeframe: Optional[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Get synchronized data for a trading pair.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            year: Year of data
            months: List of months
            provider: Data provider
            resample_timeframe: Timeframe to resample to (e.g., '5T', '1H')
            
        Returns:
            Tuple of DataFrames (symbol1_data, symbol2_data)
        """
        # Load data for both symbols
        data1 = self.get_klines_data(symbol1, year, months, provider)
        data2 = self.get_klines_data(symbol2, year, months, provider)
        
        if data1 is None or data2 is None:
            return None, None
            
        # Synchronize timestamps
        data1, data2 = self._synchronize_data(data1, data2)
        
        # Resample if requested
        if resample_timeframe:
            data1 = self._resample_data(data1, resample_timeframe)
            data2 = self._resample_data(data2, resample_timeframe)
            
        return data1, data2
        
    def _synchronize_data(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Synchronize two datasets by timestamp.
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            Tuple of synchronized datasets
        """
        # Ensure datetime index
        if 'timestamp' in data1.columns:
            data1['datetime'] = pd.to_datetime(data1['timestamp'], unit='ms')
            data1.set_index('datetime', inplace=True)
            
        if 'timestamp' in data2.columns:
            data2['datetime'] = pd.to_datetime(data2['timestamp'], unit='ms')
            data2.set_index('datetime', inplace=True)
            
        # Find common timestamps
        common_index = data1.index.intersection(data2.index)
        
        if len(common_index) == 0:
            self.logger.warning("No common timestamps found between datasets")
            return data1, data2
            
        # Filter to common timestamps
        data1_synced = data1.loc[common_index].copy()
        data2_synced = data2.loc[common_index].copy()
        
        return data1_synced, data2_synced
        
    def _resample_data(
        self,
        data: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Resample data to specified timeframe.
        
        Args:
            data: Input data
            timeframe: Target timeframe (pandas frequency string)
            
        Returns:
            Resampled data
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have datetime index for resampling")
            
        # Define aggregation rules for OHLCV data
        agg_rules = {
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Only aggregate columns that exist
        available_rules = {k: v for k, v in agg_rules.items() if k in data.columns}
        
        if not available_rules:
            # If no OHLCV columns, use mean for all numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            available_rules = {col: 'mean' for col in numeric_cols}
            
        resampled = data.resample(timeframe).agg(available_rules)
        
        # Forward fill any NaN values
        resampled = resampled.fillna(method='ffill')
        
        return resampled
        
    def get_available_symbols(self, provider: str = "binance_futures") -> List[str]:
        """
        Get list of available trading symbols.
        
        Args:
            provider: Data provider to query
            
        Returns:
            List of available symbols
        """
        if provider not in self.providers:
            self.logger.error(f"Unknown provider: {provider}")
            return []
            
        try:
            return self.providers[provider].get_available_symbols()
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []
            
    def validate_data_availability(
        self,
        symbols: List[str],
        year: int,
        months: Optional[List[int]] = None,
        provider: str = "binance_futures"
    ) -> Dict[str, bool]:
        """
        Check data availability for multiple symbols.
        
        Args:
            symbols: List of symbols to check
            year: Year to check
            months: Months to check
            provider: Data provider
            
        Returns:
            Dict mapping symbol to availability status
        """
        availability = {}
        
        for symbol in symbols:
            data = self.get_klines_data(symbol, year, months, provider)
            availability[symbol] = data is not None and len(data) > 0
            
        return availability
        
    def clear_cache(self):
        """Clear the data cache."""
        self._data_cache.clear()
        self.logger.info("Data cache cleared")
        
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache state.
        
        Returns:
            Dict with cache statistics
        """
        return {
            "enabled": self.cache_enabled,
            "cache_path": str(self.cache_path),
            "cached_datasets": len(self._data_cache),
            "memory_usage_mb": sum(
                df.memory_usage(deep=True).sum() 
                for df in self._data_cache.values()
            ) / 1024 / 1024
        }
