"""
Base data provider interface for cryptocurrency data.

This module defines the abstract interface that all data providers
must implement to ensure consistent access to market data.
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime


class BaseDataProvider(ABC):
    """
    Abstract base class for data providers.
    
    All data providers must implement these methods to provide
    a consistent interface for accessing market data.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data provider.
        
        Args:
            data_path: Path to data directory
        """
        self.data_path = data_path
        
    @abstractmethod
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
            timeframe: Timeframe for data
            
        Returns:
            DataFrame with OHLCV data or None if not available
        """
        pass
        
    @abstractmethod
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
        pass
        
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols.
        
        Returns:
            List of available symbols
        """
        pass
        
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is valid and has data available.
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        pass
