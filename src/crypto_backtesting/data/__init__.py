"""
Data management layer for cryptocurrency backtesting framework.

This module provides data loading, caching, and preprocessing capabilities
for various cryptocurrency data sources.
"""

from .manager import DataManager
from .providers.base import BaseDataProvider
from .providers.binance_futures import BinanceFuturesProvider

__all__ = [
    "DataManager",
    "BaseDataProvider", 
    "BinanceFuturesProvider",
]