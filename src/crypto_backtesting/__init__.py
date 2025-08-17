"""
Cryptocurrency Backtesting Framework

A comprehensive framework for backtesting cryptocurrency trading strategies,
with focus on pairs trading and statistical arbitrage.

This framework provides:
- Unified data management across multiple data sources
- Modular strategy architecture with standardized interfaces
- Comprehensive backtesting engine with realistic cost modeling
- Advanced performance analysis and reporting
- Support for pairs trading and statistical arbitrage strategies
"""

__version__ = "1.0.0"
__author__ = "Crypto Backtesting Team"

# Core components
from .backtesting.engine import BacktestEngine, BacktestResults, Trade, Position
from .strategies.base import BaseStrategy, PairsStrategy, Signal, SignalType, StrategyConfig
from .data.manager import DataManager

# Strategy implementations
from .strategies.pairs.mean_reversion import MeanReversionStrategy

# Analysis and reporting
from .analysis.performance import PerformanceAnalyzer, ReportGenerator, compare_strategies

# Data providers
from .data.providers.base import BaseDataProvider
from .data.providers.binance_futures import BinanceFuturesProvider

__all__ = [
    # Core backtesting
    "BacktestEngine",
    "BacktestResults", 
    "Trade",
    "Position",
    
    # Strategy framework
    "BaseStrategy",
    "PairsStrategy",
    "Signal",
    "SignalType", 
    "StrategyConfig",
    
    # Strategy implementations
    "MeanReversionStrategy",
    
    # Data management
    "DataManager",
    "BaseDataProvider",
    "BinanceFuturesProvider",
    
    # Analysis and reporting
    "PerformanceAnalyzer",
    "ReportGenerator",
    "compare_strategies",
]