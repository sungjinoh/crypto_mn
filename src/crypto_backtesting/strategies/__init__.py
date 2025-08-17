"""
Trading strategies for cryptocurrency backtesting.

This module contains base strategy classes and implementations for
various trading strategies including pairs trading, momentum, and arbitrage.
"""

from .base import BaseStrategy, PairsStrategy, Signal, SignalType, StrategyConfig, PairsStrategyConfig
from .pairs.mean_reversion import MeanReversionStrategy

__all__ = [
    "BaseStrategy",
    "PairsStrategy",
    "Signal",
    "SignalType",
    "StrategyConfig", 
    "PairsStrategyConfig",
    "MeanReversionStrategy",
]