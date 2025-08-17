"""
Core backtesting engine and related components.

This module contains the main backtesting engine, portfolio management,
execution simulation, and risk management components.
"""

from .engine import BacktestEngine, BacktestResults, Trade, Position, PortfolioManager

__all__ = [
    "BacktestEngine",
    "BacktestResults",
    "Trade",
    "Position",
    "PortfolioManager",
]