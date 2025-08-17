"""
Pairs trading strategies module.

This module contains specialized strategies for pairs trading
and statistical arbitrage.
"""

from .mean_reversion import MeanReversionStrategy

__all__ = [
    "MeanReversionStrategy",
]