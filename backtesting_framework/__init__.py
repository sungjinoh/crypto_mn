"""
Pairs Trading Backtesting Framework

A comprehensive framework for backtesting pairs trading strategies
with focus on mean reversion statistical arbitrage.
"""

from .pairs_backtester import (
    PairsBacktester,
    PairsStrategy,
    Trade,
    BacktestResults,
    plot_backtest_results,
)

from .pairs_utils import (
    find_cointegrated_pairs,
    calculate_half_life,
    calculate_optimal_lookback,
    analyze_spread_properties,
    calculate_dynamic_thresholds,
    calculate_position_sizing_kelly,
    generate_pair_report,
)

__version__ = "1.0.0"
__author__ = "Kiro AI"

__all__ = [
    # Core backtesting classes
    "PairsBacktester",
    "PairsStrategy",
    "Trade",
    "BacktestResults",
    "plot_backtest_results",
    # Utility functions
    "find_cointegrated_pairs",
    "calculate_half_life",
    "calculate_optimal_lookback",
    "analyze_spread_properties",
    "calculate_dynamic_thresholds",
    "calculate_position_sizing_kelly",
    "generate_pair_report",
]
