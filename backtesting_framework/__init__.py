"""
Backtesting Framework for Pairs Trading
"""

from .pairs_backtester import PairsBacktester
from .pairs_utils import (
    find_cointegrated_pairs,
    calculate_half_life,
    calculate_optimal_lookback,
    analyze_spread_properties,
    calculate_dynamic_thresholds,
    calculate_position_sizing_kelly,
    generate_pair_report,
    calculate_spread_and_zscore,
    create_trading_signals,
)

__version__ = "1.0.0"
__all__ = [
    "PairsBacktester",
    "find_cointegrated_pairs",
    "calculate_half_life",
    "calculate_optimal_lookback",
    "analyze_spread_properties",
    "calculate_dynamic_thresholds",
    "calculate_position_sizing_kelly",
    "generate_pair_report",
    "calculate_spread_and_zscore",
    "create_trading_signals",
]
