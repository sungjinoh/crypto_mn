"""
Analysis tools for backtesting results and market data.

This module provides statistical analysis, performance metrics,
visualization, and reporting capabilities.
"""

from .performance import PerformanceAnalyzer, ReportGenerator, compare_strategies

__all__ = [
    "PerformanceAnalyzer",
    "ReportGenerator",
    "compare_strategies",
]