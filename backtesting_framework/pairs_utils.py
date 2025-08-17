"""
Utility functions for pairs trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import warnings

warnings.filterwarnings("ignore")


def find_cointegrated_pairs(
    price_data: Dict[str, pd.Series],
    significance_level: float = 0.05,
    min_correlation: float = 0.7,
) -> List[Tuple[str, str, Dict]]:
    """
    Find all cointegrated pairs from a dictionary of price series

    Args:
        price_data: Dictionary with symbol as key and price series as value
        significance_level: P-value threshold for cointegration test
        min_correlation: Minimum correlation threshold to pre-filter pairs

    Returns:
        List of tuples (symbol1, symbol2, cointegration_results)
    """
    symbols = list(price_data.keys())
    cointegrated_pairs = []

    print(f"🔍 Testing {len(symbols)} symbols for cointegration...")
    print(f"   Significance level: {significance_level}")
    print(f"   Minimum correlation: {min_correlation}")

    total_pairs = len(symbols) * (len(symbols) - 1) // 2
    tested_pairs = 0

    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            symbol1, symbol2 = symbols[i], symbols[j]
            tested_pairs += 1

            if tested_pairs % 10 == 0:
                print(f"   Progress: {tested_pairs}/{total_pairs} pairs tested")

            # Get aligned price series
            aligned_data = pd.DataFrame(
                {"price1": price_data[symbol1], "price2": price_data[symbol2]}
            ).dropna()

            if len(aligned_data) < 50:  # Need sufficient data
                continue

            # Pre-filter by correlation
            correlation = aligned_data["price1"].corr(aligned_data["price2"])
            if abs(correlation) < min_correlation:
                continue

            # Test cointegration
            try:
                test_stat, p_value, critical_values = coint(
                    aligned_data["price1"], aligned_data["price2"]
                )

                if p_value < significance_level:
                    # Calculate hedge ratio
                    model = OLS(aligned_data["price1"], aligned_data["price2"]).fit()
                    hedge_ratio = model.params[0]

                    # Test stationarity of residuals
                    residuals = (
                        aligned_data["price1"] - hedge_ratio * aligned_data["price2"]
                    )
                    adf_stat, adf_p_value, _, _, adf_critical, _ = adfuller(residuals)

                    cointegration_results = {
                        "is_cointegrated": True,
                        "p_value": p_value,
                        "test_statistic": test_stat,
                        "critical_values": critical_values,
                        "hedge_ratio": hedge_ratio,
                        "correlation": correlation,
                        "adf_statistic": adf_stat,
                        "adf_p_value": adf_p_value,
                        "adf_critical_values": adf_critical,
                        "residuals_stationary": adf_p_value < 0.05,
                    }

                    cointegrated_pairs.append((symbol1, symbol2, cointegration_results))

            except Exception as e:
                continue

    # Sort by p-value (most significant first)
    cointegrated_pairs.sort(key=lambda x: x[2]["p_value"])

    print(f"✅ Found {len(cointegrated_pairs)} cointegrated pairs")
    return cointegrated_pairs


def calculate_half_life(spread: pd.Series) -> float:
    """
    Calculate the half-life of mean reversion for a spread series

    Args:
        spread: Price spread series

    Returns:
        Half-life in number of periods
    """
    try:
        # Lag the spread
        spread_lag = spread.shift(1)
        spread_diff = spread.diff()

        # Remove NaN values
        data = pd.DataFrame(
            {"spread": spread, "spread_lag": spread_lag, "spread_diff": spread_diff}
        ).dropna()

        if len(data) < 10:
            return np.nan

        # Run regression: spread_diff = alpha + beta * spread_lag + error
        model = OLS(data["spread_diff"], data["spread_lag"]).fit()
        beta = model.params[0]

        if beta >= 0:  # No mean reversion
            return np.inf

        # Half-life = -ln(2) / ln(1 + beta)
        half_life = -np.log(2) / np.log(1 + beta)
        return half_life

    except Exception:
        return np.nan


def calculate_optimal_lookback(
    spread: pd.Series, min_lookback: int = 20, max_lookback: int = 200, step: int = 10
) -> int:
    """
    Find optimal lookback period for z-score calculation based on half-life

    Args:
        spread: Price spread series
        min_lookback: Minimum lookback period to test
        max_lookback: Maximum lookback period to test
        step: Step size for testing

    Returns:
        Optimal lookback period
    """
    half_life = calculate_half_life(spread)

    if np.isnan(half_life) or np.isinf(half_life):
        # Fallback to testing different lookbacks
        best_lookback = min_lookback
        best_score = -np.inf

        for lookback in range(min_lookback, max_lookback + 1, step):
            if len(spread) < lookback * 2:
                continue

            # Calculate z-score with this lookback
            rolling_mean = spread.rolling(window=lookback).mean()
            rolling_std = spread.rolling(window=lookback).std()
            zscore = (spread - rolling_mean) / rolling_std

            # Score based on z-score properties (want good mean reversion)
            zscore_clean = zscore.dropna()
            if len(zscore_clean) < 50:
                continue

            # Good z-score should have:
            # 1. Mean close to 0
            # 2. Standard deviation close to 1
            # 3. Not too many extreme values
            mean_score = 1 / (1 + abs(zscore_clean.mean()))
            std_score = 1 / (1 + abs(zscore_clean.std() - 1))
            extreme_penalty = 1 / (
                1 + (abs(zscore_clean) > 3).sum() / len(zscore_clean)
            )

            score = mean_score * std_score * extreme_penalty

            if score > best_score:
                best_score = score
                best_lookback = lookback

        return best_lookback

    else:
        # Use half-life as guide (typically 2-3 times half-life)
        optimal_lookback = int(np.clip(half_life * 2.5, min_lookback, max_lookback))
        return optimal_lookback


def analyze_spread_properties(spread: pd.Series, lookback_period: int = 60) -> Dict:
    """
    Analyze statistical properties of a price spread

    Args:
        spread: Price spread series
        lookback_period: Lookback period for rolling statistics

    Returns:
        Dictionary with spread analysis results
    """
    # Basic statistics
    basic_stats = spread.describe()

    # Half-life
    half_life = calculate_half_life(spread)

    # Rolling statistics
    rolling_mean = spread.rolling(window=lookback_period).mean()
    rolling_std = spread.rolling(window=lookback_period).std()
    zscore = (spread - rolling_mean) / rolling_std

    # Z-score statistics
    zscore_clean = zscore.dropna()
    zscore_stats = zscore_clean.describe() if len(zscore_clean) > 0 else {}

    # Stationarity test
    try:
        adf_stat, adf_p_value, _, _, adf_critical, _ = adfuller(spread.dropna())
        is_stationary = adf_p_value < 0.05
    except:
        adf_stat = adf_p_value = np.nan
        adf_critical = {}
        is_stationary = False

    # Mean reversion strength (Hurst exponent approximation)
    try:
        # Simple Hurst calculation
        lags = range(2, min(100, len(spread) // 4))
        tau = [
            np.sqrt(np.std(np.subtract(spread[lag:], spread[:-lag]))) for lag in lags
        ]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0] * 2.0
    except:
        hurst = np.nan

    # Volatility clustering (ARCH effect)
    try:
        returns = spread.pct_change().dropna()
        squared_returns = returns**2
        arch_lm_stat = len(returns) * squared_returns.autocorr(lag=1) ** 2
        arch_p_value = 1 - stats.chi2.cdf(arch_lm_stat, df=1)
        has_arch_effect = arch_p_value < 0.05
    except:
        arch_lm_stat = arch_p_value = np.nan
        has_arch_effect = False

    return {
        "basic_stats": basic_stats.to_dict(),
        "half_life": half_life,
        "optimal_lookback": calculate_optimal_lookback(spread),
        "zscore_stats": zscore_stats,
        "stationarity": {
            "adf_statistic": adf_stat,
            "adf_p_value": adf_p_value,
            "adf_critical_values": adf_critical,
            "is_stationary": is_stationary,
        },
        "hurst_exponent": hurst,
        "mean_reverting": hurst < 0.5 if not np.isnan(hurst) else None,
        "arch_effect": {
            "lm_statistic": arch_lm_stat,
            "p_value": arch_p_value,
            "has_effect": has_arch_effect,
        },
    }


def calculate_dynamic_thresholds(
    zscore: pd.Series, percentile_entry: float = 95, percentile_exit: float = 50
) -> Dict[str, float]:
    """
    Calculate dynamic entry/exit thresholds based on historical z-score distribution

    Args:
        zscore: Z-score series
        percentile_entry: Percentile for entry threshold (e.g., 95 = top/bottom 5%)
        percentile_exit: Percentile for exit threshold (e.g., 50 = median)

    Returns:
        Dictionary with threshold values
    """
    zscore_clean = zscore.dropna()

    if len(zscore_clean) < 50:
        return {
            "entry_threshold_upper": 2.0,
            "entry_threshold_lower": -2.0,
            "exit_threshold": 0.0,
            "stop_loss_upper": 3.0,
            "stop_loss_lower": -3.0,
        }

    # Calculate percentile-based thresholds
    entry_upper = np.percentile(zscore_clean, percentile_entry)
    entry_lower = np.percentile(zscore_clean, 100 - percentile_entry)
    exit_threshold = np.percentile(zscore_clean, percentile_exit)

    # Stop loss at more extreme percentiles
    stop_upper = np.percentile(zscore_clean, 99)
    stop_lower = np.percentile(zscore_clean, 1)

    return {
        "entry_threshold_upper": entry_upper,
        "entry_threshold_lower": entry_lower,
        "exit_threshold": exit_threshold,
        "stop_loss_upper": stop_upper,
        "stop_loss_lower": stop_lower,
    }


def calculate_position_sizing_kelly(
    returns: pd.Series, win_rate: float, avg_win: float, avg_loss: float
) -> float:
    """
    Calculate optimal position size using Kelly criterion

    Args:
        returns: Historical returns series
        win_rate: Win rate (0-1)
        avg_win: Average winning trade return
        avg_loss: Average losing trade return (positive value)

    Returns:
        Kelly fraction (0-1)
    """
    if win_rate <= 0 or win_rate >= 1 or avg_win <= 0 or avg_loss <= 0:
        return 0.0

    # Kelly formula: f = (bp - q) / b
    # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - win_rate

    kelly_fraction = (b * p - q) / b

    # Cap at reasonable levels and ensure positive
    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Max 25% of capital

    return kelly_fraction


def generate_pair_report(
    symbol1: str,
    symbol2: str,
    price1: pd.Series,
    price2: pd.Series,
    backtest_results: Optional[object] = None,
) -> str:
    """
    Generate a comprehensive report for a trading pair

    Args:
        symbol1: First symbol name
        symbol2: Second symbol name
        price1: First symbol price series
        price2: Second symbol price series
        backtest_results: Optional backtest results object

    Returns:
        Formatted report string
    """
    report = []
    report.append(f"📊 PAIRS TRADING REPORT: {symbol1} vs {symbol2}")
    report.append("=" * 60)

    # Align data
    aligned_data = pd.DataFrame({"price1": price1, "price2": price2}).dropna()

    if len(aligned_data) < 50:
        report.append("❌ Insufficient data for analysis")
        return "\n".join(report)

    # Basic relationship metrics
    correlation = aligned_data["price1"].corr(aligned_data["price2"])
    spread = aligned_data["price1"] - aligned_data["price2"]
    ratio = aligned_data["price1"] / aligned_data["price2"]

    report.append(f"\n📈 BASIC RELATIONSHIP METRICS")
    report.append(f"Data points: {len(aligned_data):,}")
    report.append(f"Correlation: {correlation:.4f}")
    report.append(f"Price ratio mean: {ratio.mean():.4f}")
    report.append(f"Price ratio std: {ratio.std():.4f}")

    # Cointegration analysis
    try:
        from .pairs_backtester import PairsBacktester

        backtester = PairsBacktester()
        coint_result = backtester.check_cointegration(price1, price2)

        report.append(f"\n🔗 COINTEGRATION ANALYSIS")
        report.append(
            f"Is cointegrated: {'✅ YES' if coint_result['is_cointegrated'] else '❌ NO'}"
        )
        report.append(f"P-value: {coint_result['p_value']:.6f}")
        if coint_result["hedge_ratio"]:
            report.append(f"Hedge ratio: {coint_result['hedge_ratio']:.4f}")
    except:
        report.append(f"\n🔗 COINTEGRATION ANALYSIS")
        report.append("Could not perform cointegration test")

    # Spread analysis
    spread_analysis = analyze_spread_properties(spread)

    report.append(f"\n📊 SPREAD ANALYSIS")
    report.append(f"Mean: {spread.mean():.4f}")
    report.append(f"Std Dev: {spread.std():.4f}")
    report.append(
        f"Half-life: {spread_analysis['half_life']:.1f} periods"
        if not np.isnan(spread_analysis["half_life"])
        else "Half-life: N/A"
    )
    report.append(f"Optimal lookback: {spread_analysis['optimal_lookback']} periods")
    report.append(
        f"Is stationary: {'✅ YES' if spread_analysis['stationarity']['is_stationary'] else '❌ NO'}"
    )

    if spread_analysis["mean_reverting"] is not None:
        report.append(
            f"Mean reverting: {'✅ YES' if spread_analysis['mean_reverting'] else '❌ NO'}"
        )

    # Backtest results (if provided)
    if backtest_results:
        report.append(f"\n🚀 BACKTEST RESULTS")
        for metric, value in backtest_results.metrics.items():
            if isinstance(value, float):
                report.append(f"{metric}: {value:.4f}")
            else:
                report.append(f"{metric}: {value}")

        if backtest_results.trades:
            winning_trades = [t for t in backtest_results.trades if t.pnl > 0]
            win_rate = len(winning_trades) / len(backtest_results.trades)
            report.append(f"Win rate: {win_rate:.2%}")

    # Trading recommendations
    report.append(f"\n💡 TRADING RECOMMENDATIONS")

    if "coint_result" in locals() and coint_result["is_cointegrated"]:
        if spread_analysis["stationarity"]["is_stationary"]:
            report.append(
                "✅ Pair shows good statistical properties for mean reversion trading"
            )
            report.append(
                f"✅ Recommended lookback period: {spread_analysis['optimal_lookback']}"
            )

            # Dynamic thresholds
            zscore = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
            thresholds = calculate_dynamic_thresholds(zscore)
            report.append(
                f"✅ Suggested entry thresholds: ±{abs(thresholds['entry_threshold_lower']):.2f}"
            )
            report.append(
                f"✅ Suggested stop loss: ±{abs(thresholds['stop_loss_lower']):.2f}"
            )
        else:
            report.append("⚠️  Spread is not stationary - consider different approach")
    else:
        report.append(
            "❌ Pair is not cointegrated - not suitable for mean reversion trading"
        )
        report.append("💡 Consider momentum-based strategies instead")

    return "\n".join(report)
