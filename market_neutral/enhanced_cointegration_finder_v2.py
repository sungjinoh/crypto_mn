"""
Enhanced Cointegration Pair Finder for Market Neutral Trading - Version 2
This module finds cointegrated pairs from crypto futures data for mean reversion strategies.
Improvements include: stationarity checks, better half-life calculation, volume filters,
rolling cointegration analysis, and market regime detection.
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# Add parent directory to Python path to import backtesting_framework
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backtesting_framework.pairs_backtester import PairsBacktester
except ImportError:
    # Fallback: try direct import from relative path
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "backtesting_framework"
        ),
    )
    from pairs_backtester import PairsBacktester


class EnhancedCointegrationFinder:
    """
    An enhanced class to find and analyze cointegrated pairs from historical market data.
    Includes improvements for crypto futures trading.
    """

    def __init__(
        self,
        base_path: str = "binance_futures_data",
        resample_interval: str = "1H",  # Changed to 1H for more responsive signals
        min_data_points: int = 1000,
        significance_level: float = 0.05,
        min_daily_volume: float = 1000000,  # Minimum $1M daily volume
        n_jobs: int = -1,
        check_stationarity: bool = True,  # New: check for stationarity
        use_rolling_window: bool = False,  # New: rolling cointegration analysis
        rolling_window_size: int = 500,  # New: size of rolling window
        rolling_step_size: int = 100,  # New: step size for rolling window
    ):
        """
        Initialize the Enhanced CointegrationFinder.

        Args:
            base_path: Path to the data directory
            resample_interval: Resampling interval for klines
            min_data_points: Minimum number of data points required for analysis
            significance_level: P-value threshold for cointegration test
            min_daily_volume: Minimum daily trading volume in USD
            n_jobs: Number of parallel jobs (-1 for all cores)
            check_stationarity: Whether to check for stationarity before cointegration
            use_rolling_window: Whether to use rolling window analysis
            rolling_window_size: Size of rolling window for analysis
            rolling_step_size: Step size for rolling window
        """
        self.base_path = Path(base_path)
        self.klines_path = self.base_path / "klines"
        self.funding_path = self.base_path / "fundingRate"
        self.resample_interval = resample_interval
        self.min_data_points = min_data_points
        self.significance_level = significance_level
        self.min_daily_volume = min_daily_volume
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        self.check_stationarity = check_stationarity
        self.use_rolling_window = use_rolling_window
        self.rolling_window_size = rolling_window_size
        self.rolling_step_size = rolling_step_size
        self.backtester = PairsBacktester()

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from the data directory."""
        symbols = []
        if self.klines_path.exists():
            for symbol_dir in self.klines_path.iterdir():
                if symbol_dir.is_dir():
                    symbols.append(symbol_dir.name)
        return sorted(symbols)

    def read_kline(self, year: int, month: int, symbol: str) -> Optional[pd.DataFrame]:
        """Read kline data for a specific symbol, year, and month."""
        month_str = f"{month:02d}"
        file_path = self.klines_path / symbol / "1m" / f"{year}-{month_str}.parquet"

        if not file_path.exists():
            return None

        try:
            df = pd.read_parquet(file_path)

            # Process timestamp
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
            elif "open_time" in df.columns:
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                df.set_index("open_time", inplace=True)

            # Convert price and volume columns to float
            price_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
            for col in price_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except Exception as e:
            print(f"Error reading {symbol} {year}-{month_str}: {e}")
            return None

    def read_funding_rate(
        self, year: int, month: int, symbol: str
    ) -> Optional[pd.DataFrame]:
        """Read funding rate data for a specific symbol, year, and month."""
        month_str = f"{month:02d}"
        file_path = self.funding_path / symbol / f"{year}-{month_str}.parquet"

        if not file_path.exists():
            return None

        try:
            df = pd.read_parquet(file_path)

            # Process timestamp
            if "calc_time" in df.columns:
                df["calc_time"] = pd.to_datetime(df["calc_time"], unit="ms")
                df.set_index("calc_time", inplace=True)

            # Convert funding rate to float
            if "fundingRate" in df.columns:
                df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")

            return df

        except Exception as e:
            print(f"Error reading funding rate {symbol} {year}-{month_str}: {e}")
            return None

    def load_symbol_data(
        self, symbol: str, years: List[int], months
    ) -> Optional[pd.DataFrame]:
        """
        Load and combine data for a symbol across multiple years and months.

        Args:
            symbol: Trading symbol
            years: List of years of data to load
            months: Either List[int] (same months for all years) or
                   List[List[int]] (different months for each year)

        Returns:
            Combined DataFrame with kline and funding rate data
        """
        all_klines = []
        all_funding = []

        # Handle both formats: months as List[int] or List[List[int]]
        if isinstance(months[0], list):
            # Different months for each year: months = [[1,2,3], [4,5,6]]
            if len(months) != len(years):
                raise ValueError(
                    f"Length of months ({len(months)}) must match length of years ({len(years)}) when using per-year month specification"
                )

            for year, year_months in zip(years, months):
                for month in year_months:
                    # Read kline data
                    kline_df = self.read_kline(year, month, symbol)
                    if kline_df is not None:
                        all_klines.append(kline_df)

                    # Read funding rate data
                    funding_df = self.read_funding_rate(year, month, symbol)
                    if funding_df is not None:
                        all_funding.append(funding_df)
        else:
            # Same months for all years: months = [1,2,3]
            for year in years:
                for month in months:
                    # Read kline data
                    kline_df = self.read_kline(year, month, symbol)
                    if kline_df is not None:
                        all_klines.append(kline_df)

                    # Read funding rate data
                    funding_df = self.read_funding_rate(year, month, symbol)
                    if funding_df is not None:
                        all_funding.append(funding_df)

        if not all_klines:
            return None

        # Combine all years and months
        combined_klines = pd.concat(all_klines, axis=0).sort_index()

        # Remove duplicates if any (based on index)
        combined_klines = combined_klines[
            ~combined_klines.index.duplicated(keep="first")
        ]

        # Resample klines
        resampled_klines = self.resample_klines(combined_klines)

        # Combine funding rates if available
        if all_funding:
            combined_funding = pd.concat(all_funding, axis=0).sort_index()
            # Remove duplicates if any (based on index)
            combined_funding = combined_funding[
                ~combined_funding.index.duplicated(keep="first")
            ]
            # Merge with funding rates
            result = self.merge_kline_with_funding(resampled_klines, combined_funding)
        else:
            result = resampled_klines
            # Add empty funding rate columns
            result["fundingRate"] = 0.0
            result["funding_interval_hours"] = 8.0

        # Reset index and rename
        result = result.reset_index()
        if "open_time" in result.columns:
            result = result.rename(columns={"open_time": "timestamp"})
        elif result.index.name in ["timestamp", "open_time"]:
            result = result.reset_index().rename(
                columns={result.index.name: "timestamp"}
            )

        return result

    def resample_klines(self, kline_df: pd.DataFrame) -> pd.DataFrame:
        """Resample 1-minute kline data to specified interval."""
        df = kline_df.copy()

        # Aggregation rules for OHLCV data
        agg_rules = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        # Add other columns if they exist
        if "quote_volume" in df.columns:
            agg_rules["quote_volume"] = "sum"
        if "trades" in df.columns:
            agg_rules["trades"] = "sum"
        if "taker_buy_volume" in df.columns:
            agg_rules["taker_buy_volume"] = "sum"
        if "taker_buy_quote_volume" in df.columns:
            agg_rules["taker_buy_quote_volume"] = "sum"

        # Resample
        df_resampled = df.resample(self.resample_interval).agg(agg_rules)
        df_resampled = df_resampled.dropna()

        return df_resampled

    def merge_kline_with_funding(
        self, kline_df: pd.DataFrame, funding_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge kline data with funding rate data."""
        # Select only funding rate columns
        funding_cols = ["fundingRate"]
        if "funding_interval_hours" in funding_df.columns:
            funding_cols.append("funding_interval_hours")

        funding_to_merge = funding_df[funding_cols].copy()

        # Left join
        combined_df = kline_df.join(funding_to_merge, how="left")

        # Fill missing values (forward fill then backward fill)
        combined_df[funding_cols] = combined_df[funding_cols].fillna(method="ffill")
        combined_df[funding_cols] = combined_df[funding_cols].fillna(method="bfill")

        return combined_df

    def check_stationarity_adf(
        self, series: pd.Series, significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Check if a time series is stationary using Augmented Dickey-Fuller test.

        Args:
            series: Time series to test
            significance_level: Significance level for the test

        Returns:
            Dictionary with test results
        """
        try:
            # Remove NaN values
            clean_series = series.dropna()

            if len(clean_series) < 50:
                return {
                    "is_stationary": False,
                    "p_value": 1.0,
                    "test_statistic": None,
                    "critical_values": None,
                    "error": "Insufficient data",
                }

            result = adfuller(clean_series, autolag="AIC")

            return {
                "is_stationary": result[1]
                < significance_level,  # Reject null hypothesis of unit root
                "is_non_stationary": result[1] >= significance_level,  # For I(1) check
                "p_value": result[1],
                "test_statistic": result[0],
                "critical_values": result[4],
                "lags_used": result[2],
                "observations": result[3],
            }
        except Exception as e:
            return {
                "is_stationary": False,
                "p_value": 1.0,
                "test_statistic": None,
                "critical_values": None,
                "error": str(e),
            }

    def calculate_half_life_ou(self, spread: pd.Series) -> Optional[float]:
        """
        Calculate half-life of mean reversion using Ornstein-Uhlenbeck process.

        Args:
            spread: Spread time series

        Returns:
            Half-life in periods or None if calculation fails
        """
        try:
            spread = spread.dropna()

            if len(spread) < 50:
                return None

            # Lag the spread
            spread_lag = spread.shift(1)
            spread_diff = spread.diff()

            # Remove NaN values
            spread_lag = spread_lag[1:]
            spread_diff = spread_diff[1:]

            # Ensure same length
            min_len = min(len(spread_lag), len(spread_diff))
            spread_lag = spread_lag.iloc[:min_len]
            spread_diff = spread_diff.iloc[:min_len]

            # Run regression: spread_diff = theta * spread_lag + epsilon
            model = OLS(spread_diff, spread_lag).fit()

            # Calculate half-life
            theta = model.params[0]

            if theta < 0:  # Mean reverting
                half_life = -np.log(2) / theta

                # Sanity check: half-life should be positive and reasonable
                if 0 < half_life < len(spread) / 2:
                    return float(half_life)

            return None

        except Exception as e:
            print(f"Error calculating half-life: {e}")
            return None

    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect market regime (bull/bear/sideways) based on price action.

        Args:
            data: DataFrame with price data

        Returns:
            Market regime string
        """
        try:
            # Use closing prices
            if "close" in data.columns:
                prices = data["close"]
            else:
                # Find first column with 'close' in name
                close_cols = [col for col in data.columns if "close" in col.lower()]
                if close_cols:
                    prices = data[close_cols[0]]
                else:
                    return "unknown"

            # Calculate returns
            returns = prices.pct_change().dropna()

            # Calculate metrics
            avg_return = returns.mean()
            volatility = returns.std()

            # Simple 50-day and 200-day moving averages (scaled to data frequency)
            periods_per_day = 24 if "H" in self.resample_interval else 1
            ma_short = prices.rolling(
                window=50 * periods_per_day, min_periods=20
            ).mean()
            ma_long = prices.rolling(
                window=200 * periods_per_day, min_periods=50
            ).mean()

            # Current position relative to MAs
            current_price = prices.iloc[-1]
            current_ma_short = ma_short.iloc[-1] if len(ma_short) > 0 else current_price
            current_ma_long = ma_long.iloc[-1] if len(ma_long) > 0 else current_price

            # Trend strength
            trend_strength = abs(avg_return) / volatility if volatility > 0 else 0

            # Classify regime
            if (
                current_price > current_ma_short > current_ma_long
                and avg_return > 0.001
            ):
                regime = "bull"
            elif (
                current_price < current_ma_short < current_ma_long
                and avg_return < -0.001
            ):
                regime = "bear"
            elif trend_strength < 0.1:  # Low trend relative to volatility
                regime = "sideways"
            else:
                regime = "transitional"

            return regime

        except Exception as e:
            print(f"Error detecting market regime: {e}")
            return "unknown"

    def calculate_volume_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate volume-related metrics for liquidity assessment.

        Args:
            data: DataFrame with volume data

        Returns:
            Dictionary with volume metrics
        """
        try:
            volume_metrics = {}

            # Check for volume columns
            if "volume" in data.columns:
                volume = data["volume"]
                volume_metrics["avg_volume"] = float(volume.mean())
                volume_metrics["median_volume"] = float(volume.median())
                volume_metrics["min_volume"] = float(volume.min())
                volume_metrics["volume_stability"] = float(
                    volume.std() / volume.mean() if volume.mean() > 0 else np.inf
                )

            if "quote_volume" in data.columns:
                quote_volume = data["quote_volume"]
                volume_metrics["avg_quote_volume"] = float(quote_volume.mean())
                volume_metrics["median_quote_volume"] = float(quote_volume.median())

                # Calculate average price for daily volume estimate
                if "close" in data.columns and "volume" in data.columns:
                    avg_price = (data["close"] * data["volume"]).sum() / data[
                        "volume"
                    ].sum()

                    # Estimate daily volume based on resample interval
                    if "H" in self.resample_interval:
                        hours = int(
                            self.resample_interval.replace("H", "").replace("h", "")
                        )
                        periods_per_day = 24 / hours
                    elif (
                        "T" in self.resample_interval or "min" in self.resample_interval
                    ):
                        minutes = int(
                            self.resample_interval.replace("T", "").replace("min", "")
                        )
                        periods_per_day = 1440 / minutes
                    else:
                        periods_per_day = 1

                    volume_metrics["estimated_daily_volume_usd"] = float(
                        quote_volume.mean() * periods_per_day
                    )

            return volume_metrics

        except Exception as e:
            print(f"Error calculating volume metrics: {e}")
            return {}

    def test_rolling_cointegration(
        self,
        symbol1: str,
        symbol2: str,
        data_dict: Dict[str, pd.DataFrame],
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
    ) -> List[Dict]:
        """
        Test cointegration over rolling windows to assess stability.

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            data_dict: Dictionary with symbol data
            window_size: Size of rolling window
            step_size: Step size for rolling window

        Returns:
            List of cointegration results for each window
        """
        if window_size is None:
            window_size = self.rolling_window_size
        if step_size is None:
            step_size = self.rolling_step_size

        results = []

        try:
            # Get data for both symbols
            df1 = data_dict.get(symbol1)
            df2 = data_dict.get(symbol2)

            if df1 is None or df2 is None:
                return results

            # Prepare pair data
            pair_data = self.backtester.prepare_pair_data(df1, df2, symbol1, symbol2)

            if len(pair_data) < window_size:
                return results

            # Rolling windows
            for i in range(0, len(pair_data) - window_size + 1, step_size):
                window_data = pair_data.iloc[i : i + window_size]

                # Test cointegration for this window
                coint_result = self.backtester.check_cointegration(
                    window_data[f"{symbol1}_close"],
                    window_data[f"{symbol2}_close"],
                    significance_level=self.significance_level,
                )

                # Add window information
                coint_result["window_start"] = window_data.index[0].isoformat()
                coint_result["window_end"] = window_data.index[-1].isoformat()
                coint_result["window_index"] = i

                results.append(coint_result)

            return results

        except Exception as e:
            print(f"Error in rolling cointegration test for {symbol1}-{symbol2}: {e}")
            return results

    def analyze_funding_impact(
        self, data: pd.DataFrame, symbol: str
    ) -> Dict[str, float]:
        """
        Analyze funding rate impact on trading costs.

        Args:
            data: DataFrame with funding rate data
            symbol: Symbol name

        Returns:
            Dictionary with funding rate analysis
        """
        try:
            if "fundingRate" not in data.columns:
                return {}

            funding = data["fundingRate"].dropna()

            if len(funding) == 0:
                return {}

            # Calculate funding metrics
            metrics = {
                "avg_funding_rate": float(funding.mean()),
                "median_funding_rate": float(funding.median()),
                "max_funding_rate": float(funding.max()),
                "min_funding_rate": float(funding.min()),
                "funding_volatility": float(funding.std()),
                "positive_funding_pct": float((funding > 0).mean()),
                "negative_funding_pct": float((funding < 0).mean()),
                # Annualized funding cost assuming 8-hour funding periods
                "annualized_funding_long": float(funding.mean() * 3 * 365),
                "annualized_funding_short": float(-funding.mean() * 3 * 365),
            }

            return metrics

        except Exception as e:
            print(f"Error analyzing funding impact for {symbol}: {e}")
            return {}

    def test_pair_cointegration(
        self, symbol1: str, symbol2: str, data_dict: Dict[str, pd.DataFrame]
    ) -> Optional[Dict]:
        """
        Enhanced test for cointegration between two symbols.

        Returns:
            Dictionary with cointegration results or None if test fails
        """
        try:
            # Get data for both symbols
            df1 = data_dict.get(symbol1)
            df2 = data_dict.get(symbol2)

            if df1 is None or df2 is None:
                return None

            # Check volume requirements
            volume_metrics1 = self.calculate_volume_metrics(df1)
            volume_metrics2 = self.calculate_volume_metrics(df2)

            # Check if both meet minimum volume requirements
            min_vol1 = volume_metrics1.get("estimated_daily_volume_usd", 0)
            min_vol2 = volume_metrics2.get("estimated_daily_volume_usd", 0)

            if min_vol1 < self.min_daily_volume or min_vol2 < self.min_daily_volume:
                print(
                    f"Pair {symbol1}-{symbol2} fails volume requirement: ${min_vol1:.0f} / ${min_vol2:.0f}"
                )
                return None

            # Prepare pair data
            pair_data = self.backtester.prepare_pair_data(df1, df2, symbol1, symbol2)

            if len(pair_data) < self.min_data_points:
                return None

            # Check stationarity if enabled
            stationarity_results = {}
            if self.check_stationarity:
                # Check if price series are I(1) - non-stationary in levels, stationary in differences
                stat1_level = self.check_stationarity_adf(pair_data[f"{symbol1}_close"])
                stat2_level = self.check_stationarity_adf(pair_data[f"{symbol2}_close"])

                stat1_diff = self.check_stationarity_adf(
                    pair_data[f"{symbol1}_close"].diff().dropna()
                )
                stat2_diff = self.check_stationarity_adf(
                    pair_data[f"{symbol2}_close"].diff().dropna()
                )

                stationarity_results = {
                    "symbol1_is_I1": stat1_level["is_non_stationary"]
                    and stat1_diff["is_stationary"],
                    "symbol2_is_I1": stat2_level["is_non_stationary"]
                    and stat2_diff["is_stationary"],
                    "symbol1_level_pvalue": stat1_level["p_value"],
                    "symbol2_level_pvalue": stat2_level["p_value"],
                    "symbol1_diff_pvalue": stat1_diff["p_value"],
                    "symbol2_diff_pvalue": stat2_diff["p_value"],
                }

                # Both should be I(1) for cointegration
                if not (
                    stationarity_results["symbol1_is_I1"]
                    and stationarity_results["symbol2_is_I1"]
                ):
                    print(f"Pair {symbol1}-{symbol2} fails I(1) requirement")
                    # Continue anyway but flag it
                    stationarity_results["warning"] = "Series may not be I(1)"

            # Test cointegration
            coint_result = self.backtester.check_cointegration(
                pair_data[f"{symbol1}_close"],
                pair_data[f"{symbol2}_close"],
                significance_level=self.significance_level,
            )

            # Add additional information
            coint_result["symbol1"] = symbol1
            coint_result["symbol2"] = symbol2
            coint_result["data_points"] = len(pair_data)
            coint_result["start_date"] = pair_data.index[0].isoformat()
            coint_result["end_date"] = pair_data.index[-1].isoformat()

            # Calculate correlation
            correlation = pair_data[f"{symbol1}_close"].corr(
                pair_data[f"{symbol2}_close"]
            )
            coint_result["correlation"] = correlation

            # Add stationarity results
            if stationarity_results:
                coint_result["stationarity"] = stationarity_results

            # Add volume metrics
            coint_result["volume_metrics"] = {
                "symbol1": volume_metrics1,
                "symbol2": volume_metrics2,
            }

            # Detect market regime
            coint_result["market_regime"] = self.detect_market_regime(pair_data)

            # Analyze funding impact
            coint_result["funding_analysis"] = {
                "symbol1": self.analyze_funding_impact(df1, symbol1),
                "symbol2": self.analyze_funding_impact(df2, symbol2),
            }

            # Analyze spread properties if cointegrated
            if coint_result["is_cointegrated"]:
                # Use consistent spread calculation method
                hedge_ratio = coint_result["hedge_ratio"]
                intercept = coint_result.get("intercept", 0)
                use_log_prices = coint_result.get("use_log_prices", False)

                if use_log_prices:
                    spread = (
                        np.log(pair_data[f"{symbol1}_close"])
                        - hedge_ratio * np.log(pair_data[f"{symbol2}_close"])
                        - intercept
                    )
                else:
                    spread = (
                        pair_data[f"{symbol1}_close"]
                        - hedge_ratio * pair_data[f"{symbol2}_close"]
                        - intercept
                    )

                # Enhanced spread properties with OU half-life
                spread_props = analyze_spread_properties_enhanced(spread)
                spread_props["half_life_ou"] = self.calculate_half_life_ou(spread)

                # Check spread stationarity
                spread_stat = self.check_stationarity_adf(spread)
                spread_props["spread_is_stationary"] = spread_stat["is_stationary"]
                spread_props["spread_stationarity_pvalue"] = spread_stat["p_value"]

                coint_result["spread_properties"] = spread_props

                # Test rolling cointegration if enabled
                if self.use_rolling_window:
                    rolling_results = self.test_rolling_cointegration(
                        symbol1, symbol2, data_dict
                    )
                    if rolling_results:
                        # Calculate stability metrics
                        cointegrated_windows = [
                            r for r in rolling_results if r["is_cointegrated"]
                        ]
                        coint_result["rolling_stability"] = {
                            "total_windows": len(rolling_results),
                            "cointegrated_windows": len(cointegrated_windows),
                            "stability_ratio": (
                                len(cointegrated_windows) / len(rolling_results)
                                if rolling_results
                                else 0
                            ),
                            "hedge_ratio_std": np.std(
                                [
                                    r["hedge_ratio"]
                                    for r in rolling_results
                                    if r.get("hedge_ratio")
                                ]
                            ),
                        }

            return coint_result

        except Exception as e:
            print(f"Error testing {symbol1}-{symbol2}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def find_all_cointegrated_pairs(
        self,
        years: List[int] = [2024],
        months=[1, 2, 3],
        symbols: Optional[List[str]] = None,
        max_symbols: Optional[int] = None,
        use_parallel: bool = True,
        filter_by_sector: bool = False,  # New: filter by sector/correlation
        max_correlation: float = 0.95,  # New: filter out perfect correlation
    ) -> Dict:
        """
        Find all cointegrated pairs from available symbols across multiple years.

        Args:
            years: List of years of data to analyze
            months: Either List[int] (same months for all years) or
                   List[List[int]] (different months for each year)
            symbols: List of symbols to analyze (None for all available)
            max_symbols: Maximum number of symbols to analyze (for testing)
            use_parallel: Whether to use parallel processing
            filter_by_sector: Whether to filter pairs by sector similarity
            max_correlation: Maximum correlation allowed (filter out near-perfect correlations)

        Returns:
            Dictionary with results and metadata
        """
        # Get symbols to analyze
        if symbols is None:
            symbols = self.get_available_symbols()

        if max_symbols:
            symbols = symbols[:max_symbols]

        print(f"Found {len(symbols)} symbols to analyze")
        print(f"Loading data for years {years}, months {months}...")
        print(f"Resample interval: {self.resample_interval}")
        print(f"Minimum daily volume: ${self.min_daily_volume:,.0f}")

        # Load data for all symbols
        data_dict = {}
        for symbol in tqdm(symbols, desc="Loading symbol data"):
            df = self.load_symbol_data(symbol, years, months)
            if df is not None and len(df) >= self.min_data_points:
                # Pre-check volume to save computation
                volume_metrics = self.calculate_volume_metrics(df)
                if (
                    volume_metrics.get("estimated_daily_volume_usd", 0)
                    >= self.min_daily_volume
                ):
                    data_dict[symbol] = df
                else:
                    print(f"Symbol {symbol} excluded due to low volume")

        print(
            f"Successfully loaded data for {len(data_dict)} symbols (after volume filter)"
        )

        # Optional: Filter by sector similarity (e.g., don't pair BTC with BTCUSDT)
        if filter_by_sector:
            symbol_pairs = self._filter_pairs_by_sector(list(data_dict.keys()))
        else:
            symbol_pairs = list(combinations(data_dict.keys(), 2))

        print(f"Testing {len(symbol_pairs)} symbol pairs for cointegration...")

        # Test all pairs
        results = []

        if use_parallel and self.n_jobs > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(self.test_pair_cointegration, s1, s2, data_dict): (
                        s1,
                        s2,
                    )
                    for s1, s2 in symbol_pairs
                }

                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Testing pairs"
                ):
                    result = future.result()
                    if result is not None:
                        # Filter by correlation
                        if abs(result.get("correlation", 0)) < max_correlation:
                            results.append(result)
                        else:
                            print(
                                f"Pair {result['symbol1']}-{result['symbol2']} filtered due to high correlation: {result['correlation']:.3f}"
                            )
        else:
            # Sequential processing
            for s1, s2 in tqdm(symbol_pairs, desc="Testing pairs"):
                result = self.test_pair_cointegration(s1, s2, data_dict)
                if result is not None:
                    # Filter by correlation
                    if abs(result.get("correlation", 0)) < max_correlation:
                        results.append(result)

        # Filter cointegrated pairs
        cointegrated_pairs = [r for r in results if r["is_cointegrated"]]

        # Sort by multiple criteria
        def sort_key(x):
            # Primary: p-value (lower is better)
            # Secondary: stability ratio if available (higher is better)
            # Tertiary: half-life (moderate values preferred)
            stability = x.get("rolling_stability", {}).get("stability_ratio", 0)
            half_life = x.get("spread_properties", {}).get("half_life_ou", 1000)
            half_life_score = (
                abs(half_life - 30) if half_life else 1000
            )  # Prefer half-life around 30 periods

            return (x["p_value"], -stability, half_life_score)

        cointegrated_pairs.sort(key=sort_key)

        # Prepare final results
        final_results = {
            "metadata": {
                "analysis_date": datetime.now().isoformat(),
                "data_years": years,
                "data_months": months,
                "resample_interval": self.resample_interval,
                "significance_level": self.significance_level,
                "min_data_points": self.min_data_points,
                "min_daily_volume": self.min_daily_volume,
                "check_stationarity": self.check_stationarity,
                "use_rolling_window": self.use_rolling_window,
                "total_symbols": len(symbols),
                "symbols_with_data": len(data_dict),
                "total_pairs_tested": len(symbol_pairs),
                "cointegrated_pairs_found": len(cointegrated_pairs),
            },
            "cointegrated_pairs": cointegrated_pairs,
            "all_results": results,
        }

        return final_results

    def _filter_pairs_by_sector(self, symbols: List[str]) -> List[Tuple[str, str]]:
        """
        Filter symbol pairs to avoid obvious non-cointegrated pairs.
        For crypto, avoid pairing spot-like products with their derivatives.
        """
        pairs = []

        for s1, s2 in combinations(symbols, 2):
            # Avoid pairing a symbol with its own derivative
            # E.g., don't pair BTC with BTCUSDT, ETH with ETHUSDT
            if s1 in s2 or s2 in s1:
                continue

            # Avoid pairing stablecoins with non-stablecoins
            stablecoins = ["USDT", "USDC", "BUSD", "DAI", "TUSD"]
            s1_is_stable = any(stable in s1.upper() for stable in stablecoins)
            s2_is_stable = any(stable in s2.upper() for stable in stablecoins)

            if s1_is_stable != s2_is_stable:
                continue

            pairs.append((s1, s2))

        return pairs

    def save_results(
        self,
        results: Dict,
        output_dir: str = "cointegration_results_v2",
        formats: List[str] = ["json", "csv", "pickle"],
    ) -> None:
        """
        Save cointegration results in multiple formats.

        Args:
            results: Results dictionary from find_all_cointegrated_pairs
            output_dir: Directory to save results
            formats: List of formats to save ('json', 'csv', 'pickle', 'parquet')
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        if "json" in formats:
            json_file = output_path / f"cointegration_results_{timestamp}.json"
            try:
                with open(json_file, "w") as f:
                    # Convert numpy types to Python types for JSON serialization
                    json_results = self._convert_for_json(results)
                    json.dump(json_results, f, indent=2)
                print(f"Saved JSON results to {json_file}")
            except TypeError as e:
                print(f"Warning: Could not save JSON due to serialization error: {e}")
                print("Falling back to pickle format for complete data preservation.")
                if "pickle" not in formats:
                    formats.append("pickle")

        # Save as CSV (cointegrated pairs only)
        if "csv" in formats:
            csv_file = output_path / f"cointegrated_pairs_{timestamp}.csv"
            if results["cointegrated_pairs"]:
                # Flatten nested dictionaries for CSV
                flattened_results = []
                for pair in results["cointegrated_pairs"]:
                    flat_pair = {
                        "symbol1": pair["symbol1"],
                        "symbol2": pair["symbol2"],
                        "p_value": pair["p_value"],
                        "hedge_ratio": pair["hedge_ratio"],
                        "correlation": pair["correlation"],
                        "data_points": pair["data_points"],
                        "start_date": pair["start_date"],
                        "end_date": pair["end_date"],
                    }

                    # Add spread properties
                    if "spread_properties" in pair:
                        for key, value in pair["spread_properties"].items():
                            flat_pair[f"spread_{key}"] = value

                    # Add rolling stability
                    if "rolling_stability" in pair:
                        for key, value in pair["rolling_stability"].items():
                            flat_pair[f"rolling_{key}"] = value

                    # Add volume metrics
                    if "volume_metrics" in pair:
                        if "symbol1" in pair["volume_metrics"]:
                            for key, value in pair["volume_metrics"]["symbol1"].items():
                                flat_pair[f"symbol1_{key}"] = value
                        if "symbol2" in pair["volume_metrics"]:
                            for key, value in pair["volume_metrics"]["symbol2"].items():
                                flat_pair[f"symbol2_{key}"] = value

                    flattened_results.append(flat_pair)

                df = pd.DataFrame(flattened_results)
                df.to_csv(csv_file, index=False)
                print(f"Saved CSV results to {csv_file}")

        # Save as Pickle (preserves all data types)
        if "pickle" in formats:
            pickle_file = output_path / f"cointegration_results_{timestamp}.pkl"
            with open(pickle_file, "wb") as f:
                pickle.dump(results, f)
            print(f"Saved pickle results to {pickle_file}")

        # Save summary report
        self._save_summary_report(results, output_path / f"summary_{timestamp}.txt")

    def _convert_for_json(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self._convert_for_json(item) for item in obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.signedinteger, np.unsignedinteger)):
                return int(obj)
            elif isinstance(obj, np.floating):
                if np.ndim(obj) == 0:
                    if np.isnan(obj) or np.isinf(obj):
                        return None
                return float(obj)
            elif isinstance(obj, np.complexfloating):
                return str(obj)
            else:
                try:
                    return obj.item()
                except:
                    return str(obj)
        if hasattr(pd, "Timestamp") and isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if hasattr(pd, "NaT") and obj is pd.NaT:
            return None
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(obj, "item"):
            try:
                if hasattr(obj, "shape") and obj.shape == ():
                    return obj.item()
            except (ValueError, AttributeError, TypeError):
                pass
        if hasattr(obj, "__class__") and "RegressionResultsWrapper" in str(type(obj)):
            try:
                return {
                    "r_squared": float(obj.rsquared),
                    "aic": float(obj.aic),
                    "bic": float(obj.bic),
                    "params": [float(p) for p in obj.params],
                    "pvalues": [float(p) for p in obj.pvalues],
                    "object_type": "RegressionResultsWrapper",
                }
            except:
                return str(obj)
        try:
            return str(obj)
        except:
            return None

    def _save_summary_report(self, results: Dict, filepath: Path) -> None:
        """Save a human-readable summary report."""
        with open(filepath, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("ENHANCED COINTEGRATION ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Metadata
            meta = results["metadata"]
            f.write("Analysis Parameters:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Analysis Date: {meta['analysis_date']}\n")
            f.write(
                f"Data Period: Years {meta['data_years']}, Months {meta['data_months']}\n"
            )
            f.write(f"Resample Interval: {meta['resample_interval']}\n")
            f.write(f"Significance Level: {meta['significance_level']}\n")
            f.write(f"Min Data Points: {meta['min_data_points']}\n")
            f.write(f"Min Daily Volume: ${meta['min_daily_volume']:,.0f}\n")
            f.write(f"Stationarity Check: {meta.get('check_stationarity', False)}\n")
            f.write(
                f"Rolling Window Analysis: {meta.get('use_rolling_window', False)}\n\n"
            )

            f.write("Results Summary:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Symbols Analyzed: {meta['total_symbols']}\n")
            f.write(f"Symbols with Sufficient Data: {meta['symbols_with_data']}\n")
            f.write(f"Total Pairs Tested: {meta['total_pairs_tested']}\n")
            f.write(f"Cointegrated Pairs Found: {meta['cointegrated_pairs_found']}\n")
            if meta["total_pairs_tested"] > 0:
                f.write(
                    f"Success Rate: {meta['cointegrated_pairs_found']/meta['total_pairs_tested']*100:.2f}%\n\n"
                )
            else:
                f.write("Success Rate: N/A (no pairs tested)\n\n")

            # Top cointegrated pairs
            if results["cointegrated_pairs"]:
                f.write(
                    "Top 20 Cointegrated Pairs (sorted by p-value and stability):\n"
                )
                f.write("-" * 40 + "\n")
                for i, pair in enumerate(results["cointegrated_pairs"][:20], 1):
                    f.write(f"{i:2d}. {pair['symbol1']:12s} - {pair['symbol2']:12s} | ")
                    f.write(f"p-value: {pair['p_value']:.6f} | ")
                    f.write(f"hedge_ratio: {pair['hedge_ratio']:.4f} | ")
                    f.write(f"correlation: {pair['correlation']:.4f}\n")

                    if "spread_properties" in pair and pair["spread_properties"]:
                        props = pair["spread_properties"]
                        half_life_ou = props.get("half_life_ou")
                        if half_life_ou is not None:
                            f.write(
                                f"    → Half-life (OU): {half_life_ou:.1f} periods | "
                            )
                        else:
                            f.write(f"    → Half-life: N/A | ")
                        f.write(f"Mean: {props.get('mean', 0):.6f} | ")
                        f.write(f"Std: {props.get('std', 0):.6f}")
                        if props.get("spread_is_stationary"):
                            f.write(" | Spread: Stationary ✓")
                        f.write("\n")

                    if "rolling_stability" in pair and pair["rolling_stability"]:
                        stability = pair["rolling_stability"]
                        f.write(
                            f"    → Stability: {stability['stability_ratio']:.1%} ({stability['cointegrated_windows']}/{stability['total_windows']} windows)\n"
                        )

                    if "market_regime" in pair:
                        f.write(f"    → Market Regime: {pair['market_regime']}\n")

        print(f"Saved summary report to {filepath}")

    def load_results(self, filepath: str) -> Dict:
        """
        Load previously saved results.

        Args:
            filepath: Path to the results file

        Returns:
            Results dictionary
        """
        filepath = Path(filepath)

        if filepath.suffix == ".json":
            with open(filepath, "r") as f:
                return json.load(f)
        elif filepath.suffix == ".pkl":
            with open(filepath, "rb") as f:
                return pickle.load(f)
        elif filepath.suffix == ".csv":
            df = pd.read_csv(filepath)
            return {"cointegrated_pairs": df.to_dict("records")}
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")


def analyze_spread_properties_enhanced(spread: pd.Series) -> Dict:
    """
    Enhanced analysis of statistical properties of the spread series.

    Args:
        spread: The spread time series

    Returns:
        Dictionary with spread properties
    """
    try:
        properties = {
            "mean": float(spread.mean()),
            "std": float(spread.std()),
            "min": float(spread.min()),
            "max": float(spread.max()),
            "skewness": float(spread.skew()),
            "kurtosis": float(spread.kurtosis()),
            "sharpe_ratio": (
                float(spread.mean() / spread.std()) if spread.std() > 0 else 0
            ),
        }

        # Calculate percentiles
        properties["percentile_25"] = float(spread.quantile(0.25))
        properties["percentile_75"] = float(spread.quantile(0.75))
        properties["iqr"] = properties["percentile_75"] - properties["percentile_25"]

        # Calculate half-life of mean reversion (simple AR(1) method)
        try:
            spread_diff = spread.diff().dropna()
            spread_lag = spread.shift(1).dropna()

            # Align the series
            min_len = min(len(spread_diff), len(spread_lag))
            spread_diff = spread_diff.iloc[:min_len]
            spread_lag = spread_lag.iloc[:min_len]

            # Calculate correlation coefficient for AR(1)
            if len(spread_diff) > 1 and spread_lag.std() > 0:
                correlation = spread_diff.corr(spread_lag)
                if correlation < 0 and correlation > -1:
                    half_life = -np.log(2) / np.log(1 + correlation)
                    properties["half_life"] = float(half_life)
                else:
                    properties["half_life"] = None
            else:
                properties["half_life"] = None
        except:
            properties["half_life"] = None

        # Calculate number of zero crossings (mean reversion frequency)
        mean_spread = spread.mean()
        zero_crossings = (
            (spread - mean_spread).shift(1) * (spread - mean_spread) < 0
        ).sum()
        properties["zero_crossings"] = int(zero_crossings)
        properties["crossing_frequency"] = zero_crossings / len(spread)

        return properties

    except Exception as e:
        print(f"Error analyzing spread properties: {e}")
        return {
            "mean": 0.0,
            "std": 1.0,
            "min": 0.0,
            "max": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "half_life": None,
        }


def main():
    """Main function to run the enhanced cointegration analysis."""

    # Initialize the enhanced finder with improved parameters
    finder = EnhancedCointegrationFinder(
        base_path="binance_futures_data",
        resample_interval="1H",  # More responsive 1-hour candles for crypto
        min_data_points=1000,
        significance_level=0.05,
        min_daily_volume=1000000,  # $1M minimum daily volume
        n_jobs=-1,
        check_stationarity=True,  # Enable stationarity checks
        use_rolling_window=True,  # Enable rolling window analysis
        rolling_window_size=500,  # ~20 days of hourly data
        rolling_step_size=100,  # ~4 days step
    )

    # Find cointegrated pairs with enhanced analysis
    results = finder.find_all_cointegrated_pairs(
        years=[2023, 2024, 2025],
        months=[
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # All months for 2023
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Jan-May for 2024
            [1, 2, 3],
        ],
        max_symbols=None,  # Set to a number for testing (e.g., 30)
        use_parallel=True,
        filter_by_sector=True,  # Filter obvious non-pairs
        max_correlation=0.95,  # Filter out near-perfect correlations
    )

    # Print summary
    print("\n" + "=" * 80)
    print("ENHANCED ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total pairs tested: {results['metadata']['total_pairs_tested']}")
    print(
        f"Cointegrated pairs found: {results['metadata']['cointegrated_pairs_found']}"
    )

    if results["cointegrated_pairs"]:
        print(f"\nTop 10 cointegrated pairs with best stability:")
        for i, pair in enumerate(results["cointegrated_pairs"][:10], 1):
            print(
                f"{i:2d}. {pair['symbol1']:10s} - {pair['symbol2']:10s} | "
                f"p-value: {pair['p_value']:.6f} | "
                f"hedge_ratio: {pair['hedge_ratio']:.4f}"
            )
            if "rolling_stability" in pair:
                stability = pair["rolling_stability"]["stability_ratio"]
                print(f"    Stability: {stability:.1%}")
            if "spread_properties" in pair:
                half_life = pair["spread_properties"].get("half_life_ou")
                if half_life:
                    print(f"    Half-life: {half_life:.1f} periods")

    # Save results in multiple formats
    finder.save_results(
        results,
        output_dir="cointegration_results_v2",
        formats=["json", "csv", "pickle"],
    )

    print("\nResults saved to 'cointegration_results_v2' directory")

    return results


if __name__ == "__main__":
    results = main()
