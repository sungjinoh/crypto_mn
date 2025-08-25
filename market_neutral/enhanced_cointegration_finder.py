"""
Enhanced Cointegration Pair Finder for Market Neutral Trading
This module finds cointegrated pairs from crypto futures data for mean reversion strategies.
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

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


class CointegrationFinder:
    """
    A class to find and analyze cointegrated pairs from historical market data.
    """

    def __init__(
        self,
        base_path: str = "binance_futures_data",
        resample_interval: str = "30T",
        min_data_points: int = 1000,
        significance_level: float = 0.05,
        n_jobs: int = -1,
    ):
        """
        Initialize the CointegrationFinder.

        Args:
            base_path: Path to the data directory
            resample_interval: Resampling interval for klines (e.g., '30T' for 30 minutes)
            min_data_points: Minimum number of data points required for analysis
            significance_level: P-value threshold for cointegration test
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.base_path = Path(base_path)
        self.klines_path = self.base_path / "klines"
        self.funding_path = self.base_path / "fundingRate"
        self.resample_interval = resample_interval
        self.min_data_points = min_data_points
        self.significance_level = significance_level
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
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

            # Convert price columns to float
            price_cols = ["open", "high", "low", "close", "volume"]
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

    def test_pair_cointegration(
        self, symbol1: str, symbol2: str, data_dict: Dict[str, pd.DataFrame]
    ) -> Optional[Dict]:
        """
        Test cointegration between two symbols.

        Returns:
            Dictionary with cointegration results or None if test fails
        """
        try:
            # Get data for both symbols
            df1 = data_dict.get(symbol1)
            df2 = data_dict.get(symbol2)

            if df1 is None or df2 is None:
                return None

            # Prepare pair data
            pair_data = self.backtester.prepare_pair_data(df1, df2, symbol1, symbol2)

            if len(pair_data) < self.min_data_points:
                return None

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

                spread_props = analyze_spread_properties(spread)
                coint_result["spread_properties"] = spread_props

            return coint_result

        except Exception as e:
            print(f"Error testing {symbol1}-{symbol2}: {e}")
            return None

    def find_all_cointegrated_pairs(
        self,
        years: List[int] = [2024],
        months=[1, 2, 3],
        symbols: Optional[List[str]] = None,
        max_symbols: Optional[int] = None,
        use_parallel: bool = True,
    ) -> Dict:
        """
        Find all cointegrated pairs from available symbols across multiple years.

        Args:
            years: List of years of data to analyze
            months: Either List[int] (same months for all years) or
                   List[List[int]] (different months for each year)
                   Examples:
                   - [1, 2, 3] -> months 1,2,3 for all years
                   - [[1,2,3,4,5,6,7,8,9,10,11,12], [1,2,3,4,5]] -> all months for first year, Jan-May for second year
            symbols: List of symbols to analyze (None for all available)
            max_symbols: Maximum number of symbols to analyze (for testing)
            use_parallel: Whether to use parallel processing

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

        # Load data for all symbols
        data_dict = {}
        for symbol in tqdm(symbols, desc="Loading symbol data"):
            df = self.load_symbol_data(symbol, years, months)
            if df is not None and len(df) >= self.min_data_points:
                data_dict[symbol] = df

        print(f"Successfully loaded data for {len(data_dict)} symbols")

        # Generate all pairs
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
                        results.append(result)
        else:
            # Sequential processing
            for s1, s2 in tqdm(symbol_pairs, desc="Testing pairs"):
                result = self.test_pair_cointegration(s1, s2, data_dict)
                if result is not None:
                    results.append(result)

        # Filter cointegrated pairs
        cointegrated_pairs = [r for r in results if r["is_cointegrated"]]

        # Sort by p-value
        cointegrated_pairs.sort(key=lambda x: x["p_value"])

        # Prepare final results
        final_results = {
            "metadata": {
                "analysis_date": datetime.now().isoformat(),
                "data_years": years,
                "data_months": months,
                "resample_interval": self.resample_interval,
                "significance_level": self.significance_level,
                "min_data_points": self.min_data_points,
                "total_symbols": len(symbols),
                "symbols_with_data": len(data_dict),
                "total_pairs_tested": len(symbol_pairs),
                "cointegrated_pairs_found": len(cointegrated_pairs),
            },
            "cointegrated_pairs": cointegrated_pairs,
            "all_results": results,
        }

        return final_results

    def save_results(
        self,
        results: Dict,
        output_dir: str = "cointegration_results",
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
                # Ensure pickle is saved if JSON fails
                if "pickle" not in formats:
                    formats.append("pickle")

        # Save as CSV (cointegrated pairs only)
        if "csv" in formats:
            csv_file = output_path / f"cointegrated_pairs_{timestamp}.csv"
            if results["cointegrated_pairs"]:
                df = pd.DataFrame(results["cointegrated_pairs"])
                # Flatten spread_properties if present
                if "spread_properties" in df.columns:
                    spread_props = pd.json_normalize(df["spread_properties"])
                    spread_props.columns = [
                        f"spread_{col}" for col in spread_props.columns
                    ]
                    df = pd.concat(
                        [df.drop("spread_properties", axis=1), spread_props], axis=1
                    )
                df.to_csv(csv_file, index=False)
                print(f"Saved CSV results to {csv_file}")

        # Save as Pickle (preserves all data types)
        if "pickle" in formats:
            pickle_file = output_path / f"cointegration_results_{timestamp}.pkl"
            with open(pickle_file, "wb") as f:
                pickle.dump(results, f)
            print(f"Saved pickle results to {pickle_file}")

        # Save as Parquet (efficient for large datasets)
        if "parquet" in formats:
            parquet_file = output_path / f"cointegrated_pairs_{timestamp}.parquet"
            if results["cointegrated_pairs"]:
                df = pd.DataFrame(results["cointegrated_pairs"])
                # Handle nested dictionaries
                if "spread_properties" in df.columns:
                    # Store as JSON string for parquet compatibility
                    df["spread_properties"] = df["spread_properties"].apply(json.dumps)
                df.to_parquet(parquet_file, index=False)
                print(f"Saved parquet results to {parquet_file}")

        # Save summary report
        self._save_summary_report(results, output_path / f"summary_{timestamp}.txt")

    def _convert_for_json(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        # Handle None first
        if obj is None:
            return None

        # Handle standard Python types first (most common case)
        if isinstance(obj, (str, int, float, bool)):
            return obj

        # Handle dictionaries recursively
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}

        # Handle lists recursively
        if isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]

        # Handle tuples
        if isinstance(obj, tuple):
            return tuple(self._convert_for_json(item) for item in obj)

        # Handle numpy arrays (before scalar checks)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle numpy scalar types
        if isinstance(obj, np.generic):
            # Handle numpy boolean
            if isinstance(obj, np.bool_):
                return bool(obj)
            # Handle numpy integers
            elif isinstance(obj, (np.integer, np.signedinteger, np.unsignedinteger)):
                return int(obj)
            # Handle numpy floats
            elif isinstance(obj, np.floating):
                # Check for scalar NaN or Inf
                if np.ndim(obj) == 0:  # It's a scalar
                    if np.isnan(obj) or np.isinf(obj):
                        return None  # JSON doesn't support NaN/Inf
                return float(obj)
            # Handle numpy complex (convert to string)
            elif isinstance(obj, np.complexfloating):
                return str(obj)
            # Try to extract scalar value
            else:
                try:
                    return obj.item()
                except:
                    return str(obj)

        # Handle pandas Timestamp
        if hasattr(pd, "Timestamp") and isinstance(obj, pd.Timestamp):
            return obj.isoformat()

        # Handle pandas NaT (Not a Time) - but check type first
        if hasattr(pd, "NaT") and obj is pd.NaT:
            return None

        # Handle other pandas null values
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass  # obj is not a valid input for pd.isna

        # Handle objects with .item() method (like numpy scalars)
        if hasattr(obj, "item"):
            try:
                # Check if it's truly a scalar (0-dimensional)
                if hasattr(obj, "shape") and obj.shape == ():
                    return obj.item()
            except (ValueError, AttributeError, TypeError):
                pass

        # Last resort: convert to string
        try:
            return str(obj)
        except:
            return None

    def _save_summary_report(self, results: Dict, filepath: Path) -> None:
        """Save a human-readable summary report."""
        with open(filepath, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("COINTEGRATION ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Metadata
            meta = results["metadata"]
            f.write("Analysis Parameters:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Analysis Date: {meta['analysis_date']}\n")
            # Handle both old format (single year) and new format (multiple years)
            if "data_years" in meta:
                months_display = format_months_display(
                    meta["data_years"], meta["data_months"]
                )
                f.write(f"Data Period: {months_display}\n")
            else:
                f.write(
                    f"Data Period: {meta['data_year']}, Months: {meta['data_months']}\n"
                )
            f.write(f"Resample Interval: {meta['resample_interval']}\n")
            f.write(f"Significance Level: {meta['significance_level']}\n")
            f.write(f"Min Data Points: {meta['min_data_points']}\n\n")

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
                f.write("Top 20 Cointegrated Pairs (sorted by p-value):\n")
                f.write("-" * 40 + "\n")
                for i, pair in enumerate(results["cointegrated_pairs"][:20], 1):
                    f.write(f"{i:2d}. {pair['symbol1']:12s} - {pair['symbol2']:12s} | ")
                    f.write(f"p-value: {pair['p_value']:.6f} | ")
                    f.write(f"hedge_ratio: {pair['hedge_ratio']:.4f} | ")
                    f.write(f"correlation: {pair['correlation']:.4f}\n")

                    if "spread_properties" in pair and pair["spread_properties"]:
                        props = pair["spread_properties"]
                        if props.get("half_life") is not None:
                            f.write(
                                f"    → Half-life: {props.get('half_life'):.1f} periods | "
                            )
                        else:
                            f.write(f"    → Half-life: N/A | ")
                        f.write(f"Mean: {props.get('mean', 0):.6f} | ")
                        f.write(f"Std: {props.get('std', 0):.6f}\n")

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
        elif filepath.suffix == ".parquet":
            df = pd.read_parquet(filepath)
            # Parse JSON strings back to dictionaries if needed
            if "spread_properties" in df.columns:
                df["spread_properties"] = df["spread_properties"].apply(json.loads)
            return {"cointegrated_pairs": df.to_dict("records")}
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")


def format_months_display(years: List[int], months) -> str:
    """
    Format months for display in reports.

    Args:
        years: List of years
        months: Either List[int] or List[List[int]]

    Returns:
        Formatted string describing the months
    """
    if isinstance(months[0], list):
        # Different months for each year
        parts = []
        for year, year_months in zip(years, months):
            if len(year_months) == 12 and year_months == list(range(1, 13)):
                parts.append(f"{year}: all months")
            elif len(year_months) <= 3:
                month_names = [f"{m:02d}" for m in year_months]
                parts.append(f"{year}: {','.join(month_names)}")
            else:
                parts.append(
                    f"{year}: {len(year_months)} months ({year_months[0]:02d}-{year_months[-1]:02d})"
                )
        return "; ".join(parts)
    else:
        # Same months for all years
        if len(months) <= 3:
            month_names = [f"{m:02d}" for m in months]
            return f"months {','.join(month_names)} for all years"
        else:
            return (
                f"{len(months)} months ({months[0]:02d}-{months[-1]:02d}) for all years"
            )


def analyze_spread_properties(spread: pd.Series) -> Dict:
    """
    Analyze statistical properties of the spread series.

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
        }

        # Calculate half-life of mean reversion
        try:
            # Simple method: fit AR(1) model
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
    """Main function to run the cointegration analysis."""

    # Initialize the finder
    finder = CointegrationFinder(
        base_path="binance_futures_data",
        resample_interval="30T",  # 30-minute candles
        min_data_points=1000,  # Minimum data points required
        significance_level=0.05,  # 5% significance level
        n_jobs=-1,  # Use all CPU cores
    )

    # Find cointegrated pairs with per-year month specification
    results = finder.find_all_cointegrated_pairs(
        years=[2023, 2024],  # Multiple years
        months=[
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [1, 2, 3, 4, 5],
        ],  # All months for 2023, Jan-May for 2024
        max_symbols=None,  # Set to a number for testing (e.g., 20)
        use_parallel=True,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total pairs tested: {results['metadata']['total_pairs_tested']}")
    print(
        f"Cointegrated pairs found: {results['metadata']['cointegrated_pairs_found']}"
    )

    if results["cointegrated_pairs"]:
        print(f"\nTop 10 cointegrated pairs:")
        for i, pair in enumerate(results["cointegrated_pairs"][:10], 1):
            print(
                f"{i:2d}. {pair['symbol1']:10s} - {pair['symbol2']:10s} | "
                f"p-value: {pair['p_value']:.6f} | "
                f"hedge_ratio: {pair['hedge_ratio']:.4f}"
            )

    # Save results in multiple formats
    finder.save_results(
        results,
        output_dir="cointegration_results",
        formats=["json", "csv", "pickle", "parquet"],
    )

    print("\nResults saved to 'cointegration_results' directory")

    return results


if __name__ == "__main__":
    results = main()
