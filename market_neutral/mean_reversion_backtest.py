"""
Mean Reversion Backtesting Script
This script loads cointegration results, filters top pairs, and runs mean reversion backtests.
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from backtesting_framework.pairs_backtester import (
    PairsBacktester,
    plot_backtest_results,
)
from mean_reversion_strategy import MeanReversionStrategy
from enhanced_cointegration_finder import CointegrationFinder


class MeanReversionBacktester:
    """
    Class to run mean reversion backtests on cointegrated pairs.
    """

    def __init__(
        self,
        base_path: str = "binance_futures_data",
        results_dir: str = "cointegration_results",
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        position_size: float = 0.5,
        resample_timeframe: Optional[str] = None,  # e.g., '5T', '15T', '1H'
        save_plots: bool = True,
        plots_dir: str = "backtest_plots",
    ):
        """
        Initialize the backtester.

        Args:
            base_path: Path to the data directory
            results_dir: Directory containing cointegration results
            initial_capital: Initial capital for backtesting
            transaction_cost: Transaction cost as a fraction
            position_size: Position size as a fraction of capital
            resample_timeframe: Timeframe for resampling (None for 1min data)
            save_plots: Whether to save plot images
            plots_dir: Directory to save plots
        """
        self.base_path = Path(base_path)
        self.results_dir = Path(results_dir)
        self.save_plots = save_plots
        self.plots_dir = plots_dir
        self.resample_timeframe = resample_timeframe

        # Resampling functionality note
        if resample_timeframe is not None:
            print(
                f"⚠️ Note: Timeframe resampling to {resample_timeframe} is stored but not currently implemented in PairsBacktester"
            )
            print(f"   Data will be processed at its original frequency")

        self.finder = CointegrationFinder(base_path=base_path)
        self.backtester = PairsBacktester(
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            position_size=position_size,
        )

        # Trade logging settings
        self.save_trades = True
        self.trades_dir = "trade_logs"

    def load_cointegration_results(self, filepath: Optional[str] = None) -> Dict:
        """
        Load cointegration results from file.

        Args:
            filepath: Specific file to load, or None to load most recent

        Returns:
            Dictionary with cointegration results
        """
        if filepath:
            filepath = Path(filepath)
        else:
            # Find most recent results file
            json_files = list(self.results_dir.glob("cointegration_results_*.json"))
            pkl_files = list(self.results_dir.glob("cointegration_results_*.pkl"))

            all_files = json_files + pkl_files
            if not all_files:
                raise FileNotFoundError(f"No results files found in {self.results_dir}")

            filepath = max(all_files, key=lambda p: p.stat().st_mtime)
            print(f"Loading most recent results from: {filepath}")

        # Load based on file type
        if filepath.suffix == ".json":
            with open(filepath, "r") as f:
                return json.load(f)
        elif filepath.suffix == ".pkl":
            with open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def filter_top_pairs(
        self,
        results: Dict,
        n_pairs: int = 10,
        max_p_value: float = 0.05,
        min_correlation: float = 0.6,
        max_half_life: Optional[float] = 100,
    ) -> List[Dict]:
        """
        Filter and rank pairs based on multiple criteria.

        Args:
            results: Cointegration results dictionary
            n_pairs: Number of top pairs to return
            max_p_value: Maximum p-value threshold
            min_correlation: Minimum correlation threshold
            max_half_life: Maximum half-life threshold

        Returns:
            List of top pairs with scoring
        """
        pairs = results.get("cointegrated_pairs", [])

        if not pairs:
            print("No cointegrated pairs found in results")
            return []

        # Filter pairs
        filtered_pairs = []
        for pair in pairs:
            # Check p-value
            if pair["p_value"] > max_p_value:
                continue

            # Check correlation
            if abs(pair["correlation"]) < min_correlation:
                continue

            # Check half-life if available
            if (
                max_half_life
                and "spread_properties" in pair
                and pair["spread_properties"]
            ):
                half_life = pair["spread_properties"].get("half_life")
                if half_life and half_life > max_half_life:
                    continue

            filtered_pairs.append(pair)

        # Score and rank pairs
        scored_pairs = []
        for pair in filtered_pairs:
            # Calculate composite score
            p_value_score = 1 - pair["p_value"]  # Lower is better
            correlation_score = abs(pair["correlation"])  # Higher is better

            # Half-life score (lower is better for faster mean reversion)
            half_life_score = 0.5  # Default
            if "spread_properties" in pair and pair["spread_properties"]:
                half_life = pair["spread_properties"].get("half_life")
                if half_life and half_life > 0:
                    half_life_score = max(0, 1 - (half_life / 100))

            # Weighted composite score
            composite_score = (
                0.3 * p_value_score
                + 0.3 * correlation_score
                + 0.4 * half_life_score  # Prioritize fast mean reversion
            )

            pair_with_score = pair.copy()
            pair_with_score["composite_score"] = composite_score
            scored_pairs.append(pair_with_score)

        # Sort by composite score
        scored_pairs.sort(key=lambda x: x["composite_score"], reverse=True)

        # Return top N pairs
        return scored_pairs[:n_pairs]

    def load_pair_data(
        self,
        symbol1: str,
        symbol2: str,
        years: List[int] = [2024, 2025],
        months: List[int] = [4, 5, 6],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data for a specific pair for backtesting.

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            year: Year of data
            months: Months to load (for out-of-sample testing)

        Returns:
            Tuple of (symbol1_data, symbol2_data)
        """
        # Load data for both symbols
        df1 = self.finder.load_symbol_data(symbol1, years, months)
        df2 = self.finder.load_symbol_data(symbol2, years, months)

        if df1 is None or df2 is None:
            raise ValueError(f"Could not load data for {symbol1} or {symbol2}")

        return df1, df2

    def run_backtest_with_strategy(
        self,
        symbol1_data: pd.DataFrame,
        symbol2_data: pd.DataFrame,
        symbol1: str,
        symbol2: str,
        strategy: MeanReversionStrategy,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_plot: Optional[bool] = None,
        plot_dir: Optional[str] = None,
    ):
        """
        Custom run_backtest method that properly handles strategy parameters.

        Args:
            symbol1_data: OHLCV data for first symbol
            symbol2_data: OHLCV data for second symbol
            symbol1: Symbol name for first asset
            symbol2: Symbol name for second asset
            strategy: Trading strategy instance
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)
            save_plot: Override default plot saving setting (optional)
            plot_dir: Override default plot directory (optional)

        Returns:
            BacktestResults object with all results
        """
        self.backtester.reset()

        # Prepare data
        data = self.backtester.prepare_pair_data(
            symbol1_data, symbol2_data, symbol1, symbol2
        )

        # Filter by date range if specified
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        # Add indicators using strategy's lookback period
        data = self.backtester.add_indicators(data, strategy.lookback_period)

        # Check cointegration using the transformation flag from prepared data
        use_log_flag = data.attrs.get("use_log_prices", None)
        cointegration_results = self.backtester.check_cointegration(
            data[f"{symbol1}_close"],
            data[f"{symbol2}_close"],
            use_log_prices=use_log_flag,
        )

        # If cointegrated, calculate consistent spread and signals
        if cointegration_results.get("is_cointegrated", False):
            # Use the hedge ratio and other parameters from cointegration analysis
            data = self.backtester.calculate_spread_and_signals(
                data,
                hedge_ratio=cointegration_results["hedge_ratio"],
                intercept=cointegration_results.get("intercept", 0),
                use_log_prices=cointegration_results.get("use_log_prices", False),
                lookback_period=strategy.lookback_period,
            )

        # Generate signals
        signals = strategy.generate_signals(data)

        # Combine data and signals
        combined_data = pd.concat([data, signals], axis=1)

        # Run backtest
        portfolio_values = []
        positions_data = []

        for idx, row in combined_data.iterrows():
            if pd.isna(row["zscore"]):  # Skip rows without indicators
                continue

            # Execute trade if signal present
            if row["signal"] != 0:
                self.backtester.execute_trade(
                    row,
                    row["signal"],
                    row["position1"],
                    row["position2"],
                    symbol1,
                    symbol2,
                )

            # Record portfolio value
            current_portfolio_value = (
                self.backtester.cash + self.backtester.get_portfolio_value()
            )
            portfolio_values.append(
                {
                    "datetime": idx,
                    "portfolio_value": current_portfolio_value,
                    "cash": self.backtester.cash,
                }
            )

            # Record positions
            if self.backtester.current_trade:
                positions_data.append(
                    {
                        "datetime": idx,
                        "symbol1_position": self.backtester.current_trade.position1,
                        "symbol2_position": self.backtester.current_trade.position2,
                        "in_trade": True,
                    }
                )
            else:
                positions_data.append(
                    {
                        "datetime": idx,
                        "symbol1_position": 0,
                        "symbol2_position": 0,
                        "in_trade": False,
                    }
                )

        # Create results DataFrames
        from backtesting_framework.pairs_backtester import BacktestResults

        portfolio_df = pd.DataFrame(portfolio_values).set_index("datetime")
        positions_df = pd.DataFrame(positions_data).set_index("datetime")

        # Calculate returns
        returns = portfolio_df["portfolio_value"].pct_change().dropna()

        # Calculate metrics
        metrics = self.backtester.calculate_metrics(portfolio_df, returns)

        # Prepare pair statistics
        pair_stats = {
            "cointegration": cointegration_results,
            "correlation": data[f"{symbol1}_close"].corr(data[f"{symbol2}_close"]),
            "spread_stats": {
                "mean": data["spread"].mean(),
                "std": data["spread"].std(),
                "min": data["spread"].min(),
                "max": data["spread"].max(),
            },
        }

        # Create BacktestResults object
        backtest_result = BacktestResults(
            trades=self.backtester.trades,
            portfolio_value=portfolio_df["portfolio_value"],
            returns=returns,
            positions=positions_df,
            metrics=metrics,
            pair_stats=pair_stats,
        )

        # Generate and save plots if enabled
        should_save_plot = save_plot if save_plot is not None else self.save_plots
        plot_directory = plot_dir if plot_dir is not None else self.plots_dir

        if should_save_plot:
            try:
                from backtesting_framework.pairs_backtester import plot_backtest_results

                plot_backtest_results(
                    backtest_result,
                    symbol1,
                    symbol2,
                    save_plot=should_save_plot,
                    output_dir=plot_directory,
                )
                print(f"📊 Plot saved for {symbol1}-{symbol2}")
            except Exception as e:
                print(f"⚠️ Plotting failed for {symbol1}-{symbol2}: {e}")

        # Save detailed trade log if enabled
        if self.save_trades:
            try:
                self.save_trade_details(
                    backtest_result.trades, symbol1, symbol2, strategy, plot_directory
                )
            except Exception as e:
                print(f"⚠️ Failed to save trade details for {symbol1}-{symbol2}: {e}")

        # Calculate funding costs for trades if enabled
        if self.backtester.include_funding_costs and backtest_result.trades:
            self.calculate_funding_costs_for_trades(
                backtest_result.trades, symbol1, symbol2
            )

        return backtest_result

    def calculate_funding_costs_for_trades(self, trades, symbol1, symbol2):
        """
        Calculate funding costs for trades using the FundingRateCalculator
        """
        for trade in trades:
            if trade.is_closed and trade.entry_time and trade.exit_time:
                try:
                    # Use the backtester's funding calculator
                    if hasattr(self.backtester, "calculate_funding_costs_for_trade"):
                        updated_trade = (
                            self.backtester.calculate_funding_costs_for_trade(trade)
                        )

                        # Update the trade object with calculated costs
                        trade.funding_cost1 = getattr(
                            updated_trade, "funding_cost1", 0.0
                        )
                        trade.funding_cost2 = getattr(
                            updated_trade, "funding_cost2", 0.0
                        )
                        trade.total_funding_cost = getattr(
                            updated_trade, "total_funding_cost", 0.0
                        )
                        trade.net_pnl = getattr(
                            updated_trade, "net_pnl", trade.pnl or 0.0
                        )
                        trade.funding_payments = getattr(
                            updated_trade, "funding_payments", []
                        )

                        print(
                            f"   💰 Calculated funding costs for trade: ${trade.total_funding_cost:.2f}"
                        )

                    else:
                        # Fallback: set default values
                        trade.funding_cost1 = 0.0
                        trade.funding_cost2 = 0.0
                        trade.total_funding_cost = 0.0
                        trade.net_pnl = trade.pnl or 0.0
                        trade.funding_payments = []

                except Exception as e:
                    print(f"   ⚠️ Error calculating funding costs: {e}")
                    # Set default values on error
                    trade.funding_cost1 = 0.0
                    trade.funding_cost2 = 0.0
                    trade.total_funding_cost = 0.0
                    trade.net_pnl = trade.pnl or 0.0
                    trade.funding_payments = []

    def save_trade_details(
        self,
        trades: List,
        symbol1: str,
        symbol2: str,
        strategy: MeanReversionStrategy,
        output_dir: str = None,
    ):
        """
        Save detailed trade information to CSV file

        Args:
            trades: List of Trade objects
            symbol1: First symbol name
            symbol2: Second symbol name
            strategy: Strategy used for the backtest
            output_dir: Directory to save trade details
        """
        if not trades:
            print(f"   📝 No trades to save for {symbol1}-{symbol2}")
            return

        # Use provided directory or default
        save_dir = output_dir if output_dir else self.trades_dir
        os.makedirs(save_dir, exist_ok=True)

        # Convert trades to detailed DataFrame
        trade_details = []

        for i, trade in enumerate(trades, 1):
            # Calculate trade duration
            duration = None
            if trade.entry_time and trade.exit_time:
                duration = (
                    trade.exit_time - trade.entry_time
                ).total_seconds() / 3600  # hours

            # Calculate returns (with comprehensive None safety)
            try:
                entry_price1 = (
                    trade.entry_price1 if trade.entry_price1 is not None else 0
                )
                entry_price2 = (
                    trade.entry_price2 if trade.entry_price2 is not None else 0
                )
                entry_value = abs(entry_price1) + abs(entry_price2)

                if entry_value > 0 and trade.pnl is not None:
                    return_pct = trade.pnl / entry_value * 100
                else:
                    return_pct = 0.0
            except (TypeError, ZeroDivisionError):
                return_pct = 0.0

            try:
                trade_detail = {
                    # Trade identification
                    "trade_id": i,
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "pair": f"{symbol1}-{symbol2}",
                    # Strategy parameters
                    "lookback_period": strategy.lookback_period,
                    "entry_threshold": strategy.entry_threshold,
                    "exit_threshold": strategy.exit_threshold,
                    "stop_loss_threshold": strategy.stop_loss_threshold,
                    # Entry details
                    "entry_time": trade.entry_time,
                    "entry_price1": trade.entry_price1,
                    "entry_price2": trade.entry_price2,
                    "entry_spread": trade.entry_spread,
                    "entry_zscore": trade.entry_zscore,
                    "position1": trade.position1,  # +1 long, -1 short
                    "position2": trade.position2,  # +1 long, -1 short
                    # Exit details
                    "exit_time": trade.exit_time,
                    "exit_price1": trade.exit_price1,
                    "exit_price2": trade.exit_price2,
                    "exit_spread": trade.exit_spread,
                    "exit_zscore": trade.exit_zscore,
                    # Trade performance
                    "duration_hours": duration,
                    "pnl": trade.pnl or 0.0,
                    "return_pct": return_pct,
                    "is_profitable": (trade.pnl is not None and trade.pnl > 0),
                    "is_closed": trade.is_closed,
                    # Funding rate information
                    "funding_cost1": getattr(trade, "funding_cost1", 0.0),
                    "funding_cost2": getattr(trade, "funding_cost2", 0.0),
                    "total_funding_cost": getattr(trade, "total_funding_cost", 0.0),
                    "net_pnl": getattr(trade, "net_pnl", trade.pnl or 0.0),
                    "funding_payments_count": len(
                        getattr(trade, "funding_payments", [])
                    ),
                    "avg_funding_rate": (
                        np.mean([p["funding_rate"] for p in trade.funding_payments])
                        if hasattr(trade, "funding_payments") and trade.funding_payments
                        else 0.0
                    ),
                    "net_is_profitable": (
                        getattr(trade, "net_pnl", trade.pnl) is not None
                        and getattr(trade, "net_pnl", trade.pnl) > 0
                    ),
                    "funding_impact_pct": (
                        (
                            getattr(trade, "total_funding_cost", 0.0)
                            / abs(trade.pnl)
                            * 100
                        )
                        if trade.pnl is not None
                        and trade.pnl != 0
                        and abs(trade.pnl) > 0
                        else 0.0
                    ),
                    # Trade analysis
                    "spread_change": (
                        (trade.exit_spread - trade.entry_spread)
                        if trade.exit_spread
                        else None
                    ),
                    "zscore_change": (
                        (trade.exit_zscore - trade.entry_zscore)
                        if trade.exit_zscore
                        else None
                    ),
                    "price1_change": (
                        (trade.exit_price1 - trade.entry_price1)
                        if trade.exit_price1 is not None
                        and trade.entry_price1 is not None
                        else 0.0
                    ),
                    "price2_change": (
                        (trade.exit_price2 - trade.entry_price2)
                        if trade.exit_price2 is not None
                        and trade.entry_price2 is not None
                        else 0.0
                    ),
                    "price1_return": (
                        ((trade.exit_price1 / trade.entry_price1 - 1) * 100)
                        if (
                            trade.exit_price1 is not None
                            and trade.entry_price1 is not None
                            and trade.entry_price1 != 0
                        )
                        else 0.0
                    ),
                    "price2_return": (
                        ((trade.exit_price2 / trade.entry_price2 - 1) * 100)
                        if (
                            trade.exit_price2 is not None
                            and trade.entry_price2 is not None
                            and trade.entry_price2 != 0
                        )
                        else 0.0
                    ),
                }
                trade_details.append(trade_detail)
            except Exception as e:
                print(f"   ⚠️ Error processing trade {i}: {e}")
                # Add a minimal trade record
                trade_details.append(
                    {
                        "trade_id": i,
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                        "pair": f"{symbol1}-{symbol2}",
                        "pnl": 0.0,
                        "is_profitable": False,
                        "error": str(e),
                    }
                )

        # Create DataFrame
        trades_df = pd.DataFrame(trade_details)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trades_{symbol1}_{symbol2}_{timestamp}.csv"
        filepath = os.path.join(save_dir, filename)

        # Save to CSV
        trades_df.to_csv(filepath, index=False)

        # Print summary (with None safety)
        profitable_trades = trades_df["is_profitable"].sum()
        net_profitable_trades = trades_df["net_is_profitable"].sum()
        total_trades = len(trades_df)
        total_pnl = trades_df["pnl"].fillna(0).sum()
        total_funding_cost = trades_df["total_funding_cost"].fillna(0).sum()
        net_pnl = trades_df["net_pnl"].fillna(0).sum()
        avg_duration = trades_df["duration_hours"].fillna(0).mean()

        print(f"   📝 Trade details saved: {filepath}")
        print(f"      • Total trades: {total_trades}")
        print(
            f"      • Profitable (before funding): {profitable_trades} ({profitable_trades/total_trades:.1%})"
        )
        print(
            f"      • Profitable (after funding): {net_profitable_trades} ({net_profitable_trades/total_trades:.1%})"
        )
        print(f"      • Total PnL (before funding): ${total_pnl:.2f}")
        print(f"      • Total Funding Cost: ${total_funding_cost:.2f}")
        print(f"      • Net PnL (after funding): ${net_pnl:.2f}")
        print(f"      • Avg duration: {avg_duration:.1f} hours")
        if total_pnl is not None and abs(total_pnl) > 0:
            print(
                f"      • Funding impact: {(total_funding_cost/abs(total_pnl)*100):.1f}% of gross PnL"
            )

        return filepath

    def create_trade_summary_report(self, trades_df: pd.DataFrame) -> Dict:
        """
        Create a comprehensive trade summary report

        Args:
            trades_df: DataFrame with trade details

        Returns:
            Dictionary with summary statistics
        """
        if trades_df.empty:
            return {}

        # Basic statistics
        total_trades = len(trades_df)
        profitable_trades = trades_df["is_profitable"].sum()
        losing_trades = total_trades - profitable_trades
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0

        # PnL statistics
        total_pnl = trades_df["pnl"].sum()
        avg_pnl = trades_df["pnl"].mean()
        profitable_pnl = trades_df[trades_df["is_profitable"]]["pnl"].sum()
        losing_pnl = trades_df[~trades_df["is_profitable"]]["pnl"].sum()

        # Duration statistics
        avg_duration = trades_df["duration_hours"].mean()
        median_duration = trades_df["duration_hours"].median()
        max_duration = trades_df["duration_hours"].max()
        min_duration = trades_df["duration_hours"].min()

        # Return statistics
        avg_return = trades_df["return_pct"].mean()
        best_trade = trades_df["pnl"].max()
        worst_trade = trades_df["pnl"].min()

        # Z-score analysis
        avg_entry_zscore = abs(trades_df["entry_zscore"]).mean()
        avg_exit_zscore = abs(trades_df["exit_zscore"]).mean()

        summary = {
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": avg_pnl,
            "profitable_pnl": profitable_pnl,
            "losing_pnl": losing_pnl,
            "avg_duration_hours": avg_duration,
            "median_duration_hours": median_duration,
            "max_duration_hours": max_duration,
            "min_duration_hours": min_duration,
            "avg_return_pct": avg_return,
            "best_trade_pnl": best_trade,
            "worst_trade_pnl": worst_trade,
            "avg_entry_zscore": avg_entry_zscore,
            "avg_exit_zscore": avg_exit_zscore,
        }

        return summary

    def create_optimized_backtester(
        self,
        optimization_config: Dict,
    ) -> "MeanReversionBacktester":
        """
        Create a new backtester instance with optimized configuration.

        Args:
            optimization_config: Configuration for optimization

        Returns:
            New MeanReversionBacktester instance
        """
        return MeanReversionBacktester(
            base_path=str(self.base_path),
            results_dir=str(self.results_dir),
            initial_capital=optimization_config.get(
                "initial_capital", self.backtester.initial_capital
            ),
            transaction_cost=optimization_config.get(
                "transaction_cost", self.backtester.transaction_cost
            ),
            position_size=optimization_config.get(
                "position_size", self.backtester.position_size
            ),
            resample_timeframe=optimization_config.get(
                "resample_timeframe", self.resample_timeframe
            ),
            save_plots=optimization_config.get("save_plots", self.save_plots),
            plots_dir=optimization_config.get("plots_dir", self.plots_dir),
        )

    def optimize_backtesting_config(
        self,
        symbol1_data: pd.DataFrame,
        symbol2_data: pd.DataFrame,
        symbol1: str,
        symbol2: str,
        config_ranges: Optional[Dict] = None,
        strategy_params: Optional[Dict] = None,
    ) -> Dict:
        """
        Optimize backtesting configuration parameters (timeframe, costs, etc.).

        Args:
            symbol1_data: Data for first symbol
            symbol2_data: Data for second symbol
            symbol1: First symbol name
            symbol2: Second symbol name
            config_ranges: Ranges for backtesting configuration
            strategy_params: Fixed strategy parameters to use

        Returns:
            Dictionary with best configuration and results
        """
        from itertools import product

        # Default configuration ranges
        if config_ranges is None:
            config_ranges = {
                "resample_timeframe": [None, "5T", "15T", "1H"],
                "transaction_cost": [0.0005, 0.001, 0.002],
                "position_size": [0.3, 0.5, 0.8],
            }

        # Default strategy parameters
        if strategy_params is None:
            strategy_params = {
                "lookback_period": 60,
                "entry_threshold": 2.0,
                "exit_threshold": 0.0,
                "stop_loss_threshold": 3.0,
            }

        # Generate all combinations
        config_combinations = list(product(*config_ranges.values()))
        config_names = list(config_ranges.keys())

        best_result = None
        best_sharpe = -np.inf
        all_results = []

        print(f"🔧 Testing {len(config_combinations)} backtesting configurations...")
        print(f"📋 Fixed strategy parameters: {strategy_params}")

        for combo in tqdm(config_combinations, desc="Testing configs"):
            config = dict(zip(config_names, combo))

            try:
                # Create backtester with this configuration
                test_backtester = self.create_optimized_backtester(config)

                # Create strategy
                strategy = MeanReversionStrategy(**strategy_params)

                # Run backtest (no plots during config optimization)
                backtest_result = test_backtester.run_backtest_with_strategy(
                    symbol1_data=symbol1_data,
                    symbol2_data=symbol2_data,
                    symbol1=symbol1,
                    symbol2=symbol2,
                    strategy=strategy,
                    save_plot=False,  # Disable plotting during optimization for speed
                )

                # Extract metrics
                metrics = backtest_result.metrics
                sharpe_ratio = metrics.get("Sharpe Ratio", -np.inf)

                result = {
                    **config,
                    "sharpe_ratio": sharpe_ratio,
                    "total_return": metrics.get("Total Return", 0),
                    "max_drawdown": metrics.get("Max Drawdown", 0),
                    "win_rate": metrics.get("Win Rate", 0),
                    "num_trades": len(backtest_result.trades),
                }

                all_results.append(result)

                # Track best result
                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_result = result

            except Exception as e:
                print(f"   ⚠️ Config failed: {config} - {e}")
                continue

        print(f"✅ Configuration optimization completed!")
        if best_result:
            print(f"   • Best Sharpe ratio: {best_sharpe:.4f}")
            print(
                f"   • Best config: {dict((k, v) for k, v in best_result.items() if k in config_names)}"
            )

        return {
            "best_config": best_result,
            "all_results": all_results,
            "strategy_params": strategy_params,
        }

    def optimize_strategy_params(
        self,
        symbol1_data: pd.DataFrame,
        symbol2_data: pd.DataFrame,
        symbol1: str,
        symbol2: str,
        param_ranges: Optional[Dict] = None,
        optimization_metric: str = "sharpe_ratio",
        max_combinations: Optional[int] = None,
    ) -> Dict:
        """
        Optimize strategy parameters for a specific pair.

        Args:
            symbol1_data: Data for first symbol
            symbol2_data: Data for second symbol
            symbol1: First symbol name
            symbol2: Second symbol name
            param_ranges: Parameter ranges for optimization
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', 'calmar_ratio')
            max_combinations: Maximum number of combinations to test (None for all)

        Returns:
            Dictionary with best parameters and results
        """
        from itertools import product

        # Default parameter ranges
        if param_ranges is None:
            param_ranges = {
                "lookback_period": [30, 60, 90, 120],
                "entry_threshold": [1.5, 2.0, 2.5],
                "exit_threshold": [0.0, 0.5],
                "stop_loss_threshold": [2.5, 3.0, 3.5],
            }

        # Generate all combinations
        param_combinations = list(product(*param_ranges.values()))
        param_names = list(param_ranges.keys())

        # Limit combinations if specified
        if max_combinations and len(param_combinations) > max_combinations:
            import random

            random.seed(42)  # For reproducibility
            param_combinations = random.sample(param_combinations, max_combinations)
            print(
                f"⚠️ Limited to {max_combinations} random combinations out of {len(list(product(*param_ranges.values())))}"
            )

        best_result = None
        best_score = -np.inf
        all_results = []
        failed_combinations = 0

        print(f"🔍 Testing {len(param_combinations)} parameter combinations...")
        print(f"📊 Optimization metric: {optimization_metric}")
        print(f"⚙️ Backtesting configuration:")
        print(f"   • Timeframe: {self.resample_timeframe or '1T (1-minute)'}")
        print(f"   • Initial capital: ${self.backtester.initial_capital:,.0f}")
        print(f"   • Transaction cost: {self.backtester.transaction_cost:.4f}")
        print(f"   • Position size: {self.backtester.position_size:.2f}")

        for i, combo in enumerate(tqdm(param_combinations, desc="Optimizing"), 1):
            params = dict(zip(param_names, combo))

            try:
                # Create strategy with current parameters
                strategy = MeanReversionStrategy(
                    lookback_period=params["lookback_period"],
                    entry_threshold=params["entry_threshold"],
                    exit_threshold=params.get("exit_threshold", 0.0),
                    stop_loss_threshold=params["stop_loss_threshold"],
                )

                # Run backtest with all current backtester settings (no plots during optimization)
                backtest_result = self.run_backtest_with_strategy(
                    symbol1_data=symbol1_data,
                    symbol2_data=symbol2_data,
                    symbol1=symbol1,
                    symbol2=symbol2,
                    strategy=strategy,
                    save_plot=False,  # Disable plotting during optimization for speed
                )

                # Extract all available metrics
                metrics = backtest_result.metrics
                sharpe_ratio = metrics.get("Sharpe Ratio", -np.inf)
                total_return = metrics.get("Total Return", 0)
                max_drawdown = metrics.get("Max Drawdown", 0)
                win_rate = metrics.get("Win Rate", 0)
                volatility = metrics.get("Volatility", 0)

                # Calculate additional metrics
                calmar_ratio = (
                    total_return / abs(max_drawdown) if max_drawdown != 0 else 0
                )

                result = {
                    **params,
                    "sharpe_ratio": sharpe_ratio,
                    "total_return": total_return,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate,
                    "volatility": volatility,
                    "calmar_ratio": calmar_ratio,
                    "num_trades": len(backtest_result.trades),
                    "avg_trade_pnl": (
                        np.mean(
                            [t.pnl for t in backtest_result.trades if t.pnl is not None]
                        )
                        if backtest_result.trades
                        else 0
                    ),
                }

                all_results.append(result)

                # Determine optimization score based on selected metric
                if optimization_metric == "sharpe_ratio":
                    score = sharpe_ratio
                elif optimization_metric == "total_return":
                    score = total_return
                elif optimization_metric == "calmar_ratio":
                    score = calmar_ratio
                elif optimization_metric == "win_rate":
                    score = win_rate
                else:
                    score = sharpe_ratio  # Default fallback

                # Track best result
                if score > best_score and not np.isnan(score) and not np.isinf(score):
                    best_score = score
                    best_result = result

                # Progress update every 25% of combinations
                if i % max(1, len(param_combinations) // 4) == 0:
                    progress = (i / len(param_combinations)) * 100
                    print(
                        f"   Progress: {progress:.0f}% | Best {optimization_metric}: {best_score:.4f}"
                    )

            except Exception as e:
                failed_combinations += 1
                if failed_combinations <= 5:  # Only show first few errors
                    print(f"   ⚠️ Failed combination {i}: {e}")
                continue

        # Summary
        successful_combinations = len(all_results)
        print(f"\n✅ Optimization completed:")
        print(f"   • Successful: {successful_combinations}/{len(param_combinations)}")
        print(f"   • Failed: {failed_combinations}")

        if best_result:
            print(f"   • Best {optimization_metric}: {best_score:.4f}")
            print(
                f"   • Best parameters: {dict((k, v) for k, v in best_result.items() if k in param_names)}"
            )
        else:
            print(f"   • No successful combinations found!")

        return {
            "best_params": best_result,
            "all_results": all_results,
            "optimization_metric": optimization_metric,
            "successful_combinations": successful_combinations,
            "failed_combinations": failed_combinations,
        }

    def run_backtest_for_pair(
        self,
        pair_info: Dict,
        test_year: int = 2024,
        test_months: List[int] = [4, 5, 6],
        optimize_params: bool = True,
    ) -> Dict:
        """
        Run backtest for a single pair.

        Args:
            pair_info: Dictionary with pair information from cointegration results
            test_year: Year for backtesting (out-of-sample)
            test_months: Months for backtesting
            optimize_params: Whether to optimize strategy parameters

        Returns:
            Dictionary with backtest results
        """
        symbol1 = pair_info["symbol1"]
        symbol2 = pair_info["symbol2"]

        print(f"\n{'='*60}")
        print(f"Running backtest for {symbol1} - {symbol2}")
        print(f"Cointegration p-value: {pair_info['p_value']:.6f}")
        print(f"Hedge ratio: {pair_info['hedge_ratio']:.4f}")
        print(f"Correlation: {pair_info['correlation']:.4f}")

        try:
            # Load data
            print(f"Loading data for {test_year}, months {test_months}...")
            df1, df2 = self.load_pair_data(symbol1, symbol2, test_year, test_months)

            # Determine strategy parameters
            if optimize_params:
                print("🔍 Optimizing strategy parameters...")

                # Define optimization parameters based on data size
                data_size = len(df1)
                if data_size > 50000:  # Large dataset
                    max_combinations = 50  # Limit for speed
                    param_ranges = {
                        "lookback_period": [30, 60, 90],
                        "entry_threshold": [1.5, 2.0, 2.5],
                        "exit_threshold": [0.0, 0.5],
                        "stop_loss_threshold": [2.5, 3.0, 3.5],
                    }
                elif data_size > 20000:  # Medium dataset
                    max_combinations = 100
                    param_ranges = {
                        "lookback_period": [30, 60, 90, 120],
                        "entry_threshold": [1.5, 2.0, 2.5],
                        "exit_threshold": [0.0, 0.5],
                        "stop_loss_threshold": [2.5, 3.0, 3.5],
                    }
                else:  # Small dataset
                    max_combinations = None  # Test all
                    param_ranges = {
                        "lookback_period": [20, 40, 60, 80],
                        "entry_threshold": [1.0, 1.5, 2.0, 2.5],
                        "exit_threshold": [0.0, 0.5, 1.0],
                        "stop_loss_threshold": [2.0, 2.5, 3.0],
                    }

                optimization_result = self.optimize_strategy_params(
                    df1,
                    df2,
                    symbol1,
                    symbol2,
                    param_ranges=param_ranges,
                    optimization_metric="sharpe_ratio",
                    max_combinations=max_combinations,
                )

                if optimization_result["best_params"]:
                    strategy_params = optimization_result["best_params"]
                    print(f"✅ Best parameters found:")
                    print(f"   • Lookback: {strategy_params['lookback_period']}")
                    print(
                        f"   • Entry threshold: {strategy_params['entry_threshold']:.2f}"
                    )
                    print(
                        f"   • Exit threshold: {strategy_params['exit_threshold']:.2f}"
                    )
                    print(
                        f"   • Stop loss: {strategy_params['stop_loss_threshold']:.2f}"
                    )
                    print(
                        f"   • Expected Sharpe: {strategy_params['sharpe_ratio']:.3f}"
                    )
                else:
                    print("⚠️ Optimization failed, using default parameters")
                    # Use default parameters
                    strategy_params = {
                        "lookback_period": 60,
                        "entry_threshold": 2.0,
                        "exit_threshold": 0.0,
                        "stop_loss_threshold": 3.0,
                    }
            else:
                print("📋 Using default strategy parameters")
                # Use default or provided parameters
                strategy_params = {
                    "lookback_period": 60,
                    "entry_threshold": 2.0,
                    "exit_threshold": 0.0,
                    "stop_loss_threshold": 3.0,
                }

            # Create strategy
            strategy = MeanReversionStrategy(
                lookback_period=strategy_params["lookback_period"],
                entry_threshold=strategy_params["entry_threshold"],
                exit_threshold=strategy_params.get("exit_threshold", 0.0),
                stop_loss_threshold=strategy_params["stop_loss_threshold"],
            )

            # Run final backtest
            print("Running final backtest...")
            backtest_result = self.run_backtest_with_strategy(
                symbol1_data=df1,
                symbol2_data=df2,
                symbol1=symbol1,
                symbol2=symbol2,
                strategy=strategy,
            )

            # Compile results
            result = {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "cointegration_p_value": pair_info["p_value"],
                "hedge_ratio": pair_info["hedge_ratio"],
                "correlation": pair_info["correlation"],
                "strategy_params": strategy_params,
                "metrics": backtest_result.metrics,
                "num_trades": len(backtest_result.trades),
                "success": True,
            }

            # Print summary
            print(f"\n📊 Backtest Results:")
            print(
                f"  Total Return: {backtest_result.metrics.get('Total Return', 0):.2%}"
            )
            print(
                f"  Sharpe Ratio: {backtest_result.metrics.get('Sharpe Ratio', 0):.3f}"
            )
            print(
                f"  Max Drawdown: {backtest_result.metrics.get('Max Drawdown', 0):.2%}"
            )
            print(f"  Win Rate: {backtest_result.metrics.get('Win Rate', 0):.2%}")
            print(f"  Number of Trades: {len(backtest_result.trades)}")

            # Plot results (optional)
            try:
                plot_backtest_results(
                    backtest_result,
                    symbol1,
                    symbol2,
                    save_plot=self.save_plots,
                    output_dir=self.plots_dir,
                )
            except Exception as e:
                print(f"⚠️ Plotting failed: {e}")  # More informative error

            return result

        except Exception as e:
            print(f"❌ Error in backtest: {e}")
            return {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "success": False,
                "error": str(e),
            }

    def run_all_backtests(
        self,
        n_pairs: int = 10,
        test_year: int = 2024,
        test_months: List[int] = [4, 5, 6],
        optimize_params: bool = True,
        save_results: bool = True,
    ) -> pd.DataFrame:
        """
        Run backtests for top cointegrated pairs.

        Args:
            n_pairs: Number of top pairs to backtest
            test_year: Year for backtesting
            test_months: Months for backtesting
            optimize_params: Whether to optimize parameters for each pair
            save_results: Whether to save results to file

        Returns:
            DataFrame with all backtest results
        """
        print("=" * 80)
        print("MEAN REVERSION BACKTESTING")
        print("=" * 80)
        print(f"🔧 Backtesting Configuration:")
        print(f"   • Timeframe: {self.resample_timeframe or '1T (1-minute)'}")
        print(f"   • Initial Capital: ${self.backtester.initial_capital:,.0f}")
        print(f"   • Transaction Cost: {self.backtester.transaction_cost:.4f}")
        print(f"   • Position Size: {self.backtester.position_size:.2f}")
        print(f"   • Save Plots: {self.save_plots}")
        if self.save_plots:
            print(f"   • Plots Directory: {self.plots_dir}")

        # Load cointegration results
        print("\n1. Loading cointegration results...")
        coint_results = self.load_cointegration_results()

        # Filter top pairs
        print(f"\n2. Filtering top {n_pairs} pairs...")
        top_pairs = self.filter_top_pairs(
            coint_results,
            n_pairs=n_pairs,
            max_p_value=0.08,
            min_correlation=0.6,
            max_half_life=90,
        )

        if not top_pairs:
            print("No suitable pairs found!")
            return pd.DataFrame()

        print(f"Found {len(top_pairs)} pairs meeting criteria:")
        for i, pair in enumerate(top_pairs, 1):
            print(
                f"  {i}. {pair['symbol1']:10s} - {pair['symbol2']:10s} | "
                f"p-value: {pair['p_value']:.6f} | "
                f"score: {pair['composite_score']:.4f}"
            )

        # Run backtests
        print(f"\n3. Running backtests on {test_year} months {test_months}...")
        print(f"   (Out-of-sample testing - cointegration was on months 1-3)")

        all_results = []
        for i, pair in enumerate(top_pairs, 1):
            print(f"\n[{i}/{len(top_pairs)}] ", end="")
            result = self.run_backtest_for_pair(
                pair,
                test_year=test_year,
                test_months=test_months,
                optimize_params=optimize_params,
            )
            all_results.append(result)

        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)

        # Filter successful backtests
        successful = results_df[results_df["success"] == True].copy()

        if len(successful) > 0:
            # Extract metrics for sorting
            for idx, row in successful.iterrows():
                if isinstance(row["metrics"], dict):
                    successful.loc[idx, "total_return"] = row["metrics"].get(
                        "Total Return", 0
                    )
                    successful.loc[idx, "sharpe_ratio"] = row["metrics"].get(
                        "Sharpe Ratio", 0
                    )
                    successful.loc[idx, "max_drawdown"] = row["metrics"].get(
                        "Max Drawdown", 0
                    )
                    successful.loc[idx, "win_rate"] = row["metrics"].get("Win Rate", 0)

            # Sort by Sharpe ratio
            successful = successful.sort_values("sharpe_ratio", ascending=False)

            # Print summary
            print("\n" + "=" * 80)
            print("BACKTEST SUMMARY")
            print("=" * 80)
            print(f"Successful backtests: {len(successful)}/{len(top_pairs)}")

            if len(successful) > 0:
                print("\nTop 5 Performing Pairs:")
                for i, (_, row) in enumerate(successful.head().iterrows(), 1):
                    print(f"\n{i}. {row['symbol1']} - {row['symbol2']}")
                    print(f"   Total Return: {row.get('total_return', 0):.2%}")
                    print(f"   Sharpe Ratio: {row.get('sharpe_ratio', 0):.3f}")
                    print(f"   Max Drawdown: {row.get('max_drawdown', 0):.2%}")
                    print(f"   Win Rate: {row.get('win_rate', 0):.2%}")
                    print(f"   Trades: {row.get('num_trades', 0)}")

                # Overall statistics
                print("\nOverall Statistics:")
                print(
                    f"  Average Sharpe Ratio: {successful['sharpe_ratio'].mean():.3f}"
                )
                print(
                    f"  Average Total Return: {successful['total_return'].mean():.2%}"
                )
                print(f"  Average Win Rate: {successful['win_rate'].mean():.2%}")
                print(
                    f"  Total Profitable Pairs: {len(successful[successful['total_return'] > 0])}"
                )

        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"backtest_results_{timestamp}.csv"
            results_df.to_csv(output_file, index=False)
            print(f"\n💾 Results saved to {output_file}")

        return results_df


def main():
    """Main function to run mean reversion backtesting."""

    # Initialize backtester with configuration options
    backtester = MeanReversionBacktester(
        base_path="binance_futures_data",
        results_dir="cointegration_results",
        initial_capital=100000.0,
        transaction_cost=0.0005,
        position_size=0.5,
        resample_timeframe="1H",  # Use 5-minute bars instead of 1-minute
        save_plots=True,  # Save plot images
        plots_dir="backtest_plots",  # Directory for saved plots
    )

    # Run backtests
    # Using April-June 2024 for out-of-sample testing
    # (Cointegration was calculated on Jan-March 2024)
    results = backtester.run_all_backtests(
        n_pairs=10,  # Test top 10 pairs
        test_year=2024,
        test_months=[4, 5, 6, 7, 8, 9, 10],  # Out-of-sample months
        optimize_params=True,  # Optimize strategy parameters
        save_results=True,  # Save results to CSV
    )

    # Additional analysis
    if len(results) > 0:
        successful = results[results["success"] == True]
        if len(successful) > 0:
            print("\n" + "=" * 80)
            print("FINAL RECOMMENDATIONS")
            print("=" * 80)

            # Find consistently profitable pairs
            profitable = successful[successful["total_return"] > 0.05]  # >5% return
            stable = successful[successful["sharpe_ratio"] > 1.0]  # Sharpe > 1

            recommended = profitable[profitable.index.isin(stable.index)]

            if len(recommended) > 0:
                print(
                    f"\n✅ Recommended pairs for live trading ({len(recommended)} pairs):"
                )
                for _, row in recommended.iterrows():
                    print(f"  • {row['symbol1']} - {row['symbol2']}")
                    print(
                        f"    Strategy: Lookback={row['strategy_params']['lookback_period']}, "
                        f"Entry={row['strategy_params']['entry_threshold']}, "
                        f"StopLoss={row['strategy_params']['stop_loss_threshold']}"
                    )
            else:
                print("\n⚠️ No pairs meet the recommended criteria for live trading")
                print(
                    "  Consider adjusting parameters or waiting for better market conditions"
                )

    return results


if __name__ == "__main__":
    results = main()
