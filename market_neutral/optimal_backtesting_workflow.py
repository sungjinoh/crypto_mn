#!/usr/bin/env python3
"""
Optimal Backtesting Workflow
This script implements the recommended approach for parameter optimization and backtesting.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mean_reversion_backtest import MeanReversionBacktester
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class OptimalBacktestingWorkflow:
    """
    Implements the optimal workflow for pairs trading backtesting:
    1. Global parameter screening
    2. Per-pair optimization
    3. Validation and selection
    """

    def __init__(
        self,
        base_path: str = "binance_futures_data",
        results_dir: str = "cointegration_results",
    ):
        self.base_path = base_path
        self.results_dir = results_dir
        self.global_config = None
        self.pair_results = []

    def step1_global_screening(self, n_sample_pairs: int = 3) -> dict:
        """
        Step 1: Global parameter screening to find good parameter ranges
        Tests a few representative pairs to understand what works generally.
        """
        print("=" * 80)
        print("STEP 1: GLOBAL PARAMETER SCREENING")
        print("=" * 80)
        print(
            f"🔍 Testing {n_sample_pairs} representative pairs to find good parameter ranges"
        )

        # Initialize backtester for screening
        backtester = MeanReversionBacktester(
            base_path=self.base_path,
            results_dir=self.results_dir,
            save_plots=False,  # No plots during screening
        )

        # Load cointegration results
        coint_results = backtester.load_cointegration_results()
        sample_pairs = backtester.filter_top_pairs(
            coint_results, n_pairs=n_sample_pairs
        )

        if not sample_pairs:
            raise ValueError("No suitable pairs found for screening")

        print(f"📊 Sample pairs for screening:")
        for i, pair in enumerate(sample_pairs, 1):
            print(
                f"   {i}. {pair['symbol1']} - {pair['symbol2']} (p-value: {pair['p_value']:.6f})"
            )

        # Test different configurations first
        print(f"\n🔧 Testing backtesting configurations...")
        config_ranges = {
            "transaction_cost": [0.001, 0.002],
            "position_size": [0.5, 0.8],
        }

        config_results = []
        for pair in sample_pairs[:2]:  # Test config on first 2 pairs
            try:
                symbol1, symbol2 = pair["symbol1"], pair["symbol2"]
                df1, df2 = backtester.load_pair_data(symbol1, symbol2, [2024], [4, 5])

                config_result = backtester.optimize_backtesting_config(
                    df1, df2, symbol1, symbol2, config_ranges=config_ranges
                )

                if config_result["best_config"]:
                    config_results.append(config_result["best_config"])

            except Exception as e:
                print(f"   ⚠️ Config test failed for {symbol1}-{symbol2}: {e}")
                continue

        # Find most common best configuration
        if config_results:
            # Simple voting for best config
            costs = [c["transaction_cost"] for c in config_results]
            sizes = [c["position_size"] for c in config_results]

            best_global_config = {
                "transaction_cost": np.median(costs),
                "position_size": np.median(sizes),
            }
        else:
            # Fallback configuration
            best_global_config = {
                "transaction_cost": 0.001,
                "position_size": 0.5,
            }

        print(f"✅ Best global configuration:")
        print(f"   • Transaction Cost: {best_global_config['transaction_cost']:.4f}")
        print(f"   • Position Size: {best_global_config['position_size']:.2f}")

        # Now test strategy parameters with best config
        print(f"\n⚙️ Testing strategy parameter ranges...")
        optimized_backtester = MeanReversionBacktester(
            base_path=self.base_path,
            results_dir=self.results_dir,
            transaction_cost=best_global_config["transaction_cost"],
            position_size=best_global_config["position_size"],
            save_plots=False,
        )

        # Test broad parameter ranges
        broad_param_ranges = {
            "lookback_period": [30, 60, 90, 120],
            "entry_threshold": [1.5, 2.0, 2.5, 3.0],
            "exit_threshold": [0.0, 0.5, 1.0],
            "stop_loss_threshold": [2.5, 3.0, 3.5, 4.0],
        }

        all_param_results = []
        for pair in sample_pairs:
            try:
                symbol1, symbol2 = pair["symbol1"], pair["symbol2"]
                df1, df2 = optimized_backtester.load_pair_data(
                    symbol1, symbol2, [2024], [4, 5]
                )

                param_result = optimized_backtester.optimize_strategy_params(
                    df1,
                    df2,
                    symbol1,
                    symbol2,
                    param_ranges=broad_param_ranges,
                    max_combinations=50,  # Limit for screening
                )

                if param_result["best_params"]:
                    all_param_results.extend(param_result["all_results"])

            except Exception as e:
                print(f"   ⚠️ Parameter test failed for {symbol1}-{symbol2}: {e}")
                continue

        # Analyze parameter distributions
        if all_param_results:
            # Get top 20% of results
            sorted_results = sorted(
                all_param_results, key=lambda x: x["sharpe_ratio"], reverse=True
            )
            top_results = sorted_results[: max(1, len(sorted_results) // 5)]

            # Find parameter ranges that work well
            good_ranges = {
                "lookback_period": [
                    int(np.percentile([r["lookback_period"] for r in top_results], 25)),
                    int(np.percentile([r["lookback_period"] for r in top_results], 75)),
                ],
                "entry_threshold": [
                    round(
                        np.percentile([r["entry_threshold"] for r in top_results], 25),
                        1,
                    ),
                    round(
                        np.percentile([r["entry_threshold"] for r in top_results], 75),
                        1,
                    ),
                ],
                "exit_threshold": [
                    round(
                        np.percentile([r["exit_threshold"] for r in top_results], 25), 1
                    ),
                    round(
                        np.percentile([r["exit_threshold"] for r in top_results], 75), 1
                    ),
                ],
                "stop_loss_threshold": [
                    round(
                        np.percentile(
                            [r["stop_loss_threshold"] for r in top_results], 25
                        ),
                        1,
                    ),
                    round(
                        np.percentile(
                            [r["stop_loss_threshold"] for r in top_results], 75
                        ),
                        1,
                    ),
                ],
            }
        else:
            # Fallback ranges
            good_ranges = {
                "lookback_period": [45, 90],
                "entry_threshold": [1.5, 2.5],
                "exit_threshold": [0.0, 0.5],
                "stop_loss_threshold": [2.5, 3.5],
            }

        print(f"✅ Good parameter ranges identified:")
        for param, (low, high) in good_ranges.items():
            print(f"   • {param}: {low} - {high}")

        self.global_config = {
            "backtesting_config": best_global_config,
            "parameter_ranges": good_ranges,
            "sample_results": all_param_results,
        }

        return self.global_config

    def step2_per_pair_optimization(self, n_pairs: int = 10) -> list:
        """
        Step 2: Per-pair optimization using insights from global screening
        Each pair gets its own optimal parameters within the good ranges.
        """
        print(f"\n{'=' * 80}")
        print("STEP 2: PER-PAIR OPTIMIZATION")
        print("=" * 80)
        print(f"🎯 Optimizing parameters individually for top {n_pairs} pairs")

        if not self.global_config:
            raise ValueError("Must run step1_global_screening first!")

        # Create optimized backtester with global config
        backtester = MeanReversionBacktester(
            base_path=self.base_path,
            results_dir=self.results_dir,
            **self.global_config["backtesting_config"],
            save_plots=True,  # Enable plots for final results
            plots_dir="optimized_pairs_plots",
        )

        # Load all pairs
        coint_results = backtester.load_cointegration_results()
        top_pairs = backtester.filter_top_pairs(coint_results, n_pairs=n_pairs)

        print(f"📊 Optimizing {len(top_pairs)} pairs:")
        for i, pair in enumerate(top_pairs, 1):
            print(f"   {i}. {pair['symbol1']} - {pair['symbol2']}")

        # Create focused parameter ranges from global screening
        param_ranges = {}
        for param, (low, high) in self.global_config["parameter_ranges"].items():
            if param == "lookback_period":
                param_ranges[param] = list(range(low, high + 15, 15))  # Steps of 15
            elif param in ["entry_threshold", "exit_threshold", "stop_loss_threshold"]:
                param_ranges[param] = [
                    low + i * 0.25 for i in range(int((high - low) / 0.25) + 1)
                ]

        print(f"\n🔧 Focused parameter ranges:")
        for param, values in param_ranges.items():
            print(f"   • {param}: {values}")

        # Optimize each pair individually
        pair_results = []
        for i, pair in enumerate(top_pairs, 1):
            print(
                f"\n[{i}/{len(top_pairs)}] Optimizing {pair['symbol1']} - {pair['symbol2']}"
            )

            try:
                # Load data
                symbol1, symbol2 = pair["symbol1"], pair["symbol2"]
                df1, df2 = backtester.load_pair_data(
                    symbol1, symbol2, [2024], [4, 5, 6]
                )

                # Optimize parameters for this specific pair
                optimization_result = backtester.optimize_strategy_params(
                    df1,
                    df2,
                    symbol1,
                    symbol2,
                    param_ranges=param_ranges,
                    optimization_metric="sharpe_ratio",
                    max_combinations=None,  # Test all combinations in focused range
                )

                if optimization_result["best_params"]:
                    best_params = optimization_result["best_params"]

                    # Run final backtest with optimal parameters
                    from mean_reversion_strategy import MeanReversionStrategy

                    strategy = MeanReversionStrategy(
                        lookback_period=best_params["lookback_period"],
                        entry_threshold=best_params["entry_threshold"],
                        exit_threshold=best_params["exit_threshold"],
                        stop_loss_threshold=best_params["stop_loss_threshold"],
                    )

                    final_result = backtester.run_backtest_with_strategy(
                        df1, df2, symbol1, symbol2, strategy
                    )

                    # Store comprehensive results
                    pair_result = {
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                        "cointegration_p_value": pair["p_value"],
                        "correlation": pair["correlation"],
                        "optimal_params": best_params,
                        "performance": {
                            "sharpe_ratio": final_result.metrics.get("Sharpe Ratio", 0),
                            "total_return": final_result.metrics.get("Total Return", 0),
                            "max_drawdown": final_result.metrics.get("Max Drawdown", 0),
                            "win_rate": final_result.metrics.get("Win Rate", 0),
                            "num_trades": len(final_result.trades),
                        },
                        "backtest_result": final_result,
                        "optimization_details": optimization_result,
                    }

                    pair_results.append(pair_result)

                    print(
                        f"   ✅ Optimized - Sharpe: {best_params['sharpe_ratio']:.3f}"
                    )
                else:
                    print(f"   ❌ Optimization failed")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue

        self.pair_results = pair_results
        print(f"\n✅ Per-pair optimization completed: {len(pair_results)} successful")

        return pair_results

    def step3_validation_and_selection(
        self, min_sharpe: float = 1.0, min_return: float = 0.05
    ) -> list:
        """
        Step 3: Validation and selection of best pairs
        Applies filters and selects pairs suitable for live trading.
        """
        print(f"\n{'=' * 80}")
        print("STEP 3: VALIDATION AND SELECTION")
        print("=" * 80)
        print(
            f"🔍 Applying filters: Sharpe ≥ {min_sharpe:.1f}, Return ≥ {min_return:.1%}"
        )

        if not self.pair_results:
            raise ValueError("Must run step2_per_pair_optimization first!")

        # Apply filters
        filtered_pairs = []
        for pair in self.pair_results:
            perf = pair["performance"]

            # Basic performance filters
            if (
                perf["sharpe_ratio"] >= min_sharpe
                and perf["total_return"] >= min_return
                and perf["num_trades"] >= 5
            ):  # Minimum activity

                filtered_pairs.append(pair)

        print(f"📊 Filtering results:")
        print(f"   • Total pairs tested: {len(self.pair_results)}")
        print(f"   • Pairs passing filters: {len(filtered_pairs)}")

        if not filtered_pairs:
            print("⚠️ No pairs meet the criteria. Consider relaxing filters.")
            return []

        # Sort by Sharpe ratio
        filtered_pairs.sort(
            key=lambda x: x["performance"]["sharpe_ratio"], reverse=True
        )

        # Display results
        print(f"\n🏆 SELECTED PAIRS FOR LIVE TRADING:")
        print(
            f"{'Rank':<5} {'Pair':<20} {'Sharpe':<8} {'Return':<8} {'Drawdown':<10} {'Trades':<7}"
        )
        print("-" * 70)

        for i, pair in enumerate(filtered_pairs, 1):
            perf = pair["performance"]
            pair_name = f"{pair['symbol1']}-{pair['symbol2']}"
            print(
                f"{i:<5} {pair_name:<20} {perf['sharpe_ratio']:<8.3f} "
                f"{perf['total_return']:<8.2%} {perf['max_drawdown']:<10.2%} {perf['num_trades']:<7}"
            )

        # Show optimal parameters for top pairs
        print(f"\n⚙️ OPTIMAL PARAMETERS FOR TOP 5 PAIRS:")
        for i, pair in enumerate(filtered_pairs[:5], 1):
            params = pair["optimal_params"]
            print(f"\n{i}. {pair['symbol1']}-{pair['symbol2']}:")
            print(f"   • Lookback: {params['lookback_period']}")
            print(f"   • Entry: {params['entry_threshold']:.2f}")
            print(f"   • Exit: {params['exit_threshold']:.2f}")
            print(f"   • Stop: {params['stop_loss_threshold']:.2f}")

        # Save results
        self.save_final_results(filtered_pairs)

        return filtered_pairs

    def save_final_results(self, selected_pairs: list):
        """Save final results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary CSV
        summary_data = []
        for pair in selected_pairs:
            perf = pair["performance"]
            params = pair["optimal_params"]

            summary_data.append(
                {
                    "symbol1": pair["symbol1"],
                    "symbol2": pair["symbol2"],
                    "sharpe_ratio": perf["sharpe_ratio"],
                    "total_return": perf["total_return"],
                    "max_drawdown": perf["max_drawdown"],
                    "win_rate": perf["win_rate"],
                    "num_trades": perf["num_trades"],
                    "lookback_period": params["lookback_period"],
                    "entry_threshold": params["entry_threshold"],
                    "exit_threshold": params["exit_threshold"],
                    "stop_loss_threshold": params["stop_loss_threshold"],
                    "cointegration_p_value": pair["cointegration_p_value"],
                    "correlation": pair["correlation"],
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_file = f"optimal_pairs_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)

        print(f"\n💾 Results saved:")
        print(f"   • Summary: {summary_file}")
        print(f"   • Plots: optimized_pairs_plots/")

    def run_complete_workflow(
        self,
        n_sample_pairs: int = 3,
        n_final_pairs: int = 10,
        min_sharpe: float = 1.0,
        min_return: float = 0.05,
    ) -> list:
        """
        Run the complete optimal workflow
        """
        print("🚀 STARTING OPTIMAL BACKTESTING WORKFLOW")
        print(f"   • Sample pairs for screening: {n_sample_pairs}")
        print(f"   • Final pairs to optimize: {n_final_pairs}")
        print(f"   • Minimum Sharpe ratio: {min_sharpe}")
        print(f"   • Minimum return: {min_return:.1%}")

        # Step 1: Global screening
        global_config = self.step1_global_screening(n_sample_pairs)

        # Step 2: Per-pair optimization
        pair_results = self.step2_per_pair_optimization(n_final_pairs)

        # Step 3: Validation and selection
        selected_pairs = self.step3_validation_and_selection(min_sharpe, min_return)

        print(f"\n🎉 WORKFLOW COMPLETED!")
        print(f"   • {len(selected_pairs)} pairs selected for live trading")
        print(f"   • Global configuration optimized")
        print(f"   • Individual parameters optimized per pair")
        print(f"   • Results validated and filtered")

        return selected_pairs


def main():
    """Run the optimal backtesting workflow"""

    workflow = OptimalBacktestingWorkflow()

    # Run complete workflow
    selected_pairs = workflow.run_complete_workflow(
        n_sample_pairs=3,  # Test 3 pairs for global screening
        n_final_pairs=10,  # Optimize top 10 pairs individually
        min_sharpe=0.8,  # Minimum Sharpe ratio (relaxed for demo)
        min_return=0.03,  # Minimum 3% return (relaxed for demo)
    )

    if selected_pairs:
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Review the saved results and plots")
        print(f"   2. Consider paper trading the top pairs")
        print(f"   3. Monitor performance and adjust parameters as needed")
        print(f"   4. Implement risk management and position sizing")
    else:
        print(f"\n⚠️ No pairs met the selection criteria.")
        print(f"   Consider relaxing the filters or improving the strategy.")


if __name__ == "__main__":
    main()
