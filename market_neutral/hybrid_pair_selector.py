"""
Hybrid Cointegration Filtering System
Combines statistical filtering with targeted backtesting for optimal pair selection
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apply_optimal_filters import CointegrationFilter
from market_neutral.run_fixed_parameters import run_fixed_parameters_backtest


class HybridPairSelector:
    """
    Hybrid approach: Statistical filtering → Targeted backtesting → Final selection
    """

    def __init__(self):
        self.cointegration_results = None
        self.statistical_filtered = None
        self.backtest_results = None
        self.final_selection = None

    def step1_statistical_filtering(
        self,
        cointegration_file: str,
        initial_filter_strategy: str = "aggressive",
        target_pairs_for_backtest: int = 100,
    ) -> List[Dict]:
        """
        Step 1: Apply statistical filters to reduce candidate pairs

        Args:
            cointegration_file: Path to cointegration results
            initial_filter_strategy: 'aggressive' to get more pairs for testing
            target_pairs_for_backtest: How many pairs to pass to backtesting

        Returns:
            List of statistically filtered pairs
        """
        print("\n" + "=" * 80)
        print("📊 STEP 1: STATISTICAL FILTERING")
        print("=" * 80)
        print(
            f"Goal: Reduce pairs to top {target_pairs_for_backtest} candidates for backtesting"
        )

        # Load cointegration results
        filter = CointegrationFilter(initial_filter_strategy)
        self.cointegration_results = filter.load_cointegration_results(
            cointegration_file
        )

        print(f"Loaded {len(self.cointegration_results)} total pairs")

        # Calculate quality scores for all pairs
        for pair in self.cointegration_results:
            pair["quality_score"] = filter.calculate_quality_score(pair)

        # Sort by quality score
        sorted_pairs = sorted(
            self.cointegration_results,
            key=lambda x: x.get("quality_score", 0),
            reverse=True,
        )

        # Apply loose filters first
        pre_filtered = []
        for pair in sorted_pairs:
            if not pair.get("is_cointegrated", False):
                continue
            if pair.get("p_value", 1.0) > 0.1:  # Very loose p-value
                continue
            if abs(pair.get("correlation", 0)) < 0.5:  # Very loose correlation
                continue
            pre_filtered.append(pair)

        print(f"After basic filters: {len(pre_filtered)} pairs")

        # Take top N by quality score
        self.statistical_filtered = pre_filtered[:target_pairs_for_backtest]

        print(
            f"\n✅ Selected top {len(self.statistical_filtered)} pairs for backtesting"
        )

        # Show quality distribution
        qualities = [p["quality_score"] for p in self.statistical_filtered]
        print(
            f"Quality scores: Min={min(qualities):.1f}, "
            f"Median={np.median(qualities):.1f}, Max={max(qualities):.1f}"
        )

        # Save intermediate results
        self._save_intermediate_results(
            self.statistical_filtered, "hybrid_step1_statistical_filtered.json"
        )

        return self.statistical_filtered

    def step2_targeted_backtesting(
        self,
        filtered_pairs: List[Dict] = None,
        test_years: List[int] = [2024],
        test_months: List[int] = [1, 2, 3, 4, 5, 6],
        parameter_sets: List[Dict] = None,
    ) -> pd.DataFrame:
        """
        Step 2: Run backtests ONLY on statistically filtered pairs

        Args:
            filtered_pairs: Pairs to backtest (uses self.statistical_filtered if None)
            test_years: Years for backtesting
            test_months: Months for backtesting
            parameter_sets: Different parameter combinations to test

        Returns:
            DataFrame with backtest results
        """
        print("\n" + "=" * 80)
        print("🎯 STEP 2: TARGETED BACKTESTING")
        print("=" * 80)

        if filtered_pairs is None:
            filtered_pairs = self.statistical_filtered

        if filtered_pairs is None:
            print("❌ No filtered pairs available! Run step 1 first.")
            return pd.DataFrame()

        print(f"Running backtests on {len(filtered_pairs)} pre-filtered pairs")

        # Default parameter sets - test a few good combinations
        if parameter_sets is None:
            parameter_sets = [
                # Conservative
                {
                    "lookback_period": 60,
                    "entry_threshold": 2.5,
                    "exit_threshold": 0.5,
                    "stop_loss_threshold": 4.0,
                },
                # Moderate
                {
                    "lookback_period": 40,
                    "entry_threshold": 2.0,
                    "exit_threshold": 0.5,
                    "stop_loss_threshold": 3.5,
                },
                # Aggressive
                {
                    "lookback_period": 20,
                    "entry_threshold": 1.5,
                    "exit_threshold": 0.0,
                    "stop_loss_threshold": 3.0,
                },
            ]

        print(f"Testing {len(parameter_sets)} parameter combinations")

        all_results = []

        # Convert filtered pairs to format expected by backtester
        pair_list = [(p["symbol1"], p["symbol2"]) for p in filtered_pairs]

        for i, params in enumerate(parameter_sets, 1):
            print(f"\n📈 Testing parameter set {i}/{len(parameter_sets)}:")
            print(
                f"   Lookback: {params['lookback_period']}, "
                f"Entry: {params['entry_threshold']}, "
                f"Exit: {params['exit_threshold']}, "
                f"Stop Loss: {params['stop_loss_threshold']}"
            )

            # Run backtest
            results_df = run_fixed_parameters_backtest(
                fixed_params=params,
                specific_pairs=pair_list,  # Only test our filtered pairs
                test_years=test_years,
                test_months=test_months,
                save_results=False,
                save_plots=False,
            )

            # Add parameter info to results
            for col, val in params.items():
                results_df[col] = val

            all_results.append(results_df)

        # Combine all results
        self.backtest_results = pd.concat(all_results, ignore_index=True)

        # Add quality scores from statistical filtering
        quality_map = {
            f"{p['symbol1']}-{p['symbol2']}": p["quality_score"] for p in filtered_pairs
        }

        self.backtest_results["quality_score"] = self.backtest_results["pair"].map(
            quality_map
        )

        print(f"\n✅ Completed {len(self.backtest_results)} backtest runs")

        # Show summary statistics
        successful = self.backtest_results[self.backtest_results["success"] == True]
        print(
            f"Successful backtests: {len(successful)} "
            f"({len(successful)/len(self.backtest_results)*100:.1f}%)"
        )

        if len(successful) > 0:
            print(f"\nPerformance Summary:")
            print(f"  Sharpe Ratio: {successful['sharpe_ratio'].mean():.2f} (avg)")
            print(f"  Total Return: {successful['total_return'].mean():.3f} (avg)")
            print(f"  Win Rate: {successful['win_rate'].mean():.3f} (avg)")

        # Save intermediate results
        self.backtest_results.to_csv("hybrid_step2_backtest_results.csv", index=False)
        print(f"💾 Saved backtest results to hybrid_step2_backtest_results.csv")

        return self.backtest_results

    def step3_final_selection(
        self,
        target_pairs: int = 10,
        min_sharpe: float = 1.0,
        min_return: float = 0.1,
        weight_quality: float = 0.3,
        weight_sharpe: float = 0.4,
        weight_return: float = 0.3,
    ) -> List[Dict]:
        """
        Step 3: Final selection based on combined metrics

        Args:
            target_pairs: Number of pairs to select
            min_sharpe: Minimum Sharpe ratio
            min_return: Minimum total return
            weight_quality: Weight for statistical quality score
            weight_sharpe: Weight for Sharpe ratio
            weight_return: Weight for total return

        Returns:
            List of final selected pairs
        """
        print("\n" + "=" * 80)
        print("🏆 STEP 3: FINAL SELECTION")
        print("=" * 80)

        if self.backtest_results is None or len(self.backtest_results) == 0:
            print("❌ No backtest results available! Run step 2 first.")
            return []

        # Filter successful backtests
        successful = self.backtest_results[
            (self.backtest_results["success"] == True)
            & (self.backtest_results["sharpe_ratio"] >= min_sharpe)
            & (self.backtest_results["total_return"] >= min_return)
        ].copy()

        print(f"Pairs meeting minimum criteria: {len(successful)}")
        print(f"  Min Sharpe >= {min_sharpe}")
        print(f"  Min Return >= {min_return}")

        if len(successful) == 0:
            print("\n⚠️ No pairs meet the criteria. Relaxing thresholds...")
            # Relax criteria
            successful = self.backtest_results[
                self.backtest_results["success"] == True
            ].copy()

        # Group by pair and take best performance
        pair_performance = (
            successful.groupby("pair")
            .agg(
                {
                    "sharpe_ratio": "max",
                    "total_return": "max",
                    "win_rate": "mean",
                    "num_trades": "mean",
                    "max_drawdown": "min",
                    "quality_score": "first",  # Same for all rows of a pair
                }
            )
            .reset_index()
        )

        # Calculate combined score
        # Normalize metrics to 0-1 scale
        if len(pair_performance) > 0:
            pair_performance["norm_quality"] = (
                pair_performance["quality_score"]
                / pair_performance["quality_score"].max()
            )
            pair_performance["norm_sharpe"] = (
                pair_performance["sharpe_ratio"]
                / pair_performance["sharpe_ratio"].max()
            )
            pair_performance["norm_return"] = (
                pair_performance["total_return"]
                / pair_performance["total_return"].max()
            )

            # Combined score
            pair_performance["combined_score"] = (
                weight_quality * pair_performance["norm_quality"]
                + weight_sharpe * pair_performance["norm_sharpe"]
                + weight_return * pair_performance["norm_return"]
            )

            # Sort by combined score
            pair_performance = pair_performance.sort_values(
                "combined_score", ascending=False
            )

            # Select top N
            final_pairs_df = pair_performance.head(target_pairs)

            # Convert back to original format with all info
            self.final_selection = []
            for _, row in final_pairs_df.iterrows():
                # Find original pair info
                symbol1, symbol2 = row["pair"].split("-")

                # Find in statistical filtered
                for pair in self.statistical_filtered:
                    if pair["symbol1"] == symbol1 and pair["symbol2"] == symbol2:
                        # Add performance metrics
                        pair["backtest_sharpe"] = row["sharpe_ratio"]
                        pair["backtest_return"] = row["total_return"]
                        pair["backtest_win_rate"] = row["win_rate"]
                        pair["combined_score"] = row["combined_score"]
                        self.final_selection.append(pair)
                        break

            print(f"\n✅ Selected {len(self.final_selection)} final pairs")

            # Display final selection
            self._display_final_selection()

            # Save final results
            self._save_final_results()

        return self.final_selection

    def _display_final_selection(self):
        """Display the final selected pairs"""
        print("\n" + "=" * 80)
        print("🏆 FINAL SELECTED PAIRS")
        print("=" * 80)

        print(
            f"{'Rank':<5} {'Symbol1':<10} {'Symbol2':<10} {'Quality':<8} "
            f"{'Sharpe':<8} {'Return':<8} {'Combined':<10}"
        )
        print("-" * 80)

        for i, pair in enumerate(self.final_selection, 1):
            print(
                f"{i:<5} {pair['symbol1'][:8]:<10} {pair['symbol2'][:8]:<10} "
                f"{pair['quality_score']:<8.1f} "
                f"{pair.get('backtest_sharpe', 0):<8.2f} "
                f"{pair.get('backtest_return', 0):<8.3f} "
                f"{pair.get('combined_score', 0):<10.3f}"
            )

    def _save_intermediate_results(self, data: List[Dict], filename: str):
        """Save intermediate results to JSON"""
        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"💾 Saved to {filename}")

    def _save_final_results(self):
        """Save final results in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        with open(f"hybrid_final_selection_{timestamp}.json", "w") as f:
            json.dump(self.final_selection, f, indent=2, default=str)

        # Save as CSV
        df = pd.DataFrame(self.final_selection)

        # Keep only important columns for CSV
        important_cols = [
            "symbol1",
            "symbol2",
            "quality_score",
            "p_value",
            "correlation",
            "backtest_sharpe",
            "backtest_return",
            "backtest_win_rate",
            "combined_score",
        ]

        df_clean = df[[col for col in important_cols if col in df.columns]]
        df_clean.to_csv(f"hybrid_final_selection_{timestamp}.csv", index=False)

        print(f"\n💾 Final results saved:")
        print(f"   - hybrid_final_selection_{timestamp}.json")
        print(f"   - hybrid_final_selection_{timestamp}.csv")

    def run_complete_hybrid_analysis(
        self,
        cointegration_file: str,
        target_final_pairs: int = 10,
        pairs_to_backtest: int = 50,
        test_years: List[int] = [2024],
        test_months: List[int] = [1, 2, 3, 4, 5, 6],
    ) -> List[Dict]:
        """
        Run the complete hybrid analysis pipeline

        Args:
            cointegration_file: Path to cointegration results
            target_final_pairs: Number of pairs to finally select
            pairs_to_backtest: Number of pairs to test with backtesting
            test_years: Years for backtesting
            test_months: Months for backtesting

        Returns:
            List of final selected pairs
        """
        print("\n" + "=" * 80)
        print("🚀 HYBRID PAIR SELECTION SYSTEM")
        print("=" * 80)
        print(
            f"Pipeline: {len(self.cointegration_results) if self.cointegration_results else '???'} pairs "
            f"→ {pairs_to_backtest} filtered → {target_final_pairs} final"
        )

        # Step 1: Statistical Filtering
        self.step1_statistical_filtering(
            cointegration_file=cointegration_file,
            initial_filter_strategy="aggressive",
            target_pairs_for_backtest=pairs_to_backtest,
        )

        # Step 2: Targeted Backtesting
        self.step2_targeted_backtesting(test_years=test_years, test_months=test_months)

        # Step 3: Final Selection
        self.step3_final_selection(
            target_pairs=target_final_pairs, min_sharpe=1.0, min_return=0.1
        )

        # Summary
        print("\n" + "=" * 80)
        print("📊 HYBRID ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Started with: {len(self.cointegration_results)} pairs")
        print(f"Statistically filtered to: {len(self.statistical_filtered)} pairs")
        print(f"Backtested: {len(self.statistical_filtered)} pairs")
        print(f"Final selection: {len(self.final_selection)} pairs")

        # Time estimate
        total_time = len(self.statistical_filtered) * 3 * 0.5  # Approximate minutes
        print(
            f"\n⏱️ Estimated time saved vs full backtesting: "
            f"{(len(self.cointegration_results) - len(self.statistical_filtered)) * 3 * 0.5:.0f} minutes"
        )

        return self.final_selection


def quick_hybrid_run():
    """
    Quick run with sensible defaults
    """
    print("=" * 80)
    print("🚀 QUICK HYBRID ANALYSIS")
    print("=" * 80)

    # Find latest cointegration results
    coint_files = glob.glob("cointegration_results*/cointegration_results_*.json")
    coint_files.extend(glob.glob("cointegration_results*/cointegration_results_*.pkl"))

    if not coint_files:
        print("❌ No cointegration results found!")
        return

    latest_file = max(coint_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"📁 Using: {latest_file}")

    # Run hybrid analysis
    selector = HybridPairSelector()
    final_pairs = selector.run_complete_hybrid_analysis(
        cointegration_file=latest_file,
        target_final_pairs=10,
        pairs_to_backtest=50,
        test_years=[2024],
        test_months=[1, 2, 3, 4, 5, 6],
    )

    return final_pairs


def custom_hybrid_run():
    """
    Custom run with user inputs
    """
    print("=" * 80)
    print("🔧 CUSTOM HYBRID ANALYSIS")
    print("=" * 80)

    # Find cointegration files
    coint_files = glob.glob("cointegration_results*/cointegration_results_*.json")
    coint_files.extend(glob.glob("cointegration_results*/cointegration_results_*.pkl"))

    if not coint_files:
        print("❌ No cointegration results found!")
        return

    # Select file
    print("\n📁 Available cointegration results:")
    for i, file in enumerate(coint_files, 1):
        print(f"   {i}. {file}")

    if len(coint_files) == 1:
        selected_file = coint_files[0]
    else:
        choice = input(f"Select file (1-{len(coint_files)}): ").strip()
        try:
            selected_file = coint_files[int(choice) - 1]
        except:
            selected_file = coint_files[0]

    print(f"✅ Selected: {selected_file}")

    # Get parameters
    print("\n📊 PARAMETERS:")

    pairs_to_backtest = input("Pairs to backtest (default=50): ").strip()
    pairs_to_backtest = int(pairs_to_backtest) if pairs_to_backtest else 50

    target_final = input("Final pairs to select (default=10): ").strip()
    target_final = int(target_final) if target_final else 10

    # Test period
    print("\nTest period (for backtesting):")
    test_year = input("Year (default=2024): ").strip()
    test_year = int(test_year) if test_year else 2024

    test_months_str = input("Months (comma-separated, default=1,2,3,4,5,6): ").strip()
    if test_months_str:
        test_months = [int(m.strip()) for m in test_months_str.split(",")]
    else:
        test_months = [1, 2, 3, 4, 5, 6]

    # Run analysis
    selector = HybridPairSelector()
    final_pairs = selector.run_complete_hybrid_analysis(
        cointegration_file=selected_file,
        target_final_pairs=target_final,
        pairs_to_backtest=pairs_to_backtest,
        test_years=[test_year],
        test_months=test_months,
    )

    return final_pairs


def load_and_continue():
    """
    Load previous results and continue from where you left off
    """
    print("=" * 80)
    print("📂 CONTINUE FROM PREVIOUS RESULTS")
    print("=" * 80)

    # Check for intermediate files
    stat_file = "hybrid_step1_statistical_filtered.json"
    backtest_file = "hybrid_step2_backtest_results.csv"

    selector = HybridPairSelector()

    if Path(stat_file).exists():
        print(f"✅ Found statistical filtering results: {stat_file}")
        with open(stat_file, "r") as f:
            selector.statistical_filtered = json.load(f)
        print(f"   Loaded {len(selector.statistical_filtered)} filtered pairs")

    if Path(backtest_file).exists():
        print(f"✅ Found backtest results: {backtest_file}")
        selector.backtest_results = pd.read_csv(backtest_file)
        print(f"   Loaded {len(selector.backtest_results)} backtest results")

        # Can jump to final selection
        if selector.statistical_filtered and len(selector.backtest_results) > 0:
            print("\n📊 Ready for final selection")

            target = input("Number of final pairs (default=10): ").strip()
            target = int(target) if target else 10

            selector.step3_final_selection(target_pairs=target)
            return selector.final_selection

    print("\n⚠️ No intermediate results found. Please run full analysis.")
    return None


def main():
    """
    Main function with menu
    """
    print("=" * 80)
    print("🔬 HYBRID COINTEGRATION PAIR SELECTION")
    print("=" * 80)
    print("\nCombines statistical filtering with targeted backtesting")
    print("Much faster than full backtesting while maintaining accuracy")

    print("\n📋 OPTIONS:")
    print("1. Quick run (sensible defaults)")
    print("2. Custom run (specify parameters)")
    print("3. Continue from previous results")
    print("4. Run step-by-step (interactive)")

    choice = input("\nSelect option (1-4, default=1): ").strip()

    if choice == "2":
        custom_hybrid_run()
    elif choice == "3":
        load_and_continue()
    elif choice == "4":
        # Step by step
        print("\n📝 STEP-BY-STEP MODE")

        selector = HybridPairSelector()

        # Find cointegration file
        coint_files = glob.glob("cointegration_results*/cointegration_results_*.json")
        if not coint_files:
            print("❌ No cointegration results found!")
            return

        latest_file = max(coint_files, key=lambda x: Path(x).stat().st_mtime)

        # Step 1
        input("\n▶️ Press Enter to run Step 1 (Statistical Filtering)...")
        selector.step1_statistical_filtering(latest_file)

        # Step 2
        input("\n▶️ Press Enter to run Step 2 (Targeted Backtesting)...")
        selector.step2_targeted_backtesting()

        # Step 3
        input("\n▶️ Press Enter to run Step 3 (Final Selection)...")
        selector.step3_final_selection()
    else:
        quick_hybrid_run()

    print("\n✅ Hybrid analysis complete!")


if __name__ == "__main__":
    main()
