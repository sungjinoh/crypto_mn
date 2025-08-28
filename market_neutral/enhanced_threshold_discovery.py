"""
Enhanced Threshold Discovery Analysis - Version 2
Improved version with both statistical and backtest-based parameter discovery
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_neutral.run_fixed_parameters import run_fixed_parameters_backtest


class EnhancedThresholdDiscovery:
    """
    Enhanced threshold discovery combining statistical analysis and backtesting
    """

    def __init__(self):
        self.backtest_results = None
        self.cointegration_results = None
        self.combined_results = None

    def run_smart_parameter_scan(self, n_pairs: int = 100) -> pd.DataFrame:
        """
        Run a smart parameter scan with fewer but more informative parameter combinations
        """
        print("🔍 SMART PARAMETER SCAN")
        print("=" * 80)

        # Define parameter grid with key inflection points
        param_grid = {
            "lookback_period": [20, 40, 60],  # Short, medium, long
            "entry_threshold": [1.5, 2.0, 2.5, 3.0],  # Various entry sensitivities
            "exit_threshold": [0.0, 0.5, 1.0],  # Exit strategies
            "stop_loss_threshold": [3.0, 4.0, 5.0],  # Risk levels
        }

        # Calculate total combinations
        total_combinations = (
            len(param_grid["lookback_period"])
            * len(param_grid["entry_threshold"])
            * len(param_grid["exit_threshold"])
            * len(param_grid["stop_loss_threshold"])
        )

        print(f"Testing {total_combinations} parameter combinations on {n_pairs} pairs")
        print(f"Total tests: {total_combinations * n_pairs}")

        results = []

        # Test each parameter combination
        for lookback in param_grid["lookback_period"]:
            for entry in param_grid["entry_threshold"]:
                for exit_t in param_grid["exit_threshold"]:
                    for stop_loss in param_grid["stop_loss_threshold"]:

                        params = {
                            "lookback_period": lookback,
                            "entry_threshold": entry,
                            "exit_threshold": exit_t,
                            "stop_loss_threshold": stop_loss,
                        }

                        print(
                            f"\nTesting: LB={lookback}, Entry={entry}, Exit={exit_t}, SL={stop_loss}"
                        )

                        # Run backtest with these parameters
                        backtest_results = run_fixed_parameters_backtest(
                            fixed_params=params,
                            n_pairs=n_pairs,
                            test_years=[2025],
                            test_months=[4, 5, 6, 7],
                            save_results=False,
                            save_plots=False,
                        )

                        # Aggregate results
                        if len(backtest_results) > 0:
                            successful = backtest_results[
                                backtest_results["success"] == True
                            ]

                            summary = {
                                **params,
                                "success_rate": len(successful) / len(backtest_results),
                                "avg_sharpe": (
                                    successful["sharpe_ratio"].mean()
                                    if len(successful) > 0
                                    else 0
                                ),
                                "avg_return": (
                                    successful["total_return"].mean()
                                    if len(successful) > 0
                                    else 0
                                ),
                                "avg_trades": (
                                    successful["num_trades"].mean()
                                    if len(successful) > 0
                                    else 0
                                ),
                                "successful_pairs": len(successful),
                            }

                            results.append(summary)

        return pd.DataFrame(results)

    def analyze_cointegration_quality(self, cointegration_file: str) -> pd.DataFrame:
        """
        Analyze cointegration quality metrics without backtesting
        """
        print(f"\n🔬 COINTEGRATION QUALITY ANALYSIS")
        print("=" * 60)

        # Load cointegration results
        if cointegration_file.endswith(".json"):
            with open(cointegration_file, "r") as f:
                data = json.load(f)
            pairs = data.get("cointegrated_pairs", [])
        elif cointegration_file.endswith(".parquet"):
            df = pd.read_parquet(cointegration_file)
            pairs = df.to_dict("records")
        else:
            pairs = []

        if not pairs:
            print("❌ No cointegrated pairs found!")
            return pd.DataFrame()

        # Calculate quality metrics for each pair
        quality_metrics = []

        for pair in pairs:
            metrics = {
                "symbol1": pair.get("symbol1"),
                "symbol2": pair.get("symbol2"),
                "p_value": pair.get("p_value", 1.0),
                "correlation": abs(pair.get("correlation", 0)),
                "hedge_ratio": pair.get("hedge_ratio", 1.0),
            }

            # Extract spread properties
            if "spread_properties" in pair:
                props = pair["spread_properties"]
                if isinstance(props, dict):
                    metrics["half_life"] = props.get("half_life_ou") or props.get(
                        "half_life"
                    )
                    metrics["spread_mean"] = props.get("mean", 0)
                    metrics["spread_std"] = props.get("std", 1)
                    metrics["spread_sharpe"] = props.get("sharpe_ratio", 0)
                    metrics["zero_crossings"] = props.get("zero_crossings", 0)
                    metrics["is_stationary"] = props.get("spread_is_stationary", False)

            # Extract rolling stability
            if "rolling_stability" in pair:
                stability = pair["rolling_stability"]
                if isinstance(stability, dict):
                    metrics["stability_ratio"] = stability.get("stability_ratio", 0)
                    metrics["hedge_ratio_std"] = stability.get("hedge_ratio_std", 0)

            # Calculate composite quality score
            score = 0

            # P-value contribution (lower is better)
            if metrics["p_value"] <= 0.001:
                score += 30
            elif metrics["p_value"] <= 0.01:
                score += 20
            elif metrics["p_value"] <= 0.05:
                score += 10

            # Correlation contribution (0.7-0.9 is ideal)
            if 0.7 <= metrics["correlation"] <= 0.9:
                score += 20
            elif 0.6 <= metrics["correlation"] <= 0.95:
                score += 10

            # Half-life contribution (10-50 is ideal)
            half_life = metrics.get("half_life")
            if half_life and 10 <= half_life <= 50:
                score += 20
            elif half_life and 5 <= half_life <= 100:
                score += 10

            # Stationarity bonus
            if metrics.get("is_stationary", False):
                score += 15

            # Stability bonus
            stability_ratio = metrics.get("stability_ratio", 0)
            score += stability_ratio * 15

            metrics["quality_score"] = score
            quality_metrics.append(metrics)

        df = pd.DataFrame(quality_metrics)

        # Print quality distribution
        print(f"Analyzed {len(df)} cointegrated pairs")
        print(f"\nQuality Score Distribution:")
        print(df["quality_score"].describe())

        return df

    def combine_analyses(
        self, cointegration_df: pd.DataFrame, backtest_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine cointegration quality with backtest results
        """
        print(f"\n🔗 COMBINING ANALYSES")
        print("=" * 60)

        # Create pair identifiers
        cointegration_df["pair_id"] = (
            cointegration_df["symbol1"] + "-" + cointegration_df["symbol2"]
        )

        if "pair" in backtest_df.columns:
            backtest_df["pair_id"] = backtest_df["pair"]
        else:
            # Assume index contains pair information
            backtest_df["pair_id"] = backtest_df.index

        # Merge on pair_id
        combined = pd.merge(
            cointegration_df,
            backtest_df[
                ["pair_id", "sharpe_ratio", "total_return", "num_trades", "success"]
            ],
            on="pair_id",
            how="inner",
        )

        print(f"Found {len(combined)} pairs with both cointegration and backtest data")

        # Calculate combined score
        combined["combined_score"] = (
            combined["quality_score"] * 0.4  # 40% weight on statistical quality
            + combined["sharpe_ratio"] * 20  # Sharpe contribution
            + combined["total_return"] * 10  # Return contribution
        )

        return combined

    def find_optimal_cutoffs(self, df: pd.DataFrame, target_pairs: int = 10) -> Dict:
        """
        Find optimal cutoff values for filtering
        """
        print(f"\n🎯 FINDING OPTIMAL CUTOFFS")
        print("=" * 60)

        # Sort by combined score if available, otherwise by quality score
        sort_col = (
            "combined_score" if "combined_score" in df.columns else "quality_score"
        )
        df_sorted = df.sort_values(sort_col, ascending=False)

        # Get top N pairs
        top_n = df_sorted.head(target_pairs)

        cutoffs = {
            "statistical": {
                "min_quality_score": (
                    top_n["quality_score"].min()
                    if "quality_score" in top_n.columns
                    else 0
                ),
                "max_p_value": (
                    top_n["p_value"].max() if "p_value" in top_n.columns else 0.05
                ),
                "min_correlation": (
                    top_n["correlation"].min()
                    if "correlation" in top_n.columns
                    else 0.6
                ),
                "max_correlation": (
                    top_n["correlation"].max()
                    if "correlation" in top_n.columns
                    else 0.95
                ),
            }
        }

        # Add half-life cutoffs if available
        if "half_life" in top_n.columns:
            half_lives = top_n["half_life"].dropna()
            if len(half_lives) > 0:
                cutoffs["mean_reversion"] = {
                    "min_half_life": half_lives.min(),
                    "max_half_life": half_lives.max(),
                    "median_half_life": half_lives.median(),
                }

        # Add performance cutoffs if available
        if "sharpe_ratio" in top_n.columns:
            cutoffs["performance"] = {
                "min_sharpe": top_n["sharpe_ratio"].min(),
                "min_return": (
                    top_n["total_return"].min()
                    if "total_return" in top_n.columns
                    else 0
                ),
                "min_trades": (
                    top_n["num_trades"].min() if "num_trades" in top_n.columns else 1
                ),
            }

        # Add stability cutoffs if available
        if "stability_ratio" in top_n.columns:
            cutoffs["stability"] = {
                "min_stability": top_n["stability_ratio"].min(),
            }

        return cutoffs

    def create_comprehensive_plots(
        self, df: pd.DataFrame, save_dir: str = "threshold_discovery_plots"
    ):
        """
        Create comprehensive visualization plots
        """
        os.makedirs(save_dir, exist_ok=True)

        # Determine available columns
        has_backtest = "sharpe_ratio" in df.columns
        has_quality = "quality_score" in df.columns

        if has_backtest and has_quality:
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

        plot_idx = 0

        # Quality score distribution
        if has_quality:
            axes[plot_idx].hist(
                df["quality_score"], bins=30, alpha=0.7, edgecolor="black"
            )
            axes[plot_idx].set_title("Cointegration Quality Score Distribution")
            axes[plot_idx].set_xlabel("Quality Score")
            axes[plot_idx].set_ylabel("Frequency")
            plot_idx += 1

        # P-value distribution
        if "p_value" in df.columns:
            axes[plot_idx].hist(
                np.log10(df["p_value"]), bins=30, alpha=0.7, edgecolor="black"
            )
            axes[plot_idx].set_title("P-value Distribution (log scale)")
            axes[plot_idx].set_xlabel("log10(p-value)")
            axes[plot_idx].set_ylabel("Frequency")
            plot_idx += 1

        # Sharpe ratio distribution
        if has_backtest:
            axes[plot_idx].hist(
                df["sharpe_ratio"], bins=30, alpha=0.7, edgecolor="black"
            )
            axes[plot_idx].set_title("Sharpe Ratio Distribution")
            axes[plot_idx].set_xlabel("Sharpe Ratio")
            axes[plot_idx].set_ylabel("Frequency")
            plot_idx += 1

            # Return distribution
            if "total_return" in df.columns:
                axes[plot_idx].hist(
                    df["total_return"], bins=30, alpha=0.7, edgecolor="black"
                )
                axes[plot_idx].set_title("Total Return Distribution")
                axes[plot_idx].set_xlabel("Total Return")
                axes[plot_idx].set_ylabel("Frequency")
                plot_idx += 1

        # Quality vs Performance scatter
        if has_backtest and has_quality:
            axes[plot_idx].scatter(df["quality_score"], df["sharpe_ratio"], alpha=0.6)
            axes[plot_idx].set_xlabel("Quality Score")
            axes[plot_idx].set_ylabel("Sharpe Ratio")
            axes[plot_idx].set_title("Quality Score vs Sharpe Ratio")

            # Add correlation text
            corr = df["quality_score"].corr(df["sharpe_ratio"])
            axes[plot_idx].text(
                0.05,
                0.95,
                f"Correlation: {corr:.3f}",
                transform=axes[plot_idx].transAxes,
                verticalalignment="top",
            )
            plot_idx += 1

        # Half-life distribution
        if "half_life" in df.columns:
            half_lives = df["half_life"].dropna()
            if len(half_lives) > 0:
                axes[plot_idx].hist(half_lives, bins=30, alpha=0.7, edgecolor="black")
                axes[plot_idx].set_title("Half-Life Distribution")
                axes[plot_idx].set_xlabel("Half-Life (periods)")
                axes[plot_idx].set_ylabel("Frequency")
                axes[plot_idx].axvline(
                    half_lives.median(), color="red", linestyle="--", label="Median"
                )
                axes[plot_idx].legend()
                plot_idx += 1

        # Correlation distribution
        if "correlation" in df.columns:
            axes[plot_idx].hist(
                df["correlation"], bins=30, alpha=0.7, edgecolor="black"
            )
            axes[plot_idx].set_title("Correlation Distribution")
            axes[plot_idx].set_xlabel("Correlation")
            axes[plot_idx].set_ylabel("Frequency")
            plot_idx += 1

        # Combined score if available
        if "combined_score" in df.columns:
            axes[plot_idx].hist(
                df["combined_score"], bins=30, alpha=0.7, edgecolor="black"
            )
            axes[plot_idx].set_title("Combined Score Distribution")
            axes[plot_idx].set_xlabel("Combined Score")
            axes[plot_idx].set_ylabel("Frequency")
            plot_idx += 1

        # Top pairs bar chart
        if has_quality or has_backtest:
            sort_col = (
                "combined_score" if "combined_score" in df.columns else "quality_score"
            )
            if sort_col not in df.columns and has_backtest:
                sort_col = "sharpe_ratio"

            if sort_col in df.columns:
                top_20 = df.nlargest(20, sort_col)
                if "symbol1" in top_20.columns and "symbol2" in top_20.columns:
                    labels = [
                        f"{row['symbol1'][:4]}-{row['symbol2'][:4]}"
                        for _, row in top_20.iterrows()
                    ]
                else:
                    labels = [f"Pair {i+1}" for i in range(len(top_20))]

                axes[plot_idx].barh(range(len(labels)), top_20[sort_col].values)
                axes[plot_idx].set_yticks(range(len(labels)))
                axes[plot_idx].set_yticklabels(labels, fontsize=8)
                axes[plot_idx].set_xlabel(sort_col.replace("_", " ").title())
                axes[plot_idx].set_title(
                    f'Top 20 Pairs by {sort_col.replace("_", " ").title()}'
                )
                axes[plot_idx].invert_yaxis()

        # Remove empty subplots
        for i in range(plot_idx + 1, len(axes)):
            if i < len(axes):
                fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/comprehensive_analysis.png", dpi=300, bbox_inches="tight"
        )
        print(f"📊 Plots saved to {save_dir}/comprehensive_analysis.png")
        plt.close()

    def generate_recommendations(self, cutoffs: Dict, target_pairs: int = 10) -> None:
        """
        Generate and print recommendations
        """
        print(f"\n" + "=" * 80)
        print("💡 FILTERING RECOMMENDATIONS")
        print("=" * 80)

        print(f"\n🎯 To select approximately {target_pairs} high-quality pairs:")

        # Statistical filters
        if "statistical" in cutoffs:
            print(f"\n📊 STATISTICAL FILTERS:")
            stat = cutoffs["statistical"]
            print(f"   • Quality score >= {stat.get('min_quality_score', 0):.1f}")
            print(f"   • P-value <= {stat.get('max_p_value', 0.05):.4f}")
            print(
                f"   • Correlation: {stat.get('min_correlation', 0.6):.3f} - {stat.get('max_correlation', 0.95):.3f}"
            )

        # Mean reversion filters
        if "mean_reversion" in cutoffs:
            print(f"\n⏱️ MEAN REVERSION FILTERS:")
            mr = cutoffs["mean_reversion"]
            print(
                f"   • Half-life: {mr.get('min_half_life', 5):.1f} - {mr.get('max_half_life', 100):.1f} periods"
            )
            print(
                f"   • Median half-life: {mr.get('median_half_life', 30):.1f} periods"
            )

        # Performance filters
        if "performance" in cutoffs:
            print(f"\n📈 PERFORMANCE FILTERS (from backtesting):")
            perf = cutoffs["performance"]
            print(f"   • Min Sharpe ratio: {perf.get('min_sharpe', 0):.2f}")
            print(f"   • Min total return: {perf.get('min_return', 0):.3f}")
            print(f"   • Min trades: {perf.get('min_trades', 1):.0f}")

        # Stability filters
        if "stability" in cutoffs:
            print(f"\n🔄 STABILITY FILTERS:")
            stab = cutoffs["stability"]
            print(f"   • Min stability ratio: {stab.get('min_stability', 0):.3f}")

        # Generate Python code
        print(f"\n📋 PYTHON IMPLEMENTATION:")
        print("```python")
        print("def filter_pairs(pairs_df):")
        print("    filtered = pairs_df.copy()")

        if "statistical" in cutoffs:
            stat = cutoffs["statistical"]
            if "min_quality_score" in stat:
                print(
                    f"    filtered = filtered[filtered['quality_score'] >= {stat['min_quality_score']:.1f}]"
                )
            print(
                f"    filtered = filtered[filtered['p_value'] <= {stat.get('max_p_value', 0.05):.4f}]"
            )
            print(
                f"    filtered = filtered[filtered['correlation'] >= {stat.get('min_correlation', 0.6):.3f}]"
            )
            print(
                f"    filtered = filtered[filtered['correlation'] <= {stat.get('max_correlation', 0.95):.3f}]"
            )

        if "mean_reversion" in cutoffs:
            mr = cutoffs["mean_reversion"]
            print(
                f"    filtered = filtered[filtered['half_life'] >= {mr.get('min_half_life', 5):.1f}]"
            )
            print(
                f"    filtered = filtered[filtered['half_life'] <= {mr.get('max_half_life', 100):.1f}]"
            )

        if "performance" in cutoffs:
            perf = cutoffs["performance"]
            print(
                f"    filtered = filtered[filtered['sharpe_ratio'] >= {perf.get('min_sharpe', 0):.2f}]"
            )
            print(
                f"    filtered = filtered[filtered['total_return'] >= {perf.get('min_return', 0):.3f}]"
            )

        print("    return filtered")
        print("```")

        # Conservative vs Aggressive options
        print(f"\n🛡️ CONSERVATIVE OPTION (higher quality, fewer pairs):")
        print(f"   • Tighten p-value to <= 0.01")
        print(f"   • Increase min correlation to >= 0.8")
        print(f"   • Reduce max half-life to <= 30")

        print(f"\n🚀 AGGRESSIVE OPTION (more pairs, relaxed criteria):")
        print(f"   • Relax p-value to <= 0.1")
        print(f"   • Decrease min correlation to >= 0.6")
        print(f"   • Increase max half-life to <= 100")


def main():
    """
    Enhanced main function with multiple analysis options
    """
    print("=" * 80)
    print("🔬 ENHANCED THRESHOLD DISCOVERY ANALYSIS")
    print("=" * 80)

    analyzer = EnhancedThresholdDiscovery()

    # Check for existing files
    import glob

    # Look for cointegration results
    coint_files = glob.glob("cointegration_results*/cointegration_results_*.json")
    coint_files.extend(glob.glob("cointegration_results*/cointegrated_pairs_*.parquet"))

    # Look for backtest results
    backtest_files = glob.glob("*_results_*.csv")

    print("\n📁 Available Data:")
    print(f"   • Cointegration files: {len(coint_files)}")
    print(f"   • Backtest files: {len(backtest_files)}")

    # Menu
    print("\n📋 ANALYSIS OPTIONS:")
    print("1. Quick statistical analysis (no backtesting)")
    print("2. Analyze existing backtest results")
    print("3. Run new smart parameter scan")
    print("4. Combined analysis (cointegration + backtest)")

    choice = input("\nSelect option (1-4): ").strip()

    if choice == "1":
        # Quick statistical analysis
        if coint_files:
            latest_coint = max(coint_files, key=lambda x: Path(x).stat().st_mtime)
            print(f"\n📊 Analyzing: {latest_coint}")

            quality_df = analyzer.analyze_cointegration_quality(latest_coint)
            if len(quality_df) > 0:
                cutoffs = analyzer.find_optimal_cutoffs(quality_df, target_pairs=10)
                analyzer.create_comprehensive_plots(quality_df)
                analyzer.generate_recommendations(cutoffs)
        else:
            print("❌ No cointegration results found!")

    elif choice == "2":
        # Analyze existing backtest
        if backtest_files:
            latest_backtest = max(backtest_files, key=lambda x: Path(x).stat().st_mtime)
            print(f"\n📊 Analyzing: {latest_backtest}")

            backtest_df = pd.read_csv(latest_backtest)
            if "success" in backtest_df.columns:
                backtest_df = backtest_df[backtest_df["success"] == True]

            cutoffs = analyzer.find_optimal_cutoffs(backtest_df, target_pairs=10)
            analyzer.create_comprehensive_plots(backtest_df)
            analyzer.generate_recommendations(cutoffs)
        else:
            print("❌ No backtest results found!")

    elif choice == "3":
        # Run new parameter scan
        n_pairs = input("Number of pairs to test (default: 50): ").strip()
        n_pairs = int(n_pairs) if n_pairs else 50

        param_results = analyzer.run_smart_parameter_scan(n_pairs=n_pairs)

        print("\n📊 PARAMETER SCAN RESULTS:")
        print(param_results.sort_values("avg_sharpe", ascending=False).head(10))

        # Save results
        param_results.to_csv("parameter_scan_results.csv", index=False)
        print("\n✅ Results saved to parameter_scan_results.csv")

    elif choice == "4":
        # Combined analysis
        if coint_files and backtest_files:
            latest_coint = max(coint_files, key=lambda x: Path(x).stat().st_mtime)
            latest_backtest = max(backtest_files, key=lambda x: Path(x).stat().st_mtime)

            print(f"\n📊 Combining:")
            print(f"   • Cointegration: {latest_coint}")
            print(f"   • Backtest: {latest_backtest}")

            quality_df = analyzer.analyze_cointegration_quality(latest_coint)
            backtest_df = pd.read_csv(latest_backtest)

            if "success" in backtest_df.columns:
                backtest_df = backtest_df[backtest_df["success"] == True]

            combined_df = analyzer.combine_analyses(quality_df, backtest_df)

            if len(combined_df) > 0:
                cutoffs = analyzer.find_optimal_cutoffs(combined_df, target_pairs=10)
                analyzer.create_comprehensive_plots(combined_df)
                analyzer.generate_recommendations(cutoffs)

                # Save combined results
                combined_df.to_csv("combined_analysis_results.csv", index=False)
                print("\n✅ Combined results saved to combined_analysis_results.csv")
        else:
            print(
                "❌ Need both cointegration and backtest results for combined analysis!"
            )

    else:
        print("❌ Invalid option!")

    print("\n🎉 Analysis complete!")


if __name__ == "__main__":
    main()
