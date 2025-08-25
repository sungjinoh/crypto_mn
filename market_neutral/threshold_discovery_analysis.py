#!/usr/bin/env python3
"""
Threshold Discovery Analysis - Fixed Version
Use run_fixed_parameters.py results to find optimal filtering thresholds.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_neutral.run_fixed_parameters import run_fixed_parameters_backtest


def run_threshold_discovery_backtest():
    """
    Run comprehensive backtest on all pairs to discover optimal thresholds
    """
    print("🔍 THRESHOLD DISCOVERY ANALYSIS")
    print("=" * 80)

    # Use moderate, reasonable baseline parameters
    baseline_params = {
        "lookback_period": 60,  # Standard rolling window
        "entry_threshold": 2.0,  # Moderate entry signal
        "exit_threshold": 0.5,  # Balanced exit signal
        "stop_loss_threshold": 3.0,  # Conservative stop loss
    }

    print("📋 Using baseline parameters for threshold discovery:")
    for param, value in baseline_params.items():
        print(f"   • {param}: {value}")

    # Run on many pairs to get good distribution
    results = run_fixed_parameters_backtest(
        fixed_params=baseline_params,
        n_pairs=100,  # Test many pairs for good statistics
        test_years=[2023, 2024],
        test_months=[
            [6, 7, 8, 9, 10, 11, 12],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ],  # Out-of-sample testing
        save_results=True,
        save_plots=False,  # Disable plots for speed
    )

    return results, baseline_params


def analyze_performance_distribution(df):
    """Analyze the performance distribution to find natural thresholds"""

    print(f"\n📊 PERFORMANCE DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # Filter successful results
    successful = df[df["success"] == True]

    print(f"Total pairs tested: {len(df)}")
    print(f"Successful backtests: {len(successful)} ({len(successful)/len(df):.1%})")

    if len(successful) == 0:
        print("❌ No successful backtests found!")
        return None, None

    # Key performance metrics
    metrics = {
        "sharpe_ratio": "Sharpe Ratio",
        "total_return": "Total Return",
        "max_drawdown": "Max Drawdown",
        "win_rate": "Win Rate",
        "num_trades": "Number of Trades",
    }

    print(f"\n📈 PERFORMANCE PERCENTILES:")
    percentile_data = {}

    for metric, display_name in metrics.items():
        if metric in successful.columns:
            percentiles = successful[metric].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            percentile_data[metric] = percentiles

            print(f"\n{display_name}:")
            print(f"  10th percentile: {percentiles[0.1]:.3f}")
            print(f"  25th percentile: {percentiles[0.25]:.3f}")
            print(f"  50th percentile: {percentiles[0.5]:.3f}")
            print(f"  75th percentile: {percentiles[0.75]:.3f}")
            print(f"  90th percentile: {percentiles[0.9]:.3f}")

    return percentile_data, successful


def find_optimal_thresholds(df):
    """Find optimal filtering thresholds based on performance distribution"""

    print(f"\n🎯 OPTIMAL THRESHOLD ANALYSIS")
    print("=" * 60)

    # Sort by Sharpe ratio (primary metric)
    df_sorted = df.sort_values("sharpe_ratio", ascending=False)

    threshold_options = {}

    # Analyze different selection percentages
    selection_pcts = [0.1, 0.2, 0.3, 0.4, 0.5]  # Top 10%, 20%, etc.

    for pct in selection_pcts:
        n_pairs = max(1, int(len(df_sorted) * pct))
        top_pairs = df_sorted.head(n_pairs)

        # Calculate thresholds for this group
        threshold_set = {
            "selection_pct": pct,
            "pairs_selected": len(top_pairs),
            "min_sharpe": top_pairs["sharpe_ratio"].min(),
            "min_return": top_pairs["total_return"].min(),
            "max_drawdown": top_pairs["max_drawdown"].max(),
            "min_win_rate": (
                top_pairs["win_rate"].min() if "win_rate" in top_pairs.columns else 0.0
            ),
            "min_trades": int(top_pairs["num_trades"].min()),
            "avg_sharpe": top_pairs["sharpe_ratio"].mean(),
            "avg_return": top_pairs["total_return"].mean(),
        }

        threshold_options[f"top_{pct*100:.0f}pct"] = threshold_set

    # Display threshold options
    print(f"\n🎯 THRESHOLD OPTIONS:")
    print(
        f"{'Group':<12} {'Pairs':<6} {'Min Sharpe':<10} {'Min Return':<10} {'Max DD':<8} {'Min Trades':<10}"
    )
    print("-" * 70)

    for group_name, thresholds in threshold_options.items():
        print(
            f"{group_name:<12} {thresholds['pairs_selected']:<6} "
            f"{thresholds['min_sharpe']:<10.3f} {thresholds['min_return']:<10.3f} "
            f"{thresholds['max_drawdown']:<8.3f} {thresholds['min_trades']:<10}"
        )

    return threshold_options


def recommend_thresholds(threshold_options, target_pairs=10):
    """Recommend optimal thresholds based on target number of pairs"""

    print(f"\n💡 THRESHOLD RECOMMENDATIONS")
    print("=" * 60)

    if not threshold_options:
        print("❌ No threshold options available!")
        return None

    # Find the selection percentage that gets closest to target
    best_option = None
    min_diff = float("inf")

    for group_name, thresholds in threshold_options.items():
        diff = abs(thresholds["pairs_selected"] - target_pairs)
        if diff < min_diff:
            min_diff = diff
            best_option = (group_name, thresholds)

    if best_option:
        group_name, recommended = best_option  # Fixed: properly unpack the tuple

        print(f"🎯 RECOMMENDED THRESHOLDS (targeting ~{target_pairs} pairs):")
        print(f"   • Min Sharpe Ratio: {recommended['min_sharpe']:.3f}")
        print(f"   • Min Total Return: {recommended['min_return']:.3f}")
        print(f"   • Min Win Rate: {recommended['min_win_rate']:.3f}")
        print(f"   • Min Trades: {recommended['min_trades']}")
        print(f"   • Expected pairs selected: {recommended['pairs_selected']}")

        print(f"\n📋 USE THESE THRESHOLDS IN OPTIMAL WORKFLOW:")
        print(f"```python")
        print(f"selected_pairs = workflow.step3_validation_and_selection(")
        print(f"    min_sharpe={recommended['min_sharpe']:.3f},")
        print(f"    min_return={recommended['min_return']:.3f}")
        print(f")")
        print(f"```")

        # Also show conservative and aggressive options
        group_names = list(threshold_options.keys())
        current_idx = group_names.index(group_name)

        if current_idx > 0:
            conservative_name = group_names[current_idx - 1]
            conservative = threshold_options[conservative_name]
            print(f"\n🛡️ CONSERVATIVE OPTION (fewer pairs, higher quality):")
            print(
                f"   min_sharpe={conservative['min_sharpe']:.3f}, min_return={conservative['min_return']:.3f}"
            )

        if current_idx < len(group_names) - 1:
            aggressive_name = group_names[current_idx + 1]
            aggressive = threshold_options[aggressive_name]
            print(f"\n🚀 AGGRESSIVE OPTION (more pairs, lower standards):")
            print(
                f"   min_sharpe={aggressive['min_sharpe']:.3f}, min_return={aggressive['min_return']:.3f}"
            )

    # Add cointegration-specific filtering recommendations
    print(f"\n🔬 COINTEGRATION FILTERING RECOMMENDATIONS")
    print("=" * 60)
    print(
        f"For filtering cointegrated pairs before backtesting, consider these criteria:"
    )
    print(f"")
    print(f"📊 STATISTICAL SIGNIFICANCE:")
    print(f"   • max_p_value = 0.01      # Very strong cointegration (1% significance)")
    print(f"   • max_p_value = 0.05      # Standard significance (5%)")
    print(f"")
    print(f"🔗 CORRELATION STRENGTH:")
    print(f"   • min_correlation = 0.8   # High correlation (recommended)")
    print(f"   • min_correlation = 0.7   # Moderate correlation")
    print(f"   • min_correlation = 0.6   # Lower threshold")
    print(f"")
    print(f"⏱️ MEAN REVERSION SPEED:")
    print(f"   • max_half_life = 20      # Fast mean reversion (aggressive)")
    print(f"   • max_half_life = 50      # Moderate mean reversion")
    print(f"   • max_half_life = 100     # Slower mean reversion (conservative)")
    print(f"")
    print(f"📈 RESIDUALS STATIONARITY:")
    print(f"   • require_stationary_residuals = True   # ADF test p-value < 0.05")
    print(f"")
    print(f"💰 LOG PRICE TRANSFORMATION:")
    print(
        f"   • allow_log_prices = True/False        # Include/exclude log-transformed pairs"
    )
    print(f"   • max_price_ratio = 10                 # Maximum price level difference")
    print(f"")
    print(f"📋 EXAMPLE FILTERING FUNCTION:")
    print(f"```python")
    print(f"def filter_top_pairs(cointegration_results, backtesting_results):")
    print(f'    """Filter pairs using both cointegration and performance criteria"""')
    print(f"    ")
    print(f"    # Step 1: Filter by cointegration quality")
    print(f"    coint_filtered = []")
    print(f"    for pair in cointegration_results:")
    print(
        f"        if (pair['p_value'] <= 0.01 and                    # Strong cointegration"
    )
    print(
        f"            pair['correlation'] >= 0.8 and                 # High correlation"
    )
    print(
        f"            pair['spread_properties']['half_life'] <= 50 and  # Fast mean reversion"
    )
    print(
        f"            pair['residuals_stationary'] == True):         # Stationary residuals"
    )
    print(f"            coint_filtered.append(pair)")
    print(f"    ")
    print(f"    # Step 2: Filter by backtest performance")
    print(f"    final_pairs = []")
    print(f"    for pair in coint_filtered:")
    print(f"        pair_symbol = f\"{{pair['symbol1']}}-{{pair['symbol2']}}\"")
    print(f"        backtest_result = backtesting_results.get(pair_symbol)")
    print(f"        if (backtest_result and")
    print(
        f"            backtest_result['sharpe_ratio'] >= {recommended['min_sharpe']:.3f} and"
    )
    print(
        f"            backtest_result['total_return'] >= {recommended['min_return']:.3f}):"
    )
    print(f"            final_pairs.append(pair)")
    print(f"    ")
    print(f"    return final_pairs")
    print(f"```")
    print(f"")
    print(f"🎯 RECOMMENDED COINTEGRATION FILTERS (CONSERVATIVE):")
    print(f"   • max_p_value = 0.01")
    print(f"   • min_correlation = 0.8")
    print(f"   • max_half_life = 30")
    print(f"   • require_stationary_residuals = True")
    print(f"")
    print(f"🚀 AGGRESSIVE COINTEGRATION FILTERS (MORE PAIRS):")
    print(f"   • max_p_value = 0.05")
    print(f"   • min_correlation = 0.7")
    print(f"   • max_half_life = 100")
    print(f"   • require_stationary_residuals = False")

    return recommended if best_option else None


def create_threshold_plots(df, save_dir="threshold_analysis_plots"):
    """Create visualization plots for threshold analysis"""

    os.makedirs(save_dir, exist_ok=True)

    # Performance distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Sharpe ratio distribution
    axes[0, 0].hist(df["sharpe_ratio"], bins=30, alpha=0.7, edgecolor="black")
    axes[0, 0].set_title("Sharpe Ratio Distribution")
    axes[0, 0].set_xlabel("Sharpe Ratio")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(
        df["sharpe_ratio"].median(), color="red", linestyle="--", label="Median"
    )
    axes[0, 0].legend()

    # Return distribution
    axes[0, 1].hist(df["total_return"], bins=30, alpha=0.7, edgecolor="black")
    axes[0, 1].set_title("Total Return Distribution")
    axes[0, 1].set_xlabel("Total Return")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].axvline(
        df["total_return"].median(), color="red", linestyle="--", label="Median"
    )
    axes[0, 1].legend()

    # Scatter plot: Sharpe vs Return
    axes[1, 0].scatter(df["sharpe_ratio"], df["total_return"], alpha=0.6)
    axes[1, 0].set_xlabel("Sharpe Ratio")
    axes[1, 0].set_ylabel("Total Return")
    axes[1, 0].set_title("Sharpe Ratio vs Total Return")

    # Number of trades distribution
    axes[1, 1].hist(df["num_trades"], bins=20, alpha=0.7, edgecolor="black")
    axes[1, 1].set_title("Number of Trades Distribution")
    axes[1, 1].set_xlabel("Number of Trades")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/performance_distributions.png", dpi=300, bbox_inches="tight"
    )
    print(f"📊 Plots saved to {save_dir}/performance_distributions.png")

    plt.close()


def analyze_cointegration_filters(cointegration_file, backtesting_file):
    """
    Analyze how different cointegration filters affect the final pair selection

    Args:
        cointegration_file: Path to cointegration results (JSON or Parquet)
        backtesting_file: Path to backtesting results (CSV)
    """
    print(f"🔬 COINTEGRATION FILTER ANALYSIS")
    print("=" * 60)

    try:
        # Load cointegration results
        if cointegration_file.endswith(".json"):
            import json

            with open(cointegration_file, "r") as f:
                coint_data = json.load(f)
            if isinstance(coint_data, dict) and "cointegrated_pairs" in coint_data:
                coint_pairs = coint_data["cointegrated_pairs"]
            else:
                coint_pairs = coint_data
        elif cointegration_file.endswith(".parquet"):
            coint_df = pd.read_parquet(cointegration_file)
            coint_pairs = coint_df.to_dict("records")
        else:
            print(f"❌ Unsupported cointegration file format: {cointegration_file}")
            return

        # Load backtesting results
        backtest_df = pd.read_csv(backtesting_file)
        if "success" in backtest_df.columns:
            successful_backtests = backtest_df[backtest_df["success"] == True]
        else:
            successful_backtests = backtest_df

        print(f"📊 Loaded {len(coint_pairs)} cointegrated pairs")
        print(f"📊 Loaded {len(successful_backtests)} successful backtests")

        # Test different filter combinations
        filter_configs = [
            {
                "name": "Very Conservative",
                "max_p_value": 0.001,
                "min_correlation": 0.9,
                "max_half_life": 20,
            },
            {
                "name": "Conservative",
                "max_p_value": 0.01,
                "min_correlation": 0.8,
                "max_half_life": 30,
            },
            {
                "name": "Moderate",
                "max_p_value": 0.05,
                "min_correlation": 0.7,
                "max_half_life": 50,
            },
            {
                "name": "Aggressive",
                "max_p_value": 0.1,
                "min_correlation": 0.6,
                "max_half_life": 100,
            },
        ]

        print(f"\n📋 FILTER ANALYSIS RESULTS:")
        print(
            f"{'Filter':<15} {'Coint':<6} {'Backtest':<9} {'Overlap':<7} {'Success %':<10}"
        )
        print("-" * 60)

        for config in filter_configs:
            # Apply cointegration filters
            filtered_coint = []
            for pair in coint_pairs:
                if (
                    pair.get("p_value", 1.0) <= config["max_p_value"]
                    and pair.get("correlation", 0.0) >= config["min_correlation"]
                ):

                    # Check half-life if available
                    half_life = None
                    if "spread_properties" in pair and pair["spread_properties"]:
                        if isinstance(pair["spread_properties"], str):
                            try:
                                import json

                                props = json.loads(pair["spread_properties"])
                                half_life = props.get("half_life")
                            except:
                                pass
                        else:
                            half_life = pair["spread_properties"].get("half_life")

                    if half_life is None or half_life <= config["max_half_life"]:
                        filtered_coint.append(pair)

            # Find overlap with successful backtests
            overlap_count = 0
            coint_symbols = set()
            for pair in filtered_coint:
                pair_symbol = f"{pair['symbol1']}-{pair['symbol2']}"
                coint_symbols.add(pair_symbol)

            backtest_symbols = set(
                successful_backtests["pair"].unique()
                if "pair" in successful_backtests.columns
                else successful_backtests.index
            )

            overlap = coint_symbols.intersection(backtest_symbols)
            overlap_count = len(overlap)

            success_rate = (
                (overlap_count / len(filtered_coint) * 100) if filtered_coint else 0
            )

            print(
                f"{config['name']:<15} {len(filtered_coint):<6} {len(successful_backtests):<9} "
                f"{overlap_count:<7} {success_rate:<10.1f}%"
            )

        print(f"\n💡 INSIGHTS:")
        print(f"   • Use conservative filters to focus on highest-quality pairs")
        print(f"   • Monitor success rate vs. number of pairs trade-off")
        print(f"   • Consider moderate filters for balanced approach")

    except Exception as e:
        print(f"❌ Error in cointegration filter analysis: {e}")


def analyze_existing_results(csv_file):
    """Analyze existing results from run_fixed_parameters.py"""

    print(f"📁 ANALYZING EXISTING RESULTS: {csv_file}")
    print("=" * 60)

    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} results from {csv_file}")

        # Analyze the successful results
        if "success" in df.columns:
            successful = df[df["success"] == True]
        else:
            # Assume all results are successful if no success column
            successful = df

        if len(successful) == 0:
            print("❌ No successful results found!")
            return

        print(f"📊 Found {len(successful)} successful backtests")

        # Run analysis on existing data
        percentiles, _ = analyze_performance_distribution(df)
        threshold_options = find_optimal_thresholds(successful)
        recommended = recommend_thresholds(threshold_options, target_pairs=10)

        # Create plots
        create_threshold_plots(successful)

        print(f"\n🎉 ANALYSIS OF EXISTING RESULTS COMPLETED!")

        # Check for cointegration results to do combined analysis
        import glob

        coint_files = glob.glob("cointegration_results/cointegration_results_*.json")
        coint_files.extend(
            glob.glob("cointegration_results/cointegration_results_*.parquet")
        )

        if coint_files:
            latest_coint = max(coint_files, key=os.path.getctime)
            print(f"\n🔬 Found cointegration results: {latest_coint}")
            print("Running combined cointegration + backtest analysis...")
            analyze_cointegration_filters(latest_coint, csv_file)

    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
        print("   Please run run_fixed_parameters.py first to generate results")
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")


def main():
    """Main function for threshold discovery"""

    # Check if there are existing results files to analyze
    import glob

    existing_files = glob.glob("fixed_params_results_*.csv")

    if existing_files:
        print("🔍 Found existing results files:")
        for i, file in enumerate(existing_files):
            print(f"   {i+1}. {file}")

        choice = (
            input("\nWould you like to analyze existing results? (y/n): ")
            .lower()
            .strip()
        )

        if choice == "y":
            # Use the most recent file
            latest_file = max(existing_files, key=os.path.getctime)
            analyze_existing_results(latest_file)
            return

    print("🚀 Running new threshold discovery backtest...")

    # Step 1: Run comprehensive backtest
    print("Step 1: Running comprehensive backtest...")
    results_df, baseline_params = run_threshold_discovery_backtest()

    if len(results_df) == 0:
        print("❌ No results obtained from backtest!")
        return

    # Step 2: Analyze performance distribution
    print("\nStep 2: Analyzing performance distribution...")
    percentiles, successful_df = analyze_performance_distribution(results_df)

    if successful_df is None or len(successful_df) == 0:
        print("❌ No successful results to analyze!")
        return

    # Step 3: Find optimal thresholds
    print("\nStep 3: Finding optimal thresholds...")
    threshold_options = find_optimal_thresholds(successful_df)

    # Step 4: Make recommendations
    print("\nStep 4: Making threshold recommendations...")
    recommended = recommend_thresholds(threshold_options, target_pairs=10)

    # Step 5: Create visualization plots
    print("\nStep 5: Creating visualization plots...")
    create_threshold_plots(successful_df)

    print(f"\n🎉 THRESHOLD DISCOVERY COMPLETED!")
    print(f"   • Analyzed {len(successful_df)} successful pairs")
    print(f"   • Generated threshold recommendations")
    print(f"   • Created performance distribution plots")
    print(f"   • Ready to use thresholds in optimal workflow!")


if __name__ == "__main__":
    main()
