#!/usr/bin/env python3
"""
Quick Start: Optimized Backtesting
Simple script to get you started with the optimal approach.
"""

from mean_reversion_backtest import MeanReversionBacktester
import warnings

warnings.filterwarnings("ignore")


def quick_optimized_backtest(n_pairs: int = 5):
    """
    Quick optimized backtesting - the right way to do it
    """
    print("🚀 QUICK START: OPTIMIZED BACKTESTING")
    print("=" * 60)

    # Initialize backtester with good default settings
    backtester = MeanReversionBacktester(
        base_path="binance_futures_data",
        results_dir="cointegration_results",
        resample_timeframe="15T",  # 15-minute bars for good balance
        transaction_cost=0.001,  # 0.1% transaction cost
        position_size=0.5,  # 50% of capital per trade
        save_plots=True,  # Save plots for review
        plots_dir="quick_optimized_plots",
    )

    print(f"📊 Configuration:")
    print(f"   • Testing top {n_pairs} pairs")
    print(f"   • Using 15-minute timeframe")
    print(f"   • Per-pair parameter optimization enabled")
    print(f"   • Plots will be saved to quick_optimized_plots/")

    # Run backtests with per-pair optimization
    results = backtester.run_all_backtests(
        n_pairs=n_pairs,
        test_year=2024,
        test_months=[4, 5, 6],  # Out-of-sample testing
        optimize_params=True,  # ✅ This is the key - per-pair optimization
        save_results=True,
    )

    # Show results
    if len(results) > 0:
        successful = results[results["success"] == True]

        if len(successful) > 0:
            print(
                f"\n✅ SUCCESS: {len(successful)}/{len(results)} pairs optimized successfully"
            )

            # Extract performance metrics
            for idx, row in successful.iterrows():
                if isinstance(row["metrics"], dict):
                    successful.loc[idx, "sharpe_ratio"] = row["metrics"].get(
                        "Sharpe Ratio", 0
                    )
                    successful.loc[idx, "total_return"] = row["metrics"].get(
                        "Total Return", 0
                    )
                    successful.loc[idx, "max_drawdown"] = row["metrics"].get(
                        "Max Drawdown", 0
                    )
                    successful.loc[idx, "win_rate"] = row["metrics"].get("Win Rate", 0)

            # Sort by Sharpe ratio
            successful = successful.sort_values("sharpe_ratio", ascending=False)

            print(f"\n🏆 TOP PERFORMING PAIRS:")
            print(
                f"{'Rank':<5} {'Pair':<20} {'Sharpe':<8} {'Return':<8} {'Drawdown':<10}"
            )
            print("-" * 60)

            for i, (_, row) in enumerate(successful.head().iterrows(), 1):
                pair_name = f"{row['symbol1']}-{row['symbol2']}"
                sharpe = row.get("sharpe_ratio", 0)
                ret = row.get("total_return", 0)
                dd = row.get("max_drawdown", 0)

                print(f"{i:<5} {pair_name:<20} {sharpe:<8.3f} {ret:<8.2%} {dd:<10.2%}")

            # Show optimal parameters for top pair
            if len(successful) > 0:
                top_pair = successful.iloc[0]
                if isinstance(top_pair["strategy_params"], dict):
                    params = top_pair["strategy_params"]
                    print(
                        f"\n⚙️ OPTIMAL PARAMETERS FOR TOP PAIR ({top_pair['symbol1']}-{top_pair['symbol2']}):"
                    )
                    print(
                        f"   • Lookback Period: {params.get('lookback_period', 'N/A')}"
                    )
                    print(
                        f"   • Entry Threshold: {params.get('entry_threshold', 'N/A'):.2f}"
                    )
                    print(
                        f"   • Exit Threshold: {params.get('exit_threshold', 'N/A'):.2f}"
                    )
                    print(
                        f"   • Stop Loss: {params.get('stop_loss_threshold', 'N/A'):.2f}"
                    )

            # Recommendations
            profitable_pairs = successful[
                successful["total_return"] > 0.05
            ]  # >5% return
            stable_pairs = successful[successful["sharpe_ratio"] > 1.0]  # Sharpe > 1

            print(f"\n💡 RECOMMENDATIONS:")
            print(f"   • Profitable pairs (>5% return): {len(profitable_pairs)}")
            print(f"   • Stable pairs (Sharpe >1.0): {len(stable_pairs)}")

            if len(profitable_pairs) > 0 and len(stable_pairs) > 0:
                recommended = profitable_pairs[
                    profitable_pairs.index.isin(stable_pairs.index)
                ]
                if len(recommended) > 0:
                    print(
                        f"   • ✅ RECOMMENDED FOR LIVE TRADING: {len(recommended)} pairs"
                    )
                    for _, row in recommended.iterrows():
                        print(f"     - {row['symbol1']}-{row['symbol2']}")
                else:
                    print(
                        f"   • ⚠️ No pairs meet both criteria - consider relaxing thresholds"
                    )
            else:
                print(f"   • ⚠️ Consider adjusting strategy or market conditions")

        else:
            print(f"❌ No successful backtests. Check your data and parameters.")

    else:
        print(f"❌ No results generated. Check your cointegration data.")

    print(f"\n📁 FILES GENERATED:")
    print(f"   • CSV results file with timestamp")
    print(f"   • Plot images in quick_optimized_plots/")
    print(f"   • Each pair has its own optimized parameters")

    return results


if __name__ == "__main__":
    print("This script demonstrates the CORRECT way to do parameter optimization:")
    print("✅ Each pair gets its own optimal parameters")
    print("✅ No assumption that one size fits all")
    print("✅ Better performance through customization")
    print()

    # Run quick optimized backtest
    results = quick_optimized_backtest(n_pairs=5)

    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Review the generated plots and CSV results")
    print(
        f"   2. For more comprehensive optimization, run: optimal_backtesting_workflow.py"
    )
    print(f"   3. For comparison with global parameters, run: approach_comparison.py")
    print(f"   4. Consider paper trading the top recommended pairs")
