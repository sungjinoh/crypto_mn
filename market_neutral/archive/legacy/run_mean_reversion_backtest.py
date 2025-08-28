#!/usr/bin/env python3
"""
Quick Mean Reversion Backtesting Script
Loads cointegration results and runs backtests on top pairs.
"""

from mean_reversion_backtest import MeanReversionBacktester
import warnings

warnings.filterwarnings("ignore")


def main():
    print("=" * 80)
    print("MEAN REVERSION BACKTESTING - QUICK RUN")
    print("=" * 80)

    # Initialize backtester with options
    backtester = MeanReversionBacktester(
        base_path="binance_futures_data",
        results_dir="cointegration_results",
        resample_timeframe="30T",  # Use 15-minute bars for faster backtesting
        save_plots=True,  # Save plot images
        plots_dir="quick_backtest_plots",
    )

    # Configuration
    config = {
        "n_pairs": 5,  # Start with top 5 pairs for quick testing
        "test_year": 2024,
        "test_months": [
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
        ],  # Out-of-sample testing (Apr-Jun)
        "optimize_params": False,  # Skip optimization for speed
        "save_results": True,
    }

    print(f"\nConfiguration:")
    print(f"  Testing top {config['n_pairs']} pairs")
    print(f"  Test period: {config['test_year']} months {config['test_months']}")
    print(f"  Timeframe: {backtester.resample_timeframe or '1T (1-minute)'}")
    print(f"  Save plots: {backtester.save_plots}")
    print(
        f"  Parameter optimization: {'Yes' if config['optimize_params'] else 'No (using defaults)'}"
    )

    # Run backtests
    results = backtester.run_all_backtests(**config)

    # Quick summary
    if len(results) > 0:
        successful = results[results["success"] == True]
        if len(successful) > 0:
            print("\n" + "=" * 80)
            print("QUICK SUMMARY")
            print("=" * 80)
            print(f"✅ Successful backtests: {len(successful)}/{len(results)}")

            # Extract and show best performer
            best_idx = None
            best_sharpe = -float("inf")
            for idx, row in successful.iterrows():
                if isinstance(row["metrics"], dict):
                    sharpe = row["metrics"].get("Sharpe Ratio", -float("inf"))
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_idx = idx

            if best_idx is not None:
                best = successful.loc[best_idx]
                print(f"\n🏆 Best Performing Pair:")
                print(f"  Pair: {best['symbol1']} - {best['symbol2']}")
                if isinstance(best["metrics"], dict):
                    print(
                        f"  Total Return: {best['metrics'].get('Total Return', 0):.2%}"
                    )
                    print(
                        f"  Sharpe Ratio: {best['metrics'].get('Sharpe Ratio', 0):.3f}"
                    )
                    print(
                        f"  Max Drawdown: {best['metrics'].get('Max Drawdown', 0):.2%}"
                    )
                    print(f"  Win Rate: {best['metrics'].get('Win Rate', 0):.2%}")

            print("\n💡 Next Steps:")
            print("  1. Review the detailed results in the CSV file")
            print("  2. Run with optimize_params=True for better parameters")
            print("  3. Test more pairs by increasing n_pairs")
            print("  4. Consider different test periods for validation")

    return results


if __name__ == "__main__":
    results = main()
