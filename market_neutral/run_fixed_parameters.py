#!/usr/bin/env python3
"""
Run Fixed Parameters on Multiple Pairs
Test specific parameter settings across top pairs without optimization.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mean_reversion_backtest import MeanReversionBacktester
from mean_reversion_strategy import MeanReversionStrategy
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def run_fixed_parameters_backtest(
    fixed_params: dict,
    n_pairs: int = 20,
    test_years: list = [2024],
    test_months: list = [4, 5, 6],
    save_results: bool = True,
    save_plots: bool = True,
):
    """
    Run backtests with fixed parameters on multiple pairs

    Args:
        fixed_params: Dictionary with strategy parameters
        n_pairs: Number of top pairs to test
        test_years: List of years for backtesting (e.g., [2024])
        test_months: List of months for backtesting (e.g., [4, 5, 6])
        save_results: Whether to save results to CSV
        save_plots: Whether to save plot images
    """

    print("=" * 80)
    print("FIXED PARAMETERS BACKTESTING")
    print("=" * 80)

    print(f"🔧 Fixed Parameters:")
    print(f"   • Lookback Period: {fixed_params['lookback_period']}")
    print(f"   • Entry Threshold: {fixed_params['entry_threshold']}")
    print(f"   • Exit Threshold: {fixed_params['exit_threshold']}")
    print(f"   • Stop Loss Threshold: {fixed_params['stop_loss_threshold']}")

    print(f"\n📊 Test Configuration:")
    print(f"   • Number of pairs: {n_pairs}")
    print(f"   • Test period: {test_years} months {test_months}")
    print(f"   • Save results: {save_results}")
    print(f"   • Save plots: {save_plots}")

    # Initialize backtester
    backtester = MeanReversionBacktester(
        base_path="binance_futures_data",
        results_dir="cointegration_results",
        resample_timeframe="4H",  # 15-minute bars for good balance
        transaction_cost=0.0008,  # 0.1% transaction cost
        position_size=0.5,  # 50% of capital per trade
        save_plots=save_plots,
        plots_dir="fixed_params_plots",
    )

    # Enable detailed trade logging
    backtester.save_trades = True
    backtester.trades_dir = "fixed_params_trades"

    print(f"\n🔧 Backtesting Configuration:")
    print(f"   • Timeframe: {backtester.resample_timeframe or '1T (1-minute)'}")
    print(f"   • Initial Capital: ${backtester.backtester.initial_capital:,.0f}")
    print(f"   • Transaction Cost: {backtester.backtester.transaction_cost:.4f}")
    print(f"   • Position Size: {backtester.backtester.position_size:.2f}")
    print(f"   • Save Trade Details: {backtester.save_trades}")
    print(f"   • Trade Logs Directory: {backtester.trades_dir}")

    # Load cointegration results and get top pairs
    print(f"\n1. Loading cointegration results...")
    coint_results = backtester.load_cointegration_results()

    print(f"\n2. Filtering top {n_pairs} pairs...")
    top_pairs = backtester.filter_top_pairs(
        coint_results,
        n_pairs=n_pairs,
        # max_p_value=0.08,
        # min_correlation=0.6,
        # max_half_life=90,
    )

    if not top_pairs:
        print("❌ No suitable pairs found!")
        return pd.DataFrame()

    print(f"Found {len(top_pairs)} pairs meeting criteria:")
    for i, pair in enumerate(top_pairs[:10], 1):  # Show first 10
        print(
            f"  {i:2d}. {pair['symbol1']:10s} - {pair['symbol2']:10s} | "
            f"p-value: {pair['p_value']:.6f} | "
            f"corr: {pair['correlation']:.3f}"
        )

    if len(top_pairs) > 10:
        print(f"  ... and {len(top_pairs) - 10} more pairs")

    # Create strategy with fixed parameters
    strategy = MeanReversionStrategy(
        lookback_period=fixed_params["lookback_period"],
        entry_threshold=fixed_params["entry_threshold"],
        exit_threshold=fixed_params["exit_threshold"],
        stop_loss_threshold=fixed_params["stop_loss_threshold"],
    )

    print(f"\n3. Running backtests with fixed parameters...")

    # Run backtests on all pairs
    all_results = []
    successful_count = 0

    for i, pair in enumerate(top_pairs, 1):
        symbol1 = pair["symbol1"]
        symbol2 = pair["symbol2"]

        print(f"\n[{i:2d}/{len(top_pairs)}] Testing {symbol1} - {symbol2}")
        print(f"   Cointegration p-value: {pair['p_value']:.6f}")
        print(f"   Correlation: {pair['correlation']:.4f}")

        try:
            # Load data
            df1, df2 = backtester.load_pair_data(
                symbol1, symbol2, test_years, test_months
            )
            print(f"   Data loaded: {len(df1):,} bars")

            # Run backtest with plotting enabled
            backtest_result = backtester.run_backtest_with_strategy(
                symbol1_data=df1,
                symbol2_data=df2,
                symbol1=symbol1,
                symbol2=symbol2,
                strategy=strategy,
                save_plot=save_plots,  # Use the parameter from function
                plot_dir="fixed_params_trades",
            )

            # Extract metrics
            metrics = backtest_result.metrics

            result = {
                "rank": i,
                "symbol1": symbol1,
                "symbol2": symbol2,
                "cointegration_p_value": pair["p_value"],
                "hedge_ratio": pair["hedge_ratio"],
                "correlation": pair["correlation"],
                "half_life": pair["half_life"],
                "lookback_period": fixed_params["lookback_period"],
                "entry_threshold": fixed_params["entry_threshold"],
                "exit_threshold": fixed_params["exit_threshold"],
                "stop_loss_threshold": fixed_params["stop_loss_threshold"],
                "sharpe_ratio": metrics.get("Sharpe Ratio", 0),
                "total_return": metrics.get("Total Return", 0),
                "annualized_return": metrics.get("Annualized Return", 0),
                "volatility": metrics.get("Volatility", 0),
                "max_drawdown": metrics.get("Max Drawdown", 0),
                "calmar_ratio": metrics.get("Calmar Ratio", 0),
                "win_rate": metrics.get("Win Rate", 0),
                "num_trades": len(backtest_result.trades),
                "avg_trade_pnl": (
                    np.mean(
                        [t.pnl for t in backtest_result.trades if t.pnl is not None]
                    )
                    if backtest_result.trades
                    else 0
                ),
                "success": True,
            }

            all_results.append(result)
            successful_count += 1

            # Print summary
            print(f"   ✅ Results:")
            print(f"      • Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            print(f"      • Total Return: {result['total_return']:.2%}")
            print(f"      • Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"      • Win Rate: {result['win_rate']:.2%}")
            print(f"      • Trades: {result['num_trades']}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

            # Add failed result
            result = {
                "rank": i,
                "symbol1": symbol1,
                "symbol2": symbol2,
                "cointegration_p_value": pair["p_value"],
                "correlation": pair["correlation"],
                "error": str(e),
                "success": False,
            }
            all_results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Analysis and summary
    print(f"\n{'=' * 80}")
    print("BACKTEST SUMMARY")
    print("=" * 80)
    print(f"Total pairs tested: {len(top_pairs)}")
    print(f"Successful backtests: {successful_count}")
    print(f"Failed backtests: {len(top_pairs) - successful_count}")

    if successful_count > 0:
        successful_results = results_df[results_df["success"] == True].copy()

        # Sort by Sharpe ratio
        successful_results = successful_results.sort_values(
            "sharpe_ratio", ascending=False
        )

        print(f"\n🏆 TOP 10 PERFORMING PAIRS:")
        print(
            f"{'Rank':<5} {'Pair':<20} {'Sharpe':<8} {'Return':<8} {'Drawdown':<10} {'Trades':<7}"
        )
        print("-" * 70)

        for i, (_, row) in enumerate(successful_results.head(10).iterrows(), 1):
            pair_name = f"{row['symbol1']}-{row['symbol2']}"
            print(
                f"{i:<5} {pair_name:<20} {row['sharpe_ratio']:<8.3f} "
                f"{row['total_return']:<8.2%} {row['max_drawdown']:<10.2%} {row['num_trades']:<7}"
            )

        # Overall statistics
        print(f"\n📊 OVERALL STATISTICS:")
        print(
            f"   • Average Sharpe Ratio: {successful_results['sharpe_ratio'].mean():.3f}"
        )
        print(
            f"   • Median Sharpe Ratio: {successful_results['sharpe_ratio'].median():.3f}"
        )
        print(
            f"   • Average Total Return: {successful_results['total_return'].mean():.2%}"
        )
        print(
            f"   • Average Max Drawdown: {successful_results['max_drawdown'].mean():.2%}"
        )
        print(f"   • Average Win Rate: {successful_results['win_rate'].mean():.2%}")
        print(f"   • Total Trades: {successful_results['num_trades'].sum()}")

        # Performance categories
        profitable_pairs = successful_results[successful_results["total_return"] > 0]
        high_sharpe_pairs = successful_results[successful_results["sharpe_ratio"] > 1.0]
        low_drawdown_pairs = successful_results[
            successful_results["max_drawdown"] < 0.1
        ]  # <10%

        print(f"\n📈 PERFORMANCE CATEGORIES:")
        print(
            f"   • Profitable pairs (>0% return): {len(profitable_pairs)}/{successful_count} ({len(profitable_pairs)/successful_count:.1%})"
        )
        print(
            f"   • High Sharpe pairs (>1.0): {len(high_sharpe_pairs)}/{successful_count} ({len(high_sharpe_pairs)/successful_count:.1%})"
        )
        print(
            f"   • Low drawdown pairs (<10%): {len(low_drawdown_pairs)}/{successful_count} ({len(low_drawdown_pairs)/successful_count:.1%})"
        )

        # Recommended pairs
        recommended = profitable_pairs[
            (profitable_pairs["sharpe_ratio"] > 0.8)
            & (profitable_pairs["total_return"] > 0.03)
            & (profitable_pairs["num_trades"] >= 5)
        ]

        if len(recommended) > 0:
            print(
                f"\n✅ RECOMMENDED PAIRS FOR LIVE TRADING ({len(recommended)} pairs):"
            )
            for _, row in recommended.iterrows():
                print(
                    f"   • {row['symbol1']}-{row['symbol2']}: "
                    f"Sharpe={row['sharpe_ratio']:.3f}, "
                    f"Return={row['total_return']:.2%}, "
                    f"Trades={row['num_trades']}"
                )
        else:
            print(f"\n⚠️ No pairs meet the recommended criteria with these parameters")
            print(f"   Consider adjusting parameters or relaxing criteria")

    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fixed_params_results_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to: {filename}")

    if save_plots:
        print(f"📊 Plots saved to: fixed_params_plots/")

    if backtester.save_trades:
        print(f"📝 Trade details saved to: {backtester.trades_dir}/")
        print(f"   • Individual CSV files for each successful pair")
        print(f"   • Complete transaction history with entry/exit details")
        print(f"   • Performance metrics and trade analysis")

    return results_df


def main():
    """Main function to run fixed parameters backtest"""

    # Your specific parameters
    fixed_params = {
        "lookback_period": 60,
        "entry_threshold": 2.5,
        "exit_threshold": 0.5,
        "stop_loss_threshold": 3.5,
    }

    print("🚀 RUNNING FIXED PARAMETERS BACKTEST")
    print(f"Parameters: {fixed_params}")

    # Run backtest
    results = run_fixed_parameters_backtest(
        fixed_params=fixed_params,
        n_pairs=100,  # Test top 20 pairs
        test_years=[2023, 2024],
        test_months=[
            [6, 7, 8, 9, 10, 11, 12],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ],  # Out-of-sample testing
        save_results=True,
        save_plots=True,
    )

    print(f"\n🎯 ANALYSIS COMPLETE!")
    print(f"   • Tested {len(results)} pairs")
    print(f"   • Results saved to CSV file")
    print(f"   • Plots saved for successful pairs")

    if len(results) > 0:
        successful = results[results["success"] == True]
        if len(successful) > 0:
            best_pair = successful.loc[successful["sharpe_ratio"].idxmax()]
            print(f"\n🏆 BEST PERFORMING PAIR:")
            print(f"   • Pair: {best_pair['symbol1']}-{best_pair['symbol2']}")
            print(f"   • Sharpe Ratio: {best_pair['sharpe_ratio']:.3f}")
            print(f"   • Total Return: {best_pair['total_return']:.2%}")
            print(f"   • Max Drawdown: {best_pair['max_drawdown']:.2%}")

    return results


if __name__ == "__main__":
    results = main()
