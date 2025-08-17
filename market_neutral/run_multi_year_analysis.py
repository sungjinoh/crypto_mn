#!/usr/bin/env python3
"""
Multi-Year Cointegration Analysis and Backtesting
Demonstrates how to use the enhanced CointegrationFinder with multiple years
and then run backtests on the discovered pairs.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from enhanced_cointegration_finder import CointegrationFinder
from mean_reversion_backtest import MeanReversionBacktester
from mean_reversion_strategy import MeanReversionStrategy


def run_multi_year_cointegration_analysis(
    analysis_years: list = [2023, 2024],
    analysis_months=[1, 2, 3, 4, 5, 6],
    max_symbols: int = 30,
    save_results: bool = True,
):
    """
    Run cointegration analysis across multiple years.

    Args:
        analysis_years: Years to include in cointegration analysis
        analysis_months: Either list of months (same for all years) or
                        list of lists (different months for each year)
        max_symbols: Maximum number of symbols to analyze
        save_results: Whether to save results

    Returns:
        Dictionary with cointegration results
    """
    print("=" * 80)
    print("MULTI-YEAR COINTEGRATION ANALYSIS")
    print("=" * 80)

    print(f"📊 Analysis Configuration:")
    print(f"   • Years: {analysis_years}")
    print(f"   • Months: {analysis_months}")
    print(f"   • Max symbols: {max_symbols}")
    print(f"   • Save results: {save_results}")

    # Initialize the enhanced finder
    finder = CointegrationFinder(
        base_path="binance_futures_data",
        resample_interval="1H",  # 1-hour candles for good balance
        min_data_points=2000,  # Higher requirement for multi-year
        significance_level=0.05,
        n_jobs=-1,  # Use all CPU cores
    )

    print(f"\n🔧 Finder Configuration:")
    print(f"   • Resample interval: {finder.resample_interval}")
    print(f"   • Min data points: {finder.min_data_points}")
    print(f"   • Significance level: {finder.significance_level}")
    print(f"   • Parallel jobs: {finder.n_jobs}")

    # Run the analysis
    print(f"\n🔍 Running cointegration analysis...")
    results = finder.find_all_cointegrated_pairs(
        years=analysis_years,
        months=analysis_months,
        max_symbols=max_symbols,
        use_parallel=True,
    )

    # Print summary
    metadata = results["metadata"]
    print(f"\n📈 Analysis Results:")
    print(
        f"   • Data period: {metadata['data_years']} (months {metadata['data_months']})"
    )
    print(f"   • Symbols analyzed: {metadata['total_symbols']}")
    print(f"   • Symbols with data: {metadata['symbols_with_data']}")
    print(f"   • Pairs tested: {metadata['total_pairs_tested']}")
    print(f"   • Cointegrated pairs found: {metadata['cointegrated_pairs_found']}")

    if metadata["total_pairs_tested"] > 0:
        success_rate = (
            metadata["cointegrated_pairs_found"] / metadata["total_pairs_tested"]
        )
        print(f"   • Success rate: {success_rate:.2%}")

    # Show top pairs
    if results["cointegrated_pairs"]:
        print(f"\n🏆 Top 10 Cointegrated Pairs:")
        print(
            f"{'Rank':<5} {'Pair':<20} {'P-Value':<10} {'Hedge Ratio':<12} {'Correlation':<12} {'Data Points':<12}"
        )
        print("-" * 85)

        for i, pair in enumerate(results["cointegrated_pairs"][:10], 1):
            pair_name = f"{pair['symbol1']}-{pair['symbol2']}"
            print(
                f"{i:<5} {pair_name:<20} {pair['p_value']:<10.6f} "
                f"{pair['hedge_ratio']:<12.4f} {pair['correlation']:<12.4f} {pair['data_points']:<12}"
            )

    # Save results if requested
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"multi_year_cointegration_{timestamp}"

        finder.save_results(
            results, output_dir=output_dir, formats=["json", "csv", "pickle"]
        )
        print(f"\n💾 Results saved to: {output_dir}/")

    return results


def run_backtest_on_multi_year_pairs(
    cointegration_results: dict,
    backtest_years: list = [2024],
    backtest_months: list = [7, 8, 9, 10, 11, 12],
    n_pairs: int = 10,
    strategy_params: dict = None,
):
    """
    Run backtests on pairs discovered from multi-year cointegration analysis.

    Args:
        cointegration_results: Results from multi-year cointegration analysis
        backtest_years: Years to use for backtesting (out-of-sample)
        backtest_months: Months to use for backtesting
        n_pairs: Number of top pairs to backtest
        strategy_params: Strategy parameters

    Returns:
        DataFrame with backtest results
    """
    print("\n" + "=" * 80)
    print("BACKTESTING ON MULTI-YEAR DISCOVERED PAIRS")
    print("=" * 80)

    if strategy_params is None:
        strategy_params = {
            "lookback_period": 48,  # 48 hours for 1H data
            "entry_threshold": 1.5,
            "exit_threshold": 0.5,
            "stop_loss_threshold": 3.0,
        }

    print(f"🎯 Backtest Configuration:")
    print(f"   • Backtest period: {backtest_years} (months {backtest_months})")
    print(f"   • Number of pairs: {n_pairs}")
    print(f"   • Strategy params: {strategy_params}")

    # Get top pairs from cointegration results
    top_pairs = cointegration_results["cointegrated_pairs"][:n_pairs]

    if not top_pairs:
        print("❌ No cointegrated pairs found!")
        return pd.DataFrame()

    print(f"\n📋 Selected pairs for backtesting:")
    for i, pair in enumerate(top_pairs, 1):
        print(
            f"  {i:2d}. {pair['symbol1']}-{pair['symbol2']} "
            f"(p-value: {pair['p_value']:.6f})"
        )

    # Initialize backtester
    backtester = MeanReversionBacktester(
        base_path="binance_futures_data",
        resample_timeframe="1H",  # Match cointegration analysis
        transaction_cost=0.001,  # 0.1% transaction cost
        position_size=0.4,  # 40% of capital per trade
        save_plots=True,
        plots_dir="multi_year_backtest_plots",
    )

    # Enable trade logging
    backtester.save_trades = True
    backtester.trades_dir = "multi_year_backtest_trades"

    # Create strategy
    strategy = MeanReversionStrategy(**strategy_params)

    print(f"\n🔧 Backtester Configuration:")
    print(f"   • Timeframe: {backtester.resample_timeframe}")
    print(f"   • Transaction cost: {backtester.backtester.transaction_cost:.4f}")
    print(f"   • Position size: {backtester.backtester.position_size:.2f}")
    print(f"   • Save plots: {backtester.save_plots}")
    print(f"   • Save trades: {backtester.save_trades}")

    # Run backtests
    print(f"\n🚀 Running backtests...")
    all_results = []
    successful_count = 0

    for i, pair in enumerate(top_pairs, 1):
        symbol1 = pair["symbol1"]
        symbol2 = pair["symbol2"]

        print(f"\n[{i:2d}/{len(top_pairs)}] Testing {symbol1} - {symbol2}")
        print(f"   Cointegration p-value: {pair['p_value']:.6f}")
        print(f"   Hedge ratio: {pair['hedge_ratio']:.4f}")
        print(f"   Correlation: {pair['correlation']:.4f}")

        try:
            # Load backtest data (out-of-sample)
            df1_list = []
            df2_list = []

            for year in backtest_years:
                for month in backtest_months:
                    try:
                        df1_month, df2_month = backtester.load_pair_data(
                            symbol1, symbol2, year, [month]
                        )
                        if df1_month is not None and df2_month is not None:
                            df1_list.append(df1_month)
                            df2_list.append(df2_month)
                    except:
                        continue

            if not df1_list or not df2_list:
                print(f"   ❌ No backtest data available")
                continue

            # Combine data
            df1 = pd.concat(df1_list, axis=0).sort_index()
            df2 = pd.concat(df2_list, axis=0).sort_index()

            # Remove duplicates
            df1 = df1[~df1.index.duplicated(keep="first")]
            df2 = df2[~df2.index.duplicated(keep="first")]

            print(f"   Data loaded: {len(df1):,} bars")

            # Run backtest
            backtest_result = backtester.run_backtest_with_strategy(
                symbol1_data=df1,
                symbol2_data=df2,
                symbol1=symbol1,
                symbol2=symbol2,
                strategy=strategy,
                save_plot=True,
                plot_dir="multi_year_backtest_plots",
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
                "cointegration_data_points": pair["data_points"],
                "backtest_data_points": len(df1),
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

            result = {
                "rank": i,
                "symbol1": symbol1,
                "symbol2": symbol2,
                "cointegration_p_value": pair["p_value"],
                "error": str(e),
                "success": False,
            }
            all_results.append(result)

    # Convert to DataFrame and analyze results
    results_df = pd.DataFrame(all_results)

    print(f"\n{'=' * 80}")
    print("BACKTEST SUMMARY")
    print("=" * 80)
    print(f"Total pairs tested: {len(top_pairs)}")
    print(f"Successful backtests: {successful_count}")
    print(f"Failed backtests: {len(top_pairs) - successful_count}")

    if successful_count > 0:
        successful_results = results_df[results_df["success"] == True].copy()
        successful_results = successful_results.sort_values(
            "sharpe_ratio", ascending=False
        )

        print(f"\n🏆 TOP PERFORMING PAIRS:")
        print(
            f"{'Rank':<5} {'Pair':<20} {'Sharpe':<8} {'Return':<8} {'Drawdown':<10} {'Trades':<7}"
        )
        print("-" * 70)

        for i, (_, row) in enumerate(successful_results.head(5).iterrows(), 1):
            pair_name = f"{row['symbol1']}-{row['symbol2']}"
            print(
                f"{i:<5} {pair_name:<20} {row['sharpe_ratio']:<8.3f} "
                f"{row['total_return']:<8.2%} {row['max_drawdown']:<10.2%} {row['num_trades']:<7}"
            )

        # Statistics
        print(f"\n📊 OVERALL STATISTICS:")
        print(
            f"   • Average Sharpe Ratio: {successful_results['sharpe_ratio'].mean():.3f}"
        )
        print(
            f"   • Average Total Return: {successful_results['total_return'].mean():.2%}"
        )
        print(
            f"   • Average Max Drawdown: {successful_results['max_drawdown'].mean():.2%}"
        )
        print(f"   • Average Win Rate: {successful_results['win_rate'].mean():.2%}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_year_backtest_results_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to: {filename}")

    return results_df


def main():
    """Main function to run the complete multi-year analysis and backtesting."""

    print("🚀 MULTI-YEAR COINTEGRATION ANALYSIS & BACKTESTING")
    print("This script demonstrates the enhanced workflow:")
    print("1. Find cointegrated pairs using multiple years of data")
    print("2. Backtest the discovered pairs on out-of-sample data")
    print()

    # Step 1: Multi-year cointegration analysis
    print("STEP 1: Multi-year cointegration analysis")
    cointegration_results = run_multi_year_cointegration_analysis(
        analysis_years=[2023, 2024],  # Use 2023-2024 for cointegration
        analysis_months=[
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [1, 2, 3, 4, 5, 6],
        ],  # All months for 2023, Jan-June for 2024
        max_symbols=25,  # Reasonable number for demo
        save_results=True,
    )

    if not cointegration_results["cointegrated_pairs"]:
        print("❌ No cointegrated pairs found. Cannot proceed with backtesting.")
        return

    # Step 2: Backtest on out-of-sample data
    print("\nSTEP 2: Backtesting on out-of-sample data")
    backtest_results = run_backtest_on_multi_year_pairs(
        cointegration_results=cointegration_results,
        backtest_years=[2024],  # Use 2024 for backtesting
        backtest_months=[7, 8, 9, 10, 11, 12],  # Second half of 2024
        n_pairs=10,  # Test top 10 pairs
        strategy_params={
            "lookback_period": 48,  # 48 hours (2 days for 1H data)
            "entry_threshold": 1.5,
            "exit_threshold": 0.5,
            "stop_loss_threshold": 3.0,
        },
    )

    # Final summary
    print(f"\n{'=' * 80}")
    print("COMPLETE ANALYSIS SUMMARY")
    print("=" * 80)

    coint_meta = cointegration_results["metadata"]
    print(f"📊 Cointegration Analysis:")
    print(
        f"   • Analysis period: {coint_meta['data_years']} (months {coint_meta['data_months']})"
    )
    print(f"   • Pairs found: {coint_meta['cointegrated_pairs_found']}")

    if len(backtest_results) > 0:
        successful_backtests = backtest_results[backtest_results["success"] == True]
        if len(successful_backtests) > 0:
            best_pair = successful_backtests.loc[
                successful_backtests["sharpe_ratio"].idxmax()
            ]
            print(f"\n🏆 Best Performing Pair:")
            print(f"   • Pair: {best_pair['symbol1']}-{best_pair['symbol2']}")
            print(
                f"   • Cointegration p-value: {best_pair['cointegration_p_value']:.6f}"
            )
            print(f"   • Backtest Sharpe: {best_pair['sharpe_ratio']:.3f}")
            print(f"   • Backtest Return: {best_pair['total_return']:.2%}")
            print(f"   • Max Drawdown: {best_pair['max_drawdown']:.2%}")

    print(f"\n✅ Analysis complete! Check the output files for detailed results.")

    return cointegration_results, backtest_results


if __name__ == "__main__":
    try:
        coint_results, backtest_results = main()
        print("\n🎉 SUCCESS: Multi-year analysis completed successfully!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(
            "Make sure you have the required data files in 'binance_futures_data' directory"
        )
        sys.exit(1)
