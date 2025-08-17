#!/usr/bin/env python3
"""
Multi-Year Backtesting Script
Test parameters across multiple years and months for comprehensive analysis.
"""

from mean_reversion_backtest import MeanReversionBacktester
from mean_reversion_strategy import MeanReversionStrategy
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def run_multi_year_backtest(
    fixed_params: dict,
    n_pairs: int = 20,
    test_periods: list = None,
    save_results: bool = True,
    save_plots: bool = True,
):
    """
    Run backtests across multiple years and months

    Args:
        fixed_params: Dictionary with strategy parameters
        n_pairs: Number of top pairs to test
        test_periods: List of (year, months) tuples to test
        save_results: Whether to save results to CSV
        save_plots: Whether to save plot images
    """

    if test_periods is None:
        test_periods = [
            (2023, list(range(1, 13))),  # 2023: Jan-Dec
            (2024, list(range(1, 13))),  # 2024: Jan-Dec
            (2025, list(range(1, 8))),  # 2025: Jan-Jul
        ]

    print("=" * 80)
    print("MULTI-YEAR BACKTESTING ANALYSIS")
    print("=" * 80)

    print(f"🔧 Fixed Parameters:")
    for param, value in fixed_params.items():
        print(f"   • {param}: {value}")

    print(f"\n📊 Test Configuration:")
    print(f"   • Number of pairs: {n_pairs}")
    print(f"   • Test periods:")
    for year, months in test_periods:
        print(f"     - {year}: months {months[0]}-{months[-1]} ({len(months)} months)")

    # Initialize backtester
    backtester = MeanReversionBacktester(
        base_path="binance_futures_data",
        results_dir="cointegration_results",
        resample_timeframe="15T",  # 15-minute bars for good balance
        transaction_cost=0.001,
        position_size=0.5,
        save_plots=save_plots,
        plots_dir="multi_year_plots",
    )

    # Enable trade logging
    backtester.save_trades = True
    backtester.trades_dir = "multi_year_trades"

    print(f"\n🔧 Backtesting Configuration:")
    print(f"   • Timeframe: {backtester.resample_timeframe}")
    print(f"   • Initial Capital: ${backtester.backtester.initial_capital:,.0f}")
    print(f"   • Transaction Cost: {backtester.backtester.transaction_cost:.4f}")
    print(f"   • Position Size: {backtester.backtester.position_size:.2f}")
    print(f"   • Funding Costs: {backtester.backtester.include_funding_costs}")

    # Load cointegration results and get top pairs
    print(f"\n1. Loading cointegration results...")
    coint_results = backtester.load_cointegration_results()

    print(f"\n2. Filtering top {n_pairs} pairs...")
    top_pairs = backtester.filter_top_pairs(
        coint_results,
        n_pairs=n_pairs,
        max_p_value=0.08,
        min_correlation=0.6,
        max_half_life=90,
    )

    if not top_pairs:
        print("❌ No suitable pairs found!")
        return pd.DataFrame()

    print(f"Found {len(top_pairs)} pairs meeting criteria:")
    for i, pair in enumerate(top_pairs[:10], 1):  # Show first 10
        print(
            f"  {i:2d}. {pair['symbol1']:10s} - {pair['symbol2']:10s} | "
            f"p-value: {pair['p_value']:.6f}"
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

    print(f"\n3. Running multi-year backtests...")

    # Store all results
    all_results = []
    period_summaries = []

    # Test each time period
    for period_idx, (test_year, test_months) in enumerate(test_periods, 1):
        print(f"\n{'='*70}")
        print(
            f"TESTING PERIOD {period_idx}: {test_year} (Months {test_months[0]}-{test_months[-1]})"
        )
        print(f"{'='*70}")

        period_results = []
        successful_count = 0

        # Test each pair in this period
        for pair_idx, pair in enumerate(top_pairs, 1):
            symbol1 = pair["symbol1"]
            symbol2 = pair["symbol2"]

            print(
                f"\n[{pair_idx:2d}/{len(top_pairs)}] {symbol1}-{symbol2} | {test_year}"
            )

            try:
                # Load data for this period
                df1, df2 = backtester.load_pair_data(
                    symbol1, symbol2, test_year, test_months
                )

                if df1 is None or df2 is None or len(df1) < 100:
                    print(f"   ⚠️ Insufficient data for {test_year}")
                    continue

                print(f"   📊 Data loaded: {len(df1):,} bars")

                # Run backtest
                backtest_result = backtester.run_backtest_with_strategy(
                    symbol1_data=df1,
                    symbol2_data=df2,
                    symbol1=symbol1,
                    symbol2=symbol2,
                    strategy=strategy,
                    save_plot=save_plots,
                    plot_dir=f"multi_year_plots/{test_year}",
                )

                # Extract metrics
                metrics = backtest_result.metrics

                result = {
                    "period": f"{test_year}",
                    "year": test_year,
                    "months": f"{test_months[0]}-{test_months[-1]}",
                    "pair_rank": pair_idx,
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "cointegration_p_value": pair["p_value"],
                    "correlation": pair["correlation"],
                    # Strategy parameters
                    "lookback_period": fixed_params["lookback_period"],
                    "entry_threshold": fixed_params["entry_threshold"],
                    "exit_threshold": fixed_params["exit_threshold"],
                    "stop_loss_threshold": fixed_params["stop_loss_threshold"],
                    # Performance metrics
                    "sharpe_ratio": metrics.get("Sharpe Ratio", 0),
                    "total_return": metrics.get("Total Return", 0),
                    "annualized_return": metrics.get("Annualized Return", 0),
                    "volatility": metrics.get("Volatility", 0),
                    "max_drawdown": metrics.get("Max Drawdown", 0),
                    "calmar_ratio": metrics.get("Calmar Ratio", 0),
                    "win_rate": metrics.get("Win Rate", 0),
                    "num_trades": len(backtest_result.trades),
                    # Additional metrics
                    "data_points": len(df1),
                    "success": True,
                }

                # Calculate funding impact if trades exist
                if backtest_result.trades:
                    total_gross_pnl = sum(
                        trade.pnl or 0 for trade in backtest_result.trades
                    )
                    total_funding_cost = sum(
                        getattr(trade, "total_funding_cost", 0)
                        for trade in backtest_result.trades
                    )
                    total_net_pnl = sum(
                        getattr(trade, "net_pnl", trade.pnl or 0)
                        for trade in backtest_result.trades
                    )

                    result.update(
                        {
                            "gross_pnl": total_gross_pnl,
                            "funding_cost": total_funding_cost,
                            "net_pnl": total_net_pnl,
                            "funding_impact_pct": (
                                (total_funding_cost / abs(total_gross_pnl) * 100)
                                if abs(total_gross_pnl) > 0
                                else 0
                            ),
                        }
                    )
                else:
                    result.update(
                        {
                            "gross_pnl": 0,
                            "funding_cost": 0,
                            "net_pnl": 0,
                            "funding_impact_pct": 0,
                        }
                    )

                period_results.append(result)
                all_results.append(result)
                successful_count += 1

                # Print summary
                print(f"   ✅ Results:")
                print(f"      • Sharpe: {result['sharpe_ratio']:.3f}")
                print(f"      • Return: {result['total_return']:.2%}")
                print(f"      • Trades: {result['num_trades']}")
                if result["funding_cost"] != 0:
                    print(
                        f"      • Funding impact: {result['funding_impact_pct']:.1f}%"
                    )

            except Exception as e:
                print(f"   ❌ Error: {e}")

                # Add failed result
                result = {
                    "period": f"{test_year}",
                    "year": test_year,
                    "pair_rank": pair_idx,
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "error": str(e),
                    "success": False,
                }
                all_results.append(result)

        # Period summary
        if period_results:
            period_sharpe = np.mean([r["sharpe_ratio"] for r in period_results])
            period_return = np.mean([r["total_return"] for r in period_results])
            period_trades = sum([r["num_trades"] for r in period_results])
            profitable_pairs = sum(1 for r in period_results if r["total_return"] > 0)

            period_summary = {
                "period": f"{test_year}",
                "year": test_year,
                "successful_pairs": successful_count,
                "total_pairs": len(top_pairs),
                "avg_sharpe": period_sharpe,
                "avg_return": period_return,
                "total_trades": period_trades,
                "profitable_pairs": profitable_pairs,
                "success_rate": successful_count / len(top_pairs),
                "profitability_rate": (
                    profitable_pairs / successful_count if successful_count > 0 else 0
                ),
            }

            period_summaries.append(period_summary)

            print(f"\n📊 {test_year} PERIOD SUMMARY:")
            print(
                f"   • Successful pairs: {successful_count}/{len(top_pairs)} ({successful_count/len(top_pairs):.1%})"
            )
            print(f"   • Average Sharpe: {period_sharpe:.3f}")
            print(f"   • Average Return: {period_return:.2%}")
            print(f"   • Total Trades: {period_trades}")
            print(
                f"   • Profitable pairs: {profitable_pairs}/{successful_count} ({profitable_pairs/successful_count:.1%})"
            )

    # Convert to DataFrames
    results_df = pd.DataFrame(all_results)
    period_summary_df = pd.DataFrame(period_summaries)

    # Comprehensive analysis
    print(f"\n{'=' * 80}")
    print("MULTI-YEAR ANALYSIS SUMMARY")
    print("=" * 80)

    if len(results_df) > 0:
        successful_results = results_df[results_df["success"] == True].copy()

        if len(successful_results) > 0:
            print(f"📊 Overall Statistics:")
            print(f"   • Total successful backtests: {len(successful_results)}")
            print(
                f"   • Overall average Sharpe: {successful_results['sharpe_ratio'].mean():.3f}"
            )
            print(
                f"   • Overall average return: {successful_results['total_return'].mean():.2%}"
            )
            print(
                f"   • Total trades executed: {successful_results['num_trades'].sum():,}"
            )

            # Year-by-year comparison
            print(f"\n📈 YEAR-BY-YEAR PERFORMANCE:")
            print(
                f"{'Year':<6} {'Success Rate':<12} {'Avg Sharpe':<11} {'Avg Return':<11} {'Total Trades':<12}"
            )
            print("-" * 65)

            for _, period in period_summary_df.iterrows():
                print(
                    f"{int(period['year']):<6} {period['success_rate']:<12.1%} "
                    f"{period['avg_sharpe']:<11.3f} {period['avg_return']:<11.2%} "
                    f"{int(period['total_trades']):<12,}"
                )

            # Best performing pairs across all periods
            print(f"\n🏆 TOP 10 PAIRS ACROSS ALL PERIODS:")
            pair_performance = (
                successful_results.groupby(["symbol1", "symbol2"])
                .agg(
                    {
                        "sharpe_ratio": "mean",
                        "total_return": "mean",
                        "num_trades": "sum",
                        "success": "count",  # Number of periods tested
                    }
                )
                .round(3)
            )

            pair_performance = pair_performance.sort_values(
                "sharpe_ratio", ascending=False
            )

            print(
                f"{'Pair':<20} {'Periods':<8} {'Avg Sharpe':<11} {'Avg Return':<11} {'Total Trades':<12}"
            )
            print("-" * 75)

            for (s1, s2), row in pair_performance.head(10).iterrows():
                pair_name = f"{s1}-{s2}"
                print(
                    f"{pair_name:<20} {int(row['success']):<8} {row['sharpe_ratio']:<11.3f} "
                    f"{row['total_return']:<11.2%} {int(row['num_trades']):<12,}"
                )

            # Consistency analysis
            print(f"\n📊 CONSISTENCY ANALYSIS:")

            # Find pairs that performed well across multiple periods
            consistent_pairs = []
            for (s1, s2), group in successful_results.groupby(["symbol1", "symbol2"]):
                if len(group) >= 2:  # Tested in at least 2 periods
                    avg_sharpe = group["sharpe_ratio"].mean()
                    sharpe_std = group["sharpe_ratio"].std()
                    positive_periods = (group["total_return"] > 0).sum()

                    if (
                        avg_sharpe > 0.5 and positive_periods >= len(group) * 0.6
                    ):  # 60% of periods profitable
                        consistent_pairs.append(
                            {
                                "pair": f"{s1}-{s2}",
                                "periods_tested": len(group),
                                "avg_sharpe": avg_sharpe,
                                "sharpe_consistency": (
                                    1 / (1 + sharpe_std) if sharpe_std > 0 else 1
                                ),
                                "profitable_periods": positive_periods,
                                "profitability_rate": positive_periods / len(group),
                            }
                        )

            if consistent_pairs:
                consistent_df = pd.DataFrame(consistent_pairs)
                consistent_df = consistent_df.sort_values("avg_sharpe", ascending=False)

                print(f"   • Consistently profitable pairs: {len(consistent_pairs)}")
                print(f"\n🎯 MOST CONSISTENT PAIRS:")
                print(
                    f"{'Pair':<20} {'Periods':<8} {'Avg Sharpe':<11} {'Profit Rate':<11}"
                )
                print("-" * 60)

                for _, row in consistent_df.head(5).iterrows():
                    print(
                        f"{row['pair']:<20} {int(row['periods_tested']):<8} "
                        f"{row['avg_sharpe']:<11.3f} {row['profitability_rate']:<11.1%}"
                    )
            else:
                print(
                    f"   • No consistently profitable pairs found across multiple periods"
                )

            # Market condition analysis
            print(f"\n📈 MARKET CONDITION ANALYSIS:")

            # Compare performance by year
            year_performance = (
                successful_results.groupby("year")
                .agg(
                    {
                        "sharpe_ratio": ["mean", "std", "count"],
                        "total_return": ["mean", "std"],
                        "num_trades": "sum",
                    }
                )
                .round(3)
            )

            print(f"   Market conditions impact:")
            for year in [2023, 2024, 2025]:
                if year in successful_results["year"].values:
                    year_data = successful_results[successful_results["year"] == year]
                    avg_sharpe = year_data["sharpe_ratio"].mean()
                    avg_return = year_data["total_return"].mean()
                    success_count = len(year_data)

                    if avg_sharpe > 0.8:
                        condition = "🟢 Favorable"
                    elif avg_sharpe > 0.3:
                        condition = "🟡 Neutral"
                    else:
                        condition = "🔴 Challenging"

                    print(
                        f"   • {year}: {condition} (Sharpe: {avg_sharpe:.3f}, "
                        f"Return: {avg_return:.2%}, {success_count} pairs)"
                    )

            # Recommendations
            print(f"\n💡 RECOMMENDATIONS:")

            # Find best overall performers
            best_overall = (
                successful_results.groupby(["symbol1", "symbol2"])["sharpe_ratio"]
                .mean()
                .sort_values(ascending=False)
            )

            if len(best_overall) > 0:
                print(f"   🏆 Top 3 pairs for live trading:")
                for i, ((s1, s2), avg_sharpe) in enumerate(
                    best_overall.head(3).items(), 1
                ):
                    pair_data = successful_results[
                        (successful_results["symbol1"] == s1)
                        & (successful_results["symbol2"] == s2)
                    ]
                    periods_tested = len(pair_data)
                    avg_return = pair_data["total_return"].mean()

                    print(f"   {i}. {s1}-{s2}")
                    print(f"      • Average Sharpe: {avg_sharpe:.3f}")
                    print(f"      • Average Return: {avg_return:.2%}")
                    print(f"      • Tested in {periods_tested} periods")

            # Parameter effectiveness analysis
            print(f"\n⚙️ PARAMETER EFFECTIVENESS:")
            overall_avg_sharpe = successful_results["sharpe_ratio"].mean()
            overall_win_rate = (successful_results["total_return"] > 0).mean()

            print(f"   • Your parameters achieved:")
            print(f"     - Average Sharpe ratio: {overall_avg_sharpe:.3f}")
            print(f"     - Win rate: {overall_win_rate:.1%}")
            print(f"     - Tested across {len(test_periods)} different time periods")

            if overall_avg_sharpe > 1.0:
                print(f"   • ✅ Excellent parameters - suitable for live trading")
            elif overall_avg_sharpe > 0.5:
                print(
                    f"   • ✅ Good parameters - consider live trading with risk management"
                )
            elif overall_avg_sharpe > 0.0:
                print(f"   • ⚠️ Marginal parameters - consider optimization")
            else:
                print(f"   • ❌ Poor parameters - optimization needed")

    # Save comprehensive results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        detailed_filename = f"multi_year_backtest_detailed_{timestamp}.csv"
        results_df.to_csv(detailed_filename, index=False)

        # Save period summaries
        summary_filename = f"multi_year_backtest_summary_{timestamp}.csv"
        period_summary_df.to_csv(summary_filename, index=False)

        print(f"\n💾 Results saved:")
        print(f"   • Detailed results: {detailed_filename}")
        print(f"   • Period summaries: {summary_filename}")
        print(f"   • Trade logs: {backtester.trades_dir}/")
        if save_plots:
            print(f"   • Plots: multi_year_plots/")

    return results_df, period_summary_df


def analyze_multi_year_results(results_df, period_summary_df):
    """
    Analyze multi-year results for deeper insights
    """
    print(f"\n{'=' * 80}")
    print("DEEP ANALYSIS OF MULTI-YEAR RESULTS")
    print("=" * 80)

    if results_df.empty:
        print("❌ No results to analyze")
        return

    successful = results_df[results_df["success"] == True]

    if len(successful) == 0:
        print("❌ No successful backtests to analyze")
        return

    # Volatility analysis
    print(f"📊 VOLATILITY ANALYSIS:")
    sharpe_by_year = successful.groupby("year")["sharpe_ratio"].agg(
        ["mean", "std", "count"]
    )

    for year, stats in sharpe_by_year.iterrows():
        consistency_score = stats["mean"] / (
            stats["std"] + 0.001
        )  # Add small value to avoid division by zero
        print(
            f"   • {int(year)}: Avg Sharpe {stats['mean']:.3f} ± {stats['std']:.3f} "
            f"(Consistency: {consistency_score:.2f})"
        )

    # Market regime analysis
    print(f"\n📈 MARKET REGIME ANALYSIS:")

    # Compare early vs late periods
    early_results = successful[successful["year"] <= 2023]
    late_results = successful[successful["year"] >= 2024]

    if len(early_results) > 0 and len(late_results) > 0:
        early_sharpe = early_results["sharpe_ratio"].mean()
        late_sharpe = late_results["sharpe_ratio"].mean()

        print(f"   • Early period (≤2023): Avg Sharpe {early_sharpe:.3f}")
        print(f"   • Late period (≥2024): Avg Sharpe {late_sharpe:.3f}")
        print(f"   • Performance trend: {late_sharpe - early_sharpe:+.3f}")

        if late_sharpe > early_sharpe:
            print(f"   • ✅ Strategy improving over time")
        else:
            print(f"   • ⚠️ Strategy performance declining")

    # Risk analysis
    print(f"\n⚠️ RISK ANALYSIS:")
    max_drawdowns = successful["max_drawdown"].dropna()
    if len(max_drawdowns) > 0:
        avg_drawdown = max_drawdowns.mean()
        max_drawdown = max_drawdowns.max()

        print(f"   • Average max drawdown: {avg_drawdown:.2%}")
        print(f"   • Worst drawdown: {max_drawdown:.2%}")

        if max_drawdown > 0.2:
            print(f"   • ⚠️ High risk - consider position sizing adjustments")
        elif max_drawdown > 0.1:
            print(f"   • ⚠️ Moderate risk - monitor closely")
        else:
            print(f"   • ✅ Low risk profile")


def main():
    """Main function to run multi-year backtesting"""

    # Your specific parameters
    fixed_params = {
        "lookback_period": 60,
        "entry_threshold": 1.0,
        "exit_threshold": 0.5,
        "stop_loss_threshold": 3.5,
    }

    # Define test periods
    test_periods = [
        (2023, list(range(1, 13))),  # 2023: All months
        (2024, list(range(1, 13))),  # 2024: All months
        (2025, list(range(1, 8))),  # 2025: Jan-July
    ]

    print("🚀 MULTI-YEAR BACKTESTING")
    print(f"Parameters: {fixed_params}")
    print(f"Testing periods: 2023 (12 months), 2024 (12 months), 2025 (7 months)")
    print(f"Total test duration: 31 months")

    # Run comprehensive backtesting
    results_df, period_summary_df = run_multi_year_backtest(
        fixed_params=fixed_params,
        n_pairs=20,  # Test top 20 pairs
        test_periods=test_periods,
        save_results=True,
        save_plots=True,
    )

    # Deep analysis
    analyze_multi_year_results(results_df, period_summary_df)

    print(f"\n🎉 MULTI-YEAR ANALYSIS COMPLETED!")
    print(f"   • Tested across 31 months of data")
    print(f"   • Comprehensive performance analysis")
    print(f"   • Market regime identification")
    print(f"   • Risk and consistency metrics")
    print(f"   • Ready for live trading decisions")

    return results_df, period_summary_df


if __name__ == "__main__":
    results_df, summary_df = main()
