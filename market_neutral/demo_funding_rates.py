#!/usr/bin/env python3
"""
Demo: Funding Rate Integration
Shows how funding rates are calculated and included in trade analysis.
"""

from mean_reversion_backtest import MeanReversionBacktester
from mean_reversion_strategy import MeanReversionStrategy
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def demo_funding_rate_impact():
    """
    Demonstrate funding rate calculations and their impact on trading performance
    """
    print("💰 DEMO: FUNDING RATE INTEGRATION")
    print("=" * 70)

    print("🎯 This demo shows:")
    print("   • How funding rates are loaded and processed")
    print("   • Funding cost calculations for each trade")
    print("   • Impact of funding on overall profitability")
    print("   • Comparison of gross vs net PnL")

    # Test with and without funding costs
    configs = [
        {
            "name": "Without Funding Costs",
            "include_funding": False,
            "description": "Traditional backtesting (ignores funding)",
        },
        {
            "name": "With Funding Costs",
            "include_funding": True,
            "description": "Realistic futures trading (includes funding)",
        },
    ]

    results_comparison = []

    for config in configs:
        print(f"\n{'='*60}")
        print(f"TESTING: {config['name'].upper()}")
        print(f"{'='*60}")
        print(f"Description: {config['description']}")

        # Initialize backtester
        backtester = MeanReversionBacktester(
            base_path="binance_futures_data",
            results_dir="cointegration_results",
            resample_timeframe="1H",  # 1-hour bars for clearer funding impact
            save_plots=True,
            plots_dir=f"funding_demo_{config['name'].lower().replace(' ', '_')}",
        )

        # Configure funding cost calculation
        backtester.backtester.include_funding_costs = config["include_funding"]
        backtester.save_trades = True
        backtester.trades_dir = (
            f"funding_demo_trades_{config['name'].lower().replace(' ', '_')}"
        )

        print(f"⚙️ Configuration:")
        print(f"   • Include funding costs: {config['include_funding']}")
        print(f"   • Timeframe: 1-hour bars")
        print(f"   • Trade logs: {backtester.trades_dir}")

        try:
            # Load test pairs
            coint_results = backtester.load_cointegration_results()
            top_pairs = backtester.filter_top_pairs(coint_results, n_pairs=2)

            if not top_pairs:
                print("❌ No pairs found for demo")
                continue

            print(f"\n📊 Testing pairs:")
            for i, pair in enumerate(top_pairs, 1):
                print(f"   {i}. {pair['symbol1']} - {pair['symbol2']}")

            # Test strategy
            strategy = MeanReversionStrategy(
                lookback_period=60,
                entry_threshold=2.0,
                exit_threshold=0.0,
                stop_loss_threshold=3.0,
            )

            config_results = []

            for pair in top_pairs:
                symbol1, symbol2 = pair["symbol1"], pair["symbol2"]

                print(f"\n🔄 Running {symbol1}-{symbol2}...")

                try:
                    # Load data (includes funding rates if available)
                    df1, df2 = backtester.load_pair_data(symbol1, symbol2, 2024, [4, 5])

                    # Check if funding data is available
                    has_funding1 = "fundingRate" in df1.columns
                    has_funding2 = "fundingRate" in df2.columns

                    print(
                        f"   📊 Funding data available: {symbol1}={has_funding1}, {symbol2}={has_funding2}"
                    )

                    # Run backtest
                    result = backtester.run_backtest_with_strategy(
                        df1, df2, symbol1, symbol2, strategy
                    )

                    # Extract results
                    metrics = result.metrics
                    num_trades = len(result.trades)

                    # Calculate funding impact if applicable
                    total_funding_cost = 0
                    if config["include_funding"] and result.trades:
                        total_funding_cost = sum(
                            getattr(trade, "total_funding_cost", 0)
                            for trade in result.trades
                        )

                    pair_result = {
                        "pair": f"{symbol1}-{symbol2}",
                        "trades": num_trades,
                        "gross_pnl": sum(
                            trade.pnl for trade in result.trades if trade.pnl
                        ),
                        "funding_cost": total_funding_cost,
                        "net_pnl": sum(
                            getattr(trade, "net_pnl", trade.pnl)
                            for trade in result.trades
                            if trade.pnl or getattr(trade, "net_pnl", None)
                        ),
                        "sharpe": metrics.get("Sharpe Ratio", 0),
                        "return": metrics.get("Total Return", 0),
                        "win_rate": metrics.get("Win Rate", 0),
                        "has_funding_data": has_funding1 or has_funding2,
                    }

                    config_results.append(pair_result)

                    print(f"   ✅ Results:")
                    print(f"      • Trades: {num_trades}")
                    print(f"      • Gross PnL: ${pair_result['gross_pnl']:.2f}")
                    if config["include_funding"]:
                        print(f"      • Funding Cost: ${total_funding_cost:.2f}")
                        print(f"      • Net PnL: ${pair_result['net_pnl']:.2f}")
                        if pair_result["gross_pnl"] != 0:
                            impact = (
                                total_funding_cost / abs(pair_result["gross_pnl"])
                            ) * 100
                            print(f"      • Funding Impact: {impact:.1f}% of gross PnL")
                    print(f"      • Sharpe: {pair_result['sharpe']:.3f}")
                    print(f"      • Win Rate: {pair_result['win_rate']:.1%}")

                except Exception as e:
                    print(f"   ❌ Error: {e}")

            # Store results for comparison
            if config_results:
                total_gross_pnl = sum(r["gross_pnl"] for r in config_results)
                total_funding_cost = sum(r["funding_cost"] for r in config_results)
                total_net_pnl = sum(r["net_pnl"] for r in config_results)
                avg_sharpe = sum(r["sharpe"] for r in config_results) / len(
                    config_results
                )

                results_comparison.append(
                    {
                        "config": config["name"],
                        "include_funding": config["include_funding"],
                        "pairs_tested": len(config_results),
                        "total_gross_pnl": total_gross_pnl,
                        "total_funding_cost": total_funding_cost,
                        "total_net_pnl": total_net_pnl,
                        "avg_sharpe": avg_sharpe,
                        "funding_impact_pct": (
                            (total_funding_cost / abs(total_gross_pnl) * 100)
                            if total_gross_pnl != 0
                            else 0
                        ),
                    }
                )

                print(f"\n📊 {config['name']} Summary:")
                print(f"   • Total Gross PnL: ${total_gross_pnl:.2f}")
                if config["include_funding"]:
                    print(f"   • Total Funding Cost: ${total_funding_cost:.2f}")
                    print(f"   • Total Net PnL: ${total_net_pnl:.2f}")
                print(f"   • Average Sharpe: {avg_sharpe:.3f}")

        except Exception as e:
            print(f"❌ Configuration failed: {e}")

    # Final comparison
    if len(results_comparison) >= 2:
        print(f"\n{'='*70}")
        print("FUNDING RATE IMPACT ANALYSIS")
        print(f"{'='*70}")

        without_funding = next(
            (r for r in results_comparison if not r["include_funding"]), None
        )
        with_funding = next(
            (r for r in results_comparison if r["include_funding"]), None
        )

        if without_funding and with_funding:
            print(f"📊 Comparison Results:")
            print(
                f"{'Metric':<25} {'Without Funding':<15} {'With Funding':<15} {'Difference':<12}"
            )
            print("-" * 70)

            gross_diff = (
                with_funding["total_gross_pnl"] - without_funding["total_gross_pnl"]
            )
            net_diff = (
                with_funding["total_net_pnl"] - without_funding["total_gross_pnl"]
            )
            sharpe_diff = with_funding["avg_sharpe"] - without_funding["avg_sharpe"]

            print(
                f"{'Gross PnL':<25} ${without_funding['total_gross_pnl']:<14.2f} ${with_funding['total_gross_pnl']:<14.2f} ${gross_diff:<11.2f}"
            )
            print(
                f"{'Net PnL':<25} ${without_funding['total_gross_pnl']:<14.2f} ${with_funding['total_net_pnl']:<14.2f} ${net_diff:<11.2f}"
            )
            print(
                f"{'Funding Cost':<25} ${'0.00':<14} ${with_funding['total_funding_cost']:<14.2f} ${with_funding['total_funding_cost']:<11.2f}"
            )
            print(
                f"{'Average Sharpe':<25} {without_funding['avg_sharpe']:<15.3f} {with_funding['avg_sharpe']:<15.3f} {sharpe_diff:<12.3f}"
            )

            print(f"\n💡 Key Insights:")
            print(
                f"   • Funding cost impact: {with_funding['funding_impact_pct']:.1f}% of gross PnL"
            )
            print(
                f"   • Net PnL reduction: ${with_funding['total_gross_pnl'] - with_funding['total_net_pnl']:.2f}"
            )

            if with_funding["total_funding_cost"] > 0:
                print(f"   • ⚠️ Funding costs reduce profitability")
                print(f"   • Consider strategies to minimize funding exposure")
            elif with_funding["total_funding_cost"] < 0:
                print(f"   • ✅ Net funding income received!")
                print(f"   • Strategy benefits from funding rate arbitrage")

            print(f"\n🎯 Recommendations:")
            if abs(with_funding["funding_impact_pct"]) > 10:
                print(
                    f"   • High funding impact ({with_funding['funding_impact_pct']:.1f}%)"
                )
                print(f"   • Consider shorter holding periods")
                print(f"   • Monitor funding rate forecasts")
            else:
                print(
                    f"   • Low funding impact ({with_funding['funding_impact_pct']:.1f}%)"
                )
                print(f"   • Current strategy is funding-efficient")

    print(f"\n📁 FILES GENERATED:")
    print(f"   • Trade CSV files with funding details")
    print(f"   • Separate directories for each configuration")
    print(f"   • Plot comparisons showing funding impact")

    print(f"\n🔍 NEXT STEPS:")
    print(f"   1. Analyze the detailed trade CSV files")
    print(f"   2. Compare funding costs across different pairs")
    print(f"   3. Optimize strategy to minimize funding exposure")
    print(f"   4. Consider funding rate arbitrage opportunities")


def analyze_funding_patterns():
    """
    Analyze funding rate patterns and their impact on different strategies
    """
    print(f"\n{'='*70}")
    print("FUNDING RATE PATTERN ANALYSIS")
    print(f"{'='*70}")

    print("🔍 This analysis shows:")
    print("   • Which pairs have highest funding costs")
    print("   • Optimal trade durations to minimize funding")
    print("   • Long vs short position funding differences")

    # Load and analyze existing trade data
    try:
        import glob

        # Find funding demo trade files
        trade_files = glob.glob("funding_demo_trades_with_funding_costs/trades_*.csv")

        if not trade_files:
            print("❌ No funding trade data found. Run the main demo first.")
            return

        print(f"📊 Analyzing {len(trade_files)} trade files...")

        all_trades = []
        for file_path in trade_files:
            try:
                df = pd.read_csv(file_path)
                all_trades.append(df)
            except Exception as e:
                print(f"   ⚠️ Error loading {file_path}: {e}")

        if not all_trades:
            return

        combined_trades = pd.concat(all_trades, ignore_index=True)

        # Funding analysis
        print(f"\n📈 Funding Rate Analysis:")
        print(f"   • Total trades analyzed: {len(combined_trades)}")

        # Trades with funding data
        trades_with_funding = combined_trades[
            combined_trades["funding_payments_count"] > 0
        ]
        print(f"   • Trades with funding data: {len(trades_with_funding)}")

        if len(trades_with_funding) > 0:
            avg_funding_cost = trades_with_funding["total_funding_cost"].mean()
            max_funding_cost = trades_with_funding["total_funding_cost"].max()
            min_funding_cost = trades_with_funding["total_funding_cost"].min()

            print(f"   • Average funding cost per trade: ${avg_funding_cost:.2f}")
            print(f"   • Maximum funding cost: ${max_funding_cost:.2f}")
            print(f"   • Minimum funding cost: ${min_funding_cost:.2f}")

            # Analyze by position direction
            long_trades = trades_with_funding[trades_with_funding["position1"] == 1]
            short_trades = trades_with_funding[trades_with_funding["position1"] == -1]

            if len(long_trades) > 0:
                print(
                    f"   • Long positions avg funding: ${long_trades['total_funding_cost'].mean():.2f}"
                )
            if len(short_trades) > 0:
                print(
                    f"   • Short positions avg funding: ${short_trades['total_funding_cost'].mean():.2f}"
                )

            # Duration vs funding cost
            print(f"\n⏱️ Duration vs Funding Cost:")
            duration_bins = [0, 4, 8, 16, 24, float("inf")]
            duration_labels = ["0-4h", "4-8h", "8-16h", "16-24h", "24h+"]

            trades_with_funding["duration_bin"] = pd.cut(
                trades_with_funding["duration_hours"],
                bins=duration_bins,
                labels=duration_labels,
                right=False,
            )

            duration_analysis = (
                trades_with_funding.groupby("duration_bin")
                .agg(
                    {
                        "total_funding_cost": ["count", "mean"],
                        "net_pnl": "mean",
                        "net_is_profitable": "mean",
                    }
                )
                .round(3)
            )

            print(f"   Duration analysis:")
            for duration, stats in duration_analysis.iterrows():
                count = stats[("total_funding_cost", "count")]
                avg_funding = stats[("total_funding_cost", "mean")]
                avg_net_pnl = stats[("net_pnl", "mean")]
                win_rate = stats[("net_is_profitable", "mean")]

                print(
                    f"   • {duration}: {count} trades, avg funding: ${avg_funding:.2f}, "
                    f"avg net PnL: ${avg_net_pnl:.2f}, win rate: {win_rate:.1%}"
                )

    except Exception as e:
        print(f"❌ Analysis failed: {e}")


if __name__ == "__main__":
    # Run funding rate impact demo
    demo_funding_rate_impact()

    # Analyze funding patterns
    analyze_funding_patterns()

    print(f"\n🎉 FUNDING RATE DEMO COMPLETED!")
    print(f"   • Realistic futures trading simulation")
    print(f"   • Complete funding cost calculations")
    print(f"   • Detailed trade logs with funding data")
    print(f"   • Performance comparison with/without funding")
