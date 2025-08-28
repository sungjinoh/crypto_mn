#!/usr/bin/env python3
"""
Demo: Trade Logging Functionality
Shows how the detailed trade logging works with a simple example.
"""

from mean_reversion_backtest import MeanReversionBacktester
from mean_reversion_strategy import MeanReversionStrategy
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def demo_trade_logging():
    """
    Demonstrate the trade logging functionality
    """
    print("🔍 DEMO: DETAILED TRADE LOGGING")
    print("=" * 60)

    # Initialize backtester with trade logging enabled
    backtester = MeanReversionBacktester(
        base_path="binance_futures_data",
        results_dir="cointegration_results",
        resample_timeframe="15T",
        save_plots=True,
        plots_dir="demo_plots",
    )

    # Enable trade logging
    backtester.save_trades = True
    backtester.trades_dir = "demo_trades"

    print(f"📊 Configuration:")
    print(f"   • Trade logging: {backtester.save_trades}")
    print(f"   • Trade logs directory: {backtester.trades_dir}")
    print(f"   • Plots directory: {backtester.plots_dir}")

    # Load a test pair
    try:
        coint_results = backtester.load_cointegration_results()
        top_pairs = backtester.filter_top_pairs(coint_results, n_pairs=2)

        if not top_pairs:
            print("❌ No pairs found for demo")
            return

        print(f"\n🎯 Testing with {len(top_pairs)} pairs:")
        for i, pair in enumerate(top_pairs, 1):
            print(f"   {i}. {pair['symbol1']} - {pair['symbol2']}")

        # Test different strategies
        strategies = [
            {
                "name": "Conservative",
                "params": {
                    "lookback_period": 90,
                    "entry_threshold": 2.5,
                    "exit_threshold": 0.0,
                    "stop_loss_threshold": 3.5,
                },
            },
            {
                "name": "Aggressive",
                "params": {
                    "lookback_period": 45,
                    "entry_threshold": 1.5,
                    "exit_threshold": 0.5,
                    "stop_loss_threshold": 2.5,
                },
            },
        ]

        all_results = []

        for strategy_config in strategies:
            print(f"\n{'='*50}")
            print(f"TESTING {strategy_config['name'].upper()} STRATEGY")
            print(f"{'='*50}")
            print(f"Parameters: {strategy_config['params']}")

            strategy = MeanReversionStrategy(**strategy_config["params"])

            for pair in top_pairs:
                symbol1, symbol2 = pair["symbol1"], pair["symbol2"]

                print(
                    f"\n🔄 Running {symbol1}-{symbol2} with {strategy_config['name']} strategy..."
                )

                try:
                    # Load data
                    df1, df2 = backtester.load_pair_data(symbol1, symbol2, 2024, [4, 5])

                    # Run backtest (this will automatically save trade details)
                    result = backtester.run_backtest_with_strategy(
                        df1,
                        df2,
                        symbol1,
                        symbol2,
                        strategy,
                        plot_dir=f"demo_plots/{strategy_config['name'].lower()}",
                    )

                    # Show summary
                    metrics = result.metrics
                    num_trades = len(result.trades)

                    print(f"   ✅ Results:")
                    print(f"      • Trades: {num_trades}")
                    print(f"      • Sharpe: {metrics.get('Sharpe Ratio', 0):.3f}")
                    print(f"      • Return: {metrics.get('Total Return', 0):.2%}")
                    print(f"      • Win Rate: {metrics.get('Win Rate', 0):.2%}")

                    all_results.append(
                        {
                            "strategy": strategy_config["name"],
                            "pair": f"{symbol1}-{symbol2}",
                            "trades": num_trades,
                            "sharpe": metrics.get("Sharpe Ratio", 0),
                            "return": metrics.get("Total Return", 0),
                            "win_rate": metrics.get("Win Rate", 0),
                        }
                    )

                except Exception as e:
                    print(f"   ❌ Error: {e}")

        # Summary
        if all_results:
            print(f"\n{'='*60}")
            print("DEMO RESULTS SUMMARY")
            print(f"{'='*60}")

            results_df = pd.DataFrame(all_results)
            print(
                f"{'Strategy':<12} {'Pair':<15} {'Trades':<7} {'Sharpe':<7} {'Return':<8} {'Win%':<6}"
            )
            print("-" * 65)

            for _, row in results_df.iterrows():
                print(
                    f"{row['strategy']:<12} {row['pair']:<15} {row['trades']:<7} "
                    f"{row['sharpe']:<7.3f} {row['return']:<8.2%} {row['win_rate']:<6.1%}"
                )

        print(f"\n📁 FILES GENERATED:")
        print(f"   • Trade CSV files in: {backtester.trades_dir}/")
        print(f"   • Plot images in: demo_plots/")
        print(f"   • Each CSV contains detailed transaction data:")
        print(f"     - Entry/exit times and prices")
        print(f"     - Position sizes and directions")
        print(f"     - Z-scores and spread values")
        print(f"     - Trade duration and PnL")
        print(f"     - Performance metrics")

        print(f"\n🔍 NEXT STEPS:")
        print(f"   1. Check the generated CSV files for detailed trade data")
        print(f"   2. Run: python analyze_trade_details.py")
        print(f"   3. This will create comprehensive analysis and visualizations")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    demo_trade_logging()
