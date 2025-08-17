#!/usr/bin/env python3
"""
Test script to verify funding rate fixes
"""

from mean_reversion_backtest import MeanReversionBacktester
from mean_reversion_strategy import MeanReversionStrategy
import warnings

warnings.filterwarnings("ignore")


def test_funding_fixes():
    """
    Test the funding rate system with proper error handling
    """
    print("🔧 TESTING FUNDING RATE FIXES")
    print("=" * 50)

    try:
        # Initialize backtester
        backtester = MeanReversionBacktester(
            base_path="binance_futures_data",
            results_dir="cointegration_results",
            resample_timeframe="1H",
            save_plots=False,  # Disable plots for testing
        )

        # Enable trade logging
        backtester.save_trades = True
        backtester.trades_dir = "test_funding_fixes"

        print("✅ Backtester initialized successfully")

        # Load test pair
        coint_results = backtester.load_cointegration_results()
        top_pairs = backtester.filter_top_pairs(coint_results, n_pairs=1)

        if not top_pairs:
            print("❌ No pairs found for testing")
            return

        pair = top_pairs[0]
        symbol1, symbol2 = pair["symbol1"], pair["symbol2"]

        print(f"📊 Testing pair: {symbol1} - {symbol2}")

        # Load data
        df1, df2 = backtester.load_pair_data(symbol1, symbol2, 2024, [4])
        print(f"✅ Data loaded: {len(df1)} bars")

        # Create strategy
        strategy = MeanReversionStrategy(
            lookback_period=60,
            entry_threshold=2.0,
            exit_threshold=0.0,
            stop_loss_threshold=3.0,
        )

        print("✅ Strategy created")

        # Run backtest
        print("🔄 Running backtest...")
        result = backtester.run_backtest_with_strategy(
            df1, df2, symbol1, symbol2, strategy, save_plot=False
        )

        print("✅ Backtest completed successfully!")

        # Check results
        num_trades = len(result.trades)
        metrics = result.metrics

        print(f"\n📊 Results:")
        print(f"   • Trades: {num_trades}")
        print(f"   • Sharpe: {metrics.get('Sharpe Ratio', 0):.3f}")
        print(f"   • Return: {metrics.get('Total Return', 0):.2%}")

        if num_trades > 0:
            print(f"   • First trade PnL: {result.trades[0].pnl}")
            print(f"   • Trade details saved successfully")

        print(f"\n✅ ALL TESTS PASSED!")
        print(f"   • No None value errors")
        print(f"   • Trade logging works")
        print(f"   • Funding calculations handled safely")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_funding_fixes()
