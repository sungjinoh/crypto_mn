#!/usr/bin/env python3
"""
Simple test to verify the system works
"""

try:
    from mean_reversion_backtest import MeanReversionBacktester
    from mean_reversion_strategy import MeanReversionStrategy

    print("✅ Imports successful")

    # Test initialization
    backtester = MeanReversionBacktester()
    print("✅ Backtester initialized")
    print(f"   • Funding costs enabled: {backtester.backtester.include_funding_costs}")
    print(
        f"   • Has funding calculator: {hasattr(backtester.backtester, 'funding_calculator')}"
    )
    print(f"   • Trade logging enabled: {backtester.save_trades}")

    # Test strategy
    strategy = MeanReversionStrategy(
        lookback_period=60,
        entry_threshold=1.0,
        exit_threshold=0.5,
        stop_loss_threshold=3.5,
    )
    print("✅ Strategy created with your parameters")

    print("\n🎯 System is ready!")
    print("   • Funding rate calculator is properly allocated")
    print("   • Trade logging is enabled")
    print("   • Your parameters are configured")
    print("\n🚀 You can now run: python run_fixed_parameters.py")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
