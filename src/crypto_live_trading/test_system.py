#!/usr/bin/env python3
"""
Test Live Trading System
Quick test of the live trading components without full execution.
"""
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from strategies.mean_reversion import LiveMeanReversionStrategy, Signal
from execution.trade_executor import TradeExecutor
from state.position_tracker import PositionTracker
from utils.live_data_manager import LiveDataManager
from utils.scheduler import every_n_minutes, run_once


def test_imports():
    """Test that all imports work"""
    try:
        from strategies.mean_reversion import LiveMeanReversionStrategy, Signal
        from execution.trade_executor import TradeExecutor
        from state.position_tracker import PositionTracker
        from utils.live_data_manager import LiveDataManager
        from utils.scheduler import every_n_minutes, run_once

        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_components():
    """Test basic component functionality"""
    print("\n🧪 Testing Components...")

    # Test strategy
    strategy = LiveMeanReversionStrategy(
        lookback_period=60,
        entry_threshold=1.0,
        exit_threshold=0.5,
        stop_loss_threshold=3.5,
    )
    print("✅ Strategy initialized")

    # Test position tracker
    tracker = PositionTracker()
    pairs = tracker.load_cointegrated_pairs("state/cointegrated_pairs.json")
    print(f"✅ Position tracker loaded {len(pairs)} pairs")

    # Test executor (paper mode)
    executor = TradeExecutor(sandbox=True)
    print("✅ Trade executor initialized (paper mode)")

    # Test data manager (without actual API calls)
    data_manager = LiveDataManager()
    print("✅ Data manager initialized")

    return True


def test_signal_generation():
    """Test signal generation with dummy data"""
    print("\n📊 Testing Signal Generation...")

    import pandas as pd
    import numpy as np

    # Create dummy price data
    dates = pd.date_range("2024-01-01", periods=100, freq="4h")

    # Simulate cointegrated pair with mean reversion
    np.random.seed(42)
    spread = np.cumsum(np.random.randn(100) * 0.1)
    prices1 = 100 + spread + np.random.randn(100) * 0.5
    prices2 = 50 - 0.5 * spread + np.random.randn(100) * 0.3

    df1 = pd.DataFrame(
        {
            "open": prices1,
            "high": prices1 * 1.01,
            "low": prices1 * 0.99,
            "close": prices1,
            "volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )

    df2 = pd.DataFrame(
        {
            "open": prices2,
            "high": prices2 * 1.01,
            "low": prices2 * 0.99,
            "close": prices2,
            "volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )

    # Test strategy
    from strategies.mean_reversion import LiveMeanReversionStrategy

    strategy = LiveMeanReversionStrategy(
        lookback_period=50,
        entry_threshold=1.5,
        exit_threshold=0.5,
        stop_loss_threshold=3.0,
    )

    signal = strategy.generate_signal(df1, df2)

    print(f"✅ Signal generated: {signal.reason}")
    print(f"   • Is Entry: {signal.is_entry}")
    print(f"   • Is Exit: {signal.is_exit}")
    print(f"   • Side: {signal.side}")
    print(f"   • Z-Score: {signal.z_score:.3f}")

    return True


def main():
    """Run all tests"""
    print("🚀 LIVE TRADING SYSTEM TEST")
    print("=" * 50)

    # Test components
    if not test_components():
        return False

    # Test signal generation
    if not test_signal_generation():
        return False

    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("\nNext steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Generate cointegrated pairs if not done already")
    print("3. Set up Binance API keys in TradeExecutor")
    print("4. Run main.py for live trading")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
