"""
Live Mean Reversion Trading System
Entry point for live trading loop and orchestration.
"""

import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from strategies.mean_reversion import LiveMeanReversionStrategy
from execution.trade_executor import TradeExecutor
from state.position_tracker import PositionTracker
from utils.live_data_manager import LiveDataManager
from utils.scheduler import every_4h

# Load monthly cointegration pairs (update this path as needed)
COINTEGRATION_PAIRS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "crypto_live_trading",
    "state",
)

# Strategy parameters (can be loaded from config)
FIXED_PARAMS = {
    "lookback_period": 60,
    "entry_threshold": 2.5,
    "exit_threshold": 0.75,
    "stop_loss_threshold": 2.6,
}

# Trading configuration
TRADING_CONFIG = {
    "leverage": 5,  # 5x leverage for futures trading
    "max_position_size_usdt": 200,  # Maximum $200 USDT per leg
    "portfolio_value": 1000,  # Portfolio value for risk calculations
}


def main():
    print("=" * 80)
    print("LIVE MEAN REVERSION TRADING SYSTEM")
    print("=" * 80)

    # Check if API keys are loaded
    api_key_loaded = bool(os.getenv("ZZT_BINANCE_KEY"))
    api_secret_loaded = bool(os.getenv("ZZT_BINANCE_SECRET"))
    print(f"🔑 API Key loaded: {'✅' if api_key_loaded else '❌'}")
    print(f"🔐 API Secret loaded: {'✅' if api_secret_loaded else '❌'}")

    # Initialize components
    data_manager = LiveDataManager(timeframe="4h", lookback_periods=100)
    strategy = LiveMeanReversionStrategy(**FIXED_PARAMS)
    executor = TradeExecutor(
        sandbox=True,
        portfolio_value=TRADING_CONFIG["portfolio_value"],
        leverage=TRADING_CONFIG["leverage"],
        max_position_size_usdt=TRADING_CONFIG["max_position_size_usdt"],
    )  # Paper trading mode with leverage
    tracker = PositionTracker()

    # Load cointegrated pairs
    coint_results = tracker.load_cointegrated_pairs(COINTEGRATION_PAIRS_PATH)

    pairs = tracker.filter_top_pairs(
        coint_results,
        n_pairs=100,
        max_p_value=0.0276,
        min_correlation=0.884,
        max_half_life=23.4,
    )

    if not pairs:
        print(
            "❌ No cointegrated pairs found! Please run generate_cointegrated_pairs.py first."
        )
        return

    print(f"✅ Loaded {len(pairs)} cointegrated pairs for trading.")
    print(f"📊 Strategy Parameters:")
    print(f"   • Lookback Period: {FIXED_PARAMS['lookback_period']}")
    print(f"   • Entry Threshold: {FIXED_PARAMS['entry_threshold']}")
    print(f"   • Exit Threshold: {FIXED_PARAMS['exit_threshold']}")
    print(f"   • Stop Loss Threshold: {FIXED_PARAMS['stop_loss_threshold']}")
    print(f"💰 Trading Configuration:")
    print(f"   • Leverage: {TRADING_CONFIG['leverage']}x")
    print(
        f"   • Max Position Size: ${TRADING_CONFIG['max_position_size_usdt']} USDT per leg"
    )
    print(f"   • Portfolio Value: ${TRADING_CONFIG['portfolio_value']} USDT")
    print(
        f"   • Max Margin per Trade: ${TRADING_CONFIG['max_position_size_usdt']*2/TRADING_CONFIG['leverage']:.2f} USDT"
    )

    # Main live trading loop (every 4h)
    print(f"\n{'='*60}")
    print(f"Time: {datetime.now()}")
    print(f"{'='*60}")

    signals_generated = 0
    positions_opened = 0
    positions_closed = 0

    for i, pair in enumerate(pairs, 1):  # Limit to top 10 pairs for testing
        symbol1, symbol2 = pair["symbol1"], pair["symbol2"]

        print(f"\n[{i:2d}/{len(pairs)+1}] Processing {symbol1}-{symbol2}")
        try:
            # Load latest data
            df1, df2 = data_manager.get_latest_pair_data(symbol1, symbol2)

            if df1 is None or df2 is None:
                print(f"   ❌ Failed to fetch data")
                continue

            print(f"   📊 Data: {len(df1)} bars, latest: {df1.index[-1]}")

            # Generate signal
            signal = strategy.generate_signal(df1, df2, pair_info=pair)
            signals_generated += 1

            print(f"   🎯 Signal: {signal.reason}")

            if signal.is_entry:
                import pdb

                pdb.set_trace()

            # Check position state
            position = tracker.get_position(symbol1, symbol2)

            if signal.is_entry and not position:
                print(f"   🟢 Opening {signal.side} position")
                result = executor.open_position(symbol1, symbol2, signal)
                if result:
                    tracker.record_open(symbol1, symbol2, signal)
                    positions_opened += 1

            elif signal.is_exit and position:
                print(f"   🔴 Closing position (was {position.side})")
                result = executor.close_position(symbol1, symbol2, signal)
                if result:
                    tracker.record_close(symbol1, symbol2, signal)
                    positions_closed += 1

            elif position:
                print(
                    f"   ⏸️  Holding {position.side} position (opened: {position.entry_time})"
                )

            else:
                print(f"   ⏹️  No action")

        except Exception as e:
            print(f"   ❌ Error processing {symbol1}-{symbol2}: {e}")

        # Cycle summary
        open_positions = tracker.get_all_open_positions()

        print(f"\n{'='*60}")
        print(f"{'='*60}")
        print(f"📊 Pairs processed: 10")
        print(f"🎯 Signals generated: {signals_generated}")
        print(f"🟢 Positions opened: {positions_opened}")
        print(f"🔴 Positions closed: {positions_closed}")
        print(f"📈 Active positions: {len(open_positions)}")

        if open_positions:
            print(f"\nActive Positions:")
            for pair_key, pos in open_positions.items():
                print(
                    f"  • {pos.symbol1}-{pos.symbol2}: {pos.side} (opened: {pos.entry_time[:19]})"
                )

        print(f"\n⏰ Waiting for next 4-hour cycle...")

        # For testing, break after first cycle
        # Remove this break for continuous operation
        # break


if __name__ == "__main__":
    main()
