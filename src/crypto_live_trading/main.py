"""
Live Mean Reversion Trading System
Entry point for live trading loop and orchestration.
"""

import time
from strategies.mean_reversion import LiveMeanReversionStrategy
from execution.trade_executor import TradeExecutor
from state.position_tracker import PositionTracker
from utils.live_data_manager import LiveDataManager
from utils.scheduler import every_4h

# Load monthly cointegration pairs (update this path as needed)
COINTEGRATION_PAIRS_PATH = "state/cointegrated_pairs.json"

# Strategy parameters (can be loaded from config)
FIXED_PARAMS = {
    "lookback_period": 60,
    "entry_threshold": 1.0,
    "exit_threshold": 0.5,
    "stop_loss_threshold": 3.5,
}


def main():
    # Initialize components
    data_manager = LiveDataManager()
    strategy = LiveMeanReversionStrategy(**FIXED_PARAMS)
    executor = TradeExecutor()
    tracker = PositionTracker()

    # Load cointegrated pairs
    pairs = tracker.load_cointegrated_pairs(COINTEGRATION_PAIRS_PATH)
    print(f"Loaded {len(pairs)} cointegrated pairs for trading.")

    # Main live trading loop (every 4h)
    for _ in every_4h():
        print("\n[Live Trading] Running mean reversion strategy...")
        for pair in pairs:
            symbol1, symbol2 = pair["symbol1"], pair["symbol2"]
            # Load latest data
            df1, df2 = data_manager.get_latest_pair_data(symbol1, symbol2)
            # Generate signal
            signal = strategy.generate_signal(df1, df2)
            # Check position state
            position = tracker.get_position(symbol1, symbol2)
            if signal.is_entry and not position:
                print(f"Opening position for {symbol1}-{symbol2}")
                executor.open_position(symbol1, symbol2, signal)
                tracker.record_open(symbol1, symbol2, signal)
            elif signal.is_exit and position:
                print(f"Closing position for {symbol1}-{symbol2}")
                executor.close_position(symbol1, symbol2, signal)
                tracker.record_close(symbol1, symbol2, signal)
            else:
                print(f"No action for {symbol1}-{symbol2}")
        print("Sleeping until next 4h interval...")
        time.sleep(1)  # Replace with proper scheduling


if __name__ == "__main__":
    main()
