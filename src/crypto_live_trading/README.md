# Live Mean Reversion Trading System

This is a live trading system based on the mean reversion pairs trading strategy that was developed and backtested in the main framework.

## Overview

The system implements a pairs trading strategy that:

1. Uses pre-calculated cointegrated pairs from historical analysis
2. Fetches real-time market data every 4 hours
3. Calculates mean reversion signals based on z-score thresholds
4. Executes trades via Binance API (paper trading by default)
5. Tracks positions and manages risk

## Architecture

```
src/crypto_live_trading/
├── main.py                    # Main trading loop
├── generate_cointegrated_pairs.py  # Monthly pair generation
├── test_system.py            # System testing
├── config.json              # Configuration
├── requirements.txt         # Dependencies
├── strategies/
│   └── mean_reversion.py    # Live mean reversion strategy
├── execution/
│   └── trade_executor.py    # Order execution via CCXT
├── state/
│   ├── position_tracker.py  # Position management
│   ├── positions.json       # Active positions
│   ├── trades_history.json  # Trade history
│   └── cointegrated_pairs.json  # Trading pairs
└── utils/
    ├── live_data_manager.py  # Real-time data fetching
    └── scheduler.py         # 4-hour scheduling
```

## Setup

1. **Install dependencies:**

   ```bash
   cd src/crypto_live_trading
   pip install -r requirements.txt
   ```

2. **Generate cointegrated pairs:**

   ```bash
   cd /Users/fivestar/workspace/fivestar/crypto_mn
   PYTHONPATH=. python src/crypto_live_trading/generate_cointegrated_pairs.py
   ```

3. **Configure API keys (optional for paper trading):**
   Edit `execution/trade_executor.py` and add your Binance API credentials.

4. **Test the system:**

   ```bash
   PYTHONPATH=. python src/crypto_live_trading/test_system.py
   ```

5. **Run live trading:**
   ```bash
   PYTHONPATH=. python src/crypto_live_trading/main.py
   ```

## Configuration

Edit `config.json` to adjust:

- Trading parameters (portfolio size, risk per trade)
- Strategy parameters (thresholds, lookback period)
- Data settings (timeframe, number of pairs)

## Key Features

### 1. Cointegration Pair Generation

- **File:** `generate_cointegrated_pairs.py`
- **Purpose:** Monthly analysis to find statistically significant pairs
- **Output:** `state/cointegrated_pairs.json`

### 2. Live Data Management

- **File:** `utils/live_data_manager.py`
- **Purpose:** Fetch real-time 4H OHLCV data via CCXT
- **Features:** Data alignment, error handling

### 3. Mean Reversion Strategy

- **File:** `strategies/mean_reversion.py`
- **Purpose:** Generate entry/exit signals based on z-score
- **Parameters:**
  - `lookback_period`: 60 periods
  - `entry_threshold`: 1.0 (z-score)
  - `exit_threshold`: 0.5 (z-score)
  - `stop_loss_threshold`: 3.5 (z-score)

### 4. Position Management

- **File:** `state/position_tracker.py`
- **Purpose:** Track open positions, prevent double entries
- **Features:** Persistent storage, trade history

### 5. Trade Execution

- **File:** `execution/trade_executor.py`
- **Purpose:** Execute pairs trades via Binance API
- **Modes:** Paper trading (default) or live trading

### 6. Scheduling

- **File:** `utils/scheduler.py`
- **Purpose:** Run strategy every 4 hours aligned with market data
- **Times:** 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC

## Trading Logic

1. **Every 4 hours:**

   - Load cointegrated pairs
   - Fetch latest 4H data for each pair
   - Calculate z-score based on recent spread behavior
   - Generate entry/exit signals

2. **Position Management:**

   - Only one position per pair at a time
   - Entry: |z-score| >= 1.0
   - Exit: |z-score| <= 0.5 OR |z-score| >= 3.5 (stop loss)

3. **Risk Management:**
   - 2% portfolio risk per trade
   - Maximum 5 concurrent positions
   - Paper trading by default

## Paper Trading vs Live Trading

**Paper Trading (Default):**

- No real money involved
- Simulates order execution
- Safe for testing and development

**Live Trading:**

- Requires Binance API keys
- Real money at risk
- Set `sandbox=False` in TradeExecutor

## Monitoring

The system logs:

- Trading decisions and reasoning
- Position entries/exits
- Performance metrics
- Error handling

Files generated:

- `state/positions.json`: Current open positions
- `state/trades_history.json`: Complete trade history

## Safety Features

- Paper trading mode by default
- Position limits and risk controls
- Comprehensive error handling
- Data validation and alignment
- Stop-loss mechanisms

## Next Steps

1. Run in paper trading mode to validate
2. Monitor performance and adjust parameters
3. Add more sophisticated risk management
4. Implement portfolio rebalancing
5. Add notification system (email/Slack)
6. Enhance with additional indicators

## Important Notes

- Always test thoroughly in paper mode first
- Monitor positions regularly
- Keep API keys secure
- Understand the risks involved in live trading
- This is for educational/research purposes
