# Quant Strategy Development Guidelines

## Objective

Implement systematic trading strategies for backtesting and live trading. Code should be robust, vectorized, and bias-free.

---

### ✅ General Rules

- All calculations must use **vectorized pandas/numpy operations**.
- Prevent **lookahead bias**:
  - Shift signals by 1 period before applying.
- Handle **NaN values** gracefully.
- Ensure **consistent index alignment** for signals, positions, and prices.
- Include **docstrings and type hints**.

---

### ✅ Strategy Template

```python
def strategy_name(data: pd.DataFrame, param1: int, param2: int) -> pd.Series:
    """
    Generate trading signals based on given parameters.

    Args:
        data (pd.DataFrame): OHLCV data with 'close' column
        param1 (int): First parameter
        param2 (int): Second parameter

    Returns:
        pd.Series: Signals (+1 for long, -1 for short, 0 for flat)
    """
    # Example: Moving Average Crossover
    short_ma = data['close'].rolling(param1).mean()
    long_ma = data['close'].rolling(param2).mean()
    signal = np.where(short_ma > long_ma, 1, -1)
    return pd.Series(signal, index=data.index).shift(1)  # Avoid lookahead
```

---

### ✅ Backtesting Rules

- Calculate returns using data['close'].pct_change().
- Apply:
  - Commission = 0.0005
  - Slippage = 0.0002
- Track:
  - Equity curve
  - Drawdown
  - PnL
  - Trade log (entry/exit)
- Performance metrics:
  - CAGR, Sharpe ratio, Max Drawdown

---

### ✅ Risk Management

- Position sizing:
  - Use ATR-based sizing or volatility adjustment.
  - Optionally, implement Kelly criterion.
- Stop loss / take profit levels:
  - Based on ATR or percentage.

---

### ✅ Example Workflow

- Load OHLCV data.
- Generate signals using strategy function.
- Convert signals into positions.
- Backtest with transaction costs and slippage.
- Output performance report and charts.
