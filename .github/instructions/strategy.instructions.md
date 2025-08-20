---
applyTo: "**"
---

# Strategy Development Guidelines

## Objective

Implement systematic trading strategies using pandas and numpy, optimized for backtesting and forward testing.

### Core Rules

- Always use **vectorized operations** for signal calculations.
- Prevent **lookahead bias**:
  - Signals should use `shift(1)` before applying to prices.
- Ensure **timezone consistency** (`UTC` for all timestamps).
- Include **transaction costs** and **slippage** in backtest.
- Use **ATR-based position sizing** or volatility adjustment if applicable.
- All strategy functions must return:
  - `signals`: pd.Series or pd.DataFrame
  - `positions`: pd.Series or pd.DataFrame
  - `performance`: dict with metrics (CAGR, Sharpe, max drawdown)

### Example Template

```python
def moving_average_crossover(data: pd.DataFrame, short_window: int, long_window: int) -> pd.Series:
    """
    Generate signals for a moving average crossover strategy.

    Args:
        data: OHLC price data
        short_window: Short MA period
        long_window: Long MA period

    Returns:
        pd.Series: Signal (+1 for long, -1 for short, 0 for flat)
    """
    short_ma = data['close'].rolling(window=short_window).mean()
    long_ma = data['close'].rolling(window=long_window).mean()
    signal = np.where(short_ma > long_ma, 1, -1)
    return pd.Series(signal, index=data.index).shift(1)  # Prevent lookahead
```

### Backtest Rules

- Use prices.pct_change() for returns.
- Track cash, positions, and PnL.
- Apply commission = 0.0005 and slippage = 0.0002 per trade.
