# Product Overview

This is a **cryptocurrency pairs trading and statistical arbitrage system** focused on Binance futures markets. The system implements mean reversion strategies by identifying cointegrated trading pairs and executing market-neutral statistical arbitrage.

## Core Functionality

- **Data Collection**: Automated collection of Binance futures OHLCV data, trades, and funding rates using PyArrow for efficient storage
- **Cointegration Analysis**: Statistical testing to identify pairs with mean-reverting relationships suitable for pairs trading
- **Backtesting Framework**: Comprehensive backtesting engine for pairs trading strategies with performance metrics
- **Strategy Implementation**: Multiple mean reversion strategy variants including adaptive thresholds and volatility adjustment
- **Parameter Optimization**: Per-pair parameter optimization to maximize strategy performance

## Target Market

Quantitative traders and researchers working with cryptocurrency futures markets, specifically those implementing market-neutral statistical arbitrage strategies.

## Key Value Propositions

1. **Automated pair discovery** through statistical cointegration testing
2. **Robust backtesting** with realistic transaction costs and position sizing
3. **Optimized data pipeline** using PyArrow for fast data processing
4. **Multiple strategy variants** for different market conditions
5. **Comprehensive performance analysis** with detailed metrics and visualizations