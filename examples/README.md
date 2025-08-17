# Examples Directory

This directory contains practical examples demonstrating how to use the cryptocurrency backtesting framework.

## 📁 Contents

- `quick_start.py` - Basic usage example for beginners
- `parameter_optimization.py` - Strategy parameter optimization workflow
- `multi_pair_analysis.py` - Analyzing multiple trading pairs
- `custom_strategy.py` - Creating custom trading strategies
- `advanced_analysis.py` - Advanced performance analysis and reporting

## 🚀 Running Examples

Make sure you have the framework installed and your data directory is properly set up:

```bash
# Install the framework
pip install -e .

# Run examples
python examples/quick_start.py
python examples/parameter_optimization.py
```

## 📋 Example Descriptions

### quick_start.py
Demonstrates the basic workflow:
- Loading data
- Creating a strategy
- Running a backtest
- Analyzing results

### parameter_optimization.py
Shows how to:
- Define parameter ranges
- Run optimization
- Select best parameters
- Validate results

### multi_pair_analysis.py
Covers:
- Analyzing multiple pairs
- Comparing performance
- Ranking strategies
- Portfolio allocation

### custom_strategy.py
Examples of:
- Creating custom strategies
- Implementing signal logic
- Adding risk management
- Strategy validation

### advanced_analysis.py
Advanced topics:
- Detailed performance metrics
- Risk analysis
- Visualization
- Report generation

## 📊 Expected Data Structure

Examples assume your data is organized as:
```
binance_futures_data/
├── klines/
│   ├── BTCUSDT/
│   ├── ETHUSDT/
│   └── ...
└── fundingRate/
    ├── BTCUSDT/
    ├── ETHUSDT/
    └── ...
```

## ⚠️ Notes

- Adjust file paths according to your data location
- Some examples may take several minutes to complete
- Results will vary based on your data timeframe
- Examples use sample parameters for demonstration
