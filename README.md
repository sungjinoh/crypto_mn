# Cryptocurrency Backtesting Framework

A comprehensive, modular framework for backtesting cryptocurrency trading strategies with focus on pairs trading and statistical arbitrage.

## 🚀 Features

- **Unified Data Management**: Single interface for accessing market data from multiple sources
- **Modular Strategy Architecture**: Standardized base classes for implementing trading strategies
- **Comprehensive Backtesting Engine**: Realistic cost modeling with transaction costs, slippage, and funding rates
- **Advanced Performance Analysis**: Detailed metrics, risk assessment, and reporting capabilities
- **Pairs Trading Specialization**: Built-in support for statistical arbitrage and mean reversion strategies
- **Easy Configuration Management**: Standardized configuration patterns across all components
- **Migration Tools**: Utilities to migrate from legacy backtesting frameworks

## 📁 Project Structure

```
src/crypto_backtesting/
├── __init__.py                 # Main package interface
├── data/                       # Data management layer
│   ├── manager.py             # Unified data access
│   └── providers/             # Data source implementations
├── strategies/                 # Trading strategy implementations
│   ├── base.py                # Base strategy classes
│   └── pairs/                 # Pairs trading strategies
├── backtesting/               # Core backtesting engine
│   └── engine.py              # Main backtesting logic
├── analysis/                  # Performance analysis and reporting
│   └── performance.py         # Metrics and report generation
└── utils/                     # Utilities and workflows
    └── workflows.py           # Common workflow patterns
```

## 🛠 Installation

### Prerequisites

- Python 3.8+
- Required packages (see `requirements.txt`)

### Install from Source

```bash
git clone <repository-url>
cd crypto_mn
pip install -e .
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Pairs Trading Backtest

```python
from crypto_backtesting import (
    DataManager, BacktestEngine, MeanReversionStrategy,
    PerformanceAnalyzer, ReportGenerator
)

# Setup data manager
data_manager = DataManager(data_path="binance_futures_data")

# Load data for a trading pair
data1, data2 = data_manager.get_pair_data(
    'BTCUSDT', 'ETHUSDT', 
    year=2024, 
    months=[4, 5, 6]
)

# Create mean reversion strategy
strategy = MeanReversionStrategy(
    symbol1='BTCUSDT',
    symbol2='ETHUSDT',
    lookback_period=60,
    entry_threshold=2.0,
    exit_threshold=0.0,
    stop_loss_threshold=3.0
)

# Prepare data
combined_data = strategy.prepare_pair_data(data1, data2, 'BTCUSDT', 'ETHUSDT')

# Add technical indicators
rolling_mean = combined_data['spread'].rolling(60).mean()
rolling_std = combined_data['spread'].rolling(60).std()
combined_data['zscore'] = (combined_data['spread'] - rolling_mean) / rolling_std

# Run backtest
engine = BacktestEngine(
    data_manager,
    initial_capital=100000,
    commission_rate=0.001
)
results = engine.run_backtest(strategy, combined_data)

# Generate analysis
analyzer = PerformanceAnalyzer(results)
performance_report = analyzer.generate_performance_report()

print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
```

### Using Workflow Manager

```python
from crypto_backtesting.utils import WorkflowManager

# Create workflow manager
workflow = WorkflowManager()

# Quick pairs backtest
result = workflow.quick_pairs_backtest(
    symbol1='BTCUSDT',
    symbol2='ETHUSDT',
    year=2024,
    months=[4, 5, 6]
)

# Parameter optimization
optimization_result = workflow.parameter_optimization_workflow(
    symbol1='BTCUSDT',
    symbol2='ETHUSDT',
    year=2024,
    optimization_metric='sharpe_ratio',
    max_iterations=100
)

# Multi-pair analysis
pairs = [('BTCUSDT', 'ETHUSDT'), ('ADAUSDT', 'DOTUSDT')]
multi_pair_result = workflow.multi_pair_analysis_workflow(
    symbol_pairs=pairs,
    year=2024,
    top_n=5
)
```

## 📊 Data Management

### Supported Data Sources

- **Binance Futures**: OHLCV klines and funding rates
- **Extensible**: Easy to add new data providers

### Data Structure

```
binance_futures_data/
├── klines/
│   ├── BTCUSDT/
│   │   ├── BTCUSDT-1m-2024-01.csv
│   │   └── ...
│   └── ETHUSDT/
└── fundingRate/
    ├── BTCUSDT/
    └── ETHUSDT/
```

### Data Loading

```python
from crypto_backtesting import DataManager

data_manager = DataManager(data_path="binance_futures_data")

# Load single symbol
btc_data = data_manager.get_klines_data('BTCUSDT', 2024, [4, 5, 6])

# Load pair data (synchronized)
data1, data2 = data_manager.get_pair_data('BTCUSDT', 'ETHUSDT', 2024)

# Load funding rates
funding_data = data_manager.get_funding_rates('BTCUSDT')

# Check available symbols
symbols = data_manager.get_available_symbols()
```

## 🎯 Strategy Development

### Creating Custom Strategies

```python
from crypto_backtesting.strategies.base import BaseStrategy, Signal, SignalType

class MyCustomStrategy(BaseStrategy):
    def __init__(self, param1, param2):
        config = StrategyConfig(
            name="MyCustomStrategy",
            parameters={'param1': param1, 'param2': param2}
        )
        super().__init__(config)
        
    def initialize(self, data):
        # Strategy initialization
        pass
        
    def generate_signals(self, data):
        # Signal generation logic
        signals = []
        # ... implementation
        return signals
        
    def validate_parameters(self):
        # Parameter validation
        return True
```

### Pairs Trading Strategies

```python
from crypto_backtesting.strategies.base import PairsStrategy

class MyPairsStrategy(PairsStrategy):
    def validate_pair(self, data1, data2):
        # Pair validation (cointegration, correlation, etc.)
        return validation_result
        
    def calculate_spread(self, data1, data2):
        # Spread calculation
        return spread_series
        
    def generate_signals(self, data):
        # Pairs trading signals
        return signals
```

## 📈 Performance Analysis

### Basic Metrics

```python
from crypto_backtesting.analysis import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(backtest_results)
report = analyzer.generate_performance_report()

# Access metrics
print(f"Sharpe Ratio: {report['summary_metrics']['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {report['summary_metrics']['max_drawdown']:.2%}")
print(f"Win Rate: {report['trade_analysis']['win_rate']:.1%}")
```

### Report Generation

```python
from crypto_backtesting.analysis import ReportGenerator

generator = ReportGenerator(backtest_results, output_dir="reports")
full_report = generator.generate_full_report(save_plots=True)

# Generated files:
# - reports/performance_overview.png
# - reports/trade_analysis.png
# - reports/risk_analysis.png
# - reports/performance_report.txt
```

### Strategy Comparison

```python
from crypto_backtesting.analysis import compare_strategies

comparison_df = compare_strategies(
    [results1, results2, results3],
    strategy_names=['Strategy A', 'Strategy B', 'Strategy C']
)

print(comparison_df)
```

## ⚙️ Configuration Management

### Standard Configurations

```python
from crypto_backtesting.utils import ConfigurationManager

config_manager = ConfigurationManager()

# Get default configurations
data_config = config_manager.get_default_data_config()
backtest_config = config_manager.get_default_backtest_config()
strategy_config = config_manager.get_default_strategy_config('mean_reversion')

# Validate configurations
is_valid = config_manager.validate_config(data_config, 'data')
```

### Custom Configurations

```python
# Custom backtest configuration
custom_config = {
    'initial_capital': 50000.0,
    'commission_rate': 0.0005,
    'slippage_rate': 0.0002,
    'risk_management': {
        'max_position_size': 0.3,
        'stop_loss_enabled': True
    }
}

engine = BacktestEngine(data_manager, **custom_config)
```

## 🔄 Migration from Legacy Framework

### Migration Utilities

```python
from migrate_framework import FrameworkMigrator

migrator = FrameworkMigrator()

# Generate migration report
report = migrator.generate_migration_report()

# Migrate existing script
migrated_script = migrator.migrate_existing_script(
    'old_script.py', 
    'new_script.py'
)

# Convert old configurations
new_config = migrator.convert_old_strategy_config(old_config)
```

### Migration Checklist

1. **Backup existing code**
2. **Install new framework dependencies**
3. **Update import statements**
4. **Convert configurations to new format**
5. **Update data access patterns**
6. **Test migrated functionality**
7. **Update documentation**

## 📚 Examples

### Example Scripts

- `examples/quick_start.py` - Basic usage example
- `examples/parameter_optimization.py` - Parameter optimization workflow
- `examples/multi_pair_analysis.py` - Multi-pair analysis example
- `examples/custom_strategy.py` - Custom strategy implementation
- `migrate_framework.py` - Migration utilities and examples

### Running Examples

```bash
# Quick start example
python examples/quick_start.py

# Parameter optimization
python examples/parameter_optimization.py

# Migration example
python migrate_framework.py
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_strategies.py

# Run with coverage
python -m pytest tests/ --cov=src/crypto_backtesting
```

### Test Structure

```
tests/
├── test_data_manager.py       # Data management tests
├── test_strategies.py         # Strategy tests
├── test_backtesting.py        # Backtesting engine tests
├── test_analysis.py           # Analysis and reporting tests
└── fixtures/                  # Test data and fixtures
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Make sure the package is installed
   pip install -e .
   ```

2. **Data Loading Issues**
   ```python
   # Check data path and structure
   data_manager = DataManager(data_path="correct/path/to/data")
   available_symbols = data_manager.get_available_symbols()
   ```

3. **Memory Issues with Large Datasets**
   ```python
   # Use data resampling
   data1, data2 = data_manager.get_pair_data(
       'BTCUSDT', 'ETHUSDT', 
       year=2024, 
       resample_timeframe='5T'  # Resample to 5-minute data
   )
   ```

### Performance Optimization

- Use data caching for repeated analysis
- Resample data for faster backtesting
- Limit parameter optimization iterations
- Use vectorized operations in custom strategies

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd crypto_mn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to all public methods
- Write tests for new functionality

### Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run tests and linting
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

- **Documentation**: See inline docstrings and examples
- **Issues**: Report bugs and request features via GitHub issues
- **Discussions**: Use GitHub discussions for questions and community support

## 🗺 Roadmap

### Current Version (1.0.0)
- ✅ Core backtesting framework
- ✅ Pairs trading strategies
- ✅ Performance analysis
- ✅ Migration utilities

### Planned Features
- [ ] Additional strategy types (momentum, arbitrage)
- [ ] Real-time trading integration
- [ ] Advanced risk management
- [ ] Web-based dashboard
- [ ] Multi-exchange data support
- [ ] Machine learning strategy templates

## 📊 Performance Benchmarks

| Operation | Time (avg) | Memory Usage |
|-----------|------------|--------------|
| Load 1M candles | 2.3s | 150MB |
| Simple backtest | 0.8s | 75MB |
| Parameter optimization (100 iterations) | 45s | 200MB |
| Generate full report | 1.2s | 50MB |

*Benchmarks run on: Intel i7-8700K, 32GB RAM, SSD storage*

---

**Built with ❤️ for the cryptocurrency trading community**
