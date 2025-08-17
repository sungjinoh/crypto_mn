# Cryptocurrency Backtesting Framework - AI Agent Instructions

## Architecture Overview

This is a **modular cryptocurrency backtesting framework** specializing in pairs trading and statistical arbitrage. The project follows a **layered architecture** with clear separation of concerns:

```
src/crypto_backtesting/
├── data/           # Unified data access layer
├── strategies/     # Strategy implementations 
├── backtesting/    # Core engine and portfolio management
├── analysis/       # Performance metrics and reporting
└── utils/          # Common utilities and workflows
```

**Key architectural principle**: The framework uses a **dual-architecture pattern** - there's both a legacy `backtesting_framework/` and a new `src/crypto_backtesting/` structure. Always work in `src/crypto_backtesting/` for new features.

## Critical Data Flow Patterns

### 1. Data Layer Architecture
- **File Structure**: `binance_futures_data/klines/SYMBOL/TIMEFRAME/*.parquet` (e.g., `BTCUSDT/1m/`)
- **DataManager** (`src/crypto_backtesting/data/manager.py`): Central data access point
- **Column Naming**: Data uses `{SYMBOL}_close`, `{SYMBOL}_open`, etc. format for pairs

```python
# Always use DataManager for data access
data_manager = DataManager(data_path="binance_futures_data")
data1, data2 = data_manager.get_pair_data('BTCUSDT', 'ETHUSDT', year=2024, months=[4,5,6])
```

### 2. Pairs Trading Signal Pattern
**Critical**: Pairs trading uses **combined symbols** (e.g., "BTCUSDT-ETHUSDT") but requires **individual symbol prices**:

```python
# Strategy creates signal with combined symbol
signal = Signal(
    symbol="BTCUSDT-ETHUSDT",  # Combined symbol
    metadata={'position1': 1, 'position2': -1}  # Individual positions
)

# Engine detects pairs by checking:
if '-' in symbol and ('position1' in signal.metadata and 'position2' in signal.metadata):
    return self._enter_pairs_position(signal, current_prices, timestamp)
```

### 3. Strategy Implementation Pattern
All strategies inherit from `BaseStrategy` or `PairsStrategy`. **Required methods**:

```python
class CustomStrategy(PairsStrategy):
    def initialize(self, data: pd.DataFrame) -> None:
        # Validate data columns, calculate initial parameters
        
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        # Return list of Signal objects with proper metadata
        
    def prepare_pair_data(self, data1: pd.DataFrame, data2: pd.DataFrame, 
                         symbol1: str, symbol2: str) -> pd.DataFrame:
        # Combine data, calculate spread, add indicators
```

## Development Workflows

### Quick Start Pattern
```bash
# Always install in development mode
pip install -e .

# Test with the standard example
python examples/quick_start.py

# For new strategies, start from examples/custom_strategy.py
```

### Data Debugging
```python
# Check available symbols
available_symbols = data_manager.get_available_symbols()

# Verify data loading
data1, data2 = data_manager.get_pair_data('SYMBOL1', 'SYMBOL2', year=2024)
print(f"Data1 shape: {data1.shape}, Data2 shape: {data2.shape}")
```

### Backtesting Execution
The backtesting engine processes data **row-by-row**, calling `strategy.generate_signals()` with **historical data up to current timestamp**. Performance-critical: strategies should implement **incremental signal generation**.

## Key Conventions

### 1. Configuration Pattern
Use **dataclasses** for all configuration:

```python
@dataclass
class PairsStrategyConfig(StrategyConfig):
    symbol1: str
    symbol2: str
    lookback_period: int = 60
    entry_threshold: float = 2.0
```

### 2. Error Handling
- **Data missing**: Log warnings but continue execution
- **Strategy errors**: Raise `StrategyError` with clear context
- **Portfolio errors**: Log warnings for insufficient funds, skip trades

### 3. Performance Analysis
Results use **two-layer analysis**:
- `BacktestResults`: Raw trade data and basic metrics
- `PerformanceAnalyzer`: Advanced metrics and reporting

```python
results = engine.run_backtest(strategy, data)
analyzer = PerformanceAnalyzer(results)
report = analyzer.generate_performance_report()
```

## Integration Points

### Data Provider Pattern
New data sources must implement `BaseDataProvider`:

```python
class CustomProvider(BaseDataProvider):
    def get_klines(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        # Must return standardized columns: timestamp, open, high, low, close, volume
```

### Strategy Extension
- Inherit from `BaseStrategy` for single assets
- Inherit from `PairsStrategy` for pairs trading
- Always implement `prepare_pair_data()` for pairs strategies

### Testing Patterns
- Unit tests in `tests/unit/`
- Use `examples/` for integration testing
- Mock data providers for isolated testing

## Common Pitfalls

1. **Pairs Symbol Confusion**: Engine expects individual symbol prices but strategy uses combined symbols
2. **Data Column Names**: Use `{SYMBOL}_close` format, not just `close`
3. **Signal Metadata**: Pairs strategies must include `position1` and `position2` in signal metadata
4. **Historical Data**: Strategies receive **cumulative** historical data, not just current row

## Project-Specific Commands

```bash
# Run comprehensive backtest
python examples/quick_start.py

# Parameter optimization
python examples/parameter_optimization.py

# Multi-pair analysis
python examples/multi_pair_analysis.py

# Install framework
pip install -e .
```

When implementing new features, always check `examples/` for usage patterns and `src/crypto_backtesting/__init__.py` for the public API.
