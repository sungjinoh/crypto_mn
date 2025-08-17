# Project Structure

## Root Directory Organization

```
├── backtesting_framework/          # Core backtesting engine
├── binance_futures_data/          # Data storage directory (Parquet files)
├── cointegration_results/         # Analysis results output
├── *.py                          # Main scripts and utilities
└── .kiro/                        # Kiro configuration and steering
```

## Core Modules

### Backtesting Framework (`backtesting_framework/`)
- `pairs_backtester.py` - Main backtesting engine with PairsBacktester class
- `pairs_utils.py` - Utility functions for pairs analysis and statistics
- `__init__.py` - Package initialization

### Data Collection & Processing
- `binance_futures_data_collector.py` - PyArrow-optimized data collection from Binance
- `data_resampling_utils.py` - Time series resampling utilities

### Strategy Implementation
- `mean_reversion_strategy.py` - Multiple mean reversion strategy variants
- `enhanced_cointegration_finder.py` - Cointegration analysis and pair discovery

### Workflow Scripts
- `optimal_backtesting_workflow.py` - Recommended backtesting approach
- `quick_start_optimized.py` - Quick start script for new users
- `run_*.py` - Various execution scripts for different workflows

### Analysis & Utilities
- `analyze_trade_details.py` - Trade analysis and debugging
- `demo_*.py` - Demonstration scripts for different features
- `test_*.py` - Testing and validation scripts

## Data Directory Structure

### `binance_futures_data/`
```
├── klines/
│   └── {SYMBOL}/
│       └── 1m/
│           └── {YYYY-MM}.parquet
├── fundingRate/
│   └── {SYMBOL}/
│       └── {YYYY-MM}.parquet
└── trades/
    └── {SYMBOL}/
        └── {YYYY-MM}.parquet
```

### `cointegration_results/`
- Analysis results in multiple formats (JSON, CSV, Parquet, Pickle)
- Summary reports and metadata
- Timestamped result files

## Naming Conventions

### Files
- **Snake_case** for all Python files
- **Descriptive names** indicating functionality (e.g., `enhanced_cointegration_finder.py`)
- **Prefixes** for organization:
  - `run_*` - Execution scripts
  - `demo_*` - Demonstration/example scripts  
  - `test_*` - Testing scripts
  - `analyze_*` - Analysis utilities

### Classes
- **PascalCase** for class names (e.g., `PairsBacktester`, `MeanReversionStrategy`)
- **Abstract base classes** for strategy patterns (e.g., `PairsStrategy`)

### Variables & Functions
- **Snake_case** for variables and functions
- **Descriptive names** (e.g., `calculate_position_size`, `entry_threshold`)
- **Type hints** used throughout for better code clarity

## Architecture Patterns

### Strategy Pattern
- Abstract `PairsStrategy` base class
- Concrete strategy implementations (e.g., `MeanReversionStrategy`)
- Pluggable strategy system in backtester

### Data Processing Pipeline
1. **Collection** - Raw data from Binance API
2. **Storage** - PyArrow Parquet format with compression
3. **Processing** - Resampling and indicator calculation
4. **Analysis** - Cointegration testing and pair discovery
5. **Backtesting** - Strategy execution and performance measurement

### Configuration Management
- Parameters passed as dictionaries or dataclass objects
- Flexible configuration for different timeframes and markets
- Results saved with metadata for reproducibility

## Import Patterns

### Internal Imports
```python
from backtesting_framework.pairs_backtester import PairsBacktester
from backtesting_framework.pairs_utils import find_cointegrated_pairs
```

### External Dependencies
```python
import pandas as pd
import numpy as np
import pyarrow as pa
from statsmodels.tsa.stattools import coint
```

### Warning Suppression
```python
import warnings
warnings.filterwarnings('ignore')  # Common pattern for clean output
```