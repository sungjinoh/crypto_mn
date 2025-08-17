# Technology Stack

## Core Technologies

- **Python 3.x** - Primary programming language
- **PyArrow** - High-performance columnar data format for efficient storage and processing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Statsmodels** - Statistical analysis and econometric modeling (cointegration tests, ADF tests)
- **SciPy** - Scientific computing (statistical functions)

## Data Storage & Processing

- **Parquet format** - Efficient columnar storage using PyArrow with Snappy compression
- **Multiprocessing** - Parallel data collection and processing using ProcessPoolExecutor
- **CCXT** - Cryptocurrency exchange connectivity for market data
- **Requests** - HTTP client for Binance data API

## Analysis & Visualization

- **Matplotlib** - Plotting and visualization
- **Seaborn** - Statistical data visualization
- **tqdm** - Progress bars for long-running operations
- **QuantStats** (optional) - Advanced portfolio performance metrics

## Common Commands

### Data Collection
```bash
# Collect Binance futures data
python binance_futures_data_collector.py

# Run cointegration analysis
python run_cointegration_finder.py

# Quick optimized backtest
python quick_start_optimized.py
```

### Analysis & Backtesting
```bash
# Run mean reversion backtest
python run_mean_reversion_backtest.py

# Multi-year analysis
python run_multi_year_analysis.py

# Parameter optimization
python demo_parameter_optimization.py
```

### Testing & Validation
```bash
# Run simple tests
python test_simple.py

# Test trade parsing
python test_trade_parsing.py

# Analyze trade details
python analyze_trade_details.py
```

## Development Environment

- **Jupyter notebooks** - Not present but compatible for interactive analysis
- **IDE support** - Standard Python development environment
- **Version control** - Git (evidenced by .DS_Store exclusion patterns)

## Performance Considerations

- **PyArrow optimization** - Uses optimized Parquet I/O with proper compression settings
- **Parallel processing** - Multicore utilization for data processing and pair testing
- **Memory efficiency** - Chunked processing and proper data type handling
- **Caching** - Results saved in multiple formats (JSON, CSV, Parquet, Pickle)