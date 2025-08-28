# Market Neutral Strategy - File Organization

## 📂 Directory Structure

```
market_neutral/
│
├── 📄 README.md                          # This file - main documentation
│
├── 🚀 QUICK_START.md                     # Quick start guide
│
├── ⚙️ Core Workflow Scripts (USE THESE)
│   ├── comprehensive_workflow.py         # [MAIN] Complete accuracy-focused workflow
│   ├── hybrid_pair_selector.py          # [MAIN] Fast hybrid selection (statistical + backtest)
│   ├── complete_workflow.py             # [MAIN] Guided complete workflow
│   └── apply_optimal_filters.py         # [UTILITY] Apply filters to cointegration results
│
├── 🔬 Analysis & Discovery
│   ├── enhanced_cointegration_finder_v2.py  # [CORE] Find cointegrated pairs (latest version)
│   ├── enhanced_threshold_discovery.py      # [ANALYSIS] Find optimal thresholds
│   └── statistical_filter_discovery.py      # [ANALYSIS] Statistical filtering without backtest
│
├── 📊 Backtesting
│   ├── run_fixed_parameters.py          # [BACKTEST] Run with fixed parameters
│   └── mean_reversion_backtest.py       # [BACKTEST] Core backtesting engine
│
├── 🗂️ Legacy/Demo Files (can be archived)
│   ├── enhanced_cointegration_finder.py # Older version - use v2 instead
│   ├── threshold_discovery_analysis.py  # Older version - use enhanced version
│   ├── demo_*.py                        # Demo files
│   ├── test_*.py                        # Test files
│   └── fix_*.py                         # Fix scripts
│
└── 📁 Output Directories (auto-created)
    ├── cointegration_results_*/         # Cointegration results by timeframe
    ├── filtered_pairs/                  # Filtered pair selections
    ├── backtest_results/                # Backtesting results
    └── reports/                         # Analysis reports
```

## 🎯 Main Workflows

### 1. **Comprehensive Analysis (Maximum Accuracy)**
```bash
python comprehensive_workflow.py
```
- Runtime: 6-10 hours
- Tests multiple timeframes
- Optimizes all parameters
- Full cross-validation

### 2. **Hybrid Quick Analysis (Balanced)**
```bash
python hybrid_pair_selector.py
```
- Runtime: 30-60 minutes
- Statistical filtering + targeted backtesting
- Good accuracy with faster results

### 3. **Complete Guided Workflow**
```bash
python complete_workflow.py
```
- Interactive guided process
- Step-by-step instructions
- Good for first-time users

## 📊 Core Scripts Description

### Primary Workflow Scripts

1. **comprehensive_workflow.py** ⭐
   - Complete accuracy-focused analysis
   - Tests 4 timeframes, 500+ parameters
   - Walk-forward validation
   - Robustness testing

2. **hybrid_pair_selector.py** ⭐
   - Combines statistical filtering with backtesting
   - Much faster than full backtesting
   - 90% accuracy of comprehensive approach

3. **complete_workflow.py** ⭐
   - User-friendly guided workflow
   - Explains each step
   - Good for understanding the process

### Analysis Scripts

4. **enhanced_cointegration_finder_v2.py**
   - Finds cointegrated pairs
   - Includes stationarity checks
   - Volume filtering
   - Rolling window stability

5. **apply_optimal_filters.py**
   - Applies filtering criteria to pairs
   - Multiple strategy options (conservative/moderate/aggressive)
   - Exports filtered results

6. **enhanced_threshold_discovery.py**
   - Discovers optimal filtering thresholds
   - Can use backtesting or statistical analysis
   - Generates recommendations

7. **statistical_filter_discovery.py**
   - Pure statistical filtering (no backtesting)
   - Very fast analysis
   - Good for initial screening

### Backtesting Scripts

8. **run_fixed_parameters.py**
   - Runs backtests with specified parameters
   - Can test multiple pairs
   - Generates performance reports

9. **mean_reversion_backtest.py**
   - Core backtesting engine
   - Implements mean reversion strategy
   - Handles trade execution logic

## 🗑️ Files to Archive/Remove

### Legacy Versions (keep for reference but don't use)
- `enhanced_cointegration_finder.py` → Use `enhanced_cointegration_finder_v2.py`
- `threshold_discovery_analysis.py` → Use `enhanced_threshold_discovery.py`
- `run_cointegration_finder.py` → Integrated into workflow scripts
- `optimal_backtesting_workflow.py` → Use `comprehensive_workflow.py`
- `quick_start_optimized.py` → Use `hybrid_pair_selector.py`

### Demo/Test Files (can be moved to archive)
- `demo_*.py` files
- `test_*.py` files  
- `fix_*.py` files
- `clean_trade_logging.py`
- `analyze_trade_details.py`

### Utility Files (keep if needed)
- `binance_futures_data_collector.py` - Keep if collecting new data
- `data_resampling_utils.py` - Utility functions
- `load_and_use_results.py` - Utility for loading results

## 🚀 Quick Start Commands

### For New Users
```bash
# Step 1: Run complete guided workflow
python complete_workflow.py

# Follow the prompts - uses optimal defaults
```

### For Production Setup
```bash
# Step 1: Run comprehensive analysis (run overnight)
python comprehensive_workflow.py

# Step 2: Review results
cat comprehensive_report_*.json

# Step 3: Deploy with configuration
cat trading_config_final_*.json
```

### For Quick Analysis
```bash
# Step 1: Find cointegration (if not done)
python enhanced_cointegration_finder_v2.py

# Step 2: Run hybrid selection
python hybrid_pair_selector.py
```

## 📁 Output Files

### Cointegration Results
- `cointegration_results_1H/` - Results for 1-hour timeframe
- `cointegration_results_*.json` - Cointegration analysis results

### Filtered Pairs
- `filtered_pairs_*.json` - Selected pairs after filtering
- `filtered_pairs_*.csv` - CSV format for easy viewing

### Reports
- `comprehensive_report_*.json` - Full analysis report
- `trading_config_final_*.json` - Ready-to-use trading configuration
- `parameter_optimization_*.csv` - Parameter test results

## 🧹 Cleaning Recommendations

1. **Create Archive Folder**
```bash
mkdir archive
mv demo_*.py test_*.py fix_*.py archive/
mv *_old.py archive/  # Any old versions
```

2. **Keep Core Scripts**
- All scripts listed under "Core Workflow Scripts"
- Enhanced versions (v2) of scripts
- Main analysis tools

3. **Organize Outputs**
```bash
mkdir -p results/cointegration
mkdir -p results/backtests
mkdir -p results/reports
```

## 📝 Notes

- Always use `*_v2.py` versions when available
- Run `comprehensive_workflow.py` for production setup
- Use `hybrid_pair_selector.py` for daily/weekly rebalancing
- Check `__pycache__` can be safely deleted anytime

## 🔗 Dependencies

Required packages:
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scipy
- tqdm

Install all:
```bash
pip install pandas numpy matplotlib seaborn statsmodels scipy tqdm
```
