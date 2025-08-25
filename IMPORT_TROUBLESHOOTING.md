# Import Troubleshooting Guide

## Issue: `ModuleNotFoundError: No module named 'backtesting_framework'`

This error occurs when Python can't find the `backtesting_framework` module. Here are several solutions:

## Solution 1: Use the Runner Script (Recommended)

```bash
cd /Users/fivestar/workspace/fivestar/crypto_mn
python3 run_cointegration_finder.py
```

## Solution 2: Set PYTHONPATH Environment Variable

```bash
cd /Users/fivestar/workspace/fivestar/crypto_mn
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 market_neutral/enhanced_cointegration_finder.py
```

## Solution 3: Run from Project Root

Always run Python scripts from the `crypto_mn` directory:

```bash
cd /Users/fivestar/workspace/fivestar/crypto_mn
python3 -m market_neutral.enhanced_cointegration_finder
```

## Solution 4: Install as Development Package

```bash
cd /Users/fivestar/workspace/fivestar/crypto_mn
pip install -e .
```

(Note: This requires a setup.py file)

## Solution 5: Direct Python Path Setup

```python
import sys
import os
sys.path.insert(0, '/Users/fivestar/workspace/fivestar/crypto_mn')

# Now imports should work
from backtesting_framework.pairs_backtester import PairsBacktester
```

## Verification

Test that imports work:

```python
python3 -c "
import sys
sys.path.append('/Users/fivestar/workspace/fivestar/crypto_mn')
from backtesting_framework.pairs_backtester import PairsBacktester
print('✅ Import successful!')
"
```

## Project Structure

Make sure your directory structure looks like this:

```
crypto_mn/
├── backtesting_framework/
│   ├── __init__.py
│   ├── pairs_backtester.py
│   └── pairs_utils.py
├── market_neutral/
│   └── enhanced_cointegration_finder.py
├── src/
│   └── crypto_live_trading/
└── run_cointegration_finder.py
```

## Required Dependencies

Make sure these packages are installed:

```bash
pip install pandas numpy scipy statsmodels tqdm matplotlib seaborn
```

## Common Issues

1. **Wrong working directory**: Always run from `crypto_mn/`
2. **Missing **init**.py**: Make sure `backtesting_framework/__init__.py` exists
3. **Virtual environment**: If using conda/venv, make sure it's activated
4. **Python version**: Use Python 3.7+ for best compatibility

## Quick Fix

If all else fails, run this one-liner:

```bash
cd /Users/fivestar/workspace/fivestar/crypto_mn && PYTHONPATH="$(pwd):$PYTHONPATH" python3 market_neutral/enhanced_cointegration_finder.py
```
