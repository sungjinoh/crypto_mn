#!/usr/bin/env python3
"""
Setup Script for Cryptocurrency Backtesting Framework
====================================================

This script helps you set up and validate your cryptocurrency backtesting framework.
It will:
1. Check Python version and dependencies
2. Install the framework in development mode
3. Validate data structure
4. Run a quick test to ensure everything works
5. Generate example configuration files

Usage:
    python setup_framework.py
    
Or with specific options:
    python setup_framework.py --install-deps --validate-data --run-test
"""

import sys
import subprocess
import argparse
from pathlib import Path
import importlib.util


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   This framework requires Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Read requirements
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Install requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def install_framework():
    """Install the framework in development mode."""
    print("\n🔧 Installing framework...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-e", "."
        ])
        print("✅ Framework installed in development mode")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install framework: {e}")
        return False


def validate_data_structure():
    """Validate the data directory structure."""
    print("\n📊 Validating data structure...")
    
    data_dir = Path("binance_futures_data")
    if not data_dir.exists():
        print("⚠️ Data directory 'binance_futures_data' not found")
        print("   Creating example directory structure...")
        
        # Create example structure
        (data_dir / "klines").mkdir(parents=True, exist_ok=True)
        (data_dir / "fundingRate").mkdir(parents=True, exist_ok=True)
        
        # Create example subdirectories
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            (data_dir / "klines" / symbol).mkdir(exist_ok=True)
            (data_dir / "fundingRate" / symbol).mkdir(exist_ok=True)
        
        print("✅ Created example data directory structure")
        print("   Please add your actual data files to these directories")
        return True
    
    # Check for required subdirectories
    klines_dir = data_dir / "klines"
    funding_dir = data_dir / "fundingRate"
    
    if not klines_dir.exists():
        print("❌ Missing 'klines' directory")
        return False
    
    if not funding_dir.exists():
        print("❌ Missing 'fundingRate' directory")
        return False
    
    # Check for at least some data
    klines_symbols = list(klines_dir.iterdir())
    if len(klines_symbols) == 0:
        print("⚠️ No symbol directories found in klines/")
        print("   Add symbol directories (e.g., BTCUSDT, ETHUSDT) with CSV files")
        return True
    
    print(f"✅ Data structure validated ({len(klines_symbols)} symbols found)")
    return True


def test_framework():
    """Run a quick test to ensure the framework works."""
    print("\n🧪 Testing framework...")
    
    try:
        # Add src to path for testing
        src_path = Path("src")
        if src_path.exists():
            sys.path.insert(0, str(src_path))
        
        # Try importing the framework
        from crypto_backtesting import DataManager, BacktestEngine, MeanReversionStrategy
        print("✅ Framework imports working")
        
        # Try creating basic objects
        data_manager = DataManager(data_path="binance_futures_data")
        print("✅ DataManager created successfully")
        
        engine = BacktestEngine(data_manager, initial_capital=100000)
        print("✅ BacktestEngine created successfully")
        
        strategy = MeanReversionStrategy(
            symbol1='BTCUSDT',
            symbol2='ETHUSDT',
            lookback_period=60,
            entry_threshold=2.0,
            exit_threshold=0.0,
            stop_loss_threshold=3.0
        )
        print("✅ Strategy created successfully")
        
        # Try getting available symbols
        try:
            symbols = data_manager.get_available_symbols()
            print(f"✅ Found {len(symbols)} available symbols")
            if len(symbols) > 0:
                print(f"   Sample symbols: {symbols[:3]}")
        except Exception as e:
            print(f"⚠️ Could not load symbols (expected if no data): {e}")
        
        print("✅ Framework test completed successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def create_example_configs():
    """Create example configuration files."""
    print("\n📋 Creating example configuration files...")
    
    # Create configs directory
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Example strategy config
    strategy_config = """# Example Strategy Configuration
strategy_configs:
  conservative:
    lookback_period: 90
    entry_threshold: 2.5
    exit_threshold: 0.0
    stop_loss_threshold: 4.0
    
  aggressive:
    lookback_period: 30
    entry_threshold: 1.5
    exit_threshold: 0.0
    stop_loss_threshold: 2.5
    
  balanced:
    lookback_period: 60
    entry_threshold: 2.0
    exit_threshold: 0.0
    stop_loss_threshold: 3.0
"""
    
    with open(configs_dir / "strategy_configs.yaml", "w") as f:
        f.write(strategy_config)
    
    # Example backtest config
    backtest_config = """# Example Backtest Configuration
backtest_configs:
  default:
    initial_capital: 100000.0
    commission_rate: 0.001
    slippage_rate: 0.0001
    
  low_cost:
    initial_capital: 100000.0
    commission_rate: 0.0005
    slippage_rate: 0.00005
    
  high_cost:
    initial_capital: 100000.0
    commission_rate: 0.002
    slippage_rate: 0.0002
"""
    
    with open(configs_dir / "backtest_configs.yaml", "w") as f:
        f.write(backtest_config)
    
    # Example pairs config
    pairs_config = """# Example Trading Pairs Configuration
trading_pairs:
  major_pairs:
    - [BTCUSDT, ETHUSDT]
    - [ADAUSDT, DOTUSDT]
    - [LINKUSDT, AVAXUSDT]
    
  defi_pairs:
    - [UNIUSDT, SUSHIUSDT]
    - [AAVEUSDT, COMPUSDT]
    - [MKRUSDT, YFIUSDT]
    
  layer1_pairs:
    - [ETHUSDT, ADAUSDT]
    - [SOLUSDT, AVAXUSDT]
    - [DOTUSDT, ATOMUSDT]
"""
    
    with open(configs_dir / "pairs_configs.yaml", "w") as f:
        f.write(pairs_config)
    
    print("✅ Created example configuration files in 'configs/' directory")
    return True


def create_quick_start_script():
    """Create a quick start script for new users."""
    print("\n🚀 Creating quick start script...")
    
    quick_start_content = '''#!/usr/bin/env python3
"""
Quick Start Script
=================

This script demonstrates the basic usage of the cryptocurrency backtesting framework.
Run this script to quickly test the framework with default settings.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from crypto_backtesting import (
    DataManager, BacktestEngine, MeanReversionStrategy,
    PerformanceAnalyzer
)

def main():
    print("🚀 Cryptocurrency Backtesting Framework - Quick Start")
    print("=" * 55)
    
    # Setup data manager
    data_manager = DataManager(data_path="binance_futures_data")
    
    # Get available symbols
    symbols = data_manager.get_available_symbols()
    print(f"Available symbols: {len(symbols)}")
    
    if len(symbols) < 2:
        print("❌ Need at least 2 symbols to run pairs trading")
        print("Please add data files to binance_futures_data/klines/")
        return
    
    # Use first two available symbols
    symbol1, symbol2 = symbols[0], symbols[1]
    print(f"Testing with pair: {symbol1} / {symbol2}")
    
    try:
        # Load data (using 2024 data if available)
        data1, data2 = data_manager.get_pair_data(symbol1, symbol2, 2024, [4, 5, 6])
        print(f"Loaded {len(data1)} candles")
        
        # Create strategy
        strategy = MeanReversionStrategy(
            symbol1=symbol1,
            symbol2=symbol2,
            lookback_period=60,
            entry_threshold=2.0,
            exit_threshold=0.0,
            stop_loss_threshold=3.0
        )
        
        # Prepare data
        combined_data = strategy.prepare_pair_data(data1, data2, symbol1, symbol2)
        rolling_mean = combined_data['spread'].rolling(60).mean()
        rolling_std = combined_data['spread'].rolling(60).std()
        combined_data['zscore'] = (combined_data['spread'] - rolling_mean) / rolling_std
        
        # Run backtest
        engine = BacktestEngine(data_manager, initial_capital=100000)
        results = engine.run_backtest(strategy, combined_data)
        
        # Display results
        print(f"\\n📊 Results:")
        print(f"Total Return: {results.total_return:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Number of Trades: {len(results.trades)}")
        print(f"Win Rate: {results.win_rate:.1%}")
        
        print("\\n✅ Quick start completed successfully!")
        print("Check out the examples/ directory for more advanced usage.")
        
    except Exception as e:
        print(f"❌ Error running quick start: {e}")
        print("This might be due to missing or incompatible data files.")

if __name__ == "__main__":
    main()
'''
    
    with open("quick_start.py", "w") as f:
        f.write(quick_start_content)
    
    print("✅ Created 'quick_start.py' script")
    return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Cryptocurrency Backtesting Framework")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies")
    parser.add_argument("--validate-data", action="store_true", help="Validate data structure")
    parser.add_argument("--run-test", action="store_true", help="Run framework test")
    parser.add_argument("--create-configs", action="store_true", help="Create example configs")
    parser.add_argument("--all", action="store_true", help="Run all setup steps")
    
    args = parser.parse_args()
    
    print("🎯 Cryptocurrency Backtesting Framework Setup")
    print("=" * 50)
    
    success = True
    
    # Check Python version first
    if not check_python_version():
        return False
    
    # Run selected steps
    if args.all or args.install_deps:
        success &= install_dependencies()
        success &= install_framework()
    
    if args.all or args.validate_data:
        success &= validate_data_structure()
    
    if args.all or args.create_configs:
        success &= create_example_configs()
    
    if args.all or args.run_test:
        success &= test_framework()
    
    # Always create quick start script
    create_quick_start_script()
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Add your data files to binance_futures_data/")
        print("2. Run: python quick_start.py")
        print("3. Explore examples in examples/ directory")
        print("4. Check out the documentation in README.md")
    else:
        print("❌ Setup completed with some issues")
        print("Please resolve the errors above before proceeding")
    
    return success


if __name__ == "__main__":
    main()
