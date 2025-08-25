#!/usr/bin/env python3
"""
Runner script for enhanced_cointegration_finder.py
This script ensures proper Python path setup and runs the cointegration finder.
"""

import os
import sys

# Add current directory to Python path to ensure all modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import and run the enhanced cointegration finder
if __name__ == "__main__":
    try:
        # Test imports
        print("🔍 Testing imports...")
        from backtesting_framework.pairs_backtester import PairsBacktester

        print("✅ backtesting_framework imported successfully")

        from market_neutral.enhanced_cointegration_finder import CointegrationFinder

        print("✅ CointegrationFinder imported successfully")

        # You can now run your cointegration finding logic here
        print("\n🚀 Ready to run cointegration analysis!")
        print("You can now import and use CointegrationFinder:")
        print("   finder = CointegrationFinder()")
        print("   # Add your analysis code here")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're running this from the crypto_mn directory")
        print("2. Check that all required packages are installed:")
        print("   pip install pandas numpy scipy statsmodels tqdm")
        print("3. Verify the backtesting_framework directory exists")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise
