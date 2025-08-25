#!/usr/bin/env python3
"""
Example demonstrating when log transformation is used in cointegration analysis
"""

import pandas as pd
import numpy as np


def demonstrate_log_transformation():
    """Show examples of when log transformation is applied"""

    print("📊 LOG TRANSFORMATION EXAMPLES")
    print("=" * 50)

    # Example 1: Similar price levels (no log transformation needed)
    print("\n1️⃣ SIMILAR PRICE LEVELS:")
    btc_price = 45000  # $45,000
    eth_price = 3000  # $3,000
    ratio1 = max(btc_price, eth_price) / min(btc_price, eth_price)
    use_log1 = ratio1 > 10

    print(f"   BTC: ${btc_price:,}")
    print(f"   ETH: ${eth_price:,}")
    print(f"   Price ratio: {ratio1:.2f}")
    print(f"   Use log transformation: {use_log1}")
    print(f"   → Normal OLS regression with raw prices")

    # Example 2: Large price differences (log transformation applied)
    print("\n2️⃣ LARGE PRICE DIFFERENCES:")
    btc_price = 45000  # $45,000
    shib_price = 0.00002  # $0.00002
    ratio2 = max(btc_price, shib_price) / min(btc_price, shib_price)
    use_log2 = ratio2 > 10

    print(f"   BTC: ${btc_price:,}")
    print(f"   SHIB: ${shib_price:.6f}")
    print(f"   Price ratio: {ratio2:,.0f}")
    print(f"   Use log transformation: {use_log2}")
    print(f"   → Log transformation applied for better model fit")

    # Example 3: Traditional stocks (borderline case)
    print("\n3️⃣ TRADITIONAL STOCKS:")
    aapl_price = 175  # $175
    brk_a_price = 450000  # $450,000 (Berkshire Hathaway Class A)
    ratio3 = max(aapl_price, brk_a_price) / min(aapl_price, brk_a_price)
    use_log3 = ratio3 > 10

    print(f"   AAPL: ${aapl_price:,}")
    print(f"   BRK.A: ${brk_a_price:,}")
    print(f"   Price ratio: {ratio3:.2f}")
    print(f"   Use log transformation: {use_log3}")
    print(f"   → Log transformation applied due to large difference")

    print("\n📊 MODEL DIFFERENCES:")
    print("   Without constant: y = β₁ × x + ε")
    print("   With constant:    y = α + β₁ × x + ε")
    print("   ✅ Constant term captures price level differences")
    print("   ✅ Log transformation handles multiplicative relationships")

    print("\n🎯 PRACTICAL IMPACT:")
    print("   • More accurate hedge ratios")
    print("   • Better model diagnostics (R², AIC, BIC)")
    print("   • Improved trading performance")
    print("   • Proper handling of different asset classes")


if __name__ == "__main__":
    demonstrate_log_transformation()
