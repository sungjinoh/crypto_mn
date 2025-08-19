"""
Monthly Cointegration Pair Generation Script
Runs historical cointegration analysis and saves tradable pairs for live trading.
"""

import os
from datetime import datetime
from src.crypto_backtesting.data.manager import DataManager
from src.crypto_backtesting.analysis.performance import PerformanceAnalyzer
from src.crypto_backtesting.strategies.pairs.mean_reversion import MeanReversionStrategy
from market_neutral.enhanced_cointegration_finder import CointegrationFinder

OUTPUT_DIR = "src/crypto_live_trading/state"
RESULTS_FILENAME = "cointegrated_pairs.json"

# Analysis parameters
RESAMPLE_INTERVAL = "4H"
MIN_DATA_POINTS = 1000
SIGNIFICANCE_LEVEL = 0.05
MAX_SYMBOLS = 100  # Set to None for all symbols


def main():
    print("=" * 80)
    print("MONTHLY COINTEGRATION PAIR GENERATION")
    print("=" * 80)

    finder = CointegrationFinder(
        base_path="/Users/fivestar/workspace/fivestar/crypto_mn/binance_futures_data",
        resample_interval=RESAMPLE_INTERVAL,
        min_data_points=MIN_DATA_POINTS,
        significance_level=SIGNIFICANCE_LEVEL,
        n_jobs=4,
    )

    results = finder.find_all_cointegrated_pairs(
        years=[2023, 2024],
        months=[
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ],
        max_symbols=MAX_SYMBOLS,
        use_parallel=True,
    )

    pairs = results.get("cointegrated_pairs", [])
    print(f"Found {len(pairs)} cointegrated pairs.")

    # Save pairs for live trading
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, RESULTS_FILENAME)
    import json

    with open(out_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved cointegrated pairs to {out_path}")

    print("Next: Use these pairs in live trading system.")


if __name__ == "__main__":
    main()
