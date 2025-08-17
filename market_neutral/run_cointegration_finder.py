#!/usr/bin/env python3
"""
Quick runner script for cointegration analysis.
Run this to find cointegrated pairs from Jan-March 2024 data.
"""

from enhanced_cointegration_finder import CointegrationFinder
import warnings

warnings.filterwarnings("ignore")


def main():
    print("=" * 80)
    print("STARTING COINTEGRATION ANALYSIS")
    print("=" * 80)

    # Initialize the finder with your settings
    finder = CointegrationFinder(
        base_path="binance_futures_data",
        resample_interval="30T",  # 30-minute candles
        min_data_points=1000,  # Minimum 1000 data points
        significance_level=0.05,  # 5% significance level
        n_jobs=4,  # Use 4 CPU cores (adjust as needed)
    )

    # For testing, let's start with a limited number of symbols
    # You can set max_symbols=None to test all symbols
    print("\nRunning analysis with limited symbols for testing...")
    print("Set max_symbols=None to analyze all available symbols\n")

    results = finder.find_all_cointegrated_pairs(
        year=2024,
        months=[1, 2, 3],  # January to March
        max_symbols=20,  # Limit to 20 symbols for quick testing
        use_parallel=True,  # Use parallel processing
    )

    # Print results summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    meta = results["metadata"]
    print(f"Symbols analyzed: {meta['symbols_with_data']}")
    print(f"Total pairs tested: {meta['total_pairs_tested']}")
    print(f"Cointegrated pairs found: {meta['cointegrated_pairs_found']}")

    if meta["cointegrated_pairs_found"] > 0:
        success_rate = (
            meta["cointegrated_pairs_found"] / meta["total_pairs_tested"]
        ) * 100
        print(f"Success rate: {success_rate:.2f}%")

        # Show top 5 pairs
        print(f"\nTop 5 cointegrated pairs:")
        print("-" * 40)
        for i, pair in enumerate(results["cointegrated_pairs"][:5], 1):
            print(f"{i}. {pair['symbol1']:10s} - {pair['symbol2']:10s}")
            print(f"   p-value: {pair['p_value']:.6f}")
            print(f"   hedge ratio: {pair['hedge_ratio']:.4f}")
            print(f"   correlation: {pair['correlation']:.4f}")
            if "spread_properties" in pair and pair["spread_properties"]:
                props = pair["spread_properties"]
                if "half_life" in props and props["half_life"]:
                    print(f"   half-life: {props['half_life']:.1f} periods")
            print()

    # Save results
    print("Saving results...")
    finder.save_results(
        results,
        output_dir="cointegration_results",
        formats=["json", "csv", "pickle"],  # Save in multiple formats
    )

    print("\n✅ Results saved to 'cointegration_results' directory")
    print("\nNext steps:")
    print("1. Review the results in the cointegration_results directory")
    print("2. Use load_and_use_results.py to load and filter pairs")
    print("3. Run with max_symbols=None for full analysis")

    return results


if __name__ == "__main__":
    results = main()
