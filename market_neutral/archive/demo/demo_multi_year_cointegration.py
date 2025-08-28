#!/usr/bin/env python3
"""
Demo script showing how to use the enhanced CointegrationFinder with multiple years.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from enhanced_cointegration_finder import CointegrationFinder


def demo_single_year():
    """Demo with single year (backward compatibility)."""
    print("=" * 80)
    print("DEMO: Single Year Analysis (2024)")
    print("=" * 80)

    finder = CointegrationFinder(
        base_path="binance_futures_data",
        resample_interval="1H",  # 1-hour candles for faster processing
        min_data_points=500,  # Lower requirement for demo
        significance_level=0.05,
        n_jobs=4,  # Use 4 cores for demo
    )

    # Single year analysis
    results = finder.find_all_cointegrated_pairs(
        years=[2024],  # Single year in list format
        months=[1, 2],  # Jan-Feb only for faster processing
        max_symbols=10,  # Limit symbols for demo
        use_parallel=True,
    )

    print(f"\nResults for 2024:")
    print(f"- Pairs tested: {results['metadata']['total_pairs_tested']}")
    print(f"- Cointegrated pairs: {results['metadata']['cointegrated_pairs_found']}")

    if results["cointegrated_pairs"]:
        print(f"\nTop 3 pairs:")
        for i, pair in enumerate(results["cointegrated_pairs"][:3], 1):
            print(
                f"{i}. {pair['symbol1']}-{pair['symbol2']}: p-value={pair['p_value']:.6f}"
            )

    return results


def demo_multi_year():
    """Demo with multiple years."""
    print("\n" + "=" * 80)
    print("DEMO: Multi-Year Analysis (2023-2024)")
    print("=" * 80)

    finder = CointegrationFinder(
        base_path="binance_futures_data",
        resample_interval="1H",  # 1-hour candles
        min_data_points=1000,  # Higher requirement for multi-year
        significance_level=0.05,
        n_jobs=4,
    )

    # Multi-year analysis
    results = finder.find_all_cointegrated_pairs(
        years=[2023, 2024],  # Multiple years
        months=[1, 2],  # Jan-Feb for each year
        max_symbols=10,  # Limit symbols for demo
        use_parallel=True,
    )

    print(f"\nResults for 2023-2024:")
    print(f"- Data years: {results['metadata']['data_years']}")
    print(f"- Data months: {results['metadata']['data_months']}")
    print(f"- Pairs tested: {results['metadata']['total_pairs_tested']}")
    print(f"- Cointegrated pairs: {results['metadata']['cointegrated_pairs_found']}")

    if results["cointegrated_pairs"]:
        print(f"\nTop 5 pairs:")
        for i, pair in enumerate(results["cointegrated_pairs"][:5], 1):
            print(
                f"{i}. {pair['symbol1']}-{pair['symbol2']}: "
                f"p-value={pair['p_value']:.6f}, "
                f"correlation={pair['correlation']:.4f}"
            )

            # Show data range
            print(
                f"   Data: {pair['start_date'][:10]} to {pair['end_date'][:10]} "
                f"({pair['data_points']} points)"
            )

    # Save results
    finder.save_results(
        results, output_dir="multi_year_cointegration_results", formats=["json", "csv"]
    )

    return results


def demo_per_year_months():
    """Demo with different months for each year."""
    print("\n" + "=" * 80)
    print("DEMO: Per-Year Month Specification")
    print("=" * 80)

    finder = CointegrationFinder(
        base_path="binance_futures_data",
        resample_interval="2H",  # 2-hour candles
        min_data_points=800,
        significance_level=0.05,
        n_jobs=4,
    )

    # Different months for each year
    results = finder.find_all_cointegrated_pairs(
        years=[2023, 2024],  # Two years
        months=[
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [1, 2, 3, 4, 5],
        ],  # All months for 2023, Jan-May for 2024
        max_symbols=8,  # Limit symbols for demo
        use_parallel=True,
    )

    print(f"\nResults for 2023 (all months) + 2024 (Jan-May):")
    print(f"- Data years: {results['metadata']['data_years']}")
    print(f"- Data months: {results['metadata']['data_months']}")
    print(f"- Pairs tested: {results['metadata']['total_pairs_tested']}")
    print(f"- Cointegrated pairs: {results['metadata']['cointegrated_pairs_found']}")

    if results["cointegrated_pairs"]:
        print(f"\nTop 3 pairs:")
        for i, pair in enumerate(results["cointegrated_pairs"][:3], 1):
            print(
                f"{i}. {pair['symbol1']}-{pair['symbol2']}: "
                f"p-value={pair['p_value']:.6f}, "
                f"data_points={pair['data_points']}"
            )

    return results


def demo_custom_years():
    """Demo with custom year selection."""
    print("\n" + "=" * 80)
    print("DEMO: Custom Years Analysis (2023, 2024, 2025)")
    print("=" * 80)

    finder = CointegrationFinder(
        base_path="binance_futures_data",
        resample_interval="2H",  # 2-hour candles
        min_data_points=800,
        significance_level=0.01,  # Stricter significance
        n_jobs=4,
    )

    # Custom years selection
    results = finder.find_all_cointegrated_pairs(
        years=[2023, 2024, 2025],  # Three years
        months=[1],  # January only
        max_symbols=8,  # Even fewer symbols
        use_parallel=True,
    )

    print(f"\nResults for 2023, 2024, 2025:")
    print(f"- Data years: {results['metadata']['data_years']}")
    print(f"- Significance level: {results['metadata']['significance_level']}")
    print(f"- Pairs tested: {results['metadata']['total_pairs_tested']}")
    print(f"- Cointegrated pairs: {results['metadata']['cointegrated_pairs_found']}")

    return results


def main():
    """Run all demos."""
    print("Multi-Year Cointegration Finder Demo")
    print("This demo shows how to use the enhanced CointegrationFinder")
    print("with support for multiple years of data.\n")

    try:
        # Demo 1: Single year (backward compatibility)
        results1 = demo_single_year()

        # Demo 2: Multiple years
        results2 = demo_multi_year()

        # Demo 3: Per-year month specification
        results3 = demo_per_year_months()

        # Demo 4: Custom years
        results4 = demo_custom_years()

        print("\n" + "=" * 80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("Key improvements:")
        print("✓ Support for multiple years of data")
        print("✓ Per-year month specification (different months for each year)")
        print("✓ Automatic data deduplication across years")
        print("✓ Enhanced metadata tracking")
        print("✓ Backward compatibility with single year")
        print("✓ Improved spread property analysis")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print(
            "Make sure you have the required data files in 'binance_futures_data' directory"
        )
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
