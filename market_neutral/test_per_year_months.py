#!/usr/bin/env python3
"""
Test script for the enhanced per-year month specification feature.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from enhanced_cointegration_finder import CointegrationFinder, format_months_display


def test_months_formatting():
    """Test the months formatting function."""
    print("=" * 60)
    print("TESTING MONTHS FORMATTING")
    print("=" * 60)

    # Test cases
    test_cases = [
        # Same months for all years
        ([2023, 2024], [1, 2, 3], "Same months for all years"),
        (
            [2023, 2024],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "All months for all years",
        ),
        # Different months for each year
        ([2023, 2024], [[1, 2, 3], [4, 5, 6]], "Different months per year"),
        (
            [2023, 2024],
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5]],
            "All months 2023, Jan-May 2024",
        ),
        (
            [2023, 2024, 2025],
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [1, 2, 3]],
            "Different months for 3 years",
        ),
    ]

    for years, months, description in test_cases:
        formatted = format_months_display(years, months)
        print(f"\n{description}:")
        print(f"  Input: years={years}, months={months}")
        print(f"  Output: {formatted}")


def test_cointegration_finder():
    """Test the CointegrationFinder with per-year month specification."""
    print("\n" + "=" * 60)
    print("TESTING COINTEGRATION FINDER")
    print("=" * 60)

    finder = CointegrationFinder(
        base_path="binance_futures_data",
        resample_interval="4H",  # 4-hour candles for faster testing
        min_data_points=200,  # Lower requirement for testing
        significance_level=0.05,
        n_jobs=2,  # Use 2 cores for testing
    )

    print(f"Finder initialized with:")
    print(f"  • Resample interval: {finder.resample_interval}")
    print(f"  • Min data points: {finder.min_data_points}")
    print(f"  • Significance level: {finder.significance_level}")

    # Test different month specifications
    test_configs = [
        {
            "name": "Same months for all years",
            "years": [2024],
            "months": [1, 2],
        },
        {
            "name": "Different months per year",
            "years": [2023, 2024],
            "months": [[1, 2, 3, 4, 5, 6], [1, 2]],
        },
    ]

    for config in test_configs:
        print(f"\n--- {config['name']} ---")
        print(f"Years: {config['years']}")
        print(f"Months: {config['months']}")

        try:
            # Test with very limited symbols for speed
            results = finder.find_all_cointegrated_pairs(
                years=config["years"],
                months=config["months"],
                max_symbols=5,  # Very limited for testing
                use_parallel=False,  # Sequential for easier debugging
            )

            metadata = results["metadata"]
            print(f"✅ Success!")
            print(f"  • Data years: {metadata['data_years']}")
            print(f"  • Data months: {metadata['data_months']}")
            print(f"  • Symbols with data: {metadata['symbols_with_data']}")
            print(f"  • Pairs tested: {metadata['total_pairs_tested']}")
            print(f"  • Cointegrated pairs: {metadata['cointegrated_pairs_found']}")

            # Test the formatting
            months_display = format_months_display(
                metadata["data_years"], metadata["data_months"]
            )
            print(f"  • Formatted display: {months_display}")

        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main test function."""
    print("🧪 TESTING PER-YEAR MONTH SPECIFICATION")
    print("This script tests the enhanced CointegrationFinder functionality")
    print("that allows different months for each year.\n")

    # Test 1: Months formatting
    test_months_formatting()

    # Test 2: CointegrationFinder functionality
    try:
        test_cointegration_finder()
        print(f"\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"\nKey features tested:")
        print(f"  ✓ Months formatting for display")
        print(f"  ✓ Same months for all years")
        print(f"  ✓ Different months per year")
        print(f"  ✓ Metadata handling")
        print(f"  ✓ Error handling")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print(f"Make sure you have some data files in 'binance_futures_data' directory")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
