"""
Example usage of different pairs trading strategies
"""

import sys
import os

sys.path.append("../../../")

import pandas as pd
import numpy as np
from typing import Optional, Dict, List

# Import backtesting framework
from backtesting_framework import (
    PairsBacktester,
    plot_backtest_results,
    find_cointegrated_pairs,
    analyze_spread_properties,
    generate_pair_report,
)

# Import strategies
from mean_reversion_strategy import (
    MeanReversionStrategy,
    AdaptiveMeanReversionStrategy,
    VolatilityAdjustedStrategy,
    MomentumMeanReversionStrategy,
)


def load_symbol_ohlcv(
    symbol: str, data_dir: str = "../../../binance_futures_data/pickle"
) -> Optional[pd.DataFrame]:
    """Load OHLCV data for a specific symbol"""
    try:
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        filepath = f"{data_dir}/{safe_symbol}_1m_ohlcv.pkl"

        if not os.path.exists(filepath):
            return None

        df = pd.read_pickle(filepath)
        df["symbol"] = symbol
        return df

    except Exception as e:
        print(f"Error loading OHLCV for {symbol}: {e}")
        return None


def compare_strategies_example():
    """Compare different strategies on the same pair"""
    print("🔄 Comparing Different Strategies")
    print("=" * 50)

    # Load data
    symbol1 = "BTC/USDT:USDT"
    symbol2 = "ETH/USDT:USDT"

    print(f"📊 Loading data for {symbol1} vs {symbol2}")
    data1 = load_symbol_ohlcv(symbol1)
    data2 = load_symbol_ohlcv(symbol2)

    if data1 is None or data2 is None:
        print("❌ Could not load data")
        return

    # Define strategies to compare
    strategies = {
        "Basic Mean Reversion": MeanReversionStrategy(
            lookback_period=60,
            entry_threshold=2.0,
            exit_threshold=0.0,
            stop_loss_threshold=3.0,
        ),
        "Adaptive Thresholds": AdaptiveMeanReversionStrategy(
            lookback_period=60,
            threshold_lookback=252,
            entry_percentile=95,
            exit_percentile=50,
        ),
        "Volatility Adjusted": VolatilityAdjustedStrategy(
            lookback_period=60,
            entry_threshold=2.0,
            volatility_lookback=20,
            vol_target=0.15,
        ),
        "Momentum Hybrid": MomentumMeanReversionStrategy(
            lookback_period=60,
            momentum_period=20,
            entry_threshold=2.0,
            momentum_weight=0.3,
        ),
    }

    # Backtest each strategy
    results = {}
    backtester = PairsBacktester(
        initial_capital=100000.0, transaction_cost=0.001, position_size=0.5
    )

    for strategy_name, strategy in strategies.items():
        print(f"\n🚀 Testing {strategy_name}...")

        try:
            result = backtester.run_backtest(
                symbol1_data=data1,
                symbol2_data=data2,
                symbol1=symbol1,
                symbol2=symbol2,
                strategy=strategy,
                start_date="2024-01-01",
                end_date="2024-12-31",
            )
            results[strategy_name] = result

            # Print key metrics
            print(f"  Total Return: {result.metrics.get('Total Return', 0):.4f}")
            print(f"  Sharpe Ratio: {result.metrics.get('Sharpe Ratio', 0):.4f}")
            print(f"  Max Drawdown: {result.metrics.get('Max Drawdown', 0):.4f}")
            print(f"  Total Trades: {result.metrics.get('Total Trades', 0)}")
            print(f"  Win Rate: {result.metrics.get('Win Rate', 0):.2%}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Compare results
    if results:
        print(f"\n📊 STRATEGY COMPARISON SUMMARY")
        print("=" * 60)

        comparison_df = pd.DataFrame(
            {
                name: {
                    "Total Return": result.metrics.get("Total Return", 0),
                    "Sharpe Ratio": result.metrics.get("Sharpe Ratio", 0),
                    "Max Drawdown": result.metrics.get("Max Drawdown", 0),
                    "Total Trades": result.metrics.get("Total Trades", 0),
                    "Win Rate": result.metrics.get("Win Rate", 0),
                }
                for name, result in results.items()
            }
        ).T

        print(comparison_df.round(4))

        # Find best strategy by Sharpe ratio
        best_strategy = comparison_df["Sharpe Ratio"].idxmax()
        print(f"\n🏆 Best Strategy by Sharpe Ratio: {best_strategy}")

        # Plot best strategy results
        if best_strategy in results:
            print(f"\n📈 Plotting results for {best_strategy}...")
            plot_backtest_results(results[best_strategy], symbol1, symbol2)

    return results


def parameter_optimization_example():
    """Example of parameter optimization for mean reversion strategy"""
    print("\n🔧 Parameter Optimization Example")
    print("=" * 50)

    # Load data
    symbol1 = "BTC/USDT:USDT"
    symbol2 = "ETH/USDT:USDT"

    data1 = load_symbol_ohlcv(symbol1)
    data2 = load_symbol_ohlcv(symbol2)

    if data1 is None or data2 is None:
        print("❌ Could not load data")
        return

    # Parameter ranges to test
    param_ranges = {
        "lookback_period": [30, 60, 90],
        "entry_threshold": [1.5, 2.0, 2.5],
        "stop_loss_threshold": [2.5, 3.0, 3.5],
    }

    optimization_results = []
    backtester = PairsBacktester(
        initial_capital=100000.0, transaction_cost=0.001, position_size=0.5
    )

    total_combinations = (
        len(param_ranges["lookback_period"])
        * len(param_ranges["entry_threshold"])
        * len(param_ranges["stop_loss_threshold"])
    )

    print(f"🔄 Testing {total_combinations} parameter combinations...")

    combination = 0
    for lookback in param_ranges["lookback_period"]:
        for entry_thresh in param_ranges["entry_threshold"]:
            for stop_thresh in param_ranges["stop_loss_threshold"]:
                if stop_thresh <= entry_thresh:
                    continue

                combination += 1
                print(
                    f"  {combination}/{total_combinations}: lookback={lookback}, entry={entry_thresh}, stop={stop_thresh}"
                )

                strategy = MeanReversionStrategy(
                    lookback_period=lookback,
                    entry_threshold=entry_thresh,
                    exit_threshold=0.0,
                    stop_loss_threshold=stop_thresh,
                )

                try:
                    result = backtester.run_backtest(
                        symbol1_data=data1,
                        symbol2_data=data2,
                        symbol1=symbol1,
                        symbol2=symbol2,
                        strategy=strategy,
                        start_date="2024-01-01",
                        end_date="2024-12-31",
                    )

                    optimization_results.append(
                        {
                            "lookback_period": lookback,
                            "entry_threshold": entry_thresh,
                            "stop_loss_threshold": stop_thresh,
                            "total_return": result.metrics.get("Total Return", 0),
                            "sharpe_ratio": result.metrics.get("Sharpe Ratio", 0),
                            "max_drawdown": result.metrics.get("Max Drawdown", 0),
                            "total_trades": result.metrics.get("Total Trades", 0),
                            "win_rate": result.metrics.get("Win Rate", 0),
                        }
                    )

                except Exception as e:
                    print(f"    ❌ Error: {e}")

    if optimization_results:
        # Convert to DataFrame
        opt_df = pd.DataFrame(optimization_results)

        # Find best parameters
        best_sharpe = opt_df.loc[opt_df["sharpe_ratio"].idxmax()]
        best_return = opt_df.loc[opt_df["total_return"].idxmax()]

        print(f"\n🏆 OPTIMIZATION RESULTS")
        print("=" * 40)

        print(f"\nBest Sharpe Ratio: {best_sharpe['sharpe_ratio']:.4f}")
        print(
            f"  Parameters: lookback={best_sharpe['lookback_period']}, "
            f"entry={best_sharpe['entry_threshold']}, stop={best_sharpe['stop_loss_threshold']}"
        )
        print(f"  Return: {best_sharpe['total_return']:.4f}")

        print(f"\nBest Total Return: {best_return['total_return']:.4f}")
        print(
            f"  Parameters: lookback={best_return['lookback_period']}, "
            f"entry={best_return['entry_threshold']}, stop={best_return['stop_loss_threshold']}"
        )
        print(f"  Sharpe: {best_return['sharpe_ratio']:.4f}")

        # Show top 5 by Sharpe ratio
        print(f"\n📊 Top 5 Parameter Combinations (by Sharpe Ratio):")
        top_5 = opt_df.nlargest(5, "sharpe_ratio")
        print(
            top_5[
                [
                    "lookback_period",
                    "entry_threshold",
                    "stop_loss_threshold",
                    "total_return",
                    "sharpe_ratio",
                    "max_drawdown",
                ]
            ].to_string(index=False)
        )

        return opt_df

    return None


def pair_discovery_example():
    """Example of discovering cointegrated pairs"""
    print("\n🔍 Pair Discovery Example")
    print("=" * 50)

    # Define symbols to test
    test_symbols = [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "BNB/USDT:USDT",
        "ADA/USDT:USDT",
        "DOT/USDT:USDT",
        "LINK/USDT:USDT",
        "UNI/USDT:USDT",
        "SOL/USDT:USDT",
        "AVAX/USDT:USDT",
    ]

    # Load price data
    price_data = {}
    for symbol in test_symbols:
        data = load_symbol_ohlcv(symbol)
        if data is not None:
            price_data[symbol] = data["close"]
            print(f"✅ Loaded {symbol}: {len(data)} bars")
        else:
            print(f"❌ Failed to load {symbol}")

    if len(price_data) < 2:
        print("❌ Need at least 2 symbols with data")
        return

    # Find cointegrated pairs
    print(f"\n🔬 Testing {len(price_data)} symbols for cointegration...")
    cointegrated_pairs = find_cointegrated_pairs(
        price_data, significance_level=0.05, min_correlation=0.7
    )

    if cointegrated_pairs:
        print(f"\n🎉 Found {len(cointegrated_pairs)} cointegrated pairs:")

        for i, (symbol1, symbol2, results) in enumerate(
            cointegrated_pairs[:5]
        ):  # Show top 5
            print(f"\n{i+1}. {symbol1} vs {symbol2}")
            print(f"   P-value: {results['p_value']:.6f}")
            print(f"   Correlation: {results['correlation']:.4f}")
            print(f"   Hedge ratio: {results['hedge_ratio']:.4f}")
            print(f"   Residuals stationary: {results['residuals_stationary']}")

            # Generate detailed report for top pair
            if i == 0:
                print(f"\n📋 Detailed Report for Best Pair:")
                report = generate_pair_report(
                    symbol1, symbol2, price_data[symbol1], price_data[symbol2]
                )
                print(report)

        return cointegrated_pairs
    else:
        print("❌ No cointegrated pairs found with current criteria")
        return None


def main():
    """Run all examples"""
    print("🚀 Pairs Trading Strategy Examples")
    print("=" * 60)

    try:
        # 1. Compare different strategies
        strategy_results = compare_strategies_example()

        # 2. Parameter optimization
        optimization_results = parameter_optimization_example()

        # 3. Pair discovery
        cointegrated_pairs = pair_discovery_example()

        print(f"\n🎉 All examples completed successfully!")

        return {
            "strategy_results": strategy_results,
            "optimization_results": optimization_results,
            "cointegrated_pairs": cointegrated_pairs,
        }

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
