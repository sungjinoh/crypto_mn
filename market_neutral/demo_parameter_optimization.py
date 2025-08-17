#!/usr/bin/env python3
"""
Demo script showing comprehensive parameter optimization
"""

from mean_reversion_backtest import MeanReversionBacktester
import warnings

warnings.filterwarnings("ignore")


def demo_strategy_optimization():
    """Demonstrate strategy parameter optimization"""

    print("=" * 80)
    print("STRATEGY PARAMETER OPTIMIZATION DEMO")
    print("=" * 80)

    # Initialize backtester
    backtester = MeanReversionBacktester(
        base_path="binance_futures_data",
        results_dir="cointegration_results",
        resample_timeframe="15T",  # 15-minute bars for reasonable speed
        save_plots=True,
        plots_dir="optimization_plots",
    )

    print("🔍 Loading cointegration results...")
    try:
        # Load cointegration results
        coint_results = backtester.load_cointegration_results()

        # Get top pair for demonstration
        top_pairs = backtester.filter_top_pairs(coint_results, n_pairs=1)

        if not top_pairs:
            print("❌ No suitable pairs found for optimization demo")
            return

        pair = top_pairs[0]
        symbol1, symbol2 = pair["symbol1"], pair["symbol2"]

        print(f"📊 Optimizing parameters for: {symbol1} - {symbol2}")
        print(f"   • Cointegration p-value: {pair['p_value']:.6f}")
        print(f"   • Correlation: {pair['correlation']:.4f}")

        # Load data
        print("📈 Loading market data...")
        df1, df2 = backtester.load_pair_data(symbol1, symbol2, 2024, [4, 5, 6])

        print(f"   • Data points: {len(df1):,}")
        print(f"   • Date range: {df1.index.min()} to {df1.index.max()}")

        # Define comprehensive parameter ranges
        param_ranges = {
            "lookback_period": [30, 45, 60, 90, 120],
            "entry_threshold": [1.0, 1.5, 2.0, 2.5, 3.0],
            "exit_threshold": [0.0, 0.25, 0.5, 0.75],
            "stop_loss_threshold": [2.0, 2.5, 3.0, 3.5, 4.0],
        }

        print(f"🔧 Parameter ranges:")
        for param, values in param_ranges.items():
            print(f"   • {param}: {values}")

        total_combinations = 1
        for values in param_ranges.values():
            total_combinations *= len(values)
        print(f"   • Total combinations: {total_combinations:,}")

        # Run optimization
        print(f"\n🚀 Starting optimization...")
        optimization_result = backtester.optimize_strategy_params(
            df1,
            df2,
            symbol1,
            symbol2,
            param_ranges=param_ranges,
            optimization_metric="sharpe_ratio",
            max_combinations=-1,  # Limit for demo
        )

        # Display results
        if optimization_result["best_params"]:
            best = optimization_result["best_params"]
            print(f"\n🏆 OPTIMIZATION RESULTS:")
            print(f"   • Best Sharpe Ratio: {best['sharpe_ratio']:.4f}")
            print(f"   • Total Return: {best['total_return']:.2%}")
            print(f"   • Max Drawdown: {best['max_drawdown']:.2%}")
            print(f"   • Win Rate: {best['win_rate']:.2%}")
            print(f"   • Number of Trades: {best['num_trades']}")

            print(f"\n⚙️ OPTIMAL PARAMETERS:")
            print(f"   • Lookback Period: {best['lookback_period']}")
            print(f"   • Entry Threshold: {best['entry_threshold']}")
            print(f"   • Exit Threshold: {best['exit_threshold']}")
            print(f"   • Stop Loss Threshold: {best['stop_loss_threshold']}")

            # Show top 5 parameter combinations
            all_results = optimization_result["all_results"]
            if len(all_results) >= 5:
                sorted_results = sorted(
                    all_results, key=lambda x: x["sharpe_ratio"], reverse=True
                )
                print(f"\n📊 TOP 5 PARAMETER COMBINATIONS:")
                print(
                    f"{'Rank':<5} {'Sharpe':<8} {'Return':<8} {'Lookback':<9} {'Entry':<7} {'Exit':<6} {'Stop':<6}"
                )
                print("-" * 60)
                for i, result in enumerate(sorted_results[:5], 1):
                    print(
                        f"{i:<5} {result['sharpe_ratio']:<8.3f} {result['total_return']:<8.2%} "
                        f"{result['lookback_period']:<9} {result['entry_threshold']:<7.1f} "
                        f"{result['exit_threshold']:<6.1f} {result['stop_loss_threshold']:<6.1f}"
                    )
        else:
            print("❌ No successful parameter combinations found")

    except Exception as e:
        print(f"❌ Optimization demo failed: {e}")


def demo_config_optimization():
    """Demonstrate backtesting configuration optimization"""

    print(f"\n{'=' * 80}")
    print("BACKTESTING CONFIGURATION OPTIMIZATION DEMO")
    print("=" * 80)

    # Initialize base backtester
    backtester = MeanReversionBacktester(
        base_path="binance_futures_data",
        results_dir="cointegration_results",
    )

    try:
        # Load cointegration results
        coint_results = backtester.load_cointegration_results()
        top_pairs = backtester.filter_top_pairs(coint_results, n_pairs=1)

        if not top_pairs:
            print("❌ No suitable pairs found for config optimization demo")
            return

        pair = top_pairs[0]
        symbol1, symbol2 = pair["symbol1"], pair["symbol2"]

        print(f"🔧 Optimizing backtesting config for: {symbol1} - {symbol2}")

        # Load data
        df1, df2 = backtester.load_pair_data(symbol1, symbol2, 2024, [4, 5])

        # Define configuration ranges
        config_ranges = {
            "resample_timeframe": [None, "5T", "15T", "1H"],
            "transaction_cost": [0.0005, 0.001, 0.002],
            "position_size": [0.3, 0.5, 0.8],
        }

        # Fixed strategy parameters for fair comparison
        strategy_params = {
            "lookback_period": 60,
            "entry_threshold": 2.0,
            "exit_threshold": 0.0,
            "stop_loss_threshold": 3.0,
        }

        print(f"🔍 Testing configurations:")
        for param, values in config_ranges.items():
            print(f"   • {param}: {values}")

        # Run configuration optimization
        config_result = backtester.optimize_backtesting_config(
            df1,
            df2,
            symbol1,
            symbol2,
            config_ranges=config_ranges,
            strategy_params=strategy_params,
        )

        # Display results
        if config_result["best_config"]:
            best = config_result["best_config"]
            print(f"\n🏆 BEST CONFIGURATION:")
            print(f"   • Timeframe: {best['resample_timeframe'] or '1T (1-minute)'}")
            print(f"   • Transaction Cost: {best['transaction_cost']:.4f}")
            print(f"   • Position Size: {best['position_size']:.2f}")
            print(f"   • Sharpe Ratio: {best['sharpe_ratio']:.4f}")
            print(f"   • Total Return: {best['total_return']:.2%}")

            # Show all configuration results
            all_configs = config_result["all_results"]
            if len(all_configs) > 1:
                sorted_configs = sorted(
                    all_configs, key=lambda x: x["sharpe_ratio"], reverse=True
                )
                print(f"\n📊 ALL CONFIGURATION RESULTS:")
                print(
                    f"{'Timeframe':<12} {'TxnCost':<8} {'PosSize':<8} {'Sharpe':<8} {'Return':<8} {'Trades':<7}"
                )
                print("-" * 65)
                for config in sorted_configs:
                    tf = config["resample_timeframe"] or "1T"
                    print(
                        f"{tf:<12} {config['transaction_cost']:<8.4f} {config['position_size']:<8.2f} "
                        f"{config['sharpe_ratio']:<8.3f} {config['total_return']:<8.2%} {config['num_trades']:<7}"
                    )
        else:
            print("❌ No successful configurations found")

    except Exception as e:
        print(f"❌ Configuration optimization demo failed: {e}")


def demo_combined_optimization():
    """Demonstrate combined strategy and configuration optimization"""

    print(f"\n{'=' * 80}")
    print("COMBINED OPTIMIZATION DEMO")
    print("=" * 80)

    print(
        "🎯 This demo shows how to optimize both strategy parameters AND backtesting configuration"
    )
    print("   Step 1: Optimize backtesting configuration with default strategy")
    print("   Step 2: Optimize strategy parameters with best configuration")
    print("   Step 3: Final validation with optimized settings")

    # Initialize backtester
    backtester = MeanReversionBacktester(
        base_path="binance_futures_data",
        results_dir="cointegration_results",
    )

    try:
        # Load data
        coint_results = backtester.load_cointegration_results()
        top_pairs = backtester.filter_top_pairs(coint_results, n_pairs=1)

        if not top_pairs:
            print("❌ No suitable pairs found")
            return

        pair = top_pairs[0]
        symbol1, symbol2 = pair["symbol1"], pair["symbol2"]
        df1, df2 = backtester.load_pair_data(symbol1, symbol2, 2024, [4, 5])

        print(f"\n📊 Optimizing: {symbol1} - {symbol2}")

        # Step 1: Optimize configuration
        print(f"\n🔧 Step 1: Optimizing backtesting configuration...")
        config_ranges = {
            "resample_timeframe": ["5T", "15T", "1H", "4H"],
            "transaction_cost": [0.001, 0.005, 0.003],
            "position_size": [0.5, 0.8],
        }

        config_result = backtester.optimize_backtesting_config(
            df1, df2, symbol1, symbol2, config_ranges=config_ranges
        )

        if not config_result["best_config"]:
            print("❌ Configuration optimization failed")
            return

        best_config = config_result["best_config"]
        print(
            f"✅ Best config found: {best_config['resample_timeframe']}, "
            f"cost={best_config['transaction_cost']:.4f}, size={best_config['position_size']:.2f}"
        )

        # Step 2: Create optimized backtester and optimize strategy
        print(f"\n⚙️ Step 2: Optimizing strategy with best configuration...")
        optimized_backtester = backtester.create_optimized_backtester(
            {
                "resample_timeframe": best_config["resample_timeframe"],
                "transaction_cost": best_config["transaction_cost"],
                "position_size": best_config["position_size"],
                "save_plots": True,
                "plots_dir": "combined_optimization_plots",
            }
        )

        param_ranges = {
            "lookback_period": [30, 45, 60, 90, 120],
            "entry_threshold": [1.0, 1.5, 2.0, 2.5, 3.0],
            "exit_threshold": [0.0, 0.25, 0.5, 0.75],
            "stop_loss_threshold": [2.0, 2.5, 3.0, 3.5, 4.0],
        }

        strategy_result = optimized_backtester.optimize_strategy_params(
            df1,
            df2,
            symbol1,
            symbol2,
            param_ranges=param_ranges,
            max_combinations=False,
        )

        if not strategy_result["best_params"]:
            print("❌ Strategy optimization failed")
            return

        best_strategy = strategy_result["best_params"]
        print(
            f"✅ Best strategy found: lookback={best_strategy['lookback_period']}, "
            f"entry={best_strategy['entry_threshold']:.2f}"
        )

        # Step 3: Final validation
        print(f"\n🎯 Step 3: Final validation with optimized settings...")
        print(f"\n🏆 FINAL OPTIMIZED RESULTS:")
        print(f"   📊 Performance Metrics:")
        print(f"      • Sharpe Ratio: {best_strategy['sharpe_ratio']:.4f}")
        print(f"      • Total Return: {best_strategy['total_return']:.2%}")
        print(f"      • Max Drawdown: {best_strategy['max_drawdown']:.2%}")
        print(f"      • Win Rate: {best_strategy['win_rate']:.2%}")
        print(f"      • Number of Trades: {best_strategy['num_trades']}")

        print(f"\n   ⚙️ Optimal Configuration:")
        print(f"      • Timeframe: {best_config['resample_timeframe']}")
        print(f"      • Transaction Cost: {best_config['transaction_cost']:.4f}")
        print(f"      • Position Size: {best_config['position_size']:.2f}")

        print(f"\n   📋 Optimal Strategy Parameters:")
        print(f"      • Lookback Period: {best_strategy['lookback_period']}")
        print(f"      • Entry Threshold: {best_strategy['entry_threshold']:.2f}")
        print(f"      • Exit Threshold: {best_strategy['exit_threshold']:.2f}")
        print(
            f"      • Stop Loss Threshold: {best_strategy['stop_loss_threshold']:.2f}"
        )

        print(f"\n💡 This optimized configuration can now be used for live trading!")

    except Exception as e:
        print(f"❌ Combined optimization demo failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting Parameter Optimization Demos")

    # Demo 1: Strategy parameter optimization
    # demo_strategy_optimization()

    # # Demo 2: Backtesting configuration optimization
    # demo_config_optimization()

    # Demo 3: Combined optimization
    demo_combined_optimization()

    print(f"\n🎉 All optimization demos completed!")
    print(f"Check the plot directories for visual results:")
    print(f"  • optimization_plots/")
    print(f"  • combined_optimization_plots/")
