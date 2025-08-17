"""
Quick Start Example
==================

This example demonstrates the basic usage of the cryptocurrency backtesting framework.
It shows how to:
1. Load data for a trading pair
2. Create a mean reversion strategy
3. Run a backtest
4. Analyze the results

Requirements:
- Binance futures data in the correct format
- Framework installed (pip install -e .)
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto_backtesting import (
    DataManager, BacktestEngine, MeanReversionStrategy,
    PerformanceAnalyzer, ReportGenerator
)


def main():
    """Run a quick start example of pairs trading backtest."""
    
    print("🚀 Cryptocurrency Backtesting Framework - Quick Start Example")
    print("=" * 65)
    
    # 1. Setup data manager
    print("\n📊 Setting up data manager...")
    data_path = Path(__file__).parent.parent / "binance_futures_data"
    data_manager = DataManager(data_path=str(data_path))
    
    # Check available symbols
    available_symbols = data_manager.get_available_symbols()
    print(f"Available symbols: {len(available_symbols)} symbols")
    print(f"Sample symbols: {available_symbols[:5]}")
    
    # 2. Load data for a trading pair
    print("\n📈 Loading data for BTCUSDT and ETHUSDT...")
    try:
        data1, data2 = data_manager.get_pair_data(
            'BTCUSDT', 'ETHUSDT', 
            year=2024, 
            months=[4, 5, 6]  # Q2 2024
        )
        print(f"Loaded {len(data1)} candles for BTCUSDT")
        print(f"Loaded {len(data2)} candles for ETHUSDT")
        print(f"Date range: {data1.index[0]} to {data1.index[-1]}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Please ensure your data is in the correct format and location.")
        return
    
    # 3. Create mean reversion strategy
    print("\n🎯 Creating mean reversion strategy...")
    strategy = MeanReversionStrategy(
        symbol1='BTCUSDT',
        symbol2='ETHUSDT',
        lookback_period=60,
        entry_threshold=2.0,
        exit_threshold=0.0,
        stop_loss_threshold=3.0
    )
    
    # 4. Prepare data for strategy
    print("📊 Preparing data for strategy...")
    combined_data = strategy.prepare_pair_data(data1, data2, 'BTCUSDT', 'ETHUSDT')
    
    # Add technical indicators
    rolling_mean = combined_data['spread'].rolling(60).mean()
    rolling_std = combined_data['spread'].rolling(60).std()
    combined_data['zscore'] = (combined_data['spread'] - rolling_mean) / rolling_std
    
    print(f"Combined data shape: {combined_data.shape}")
    print(f"Spread range: {combined_data['spread'].min():.4f} to {combined_data['spread'].max():.4f}")
    
    # 5. Run backtest
    print("\n⚡ Running backtest...")
    engine = BacktestEngine(
        data_manager,
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0001
    )
    
    try:
        results = engine.run_backtest(strategy, combined_data)
        print("✅ Backtest completed successfully!")
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return
    
    # 6. Display basic results
    print("\n📊 Basic Results:")
    print("-" * 30)
    print(f"Initial Capital: ${results.initial_capital:,.2f}")
    print(f"Final Capital: ${results.final_capital:,.2f}")
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Number of Trades: {len(results.trades)}")
    print(f"Win Rate: {results.win_rate:.1%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    
    # 7. Generate performance analysis
    print("\n📈 Generating performance analysis...")
    analyzer = PerformanceAnalyzer(results)
    performance_report = analyzer.generate_performance_report()
    
    print("\n📋 Performance Summary:")
    print("-" * 30)
    summary_metrics = performance_report['summary_metrics']
    for metric, value in summary_metrics.items():
        if isinstance(value, float):
            if 'rate' in metric or 'ratio' in metric:
                print(f"{metric.replace('_', ' ').title()}: {value:.3f}")
            elif 'return' in metric or 'drawdown' in metric:
                print(f"{metric.replace('_', ' ').title()}: {value:.2%}")
            else:
                print(f"{metric.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{metric.replace('_', ' ').title()}: {value}")
    
    # 8. Trade analysis
    print("\n💼 Trade Analysis:")
    print("-" * 30)
    trade_analysis = performance_report['trade_analysis']
    print(f"Total Trades: {trade_analysis['total_trades']}")
    print(f"Profitable Trades: {trade_analysis['profitable_trades']}")
    print(f"Losing Trades: {trade_analysis['losing_trades']}")
    print(f"Win Rate: {trade_analysis['win_rate']:.1%}")
    print(f"Average Trade PnL: ${trade_analysis['avg_trade_pnl']:.2f}")
    print(f"Best Trade: ${trade_analysis['best_trade']:.2f}")
    print(f"Worst Trade: ${trade_analysis['worst_trade']:.2f}")
    print(f"Profit Factor: {trade_analysis['profit_factor']:.2f}")
    
    # 9. Risk metrics
    print("\n⚠️ Risk Metrics:")
    print("-" * 30)
    risk_metrics = performance_report['risk_metrics']
    print(f"VaR (99%): {risk_metrics['var_99']:.2%}")
    print(f"CVaR (99%): {risk_metrics['cvar_99']:.2%}")
    print(f"Max Consecutive Losses: {risk_metrics['max_consecutive_losses']}")
    print(f"Downside Capture: {risk_metrics['downside_capture']:.2f}")
    print(f"Upside Capture: {risk_metrics['upside_capture']:.2f}")
    
    # 10. Generate visual report (optional)
    try:
        print("\n📊 Generating visual report...")
        output_dir = Path(__file__).parent.parent / "reports"
        output_dir.mkdir(exist_ok=True)
        
        generator = ReportGenerator(results, output_dir=str(output_dir))
        full_report = generator.generate_full_report(save_plots=True)
        
        print(f"✅ Visual report saved to: {output_dir}")
        print("Generated files:")
        for file_path in output_dir.glob("*.png"):
            print(f"  - {file_path.name}")
            
    except Exception as e:
        print(f"⚠️ Could not generate visual report: {e}")
        print("(This is optional and doesn't affect the core functionality)")
    
    print("\n🎉 Quick start example completed successfully!")
    print("\nNext steps:")
    print("1. Try modifying strategy parameters")
    print("2. Test with different trading pairs")
    print("3. Explore parameter optimization")
    print("4. Create custom strategies")
    print("\nCheck out other examples in the examples/ directory!")


if __name__ == "__main__":
    main()
