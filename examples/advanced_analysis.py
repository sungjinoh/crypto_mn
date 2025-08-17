"""
Advanced Analysis Example
========================

This example demonstrates advanced performance analysis and reporting capabilities.
It shows how to:
1. Generate comprehensive performance reports
2. Create advanced visualizations
3. Perform risk analysis
4. Compare multiple strategies
5. Generate detailed PDF reports

Requirements:
- Binance futures data in the correct format
- Framework installed (pip install -e .)
- matplotlib, seaborn for visualizations
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto_backtesting import (
    DataManager, BacktestEngine, MeanReversionStrategy,
    PerformanceAnalyzer, ReportGenerator
)
from crypto_backtesting.analysis.performance import compare_strategies


def advanced_performance_analysis():
    """Demonstrate advanced performance analysis capabilities."""
    
    print("📊 Advanced Performance Analysis Example")
    print("=" * 45)
    
    # Setup
    data_path = Path(__file__).parent.parent / "binance_futures_data"
    data_manager = DataManager(data_path=str(data_path))
    
    # Load data
    print("\n📈 Loading data...")
    try:
        data1, data2 = data_manager.get_pair_data(
            'BTCUSDT', 'ETHUSDT', 
            year=2024, 
            months=[4, 5, 6]
        )
        print(f"Loaded data: {len(data1)} candles")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create multiple strategy configurations for comparison
    strategy_configs = [
        {
            'name': 'Conservative',
            'params': {
                'lookback_period': 90,
                'entry_threshold': 2.5,
                'exit_threshold': 0.0,
                'stop_loss_threshold': 4.0
            }
        },
        {
            'name': 'Aggressive',
            'params': {
                'lookback_period': 30,
                'entry_threshold': 1.5,
                'exit_threshold': 0.0,
                'stop_loss_threshold': 2.5
            }
        },
        {
            'name': 'Balanced',
            'params': {
                'lookback_period': 60,
                'entry_threshold': 2.0,
                'exit_threshold': 0.0,
                'stop_loss_threshold': 3.0
            }
        }
    ]
    
    # Run backtests for all strategies
    print("\n⚡ Running backtests for multiple strategies...")
    backtest_results = []
    strategy_names = []
    
    engine = BacktestEngine(
        data_manager,
        initial_capital=100000,
        commission_rate=0.001
    )
    
    for config in strategy_configs:
        print(f"Testing {config['name']} strategy...")
        
        # Create strategy
        strategy = MeanReversionStrategy(
            symbol1='BTCUSDT',
            symbol2='ETHUSDT',
            **config['params']
        )
        
        # Prepare data
        combined_data = strategy.prepare_pair_data(data1, data2, 'BTCUSDT', 'ETHUSDT')
        
        # Add technical indicators
        lookback = config['params']['lookback_period']
        rolling_mean = combined_data['spread'].rolling(lookback).mean()
        rolling_std = combined_data['spread'].rolling(lookback).std()
        combined_data['zscore'] = (combined_data['spread'] - rolling_mean) / rolling_std
        
        # Run backtest
        try:
            results = engine.run_backtest(strategy, combined_data)
            backtest_results.append(results)
            strategy_names.append(config['name'])
            print(f"✅ {config['name']}: {results.total_return:.2%} return")
        except Exception as e:
            print(f"❌ Error with {config['name']}: {e}")
    
    if not backtest_results:
        print("No successful backtests to analyze")
        return
    
    # Advanced Performance Analysis for each strategy
    print("\n📊 Detailed Performance Analysis:")
    print("=" * 50)
    
    for i, (results, name) in enumerate(zip(backtest_results, strategy_names)):
        print(f"\n{name} Strategy Analysis:")
        print("-" * 30)
        
        analyzer = PerformanceAnalyzer(results)
        performance_report = analyzer.generate_performance_report()
        
        # Display key metrics
        summary = performance_report['summary_metrics']
        print(f"Total Return: {summary['total_return']:.2%}")
        print(f"Annualized Return: {summary['annualized_return']:.2%}")
        print(f"Volatility: {summary['volatility']:.2%}")
        print(f"Sharpe Ratio: {summary['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {summary['sortino_ratio']:.3f}")
        print(f"Max Drawdown: {summary['max_drawdown']:.2%}")
        print(f"Calmar Ratio: {summary['calmar_ratio']:.3f}")
        
        # Risk metrics
        risk = performance_report['risk_metrics']
        print(f"VaR (95%): {risk['var_95']:.2%}")
        print(f"CVaR (95%): {risk['cvar_95']:.2%}")
        print(f"Skewness: {risk['skewness']:.3f}")
        print(f"Kurtosis: {risk['kurtosis']:.3f}")
        
        # Trade analysis
        trade = performance_report['trade_analysis']
        print(f"Total Trades: {trade['total_trades']}")
        print(f"Win Rate: {trade['win_rate']:.1%}")
        print(f"Profit Factor: {trade['profit_factor']:.2f}")
        print(f"Average Trade: {trade['avg_trade_return']:.2%}")
    
    # Strategy Comparison
    print("\n🏆 Strategy Comparison:")
    print("=" * 30)
    
    comparison_df = compare_strategies(backtest_results, strategy_names)
    print("\nComparison Table:")
    print(comparison_df.round(3))
    
    # Risk-Return Analysis
    print("\n⚠️ Risk-Return Analysis:")
    print("-" * 30)
    
    for i, (results, name) in enumerate(zip(backtest_results, strategy_names)):
        risk_return_ratio = results.total_return / results.max_drawdown if results.max_drawdown != 0 else 0
        print(f"{name}: Return/Risk = {risk_return_ratio:.2f}")
    
    # Generate Visual Reports
    print("\n📊 Generating visual reports...")
    output_dir = Path(__file__).parent.parent / "reports"
    output_dir.mkdir(exist_ok=True)
    
    for i, (results, name) in enumerate(zip(backtest_results, strategy_names)):
        try:
            strategy_output_dir = output_dir / name.lower()
            strategy_output_dir.mkdir(exist_ok=True)
            
            generator = ReportGenerator(results, output_dir=str(strategy_output_dir))
            full_report = generator.generate_full_report(save_plots=True)
            
            print(f"✅ {name} strategy report saved to: {strategy_output_dir}")
            
        except Exception as e:
            print(f"⚠️ Could not generate report for {name}: {e}")
    
    # Performance Attribution Analysis
    print("\n🔍 Performance Attribution Analysis:")
    print("-" * 40)
    
    for results, name in zip(backtest_results, strategy_names):
        if len(results.trades) > 0:
            winning_trades = [t for t in results.trades if t.pnl > 0]
            losing_trades = [t for t in results.trades if t.pnl <= 0]
            
            if winning_trades:
                avg_winner = sum(t.pnl for t in winning_trades) / len(winning_trades)
                largest_winner = max(t.pnl for t in winning_trades)
                print(f"\n{name} - Winning Trades:")
                print(f"  Count: {len(winning_trades)}")
                print(f"  Average: ${avg_winner:.2f}")
                print(f"  Largest: ${largest_winner:.2f}")
            
            if losing_trades:
                avg_loser = sum(t.pnl for t in losing_trades) / len(losing_trades)
                largest_loser = min(t.pnl for t in losing_trades)
                print(f"{name} - Losing Trades:")
                print(f"  Count: {len(losing_trades)}")
                print(f"  Average: ${avg_loser:.2f}")
                print(f"  Largest: ${largest_loser:.2f}")
    
    # Time-based Performance Analysis
    print("\n📅 Time-based Performance Analysis:")
    print("-" * 40)
    
    for results, name in zip(backtest_results, strategy_names):
        if hasattr(results, 'equity_curve') and len(results.equity_curve) > 0:
            # Calculate monthly returns (simplified)
            equity_series = results.equity_curve
            monthly_returns = equity_series.resample('M').last().pct_change().dropna()
            
            if len(monthly_returns) > 0:
                print(f"\n{name} Monthly Performance:")
                print(f"  Best Month: {monthly_returns.max():.2%}")
                print(f"  Worst Month: {monthly_returns.min():.2%}")
                print(f"  Avg Monthly: {monthly_returns.mean():.2%}")
                print(f"  Monthly Volatility: {monthly_returns.std():.2%}")
                print(f"  Positive Months: {(monthly_returns > 0).sum()}/{len(monthly_returns)}")
    
    return backtest_results, strategy_names


def risk_analysis_deep_dive():
    """Perform detailed risk analysis on a single strategy."""
    
    print("\n\n🔬 Deep Dive Risk Analysis")
    print("=" * 35)
    
    # Setup and run a single backtest
    data_path = Path(__file__).parent.parent / "binance_futures_data"
    data_manager = DataManager(data_path=str(data_path))
    
    try:
        data1, data2 = data_manager.get_pair_data('BTCUSDT', 'ETHUSDT', 2024, [4, 5, 6])
        
        strategy = MeanReversionStrategy(
            symbol1='BTCUSDT',
            symbol2='ETHUSDT',
            lookback_period=60,
            entry_threshold=2.0,
            exit_threshold=0.0,
            stop_loss_threshold=3.0
        )
        
        combined_data = strategy.prepare_pair_data(data1, data2, 'BTCUSDT', 'ETHUSDT')
        rolling_mean = combined_data['spread'].rolling(60).mean()
        rolling_std = combined_data['spread'].rolling(60).std()
        combined_data['zscore'] = (combined_data['spread'] - rolling_mean) / rolling_std
        
        engine = BacktestEngine(data_manager, initial_capital=100000)
        results = engine.run_backtest(strategy, combined_data)
        
        analyzer = PerformanceAnalyzer(results)
        
        # Detailed Risk Metrics
        print("\n📊 Detailed Risk Metrics:")
        print("-" * 30)
        
        performance_report = analyzer.generate_performance_report()
        risk_metrics = performance_report['risk_metrics']
        
        print(f"Value at Risk (VaR):")
        print(f"  95% VaR: {risk_metrics['var_95']:.2%}")
        print(f"  99% VaR: {risk_metrics.get('var_99', 0):.2%}")
        print(f"Conditional VaR (CVaR):")
        print(f"  95% CVaR: {risk_metrics['cvar_95']:.2%}")
        print(f"Distribution Metrics:")
        print(f"  Skewness: {risk_metrics['skewness']:.3f}")
        print(f"  Kurtosis: {risk_metrics['kurtosis']:.3f}")
        print(f"Drawdown Metrics:")
        print(f"  Max Drawdown: {risk_metrics['max_drawdown']:.2%}")
        print(f"  Avg Drawdown: {risk_metrics.get('avg_drawdown', 0):.2%}")
        
        # Risk-Adjusted Returns
        print(f"\nRisk-Adjusted Returns:")
        print(f"  Sharpe Ratio: {risk_metrics['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {risk_metrics['sortino_ratio']:.3f}")
        print(f"  Calmar Ratio: {risk_metrics['calmar_ratio']:.3f}")
        
        # Trade Risk Analysis
        if len(results.trades) > 0:
            trade_returns = [t.pnl / results.initial_capital for t in results.trades]
            consecutive_losses = analyzer._calculate_consecutive_losses(results.trades)
            
            print(f"\nTrade Risk Analysis:")
            print(f"  Largest Single Loss: {min(trade_returns):.2%}")
            print(f"  Max Consecutive Losses: {max(consecutive_losses) if consecutive_losses else 0}")
            print(f"  Trade Return Volatility: {analyzer._calculate_std(trade_returns):.2%}")
        
        print("✅ Risk analysis completed!")
        
    except Exception as e:
        print(f"❌ Error in risk analysis: {e}")


def main():
    """Run advanced analysis examples."""
    
    print("🎯 Cryptocurrency Backtesting Framework - Advanced Analysis")
    print("=" * 65)
    
    # Run comprehensive strategy comparison
    backtest_results, strategy_names = advanced_performance_analysis()
    
    # Perform detailed risk analysis
    risk_analysis_deep_dive()
    
    print("\n🎉 Advanced analysis examples completed!")
    print("\nKey Analysis Insights:")
    print("1. Risk-adjusted returns are more important than absolute returns")
    print("2. Drawdown analysis helps understand strategy resilience")
    print("3. Trade distribution analysis reveals strategy consistency")
    print("4. Time-based analysis shows strategy adaptability")
    print("5. Multi-strategy comparison enables portfolio construction")
    print("\nGenerated reports can be found in the 'reports' directory!")


if __name__ == "__main__":
    main()
