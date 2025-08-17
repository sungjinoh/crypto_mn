"""
Multi-Pair Analysis Example
==========================

This example demonstrates how to analyze multiple trading pairs simultaneously.
It shows how to:
1. Analyze multiple pairs for cointegration
2. Run backtests on multiple pairs
3. Compare performance across pairs
4. Create a portfolio allocation strategy

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
    PerformanceAnalyzer
)
from crypto_backtesting.utils import WorkflowManager


def analyze_multiple_pairs():
    """Analyze multiple trading pairs for cointegration and performance."""
    
    print("📊 Multi-Pair Analysis Example")
    print("=" * 35)
    
    # Setup data manager
    data_path = Path(__file__).parent.parent / "binance_futures_data"
    data_manager = DataManager(data_path=str(data_path))
    
    # Get available symbols
    available_symbols = data_manager.get_available_symbols()
    print(f"Total available symbols: {len(available_symbols)}")
    
    # Define pairs to analyze (select major cryptocurrencies)
    major_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'AVAXUSDT']
    
    # Filter to only symbols we have data for
    available_major = [s for s in major_symbols if s in available_symbols]
    print(f"Available major symbols: {available_major}")
    
    if len(available_major) < 2:
        print("❌ Need at least 2 symbols for pairs analysis")
        return
    
    # Generate all possible pairs
    pairs = []
    for i in range(len(available_major)):
        for j in range(i + 1, len(available_major)):
            pairs.append((available_major[i], available_major[j]))
    
    print(f"\nAnalyzing {len(pairs)} pairs:")
    for i, (symbol1, symbol2) in enumerate(pairs, 1):
        print(f"  {i}. {symbol1} / {symbol2}")
    
    # Analyze each pair
    print("\n🔍 Analyzing pairs for cointegration...")
    pair_analysis = []
    
    for symbol1, symbol2 in pairs:
        try:
            # Load data
            data1, data2 = data_manager.get_pair_data(
                symbol1, symbol2, 
                year=2024, 
                months=[4, 5]
            )
            
            # Create strategy for cointegration testing
            strategy = MeanReversionStrategy(
                symbol1=symbol1,
                symbol2=symbol2,
                lookback_period=60,
                entry_threshold=2.0,
                exit_threshold=0.0,
                stop_loss_threshold=3.0
            )
            
            # Test cointegration
            is_cointegrated, stats = strategy.test_cointegration(data1, data2)
            
            # Calculate correlation
            returns1 = data1['close'].pct_change().dropna()
            returns2 = data2['close'].pct_change().dropna()
            correlation = returns1.corr(returns2)
            
            pair_info = {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'cointegrated': is_cointegrated,
                'adf_statistic': stats.get('adf_statistic', None),
                'p_value': stats.get('p_value', None),
                'correlation': correlation,
                'data_points': min(len(data1), len(data2))
            }
            pair_analysis.append(pair_info)
            
            status = "✅ Cointegrated" if is_cointegrated else "❌ Not cointegrated"
            print(f"{symbol1}/{symbol2}: {status} (corr: {correlation:.3f})")
            
        except Exception as e:
            print(f"⚠️ Error analyzing {symbol1}/{symbol2}: {e}")
            continue
    
    # Filter cointegrated pairs
    cointegrated_pairs = [p for p in pair_analysis if p['cointegrated']]
    print(f"\n✅ Found {len(cointegrated_pairs)} cointegrated pairs")
    
    if len(cointegrated_pairs) == 0:
        print("No cointegrated pairs found. Proceeding with all pairs for demonstration.")
        cointegrated_pairs = pair_analysis[:3]  # Take first 3 pairs
    
    # Run backtests on cointegrated pairs
    print("\n⚡ Running backtests on selected pairs...")
    backtest_results = []
    
    engine = BacktestEngine(
        data_manager,
        initial_capital=100000,
        commission_rate=0.001
    )
    
    for pair_info in cointegrated_pairs[:5]:  # Limit to 5 pairs for demo
        symbol1 = pair_info['symbol1']
        symbol2 = pair_info['symbol2']
        
        try:
            # Load data
            data1, data2 = data_manager.get_pair_data(
                symbol1, symbol2, 
                year=2024, 
                months=[4, 5, 6]
            )
            
            # Create strategy
            strategy = MeanReversionStrategy(
                symbol1=symbol1,
                symbol2=symbol2,
                lookback_period=60,
                entry_threshold=2.0,
                exit_threshold=0.0,
                stop_loss_threshold=3.0
            )
            
            # Prepare data
            combined_data = strategy.prepare_pair_data(data1, data2, symbol1, symbol2)
            rolling_mean = combined_data['spread'].rolling(60).mean()
            rolling_std = combined_data['spread'].rolling(60).std()
            combined_data['zscore'] = (combined_data['spread'] - rolling_mean) / rolling_std
            
            # Run backtest
            result = engine.run_backtest(strategy, combined_data)
            
            # Analyze performance
            analyzer = PerformanceAnalyzer(result)
            performance = analyzer.generate_performance_report()
            
            backtest_info = {
                'pair': f"{symbol1}/{symbol2}",
                'symbol1': symbol1,
                'symbol2': symbol2,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'num_trades': len(result.trades),
                'win_rate': result.win_rate,
                'cointegration_p_value': pair_info['p_value'],
                'correlation': pair_info['correlation']
            }
            backtest_results.append(backtest_info)
            
            print(f"✅ {symbol1}/{symbol2}: Return {result.total_return:.2%}, Sharpe {result.sharpe_ratio:.3f}")
            
        except Exception as e:
            print(f"⚠️ Error backtesting {symbol1}/{symbol2}: {e}")
            continue
    
    # Analyze and rank results
    print("\n🏆 Pair Performance Ranking:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Pair':<15} {'Return':<10} {'Sharpe':<8} {'Drawdown':<10} {'Trades':<7} {'Win Rate':<8}")
    print("-" * 80)
    
    # Sort by Sharpe ratio
    sorted_results = sorted(backtest_results, key=lambda x: x['sharpe_ratio'], reverse=True)
    
    for i, result in enumerate(sorted_results, 1):
        print(f"{i:<4} {result['pair']:<15} {result['total_return']:>8.2%} "
              f"{result['sharpe_ratio']:>7.3f} {result['max_drawdown']:>8.2%} "
              f"{result['num_trades']:>6} {result['win_rate']:>7.1%}")
    
    # Portfolio allocation suggestion
    print("\n💼 Portfolio Allocation Suggestion:")
    print("-" * 40)
    
    if len(sorted_results) >= 3:
        top_3 = sorted_results[:3]
        total_sharpe = sum(r['sharpe_ratio'] for r in top_3 if r['sharpe_ratio'] > 0)
        
        if total_sharpe > 0:
            print("Top 3 pairs with suggested allocation:")
            for result in top_3:
                if result['sharpe_ratio'] > 0:
                    allocation = (result['sharpe_ratio'] / total_sharpe) * 100
                    print(f"  {result['pair']}: {allocation:.1f}% allocation")
        else:
            print("No pairs with positive Sharpe ratio for allocation")
    
    # Summary statistics
    print(f"\n📊 Multi-Pair Analysis Summary:")
    print("-" * 35)
    if backtest_results:
        avg_return = sum(r['total_return'] for r in backtest_results) / len(backtest_results)
        avg_sharpe = sum(r['sharpe_ratio'] for r in backtest_results) / len(backtest_results)
        avg_drawdown = sum(r['max_drawdown'] for r in backtest_results) / len(backtest_results)
        
        print(f"Pairs analyzed: {len(backtest_results)}")
        print(f"Average return: {avg_return:.2%}")
        print(f"Average Sharpe ratio: {avg_sharpe:.3f}")
        print(f"Average max drawdown: {avg_drawdown:.2%}")
        
        best_pair = max(backtest_results, key=lambda x: x['sharpe_ratio'])
        print(f"Best performing pair: {best_pair['pair']} (Sharpe: {best_pair['sharpe_ratio']:.3f})")
    
    return backtest_results


def workflow_multi_pair_example():
    """Example using WorkflowManager for multi-pair analysis."""
    
    print("\n\n🚀 Workflow Manager Multi-Pair Example")
    print("=" * 45)
    
    try:
        # Create workflow manager
        workflow = WorkflowManager()
        
        # Define pairs to analyze
        pairs = [
            ('BTCUSDT', 'ETHUSDT'),
            ('ADAUSDT', 'DOTUSDT'),
            ('LINKUSDT', 'AVAXUSDT')
        ]
        
        print(f"\n📊 Analyzing {len(pairs)} pairs using WorkflowManager...")
        
        # Run multi-pair analysis
        result = workflow.multi_pair_analysis_workflow(
            symbol_pairs=pairs,
            year=2024,
            months=[4, 5, 6],
            top_n=3
        )
        
        print("✅ Multi-pair analysis completed!")
        
        # Display results
        print(f"\nTop {len(result['top_pairs'])} pairs:")
        for i, pair_result in enumerate(result['top_pairs'], 1):
            print(f"{i}. {pair_result['pair']}: Sharpe {pair_result['sharpe_ratio']:.3f}")
        
        print(f"\nTotal pairs analyzed: {result['total_pairs']}")
        print(f"Analysis time: {result['analysis_time']:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Error in workflow multi-pair analysis: {e}")


def main():
    """Run multi-pair analysis examples."""
    
    print("🎯 Cryptocurrency Backtesting Framework - Multi-Pair Analysis")
    print("=" * 65)
    
    # Run detailed multi-pair analysis
    backtest_results = analyze_multiple_pairs()
    
    # Run workflow example
    workflow_multi_pair_example()
    
    print("\n🎉 Multi-pair analysis examples completed!")
    print("\nKey Insights:")
    print("1. Cointegration is crucial for pairs trading success")
    print("2. High correlation doesn't guarantee profitability")
    print("3. Diversification across multiple pairs reduces risk")
    print("4. Regular rebalancing based on performance is important")
    print("5. Transaction costs significantly impact small spreads")


if __name__ == "__main__":
    main()
