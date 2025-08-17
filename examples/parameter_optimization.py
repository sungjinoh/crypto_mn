"""
Parameter Optimization Example
=============================

This example demonstrates how to optimize strategy parameters using the framework.
It shows how to:
1. Define parameter ranges for optimization
2. Use the built-in optimization workflow
3. Analyze optimization results
4. Validate the best parameters

Requirements:
- Binance futures data in the correct format
- Framework installed (pip install -e .)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import product

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto_backtesting import (
    DataManager, BacktestEngine, MeanReversionStrategy,
    PerformanceAnalyzer
)
from crypto_backtesting.utils import WorkflowManager


def manual_optimization_example():
    """Example of manual parameter optimization."""
    
    print("🔧 Manual Parameter Optimization Example")
    print("=" * 45)
    
    # Setup
    data_path = Path(__file__).parent.parent / "binance_futures_data"
    data_manager = DataManager(data_path=str(data_path))
    
    # Load data
    print("\n📊 Loading data...")
    try:
        data1, data2 = data_manager.get_pair_data(
            'BTCUSDT', 'ETHUSDT', 
            year=2024, 
            months=[4, 5]  # 2 months for faster optimization
        )
        print(f"Loaded data: {len(data1)} candles")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Define parameter ranges
    print("\n🎯 Defining parameter ranges...")
    param_ranges = {
        'lookback_period': [30, 60, 90],
        'entry_threshold': [1.5, 2.0, 2.5],
        'exit_threshold': [0.0, 0.5],
        'stop_loss_threshold': [3.0, 4.0]
    }
    
    print("Parameter ranges:")
    for param, values in param_ranges.items():
        print(f"  {param}: {values}")
    
    # Generate all parameter combinations
    param_combinations = list(product(*param_ranges.values()))
    total_combinations = len(param_combinations)
    print(f"\nTotal combinations to test: {total_combinations}")
    
    # Setup backtest engine
    engine = BacktestEngine(
        data_manager,
        initial_capital=100000,
        commission_rate=0.001
    )
    
    # Run optimization
    print("\n⚡ Running optimization...")
    results = []
    
    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_ranges.keys(), params))
        
        # Create strategy with current parameters
        strategy = MeanReversionStrategy(
            symbol1='BTCUSDT',
            symbol2='ETHUSDT',
            **param_dict
        )
        
        try:
            # Prepare data
            combined_data = strategy.prepare_pair_data(data1, data2, 'BTCUSDT', 'ETHUSDT')
            
            # Add technical indicators
            rolling_mean = combined_data['spread'].rolling(param_dict['lookback_period']).mean()
            rolling_std = combined_data['spread'].rolling(param_dict['lookback_period']).std()
            combined_data['zscore'] = (combined_data['spread'] - rolling_mean) / rolling_std
            
            # Run backtest
            backtest_result = engine.run_backtest(strategy, combined_data)
            
            # Store results
            result_row = param_dict.copy()
            result_row.update({
                'total_return': backtest_result.total_return,
                'sharpe_ratio': backtest_result.sharpe_ratio,
                'max_drawdown': backtest_result.max_drawdown,
                'num_trades': len(backtest_result.trades),
                'win_rate': backtest_result.win_rate
            })
            results.append(result_row)
            
            # Progress update
            if (i + 1) % 5 == 0 or i == total_combinations - 1:
                print(f"Progress: {i + 1}/{total_combinations} ({(i + 1)/total_combinations*100:.1f}%)")
                
        except Exception as e:
            print(f"⚠️ Error with parameters {param_dict}: {e}")
            continue
    
    # Analyze results
    print("\n📊 Analyzing optimization results...")
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("❌ No successful optimization runs")
        return
    
    print(f"Successful runs: {len(results_df)}/{total_combinations}")
    
    # Find best parameters by different metrics
    metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
    
    print("\n🏆 Best Parameters by Metric:")
    print("-" * 40)
    
    for metric in metrics:
        if metric == 'max_drawdown':
            # For drawdown, lower is better
            best_idx = results_df[metric].idxmin()
            best_value = results_df.loc[best_idx, metric]
            print(f"\nBest {metric} (lowest): {best_value:.3f}")
        else:
            # For return and sharpe, higher is better
            best_idx = results_df[metric].idxmax()
            best_value = results_df.loc[best_idx, metric]
            print(f"\nBest {metric} (highest): {best_value:.3f}")
        
        best_params = results_df.loc[best_idx]
        print("Parameters:")
        for param in param_ranges.keys():
            print(f"  {param}: {best_params[param]}")
    
    # Summary statistics
    print("\n📈 Optimization Summary Statistics:")
    print("-" * 40)
    print(f"Average Total Return: {results_df['total_return'].mean():.3f}")
    print(f"Std Total Return: {results_df['total_return'].std():.3f}")
    print(f"Average Sharpe Ratio: {results_df['sharpe_ratio'].mean():.3f}")
    print(f"Average Max Drawdown: {results_df['max_drawdown'].mean():.3f}")
    print(f"Average Number of Trades: {results_df['num_trades'].mean():.1f}")
    print(f"Average Win Rate: {results_df['win_rate'].mean():.3f}")
    
    # Save results
    output_path = Path(__file__).parent.parent / "optimization_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")
    
    return results_df


def workflow_optimization_example():
    """Example using the built-in optimization workflow."""
    
    print("\n\n🚀 Workflow Manager Optimization Example")
    print("=" * 45)
    
    # Create workflow manager
    workflow = WorkflowManager()
    
    try:
        # Run parameter optimization using built-in workflow
        print("\n⚡ Running parameter optimization workflow...")
        optimization_result = workflow.parameter_optimization_workflow(
            symbol1='BTCUSDT',
            symbol2='ETHUSDT',
            year=2024,
            months=[4, 5],
            optimization_metric='sharpe_ratio',
            max_iterations=20  # Limit for demo
        )
        
        print("✅ Optimization completed!")
        
        # Display results
        print("\n🏆 Optimization Results:")
        print("-" * 30)
        print(f"Best Sharpe Ratio: {optimization_result['best_score']:.3f}")
        print("Best Parameters:")
        for param, value in optimization_result['best_params'].items():
            print(f"  {param}: {value}")
        
        print(f"\nTotal Evaluations: {optimization_result['total_evaluations']}")
        print(f"Optimization Time: {optimization_result['optimization_time']:.1f} seconds")
        
        # Validation backtest
        print("\n🔍 Running validation backtest...")
        validation_result = workflow.quick_pairs_backtest(
            symbol1='BTCUSDT',
            symbol2='ETHUSDT',
            year=2024,
            months=[6],  # Different month for validation
            strategy_params=optimization_result['best_params']
        )
        
        print("Validation Results:")
        print(f"  Total Return: {validation_result['total_return']:.3f}")
        print(f"  Sharpe Ratio: {validation_result['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {validation_result['max_drawdown']:.3f}")
        
    except Exception as e:
        print(f"❌ Error in workflow optimization: {e}")


def main():
    """Run parameter optimization examples."""
    
    print("🎯 Cryptocurrency Backtesting Framework - Parameter Optimization")
    print("=" * 70)
    
    # Run manual optimization
    results_df = manual_optimization_example()
    
    # Run workflow optimization
    workflow_optimization_example()
    
    print("\n🎉 Parameter optimization examples completed!")
    print("\nKey Takeaways:")
    print("1. Always test multiple parameter combinations")
    print("2. Use different metrics for optimization (return, Sharpe, drawdown)")
    print("3. Validate results on out-of-sample data")
    print("4. Consider computational cost vs. accuracy trade-offs")
    print("5. Use the WorkflowManager for standardized optimization")


if __name__ == "__main__":
    main()
