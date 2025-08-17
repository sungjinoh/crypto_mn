"""
Example configuration and workflow management for the crypto backtesting framework.

This module provides clear entry points, example configurations, 
and workflow examples for different use cases.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

# Import the new framework components
from crypto_backtesting import (
    DataManager, BacktestEngine, MeanReversionStrategy,
    PerformanceAnalyzer, ReportGenerator
)


class ConfigurationManager:
    """
    Manages configuration for backtesting workflows.
    
    Provides standardized configuration patterns and validation
    across all framework components.
    """
    
    def __init__(self):
        """Initialize configuration manager."""
        self.logger = logging.getLogger(__name__)
        
    def get_default_data_config(self) -> Dict[str, Any]:
        """
        Get default data management configuration.
        
        Returns:
            Default data configuration
        """
        return {
            'data_path': 'binance_futures_data',
            'cache_enabled': True,
            'cache_path': None,  # Will use default
            'providers': {
                'binance_futures': {
                    'enabled': True,
                    'klines_path': 'klines',
                    'funding_path': 'fundingRate'
                }
            }
        }
        
    def get_default_backtest_config(self) -> Dict[str, Any]:
        """
        Get default backtesting configuration.
        
        Returns:
            Default backtesting configuration  
        """
        return {
            'initial_capital': 100000.0,
            'commission_rate': 0.001,
            'slippage_rate': 0.0001,
            'risk_management': {
                'max_position_size': 0.5,
                'max_portfolio_risk': 0.02,
                'stop_loss_enabled': True
            }
        }
        
    def get_default_strategy_config(self, strategy_type: str) -> Dict[str, Any]:
        """
        Get default strategy configuration.
        
        Args:
            strategy_type: Type of strategy ('mean_reversion', etc.)
            
        Returns:
            Default strategy configuration
        """
        if strategy_type == 'mean_reversion':
            return {
                'lookback_period': 60,
                'entry_threshold': 2.0,
                'exit_threshold': 0.0,
                'stop_loss_threshold': 3.0,
                'optimization_bounds': {
                    'lookback_period': (20, 120),
                    'entry_threshold': (1.5, 3.0),
                    'exit_threshold': (-0.5, 0.5),
                    'stop_loss_threshold': (2.5, 4.0)
                }
            }
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
            
    def get_default_analysis_config(self) -> Dict[str, Any]:
        """
        Get default analysis and reporting configuration.
        
        Returns:
            Default analysis configuration
        """
        return {
            'output_dir': 'reports',
            'save_plots': True,
            'plot_format': 'png',
            'plot_dpi': 300,
            'generate_full_report': True,
            'comparison_metrics': [
                'total_return', 'sharpe_ratio', 'max_drawdown',
                'win_rate', 'profit_factor'
            ]
        }
        
    def validate_config(self, config: Dict[str, Any], config_type: str) -> bool:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration to validate
            config_type: Type of configuration
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if config_type == 'data':
                required_keys = ['data_path', 'cache_enabled']
                return all(key in config for key in required_keys)
                
            elif config_type == 'backtest':
                required_keys = ['initial_capital', 'commission_rate']
                return all(key in config for key in required_keys)
                
            elif config_type == 'strategy':
                # Strategy-specific validation would go here
                return True
                
            elif config_type == 'analysis':
                required_keys = ['output_dir']
                return all(key in config for key in required_keys)
                
            else:
                self.logger.warning(f"Unknown config type: {config_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating config: {e}")
            return False


class WorkflowManager:
    """
    Manages common backtesting workflows and provides entry points
    for different types of analysis.
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize workflow manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.logger = logging.getLogger(__name__)
        
    def quick_pairs_backtest(
        self,
        symbol1: str,
        symbol2: str,
        year: int = 2024,
        months: Optional[List[int]] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
        backtest_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Quick entry point for pairs trading backtest.
        
        Args:
            symbol1: First trading symbol
            symbol2: Second trading symbol  
            year: Year of data to use
            months: Months to include (None for all)
            strategy_params: Strategy parameters override
            backtest_config: Backtesting configuration override
            
        Returns:
            Dict with backtest results and analysis
        """
        self.logger.info(f"Starting quick pairs backtest: {symbol1}-{symbol2}")
        
        try:
            # Setup data manager
            data_config = self.config_manager.get_default_data_config()
            data_manager = DataManager(**data_config)
            
            # Load data
            data1, data2 = data_manager.get_pair_data(
                symbol1, symbol2, year, months
            )
            
            if data1 is None or data2 is None:
                raise ValueError(f"Could not load data for {symbol1}/{symbol2}")
                
            # Setup strategy
            strategy_config = self.config_manager.get_default_strategy_config('mean_reversion')
            if strategy_params:
                strategy_config.update(strategy_params)
                
            strategy = MeanReversionStrategy(
                symbol1=symbol1,
                symbol2=symbol2,
                **strategy_config
            )
            
            # Prepare data for backtesting
            combined_data = strategy.prepare_pair_data(data1, data2, symbol1, symbol2)
            
            # Add technical indicators
            combined_data['zscore'] = self._calculate_zscore(
                combined_data['spread'], strategy_config['lookback_period']
            )
            
            # Setup backtesting engine
            backtest_config = backtest_config or self.config_manager.get_default_backtest_config()
            engine = BacktestEngine(data_manager, **backtest_config)
            
            # Run backtest
            results = engine.run_backtest(strategy, combined_data)
            
            # Generate analysis
            analyzer = PerformanceAnalyzer(results)
            performance_report = analyzer.generate_performance_report()
            
            self.logger.info(f"Backtest completed. Total return: {results.total_return:.2%}")
            
            return {
                'results': results,
                'performance_report': performance_report,
                'strategy_config': strategy_config,
                'backtest_config': backtest_config
            }
            
        except Exception as e:
            self.logger.error(f"Error in quick pairs backtest: {e}")
            raise
            
    def parameter_optimization_workflow(
        self,
        symbol1: str,
        symbol2: str,
        year: int = 2024,
        months: Optional[List[int]] = None,
        optimization_metric: str = 'sharpe_ratio',
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Parameter optimization workflow for pairs trading.
        
        Args:
            symbol1: First trading symbol
            symbol2: Second trading symbol
            year: Year of data
            months: Months to include
            optimization_metric: Metric to optimize
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dict with optimization results
        """
        self.logger.info(f"Starting parameter optimization: {symbol1}-{symbol2}")
        
        try:
            # Get base configuration
            strategy_config = self.config_manager.get_default_strategy_config('mean_reversion')
            bounds = strategy_config['optimization_bounds']
            
            # Setup data
            data_config = self.config_manager.get_default_data_config()
            data_manager = DataManager(**data_config)
            
            data1, data2 = data_manager.get_pair_data(symbol1, symbol2, year, months)
            
            if data1 is None or data2 is None:
                raise ValueError(f"Could not load data for {symbol1}/{symbol2}")
                
            # Parameter optimization logic
            best_params = None
            best_score = -np.inf
            all_results = []
            
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(bounds, max_iterations)
            
            backtest_config = self.config_manager.get_default_backtest_config()
            engine = BacktestEngine(data_manager, **backtest_config)
            
            for i, params in enumerate(param_combinations):
                try:
                    # Create strategy with current parameters
                    strategy = MeanReversionStrategy(
                        symbol1=symbol1,
                        symbol2=symbol2,
                        **params
                    )
                    
                    # Prepare data
                    combined_data = strategy.prepare_pair_data(data1, data2, symbol1, symbol2)
                    combined_data['zscore'] = self._calculate_zscore(
                        combined_data['spread'], params['lookback_period']
                    )
                    
                    # Run backtest
                    results = engine.run_backtest(strategy, combined_data)
                    
                    # Get optimization metric
                    if optimization_metric == 'sharpe_ratio':
                        score = results.sharpe_ratio
                    elif optimization_metric == 'total_return':
                        score = results.total_return
                    elif optimization_metric == 'calmar_ratio':
                        score = results.calmar_ratio
                    else:
                        score = results.sharpe_ratio
                        
                    # Track results
                    result_entry = {
                        'parameters': params.copy(),
                        'score': score,
                        'total_return': results.total_return,
                        'sharpe_ratio': results.sharpe_ratio,
                        'max_drawdown': results.max_drawdown,
                        'win_rate': results.win_rate,
                        'total_trades': results.total_trades
                    }
                    all_results.append(result_entry)
                    
                    # Update best
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        
                    if (i + 1) % 25 == 0:
                        self.logger.info(f"Optimization progress: {i+1}/{len(param_combinations)}")
                        
                except Exception as e:
                    self.logger.warning(f"Error in parameter combination {i}: {e}")
                    continue
                    
            # Run final backtest with best parameters
            if best_params:
                final_strategy = MeanReversionStrategy(
                    symbol1=symbol1,
                    symbol2=symbol2,
                    **best_params
                )
                
                combined_data = final_strategy.prepare_pair_data(data1, data2, symbol1, symbol2)
                combined_data['zscore'] = self._calculate_zscore(
                    combined_data['spread'], best_params['lookback_period']
                )
                
                final_results = engine.run_backtest(final_strategy, combined_data)
                
                self.logger.info(f"Optimization completed. Best {optimization_metric}: {best_score:.4f}")
                
                return {
                    'best_parameters': best_params,
                    'best_score': best_score,
                    'final_results': final_results,
                    'all_results': all_results,
                    'optimization_metric': optimization_metric
                }
            else:
                raise ValueError("No valid parameter combinations found")
                
        except Exception as e:
            self.logger.error(f"Error in parameter optimization: {e}")
            raise
            
    def multi_pair_analysis_workflow(
        self,
        symbol_pairs: List[Tuple[str, str]],
        year: int = 2024,
        months: Optional[List[int]] = None,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Multi-pair analysis workflow.
        
        Args:
            symbol_pairs: List of (symbol1, symbol2) tuples
            year: Year of data
            months: Months to include
            top_n: Number of top pairs to return
            
        Returns:
            Dict with multi-pair analysis results
        """
        self.logger.info(f"Starting multi-pair analysis for {len(symbol_pairs)} pairs")
        
        try:
            # Setup
            data_config = self.config_manager.get_default_data_config()
            data_manager = DataManager(**data_config)
            
            backtest_config = self.config_manager.get_default_backtest_config()
            engine = BacktestEngine(data_manager, **backtest_config)
            
            strategy_config = self.config_manager.get_default_strategy_config('mean_reversion')
            
            pair_results = []
            
            for i, (symbol1, symbol2) in enumerate(symbol_pairs):
                try:
                    self.logger.info(f"Analyzing pair {i+1}/{len(symbol_pairs)}: {symbol1}-{symbol2}")
                    
                    # Load data
                    data1, data2 = data_manager.get_pair_data(symbol1, symbol2, year, months)
                    
                    if data1 is None or data2 is None:
                        self.logger.warning(f"No data for {symbol1}/{symbol2}")
                        continue
                        
                    # Create strategy
                    strategy = MeanReversionStrategy(
                        symbol1=symbol1,
                        symbol2=symbol2,
                        **strategy_config
                    )
                    
                    # Validate pair
                    validation_result = strategy.validate_pair(data1, data2)
                    
                    if not validation_result.get('is_valid', False):
                        self.logger.warning(f"Pair {symbol1}-{symbol2} failed validation")
                        continue
                        
                    # Prepare data and run backtest
                    combined_data = strategy.prepare_pair_data(data1, data2, symbol1, symbol2)
                    combined_data['zscore'] = self._calculate_zscore(
                        combined_data['spread'], strategy_config['lookback_period']
                    )
                    
                    results = engine.run_backtest(strategy, combined_data)
                    
                    # Store results
                    pair_result = {
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'pair_name': f"{symbol1}-{symbol2}",
                        'validation': validation_result,
                        'results': results,
                        'total_return': results.total_return,
                        'sharpe_ratio': results.sharpe_ratio,
                        'max_drawdown': results.max_drawdown,
                        'win_rate': results.win_rate,
                        'total_trades': results.total_trades
                    }
                    pair_results.append(pair_result)
                    
                except Exception as e:
                    self.logger.warning(f"Error analyzing pair {symbol1}-{symbol2}: {e}")
                    continue
                    
            # Sort by Sharpe ratio and get top pairs
            valid_pairs = [p for p in pair_results if p['results'].total_trades > 0]
            valid_pairs.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
            top_pairs = valid_pairs[:top_n]
            
            self.logger.info(f"Multi-pair analysis completed. Found {len(valid_pairs)} valid pairs")
            
            return {
                'all_results': pair_results,
                'valid_pairs': valid_pairs,
                'top_pairs': top_pairs,
                'summary_stats': self._calculate_multi_pair_summary(valid_pairs)
            }
            
        except Exception as e:
            self.logger.error(f"Error in multi-pair analysis: {e}")
            raise
            
    def _calculate_zscore(self, spread: pd.Series, lookback_period: int) -> pd.Series:
        """Calculate rolling z-score for spread."""
        rolling_mean = spread.rolling(window=lookback_period, min_periods=lookback_period//2).mean()
        rolling_std = spread.rolling(window=lookback_period, min_periods=lookback_period//2).std()
        
        return (spread - rolling_mean) / rolling_std
        
    def _generate_parameter_combinations(
        self, 
        bounds: Dict[str, tuple], 
        max_combinations: int
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations for optimization."""
        import itertools
        
        # Create parameter grids
        param_grids = {}
        
        for param, (low, high) in bounds.items():
            if param == 'lookback_period':
                param_grids[param] = list(range(int(low), int(high) + 1, 10))
            else:
                param_grids[param] = [low + i * (high - low) / 9 for i in range(10)]
                
        # Generate all combinations
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        # Limit combinations if needed
        if len(all_combinations) > max_combinations:
            # Sample randomly
            np.random.seed(42)  # For reproducibility
            selected_indices = np.random.choice(
                len(all_combinations), max_combinations, replace=False
            )
            all_combinations = [all_combinations[i] for i in selected_indices]
            
        # Convert to list of dicts
        combinations = []
        for combo in all_combinations:
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
            
        return combinations
        
    def _calculate_multi_pair_summary(self, valid_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for multi-pair analysis."""
        if not valid_pairs:
            return {}
            
        returns = [p['total_return'] for p in valid_pairs]
        sharpe_ratios = [p['sharpe_ratio'] for p in valid_pairs]
        drawdowns = [p['max_drawdown'] for p in valid_pairs]
        win_rates = [p['win_rate'] for p in valid_pairs]
        
        return {
            'num_pairs': len(valid_pairs),
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'median_sharpe': np.median(sharpe_ratios),
            'avg_drawdown': np.mean(drawdowns),
            'profitable_pairs': len([r for r in returns if r > 0]),
            'high_sharpe_pairs': len([s for s in sharpe_ratios if s > 1.0])
        }


# Example usage functions for documentation
def example_quick_start():
    """
    Example of quick start workflow.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create workflow manager
    workflow = WorkflowManager()
    
    # Run quick pairs backtest
    result = workflow.quick_pairs_backtest(
        symbol1='BTCUSDT',
        symbol2='ETHUSDT',
        year=2024,
        months=[4, 5, 6]
    )
    
    # Generate report
    analysis_config = workflow.config_manager.get_default_analysis_config()
    report_generator = ReportGenerator(result['results'], analysis_config['output_dir'])
    report_generator.generate_full_report()
    
    print(f"Backtest completed!")
    print(f"Total Return: {result['results'].total_return:.2%}")
    print(f"Sharpe Ratio: {result['results'].sharpe_ratio:.3f}")
    print(f"Max Drawdown: {result['results'].max_drawdown:.2%}")


def example_parameter_optimization():
    """
    Example of parameter optimization workflow.
    """
    workflow = WorkflowManager()
    
    # Run parameter optimization
    result = workflow.parameter_optimization_workflow(
        symbol1='BTCUSDT',
        symbol2='ETHUSDT',
        year=2024,
        months=[4, 5, 6],
        optimization_metric='sharpe_ratio',
        max_iterations=50
    )
    
    print("Parameter optimization completed!")
    print(f"Best parameters: {result['best_parameters']}")
    print(f"Best Sharpe ratio: {result['best_score']:.3f}")


def example_multi_pair_analysis():
    """
    Example of multi-pair analysis workflow.
    """
    workflow = WorkflowManager()
    
    # Define pairs to analyze
    pairs = [
        ('BTCUSDT', 'ETHUSDT'),
        ('ADAUSDT', 'DOTUSDT'),
        ('LINKUSDT', 'AVAXUSDT'),
        ('SOLUSDT', 'ATOMUSDT'),
        ('BNBUSDT', 'MATICUSDT')
    ]
    
    # Run multi-pair analysis
    result = workflow.multi_pair_analysis_workflow(
        symbol_pairs=pairs,
        year=2024,
        months=[4, 5, 6],
        top_n=3
    )
    
    print("Multi-pair analysis completed!")
    print(f"Valid pairs: {len(result['valid_pairs'])}")
    print("Top pairs:")
    for i, pair in enumerate(result['top_pairs'], 1):
        print(f"{i}. {pair['pair_name']}: Sharpe {pair['sharpe_ratio']:.3f}")


if __name__ == "__main__":
    # Run examples
    print("Running example workflows...")
    
    try:
        example_quick_start()
        print("\n" + "="*50 + "\n")
        
        example_parameter_optimization()
        print("\n" + "="*50 + "\n")
        
        example_multi_pair_analysis()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Note: Examples require actual data files to run successfully")
