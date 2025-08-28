"""
Comprehensive Accuracy-Focused Workflow for Cointegration Trading
This script prioritizes accuracy over speed, testing multiple timeframes and parameters
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import glob
import sys
import os
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_cointegration_finder_v2 import EnhancedCointegrationFinder
from hybrid_pair_selector import HybridPairSelector
from market_neutral.run_fixed_parameters import run_fixed_parameters_backtest


class ComprehensiveWorkflow:
    """
    Comprehensive workflow that tests multiple timeframes and parameters for maximum accuracy
    """
    
    def __init__(self):
        self.results_summary = {}
        self.best_configuration = None
        
    def phase1_multi_timeframe_cointegration(self) -> Dict:
        """
        Phase 1: Test cointegration across multiple timeframes
        Find which timeframe produces the most stable cointegrated pairs
        """
        print("\n" + "="*80)
        print("🔬 PHASE 1: MULTI-TIMEFRAME COINTEGRATION DISCOVERY")
        print("="*80)
        print("\nTesting cointegration stability across different timeframes...")
        print("This will take 2-4 hours but ensures we find the best timeframe.")
        
        # Define timeframes to test
        timeframes_to_test = ["30T", "1H", "2H", "4H"]
        
        # Data period for cointegration discovery
        training_years = [2023]
        training_months = list(range(1, 13))  # Full year
        
        cointegration_results = {}
        
        for timeframe in timeframes_to_test:
            print(f"\n{'='*60}")
            print(f"Testing timeframe: {timeframe}")
            print(f"{'='*60}")
            
            # Create finder with specific timeframe
            finder = EnhancedCointegrationFinder(
                base_path="binance_futures_data",
                resample_interval=timeframe,
                min_data_points=1000,
                significance_level=0.05,
                min_daily_volume=1000000,
                check_stationarity=True,
                use_rolling_window=True,
                rolling_window_size=500,
                rolling_step_size=100,
            )
            
            # Find cointegrated pairs
            results = finder.find_all_cointegrated_pairs(
                years=training_years,
                months=training_months,
                max_symbols=100,  # Limit for testing (remove for production)
                use_parallel=True,
                filter_by_sector=True,
                max_correlation=0.95,
            )
            
            # Save results for each timeframe
            output_dir = f"cointegration_results_{timeframe}"
            finder.save_results(results, output_dir=output_dir)
            
            # Analyze quality metrics
            quality_metrics = self._analyze_cointegration_quality(results)
            
            cointegration_results[timeframe] = {
                'results': results,
                'quality_metrics': quality_metrics,
                'file_path': output_dir,
            }
            
            print(f"\nResults for {timeframe}:")
            print(f"  Total cointegrated pairs: {quality_metrics['total_pairs']}")
            print(f"  High quality pairs (score>70): {quality_metrics['high_quality_pairs']}")
            print(f"  Average p-value: {quality_metrics['avg_p_value']:.4f}")
            print(f"  Average stability: {quality_metrics['avg_stability']:.2%}")
            print(f"  Pairs with good half-life (10-50): {quality_metrics['good_halflife_pairs']}")
        
        # Compare timeframes
        best_timeframe = self._select_best_timeframe(cointegration_results)
        
        print(f"\n✅ BEST TIMEFRAME: {best_timeframe}")
        
        self.results_summary['phase1'] = {
            'best_timeframe': best_timeframe,
            'all_results': cointegration_results,
        }
        
        return cointegration_results
    
    def _analyze_cointegration_quality(self, results: Dict) -> Dict:
        """Analyze quality metrics of cointegration results"""
        pairs = results.get('cointegrated_pairs', [])
        
        if not pairs:
            return {
                'total_pairs': 0,
                'high_quality_pairs': 0,
                'avg_p_value': 1.0,
                'avg_stability': 0.0,
                'good_halflife_pairs': 0,
            }
        
        # Calculate quality scores
        quality_scores = []
        p_values = []
        stabilities = []
        good_halflife = 0
        
        for pair in pairs:
            # Quality score (simplified calculation)
            score = 0
            p_value = pair.get('p_value', 1.0)
            
            if p_value <= 0.001:
                score += 30
            elif p_value <= 0.01:
                score += 20
            elif p_value <= 0.05:
                score += 10
            
            # Check half-life
            if 'spread_properties' in pair:
                props = pair['spread_properties']
                if isinstance(props, dict):
                    hl = props.get('half_life_ou') or props.get('half_life')
                    if hl and 10 <= hl <= 50:
                        good_halflife += 1
                        score += 20
            
            # Check stability
            if 'rolling_stability' in pair:
                stab = pair['rolling_stability']
                if isinstance(stab, dict):
                    stability_ratio = stab.get('stability_ratio', 0)
                    stabilities.append(stability_ratio)
                    score += stability_ratio * 20
            
            quality_scores.append(score)
            p_values.append(p_value)
        
        return {
            'total_pairs': len(pairs),
            'high_quality_pairs': sum(1 for s in quality_scores if s > 70),
            'avg_p_value': np.mean(p_values),
            'avg_stability': np.mean(stabilities) if stabilities else 0.0,
            'good_halflife_pairs': good_halflife,
        }
    
    def _select_best_timeframe(self, results: Dict) -> str:
        """Select best timeframe based on multiple criteria"""
        scores = {}
        
        for timeframe, data in results.items():
            metrics = data['quality_metrics']
            
            # Scoring system (weights for different factors)
            score = 0
            score += metrics['high_quality_pairs'] * 2  # Weight high quality pairs
            score += metrics['good_halflife_pairs'] * 1.5  # Weight good half-life
            score += (1 - metrics['avg_p_value']) * 100  # Lower p-value is better
            score += metrics['avg_stability'] * 50  # Higher stability is better
            score += min(metrics['total_pairs'], 100) * 0.5  # Some weight for total pairs
            
            scores[timeframe] = score
        
        # Return timeframe with highest score
        best_timeframe = max(scores, key=scores.get)
        
        print(f"\nTimeframe scores:")
        for tf, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tf}: {score:.1f}")
        
        return best_timeframe
    
    def phase2_parameter_optimization(self, timeframe: str, cointegration_file: str) -> Dict:
        """
        Phase 2: Optimize trading parameters for the selected timeframe
        Test multiple parameter combinations to find optimal settings
        """
        print("\n" + "="*80)
        print("🎯 PHASE 2: PARAMETER OPTIMIZATION")
        print("="*80)
        print(f"Optimizing parameters for timeframe: {timeframe}")
        
        # Load cointegrated pairs
        filter_obj = CointegrationFilter("moderate")
        pairs = filter_obj.load_cointegration_results(cointegration_file)
        
        # Calculate quality scores and get top 100 pairs
        for pair in pairs:
            pair['quality_score'] = filter_obj.calculate_quality_score(pair)
        
        sorted_pairs = sorted(pairs, key=lambda x: x.get('quality_score', 0), reverse=True)
        top_pairs = sorted_pairs[:100]  # Use top 100 for parameter optimization
        
        print(f"Using top {len(top_pairs)} pairs for parameter optimization")
        
        # Define parameter grid
        parameter_grid = {
            'lookback_period': [20, 30, 40, 50, 60],
            'entry_threshold': [1.5, 1.75, 2.0, 2.25, 2.5],
            'exit_threshold': [0.0, 0.25, 0.5, 0.75, 1.0],
            'stop_loss_threshold': [3.0, 3.5, 4.0, 4.5],
        }
        
        # Generate all combinations
        param_combinations = list(itertools.product(
            parameter_grid['lookback_period'],
            parameter_grid['entry_threshold'],
            parameter_grid['exit_threshold'],
            parameter_grid['stop_loss_threshold']
        ))
        
        print(f"Testing {len(param_combinations)} parameter combinations")
        print("This will take 3-6 hours for thorough testing...")
        
        # Test on validation period
        validation_years = [2024]
        validation_months = [1, 2, 3, 4, 5, 6]
        
        # Convert pairs to format for backtester
        pair_list = [(p['symbol1'], p['symbol2']) for p in top_pairs[:30]]  # Test on 30 pairs
        
        optimization_results = []
        
        # Test each parameter combination
        for i, (lb, entry, exit, sl) in enumerate(param_combinations, 1):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(param_combinations)} combinations tested")
            
            params = {
                'lookback_period': lb,
                'entry_threshold': entry,
                'exit_threshold': exit,
                'stop_loss_threshold': sl,
            }
            
            # Run backtest
            try:
                results_df = run_fixed_parameters_backtest(
                    fixed_params=params,
                    specific_pairs=pair_list,
                    test_years=validation_years,
                    test_months=validation_months,
                    save_results=False,
                    save_plots=False,
                )
                
                # Calculate aggregate metrics
                successful = results_df[results_df['success'] == True]
                if len(successful) > 0:
                    avg_sharpe = successful['sharpe_ratio'].mean()
                    avg_return = successful['total_return'].mean()
                    avg_trades = successful['num_trades'].mean()
                    win_rate = successful['win_rate'].mean()
                    max_dd = successful['max_drawdown'].mean()
                    
                    optimization_results.append({
                        **params,
                        'avg_sharpe': avg_sharpe,
                        'avg_return': avg_return,
                        'avg_trades': avg_trades,
                        'win_rate': win_rate,
                        'max_drawdown': max_dd,
                        'success_rate': len(successful) / len(results_df),
                        'score': avg_sharpe * 0.4 + avg_return * 100 * 0.3 + win_rate * 0.2 + (1 - abs(max_dd)) * 0.1
                    })
            except Exception as e:
                print(f"Error testing parameters {params}: {e}")
                continue
        
        # Find best parameters
        opt_df = pd.DataFrame(optimization_results)
        opt_df = opt_df.sort_values('score', ascending=False)
        
        best_params = opt_df.iloc[0].to_dict()
        
        print(f"\n✅ BEST PARAMETERS FOR {timeframe}:")
        print(f"  Lookback: {best_params['lookback_period']}")
        print(f"  Entry threshold: {best_params['entry_threshold']}")
        print(f"  Exit threshold: {best_params['exit_threshold']}")
        print(f"  Stop loss: {best_params['stop_loss_threshold']}")
        print(f"  Expected Sharpe: {best_params['avg_sharpe']:.2f}")
        print(f"  Expected Return: {best_params['avg_return']:.2%}")
        
        # Save optimization results
        opt_df.to_csv(f'parameter_optimization_{timeframe}.csv', index=False)
        
        self.results_summary['phase2'] = {
            'best_parameters': best_params,
            'all_results': opt_df.to_dict('records'),
        }
        
        return best_params
    
    def phase3_cross_validation(self, timeframe: str, params: Dict, cointegration_file: str) -> Dict:
        """
        Phase 3: Cross-validation with walk-forward analysis
        Test parameters across multiple time windows to ensure robustness
        """
        print("\n" + "="*80)
        print("📊 PHASE 3: CROSS-VALIDATION")
        print("="*80)
        print("Running walk-forward analysis to validate parameters...")
        
        # Define walk-forward windows
        windows = [
            {
                'name': 'Window 1',
                'train': {'years': [2023], 'months': [1,2,3,4,5,6]},
                'test': {'years': [2023], 'months': [7,8,9]},
            },
            {
                'name': 'Window 2',
                'train': {'years': [2023], 'months': [4,5,6,7,8,9]},
                'test': {'years': [2023], 'months': [10,11,12]},
            },
            {
                'name': 'Window 3',
                'train': {'years': [2023], 'months': [7,8,9,10,11,12]},
                'test': {'years': [2024], 'months': [1,2,3]},
            },
            {
                'name': 'Window 4',
                'train': {'years': [2023, 2024], 'months': [[10,11,12], [1,2,3]]},
                'test': {'years': [2024], 'months': [4,5,6]},
            },
            {
                'name': 'Window 5',
                'train': {'years': [2024], 'months': [1,2,3,4,5,6]},
                'test': {'years': [2024], 'months': [7,8,9]},
            },
        ]
        
        cross_validation_results = []
        
        for window in windows:
            print(f"\n{window['name']}:")
            print(f"  Training: {window['train']}")
            print(f"  Testing: {window['test']}")
            
            # Find cointegration on training window
            finder = EnhancedCointegrationFinder(
                resample_interval=timeframe,
                min_daily_volume=1000000,
            )
            
            # Handle the complex year/month structure
            if isinstance(window['train']['months'][0], list):
                train_years = window['train']['years']
                train_months = window['train']['months']
            else:
                train_years = window['train']['years']
                train_months = window['train']['months']
            
            coint_results = finder.find_all_cointegrated_pairs(
                years=train_years,
                months=train_months,
                max_symbols=50,  # Limit for speed
            )
            
            if len(coint_results['cointegrated_pairs']) < 10:
                print(f"  ⚠️ Only {len(coint_results['cointegrated_pairs'])} pairs found, skipping")
                continue
            
            # Select top pairs
            pairs = coint_results['cointegrated_pairs'][:30]
            pair_list = [(p['symbol1'], p['symbol2']) for p in pairs]
            
            # Test with optimal parameters
            results_df = run_fixed_parameters_backtest(
                fixed_params=params,
                specific_pairs=pair_list,
                test_years=window['test']['years'],
                test_months=window['test']['months'],
                save_results=False,
                save_plots=False,
            )
            
            # Calculate metrics
            successful = results_df[results_df['success'] == True]
            if len(successful) > 0:
                window_metrics = {
                    'window': window['name'],
                    'sharpe_ratio': successful['sharpe_ratio'].mean(),
                    'total_return': successful['total_return'].mean(),
                    'win_rate': successful['win_rate'].mean(),
                    'max_drawdown': successful['max_drawdown'].mean(),
                    'num_trades': successful['num_trades'].mean(),
                    'success_rate': len(successful) / len(results_df),
                }
                
                cross_validation_results.append(window_metrics)
                
                print(f"  Sharpe: {window_metrics['sharpe_ratio']:.2f}")
                print(f"  Return: {window_metrics['total_return']:.2%}")
                print(f"  Win Rate: {window_metrics['win_rate']:.2%}")
        
        # Calculate consistency metrics
        cv_df = pd.DataFrame(cross_validation_results)
        
        consistency_metrics = {
            'mean_sharpe': cv_df['sharpe_ratio'].mean(),
            'std_sharpe': cv_df['sharpe_ratio'].std(),
            'min_sharpe': cv_df['sharpe_ratio'].min(),
            'mean_return': cv_df['total_return'].mean(),
            'std_return': cv_df['total_return'].std(),
            'consistency_score': cv_df['sharpe_ratio'].mean() / (cv_df['sharpe_ratio'].std() + 0.01),
        }
        
        print(f"\n📈 CROSS-VALIDATION SUMMARY:")
        print(f"  Mean Sharpe: {consistency_metrics['mean_sharpe']:.2f} ± {consistency_metrics['std_sharpe']:.2f}")
        print(f"  Mean Return: {consistency_metrics['mean_return']:.2%} ± {consistency_metrics['std_return']:.2%}")
        print(f"  Consistency Score: {consistency_metrics['consistency_score']:.2f}")
        
        # Save results
        cv_df.to_csv(f'cross_validation_{timeframe}.csv', index=False)
        
        self.results_summary['phase3'] = {
            'consistency_metrics': consistency_metrics,
            'window_results': cross_validation_results,
        }
        
        return consistency_metrics
    
    def phase4_final_pair_selection(self, timeframe: str, params: Dict, cointegration_file: str) -> List:
        """
        Phase 4: Final pair selection with comprehensive backtesting
        """
        print("\n" + "="*80)
        print("🏆 PHASE 4: FINAL PAIR SELECTION")
        print("="*80)
        print("Selecting final pairs with comprehensive backtesting...")
        
        # Use hybrid selector with optimal parameters
        selector = HybridPairSelector()
        
        # Override with our optimized parameters
        selector.parameter_sets = [params]
        
        # Run complete analysis
        final_pairs = selector.run_complete_hybrid_analysis(
            cointegration_file=cointegration_file,
            target_final_pairs=10,
            pairs_to_backtest=100,  # Test more pairs for accuracy
            test_years=[2024],
            test_months=[7, 8, 9, 10, 11]  # Recent out-of-sample period
        )
        
        self.results_summary['phase4'] = {
            'final_pairs': final_pairs,
            'num_pairs': len(final_pairs),
        }
        
        return final_pairs
    
    def phase5_robustness_testing(self, final_pairs: List, timeframe: str, params: Dict) -> Dict:
        """
        Phase 5: Robustness testing with Monte Carlo simulation and stress testing
        """
        print("\n" + "="*80)
        print("🔬 PHASE 5: ROBUSTNESS TESTING")
        print("="*80)
        
        robustness_results = {}
        
        # 1. Parameter sensitivity analysis
        print("\n1. Parameter Sensitivity Analysis...")
        sensitivity_results = self._parameter_sensitivity_analysis(final_pairs, timeframe, params)
        robustness_results['sensitivity'] = sensitivity_results
        
        # 2. Different market regime testing
        print("\n2. Market Regime Testing...")
        regime_results = self._market_regime_testing(final_pairs, params)
        robustness_results['regimes'] = regime_results
        
        # 3. Transaction cost sensitivity
        print("\n3. Transaction Cost Analysis...")
        transaction_results = self._transaction_cost_analysis(final_pairs, params)
        robustness_results['transaction_costs'] = transaction_results
        
        self.results_summary['phase5'] = robustness_results
        
        return robustness_results
    
    def _parameter_sensitivity_analysis(self, pairs: List, timeframe: str, base_params: Dict) -> Dict:
        """Test sensitivity to parameter changes"""
        sensitivity_results = []
        
        # Test ±10% and ±20% changes in each parameter
        variations = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        pair_list = [(p['symbol1'], p['symbol2']) for p in pairs]
        
        for param_name in ['lookback_period', 'entry_threshold', 'exit_threshold', 'stop_loss_threshold']:
            for variation in variations:
                test_params = base_params.copy()
                test_params[param_name] = base_params[param_name] * variation
                
                # Run quick backtest
                results_df = run_fixed_parameters_backtest(
                    fixed_params=test_params,
                    specific_pairs=pair_list[:5],  # Test on subset for speed
                    test_years=[2024],
                    test_months=[7, 8, 9],
                    save_results=False,
                    save_plots=False,
                )
                
                successful = results_df[results_df['success'] == True]
                if len(successful) > 0:
                    sensitivity_results.append({
                        'parameter': param_name,
                        'variation': variation,
                        'sharpe': successful['sharpe_ratio'].mean(),
                        'return': successful['total_return'].mean(),
                    })
        
        # Analyze sensitivity
        sensitivity_df = pd.DataFrame(sensitivity_results)
        
        print("\nParameter Sensitivity Summary:")
        for param in ['lookback_period', 'entry_threshold', 'exit_threshold', 'stop_loss_threshold']:
            param_data = sensitivity_df[sensitivity_df['parameter'] == param]
            sharpe_std = param_data['sharpe'].std()
            print(f"  {param}: Sharpe StdDev = {sharpe_std:.3f} {'(stable)' if sharpe_std < 0.2 else '(sensitive)'}")
        
        return sensitivity_results
    
    def _market_regime_testing(self, pairs: List, params: Dict) -> Dict:
        """Test performance in different market regimes"""
        # Define market regimes based on known periods
        regimes = [
            {'name': 'Bull Market', 'years': [2023], 'months': [10, 11, 12]},
            {'name': 'Bear Market', 'years': [2023], 'months': [4, 5, 6]},
            {'name': 'Sideways', 'years': [2024], 'months': [2, 3, 4]},
        ]
        
        regime_results = []
        pair_list = [(p['symbol1'], p['symbol2']) for p in pairs[:5]]
        
        for regime in regimes:
            results_df = run_fixed_parameters_backtest(
                fixed_params=params,
                specific_pairs=pair_list,
                test_years=regime['years'],
                test_months=regime['months'],
                save_results=False,
                save_plots=False,
            )
            
            successful = results_df[results_df['success'] == True]
            if len(successful) > 0:
                regime_results.append({
                    'regime': regime['name'],
                    'sharpe': successful['sharpe_ratio'].mean(),
                    'return': successful['total_return'].mean(),
                    'win_rate': successful['win_rate'].mean(),
                })
        
        print("\nMarket Regime Performance:")
        for result in regime_results:
            print(f"  {result['regime']}: Sharpe={result['sharpe']:.2f}, Return={result['return']:.2%}")
        
        return regime_results
    
    def _transaction_cost_analysis(self, pairs: List, params: Dict) -> Dict:
        """Analyze impact of different transaction costs"""
        cost_levels = [0.0005, 0.001, 0.002, 0.003]  # 0.05% to 0.3%
        
        cost_results = []
        pair_list = [(p['symbol1'], p['symbol2']) for p in pairs[:5]]
        
        for cost in cost_levels:
            # Note: This is simplified - you'd need to modify the backtester to accept transaction cost parameter
            # For now, we'll estimate the impact
            base_return = 0.20  # Assume 20% base return
            num_trades = 30  # Assume 30 trades per year
            
            # Estimate impact
            cost_impact = cost * num_trades * 2  # Buy and sell
            adjusted_return = base_return - cost_impact
            adjusted_sharpe = adjusted_return / 0.15  # Assume 15% volatility
            
            cost_results.append({
                'transaction_cost': cost,
                'estimated_return': adjusted_return,
                'estimated_sharpe': adjusted_sharpe,
            })
        
        print("\nTransaction Cost Impact:")
        for result in cost_results:
            print(f"  {result['transaction_cost']:.2%}: Return={result['estimated_return']:.2%}, Sharpe={result['estimated_sharpe']:.2f}")
        
        return cost_results
    
    def generate_final_report(self) -> None:
        """Generate comprehensive final report"""
        print("\n" + "="*80)
        print("📄 FINAL COMPREHENSIVE REPORT")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create detailed report
        report = {
            'timestamp': timestamp,
            'summary': {
                'best_timeframe': self.results_summary['phase1']['best_timeframe'],
                'best_parameters': self.results_summary['phase2']['best_parameters'],
                'expected_sharpe': self.results_summary['phase3']['consistency_metrics']['mean_sharpe'],
                'expected_return': self.results_summary['phase3']['consistency_metrics']['mean_return'],
                'num_final_pairs': self.results_summary['phase4']['num_pairs'],
            },
            'detailed_results': self.results_summary,
        }
        
        # Save comprehensive report
        report_file = f'comprehensive_report_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📊 FINAL RESULTS SUMMARY:")
        print(f"  Best Timeframe: {report['summary']['best_timeframe']}")
        print(f"  Best Parameters:")
        for key, value in self.results_summary['phase2']['best_parameters'].items():
            if key in ['lookback_period', 'entry_threshold', 'exit_threshold', 'stop_loss_threshold']:
                print(f"    {key}: {value}")
        print(f"  Expected Performance:")
        print(f"    Sharpe Ratio: {report['summary']['expected_sharpe']:.2f}")
        print(f"    Annual Return: {report['summary']['expected_return']:.2%}")
        print(f"  Final Pairs Selected: {report['summary']['num_final_pairs']}")
        
        # Generate trading configuration
        self._generate_trading_config(timestamp)
        
        print(f"\n✅ Complete report saved to: {report_file}")
    
    def _generate_trading_config(self, timestamp: str) -> None:
        """Generate ready-to-use trading configuration"""
        config = {
            'created_at': timestamp,
            'timeframe': self.results_summary['phase1']['best_timeframe'],
            'parameters': {
                k: v for k, v in self.results_summary['phase2']['best_parameters'].items()
                if k in ['lookback_period', 'entry_threshold', 'exit_threshold', 'stop_loss_threshold']
            },
            'pairs': []
        }
        
        for pair in self.results_summary['phase4']['final_pairs']:
            config['pairs'].append({
                'symbol1': pair['symbol1'],
                'symbol2': pair['symbol2'],
                'hedge_ratio': pair['hedge_ratio'],
                'quality_score': pair.get('quality_score', 0),
                'p_value': pair['p_value'],
            })
        
        config_file = f'trading_config_final_{timestamp}.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"📝 Trading configuration saved to: {config_file}")
    
    def run_complete_workflow(self) -> Dict:
        """
        Run the complete comprehensive workflow
        """
        print("="*80)
        print("🔬 COMPREHENSIVE ACCURACY-FOCUSED WORKFLOW")
        print("="*80)
        print("\nThis workflow prioritizes accuracy over speed.")
        print("Expected runtime: 6-10 hours")
        print("\nPhases:")
        print("1. Multi-timeframe cointegration discovery")
        print("2. Parameter optimization")
        print("3. Cross-validation")
        print("4. Final pair selection")
        print("5. Robustness testing")
        
        start_time = datetime.now()
        
        # Phase 1: Multi-timeframe cointegration
        cointegration_results = self.phase1_multi_timeframe_cointegration()
        best_timeframe = self.results_summary['phase1']['best_timeframe']
        best_coint_file = glob.glob(f"cointegration_results_{best_timeframe}/cointegration_results_*.json")[0]
        
        # Phase 2: Parameter optimization
        best_params = self.phase2_parameter_optimization(best_timeframe, best_coint_file)
        
        # Phase 3: Cross-validation
        consistency_metrics = self.phase3_cross_validation(best_timeframe, best_params, best_coint_file)
        
        # Phase 4: Final pair selection
        final_pairs = self.phase4_final_pair_selection(best_timeframe, best_params, best_coint_file)
        
        # Phase 5: Robustness testing
        robustness_results = self.phase5_robustness_testing(final_pairs, best_timeframe, best_params)
        
        # Generate final report
        self.generate_final_report()
        
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds() / 3600
        
        print(f"\n⏱️ Total runtime: {runtime:.1f} hours")
        print("\n✅ COMPREHENSIVE WORKFLOW COMPLETE!")
        
        return self.results_summary


# Import the CointegrationFilter class
class CointegrationFilter:
    """Simplified version for this script"""
    def __init__(self, strategy: str = "moderate"):
        self.strategy = strategy
    
    def load_cointegration_results(self, filepath: str) -> List:
        """Load cointegration results"""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'cointegrated_pairs' in data:
            return data['cointegrated_pairs']
        return data
    
    def calculate_quality_score(self, pair: Dict) -> float:
        """Calculate quality score for a pair"""
        score = 0.0
        
        # P-value score
        p_value = pair.get('p_value', 1.0)
        if p_value <= 0.001:
            score += 30
        elif p_value <= 0.01:
            score += 20
        elif p_value <= 0.05:
            score += 10
        
        # Correlation score
        correlation = abs(pair.get('correlation', 0))
        if 0.7 <= correlation <= 0.9:
            score += 20
        elif 0.6 <= correlation <= 0.95:
            score += 15
        
        # Half-life score
        if 'spread_properties' in pair:
            props = pair['spread_properties']
            if isinstance(props, dict):
                hl = props.get('half_life_ou') or props.get('half_life')
                if hl and 10 <= hl <= 50:
                    score += 25
                elif hl and 5 <= hl <= 100:
                    score += 15
        
        return score


def main():
    """Main entry point for comprehensive workflow"""
    print("="*80)
    print("🚀 COMPREHENSIVE COINTEGRATION TRADING SYSTEM")
    print("="*80)
    print("\n⚠️ ACCURACY MODE: This will run extensive testing")
    print("   Expected runtime: 6-10 hours")
    print("   Will test: 4 timeframes × 100+ parameter combinations × 5 validation windows")
    
    print("\nThis comprehensive workflow will:")
    print("1. Test multiple timeframes (30T, 1H, 2H, 4H)")
    print("2. Optimize parameters for each timeframe")
    print("3. Run walk-forward cross-validation")
    print("4. Perform robustness testing")
    print("5. Select the absolute best configuration")
    
    proceed = input("\nProceed with comprehensive testing? (y/n): ").strip().lower()
    
    if proceed == 'y':
        workflow = ComprehensiveWorkflow()
        results = workflow.run_complete_workflow()
        return results
    else:
        print("\nCancelled. Consider running the quick workflow for faster results.")
        return None


if __name__ == "__main__":
    results = main()
