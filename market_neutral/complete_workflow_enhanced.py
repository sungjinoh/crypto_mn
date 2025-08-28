"""
COMPLETE WORKFLOW - Enhanced Version
Comprehensive pipeline that tests multiple timeframes and parameters systematically
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Optional
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_cointegration_finder_v2 import EnhancedCointegrationFinder
from apply_optimal_filters import CointegrationFilter
from market_neutral.run_fixed_parameters import run_fixed_parameters_backtest


class EnhancedCompleteWorkflow:
    """
    Enhanced workflow that systematically tests everything:
    1. Multiple timeframes for cointegration
    2. Multiple parameter sets for backtesting
    3. Optimal threshold discovery
    4. Cross-validation across time periods
    """
    
    def __init__(self):
        self.cointegration_results = {}
        self.backtest_results = {}
        self.optimal_configs = {}
        self.final_selection = None
        
    def step1_generate_multi_timeframe_cointegration(
        self,
        timeframes: List[str] = None,
        data_years: List[int] = None,
        data_months: List = None,
        max_symbols: Optional[int] = None
    ) -> Dict:
        """
        Step 1: Generate cointegration for multiple timeframes
        
        This creates different cointegration sets for each timeframe,
        as relationships may vary with data granularity
        """
        print("\n" + "="*80)
        print("📊 STEP 1: MULTI-TIMEFRAME COINTEGRATION GENERATION")
        print("="*80)
        
        if timeframes is None:
            timeframes = ["30T", "1H", "2H", "4H"]
        
        if data_years is None:
            data_years = [2023]
            
        if data_months is None:
            data_months = list(range(1, 13))  # Full year
        
        print(f"\n📈 Configuration:")
        print(f"  Timeframes to test: {timeframes}")
        print(f"  Training period: {data_years[0]} months {data_months[0]}-{data_months[-1]}")
        print(f"  Symbol limit: {max_symbols if max_symbols else 'All available'}")
        
        results = {}
        
        for tf in timeframes:
            print(f"\n{'='*60}")
            print(f"⏰ Generating cointegration for timeframe: {tf}")
            print(f"{'='*60}")
            
            # Check if already exists
            existing_files = glob.glob(f"cointegration_results_{tf}/cointegration_results_*.json")
            
            if existing_files and len(existing_files) > 0:
                use_existing = input(f"\n📁 Found existing results for {tf}. Use existing? (y/n): ").strip().lower()
                if use_existing != 'n':
                    latest_file = max(existing_files, key=os.path.getctime)
                    print(f"✅ Using existing: {latest_file}")
                    
                    # Load existing results
                    with open(latest_file, 'r') as f:
                        existing_data = json.load(f)
                    
                    results[tf] = {
                        'file_path': latest_file,
                        'total_pairs': len(existing_data.get('cointegrated_pairs', [])),
                        'data': existing_data
                    }
                    continue
            
            # Generate new cointegration
            finder = EnhancedCointegrationFinder(
                base_path="binance_futures_data",
                resample_interval=tf,
                min_data_points=1000,
                significance_level=0.05,
                min_daily_volume=1000000,
                check_stationarity=True,
                use_rolling_window=True,
                rolling_window_size=500,
                rolling_step_size=100,
            )
            
            coint_results = finder.find_all_cointegrated_pairs(
                years=data_years,
                months=data_months,
                max_symbols=max_symbols,
                use_parallel=True,
                filter_by_sector=True,
                max_correlation=0.95,
            )
            
            # Save results
            output_dir = f"cointegration_results_{tf}"
            finder.save_results(coint_results, output_dir=output_dir)
            
            # Get file path
            saved_files = glob.glob(f"{output_dir}/cointegration_results_*.json")
            latest_file = max(saved_files, key=os.path.getctime)
            
            results[tf] = {
                'file_path': latest_file,
                'total_pairs': len(coint_results.get('cointegrated_pairs', [])),
                'data': coint_results
            }
            
            print(f"\n✅ Generated {results[tf]['total_pairs']} cointegrated pairs for {tf}")
        
        self.cointegration_results = results
        
        # Summary
        print(f"\n📊 COINTEGRATION GENERATION SUMMARY")
        print("="*60)
        for tf, info in results.items():
            print(f"  {tf:4s}: {info['total_pairs']:4d} pairs")
        
        return results
    
    def step2_discover_optimal_thresholds(
        self,
        sample_size: int = 100,
        validation_period: Dict = None
    ) -> Dict:
        """
        Step 2: Discover optimal thresholds for filtering pairs
        
        Tests different quality thresholds to find optimal number of pairs
        """
        print("\n" + "="*80)
        print("🎯 STEP 2: OPTIMAL THRESHOLD DISCOVERY")
        print("="*80)
        
        if validation_period is None:
            validation_period = {
                'years': [2024],
                'months': [1, 2, 3, 4, 5, 6]
            }
        
        print(f"\n📈 Testing thresholds on validation period: {validation_period}")
        
        threshold_results = {}
        
        for tf, coint_info in self.cointegration_results.items():
            print(f"\n{'='*60}")
            print(f"Testing thresholds for {tf}")
            print(f"{'='*60}")
            
            # Load cointegration data
            pairs = coint_info['data'].get('cointegrated_pairs', [])
            
            if len(pairs) == 0:
                print(f"⚠️ No pairs found for {tf}, skipping")
                continue
            
            # Calculate quality scores
            filter_obj = CointegrationFilter("moderate")
            for pair in pairs:
                pair['quality_score'] = filter_obj.calculate_quality_score(pair)
            
            # Sort by quality score
            sorted_pairs = sorted(pairs, key=lambda x: x.get('quality_score', 0), reverse=True)
            
            # Test different threshold levels
            threshold_levels = [
                {'name': 'top_10', 'count': min(10, len(sorted_pairs))},
                {'name': 'top_20', 'count': min(20, len(sorted_pairs))},
                {'name': 'top_30', 'count': min(30, len(sorted_pairs))},
                {'name': 'top_50', 'count': min(50, len(sorted_pairs))},
                {'name': 'quality_80', 'pairs': [p for p in sorted_pairs if p.get('quality_score', 0) >= 80]},
                {'name': 'quality_70', 'pairs': [p for p in sorted_pairs if p.get('quality_score', 0) >= 70]},
                {'name': 'quality_60', 'pairs': [p for p in sorted_pairs if p.get('quality_score', 0) >= 60]},
            ]
            
            level_results = []
            
            for level in threshold_levels:
                # Get pairs for this threshold
                if 'count' in level:
                    test_pairs = sorted_pairs[:level['count']]
                else:
                    test_pairs = level.get('pairs', [])
                
                if len(test_pairs) == 0:
                    continue
                
                # Sample for quick testing
                if len(test_pairs) > sample_size:
                    test_pairs = test_pairs[:sample_size]
                
                print(f"  Testing {level['name']}: {len(test_pairs)} pairs")
                
                # Quick backtest with default parameters
                pair_list = [(p['symbol1'], p['symbol2']) for p in test_pairs[:10]]  # Test subset
                
                try:
                    results_df = run_fixed_parameters_backtest(
                        fixed_params={
                            'lookback_period': 40,
                            'entry_threshold': 2.0,
                            'exit_threshold': 0.5,
                            'stop_loss_threshold': 3.5
                        },
                        specific_pairs=pair_list,
                        test_years=validation_period['years'],
                        test_months=validation_period['months'],
                        save_results=False,
                        save_plots=False,
                    )
                    
                    # Calculate metrics
                    successful = results_df[results_df['success'] == True]
                    if len(successful) > 0:
                        avg_sharpe = successful['sharpe_ratio'].mean()
                        avg_return = successful['total_return'].mean()
                        success_rate = len(successful) / len(results_df)
                        
                        level_results.append({
                            'threshold': level['name'],
                            'num_pairs': len(test_pairs),
                            'min_quality': min(p.get('quality_score', 0) for p in test_pairs),
                            'avg_quality': np.mean([p.get('quality_score', 0) for p in test_pairs]),
                            'avg_sharpe': avg_sharpe,
                            'avg_return': avg_return,
                            'success_rate': success_rate,
                            'score': avg_sharpe * 0.5 + avg_return * 100 * 0.3 + success_rate * 0.2
                        })
                except Exception as e:
                    print(f"    Error testing {level['name']}: {e}")
            
            if level_results:
                # Find best threshold
                best_threshold = max(level_results, key=lambda x: x['score'])
                threshold_results[tf] = {
                    'best_threshold': best_threshold,
                    'all_results': level_results
                }
                
                print(f"\n  ✅ Best threshold for {tf}: {best_threshold['threshold']}")
                print(f"     Pairs: {best_threshold['num_pairs']}, Sharpe: {best_threshold['avg_sharpe']:.2f}")
        
        self.optimal_configs['thresholds'] = threshold_results
        
        return threshold_results
    
    def step3_optimize_parameters(
        self,
        parameter_grid: Dict = None,
        test_sample_size: int = 30
    ) -> Dict:
        """
        Step 3: Find optimal parameters for each timeframe
        
        Tests different parameter combinations to find best settings
        """
        print("\n" + "="*80)
        print("⚙️ STEP 3: PARAMETER OPTIMIZATION")
        print("="*80)
        
        if parameter_grid is None:
            parameter_grid = {
                'lookback_period': [20, 30, 40, 50, 60],
                'entry_threshold': [1.5, 1.75, 2.0, 2.25, 2.5],
                'exit_threshold': [0.0, 0.25, 0.5, 0.75],
                'stop_loss_threshold': [3.0, 3.5, 4.0],
            }
        
        # Calculate total combinations
        total_combinations = 1
        for param_values in parameter_grid.values():
            total_combinations *= len(param_values)
        
        print(f"\n📈 Parameter grid:")
        for param, values in parameter_grid.items():
            print(f"  {param}: {values}")
        print(f"  Total combinations: {total_combinations}")
        
        optimization_results = {}
        
        for tf, coint_info in self.cointegration_results.items():
            print(f"\n{'='*60}")
            print(f"Optimizing parameters for {tf}")
            print(f"{'='*60}")
            
            # Get best threshold pairs
            if 'thresholds' in self.optimal_configs and tf in self.optimal_configs['thresholds']:
                best_threshold = self.optimal_configs['thresholds'][tf]['best_threshold']
                num_pairs = best_threshold['num_pairs']
            else:
                num_pairs = 30  # Default
            
            # Get top pairs
            pairs = coint_info['data'].get('cointegrated_pairs', [])
            filter_obj = CointegrationFilter("moderate")
            for pair in pairs:
                pair['quality_score'] = filter_obj.calculate_quality_score(pair)
            
            sorted_pairs = sorted(pairs, key=lambda x: x.get('quality_score', 0), reverse=True)
            test_pairs = sorted_pairs[:min(num_pairs, test_sample_size)]
            pair_list = [(p['symbol1'], p['symbol2']) for p in test_pairs]
            
            print(f"  Testing on {len(pair_list)} pairs")
            
            # Test parameter combinations
            param_results = []
            
            # Generate all combinations
            param_combinations = list(itertools.product(
                parameter_grid['lookback_period'],
                parameter_grid['entry_threshold'],
                parameter_grid['exit_threshold'],
                parameter_grid['stop_loss_threshold']
            ))
            
            # Sample combinations for faster testing
            if len(param_combinations) > 20:
                # Test a smart subset
                smart_sample = [
                    param_combinations[0],   # First
                    param_combinations[-1],   # Last
                    param_combinations[len(param_combinations)//2],  # Middle
                ]
                # Add random samples
                import random
                random_indices = random.sample(range(len(param_combinations)), min(17, len(param_combinations)))
                for idx in random_indices:
                    if param_combinations[idx] not in smart_sample:
                        smart_sample.append(param_combinations[idx])
                param_combinations = smart_sample[:20]
            
            print(f"  Testing {len(param_combinations)} parameter combinations")
            
            for i, (lb, entry, exit_t, sl) in enumerate(param_combinations, 1):
                if i % 5 == 0:
                    print(f"    Progress: {i}/{len(param_combinations)}")
                
                params = {
                    'lookback_period': lb,
                    'entry_threshold': entry,
                    'exit_threshold': exit_t,
                    'stop_loss_threshold': sl
                }
                
                try:
                    results_df = run_fixed_parameters_backtest(
                        fixed_params=params,
                        specific_pairs=pair_list[:10],  # Test on subset for speed
                        test_years=[2024],
                        test_months=[1, 2, 3],  # Quick test
                        save_results=False,
                        save_plots=False,
                    )
                    
                    successful = results_df[results_df['success'] == True]
                    if len(successful) > 0:
                        param_results.append({
                            **params,
                            'sharpe': successful['sharpe_ratio'].mean(),
                            'return': successful['total_return'].mean(),
                            'trades': successful['num_trades'].mean(),
                            'win_rate': successful['win_rate'].mean(),
                            'max_dd': successful['max_drawdown'].mean(),
                            'score': (successful['sharpe_ratio'].mean() * 0.4 + 
                                    successful['total_return'].mean() * 100 * 0.3 +
                                    successful['win_rate'].mean() * 0.2 +
                                    (1 - abs(successful['max_drawdown'].mean())) * 0.1)
                        })
                except Exception as e:
                    continue
            
            if param_results:
                # Find best parameters
                best_params = max(param_results, key=lambda x: x['score'])
                optimization_results[tf] = {
                    'best_params': best_params,
                    'all_results': param_results
                }
                
                print(f"\n  ✅ Best parameters for {tf}:")
                print(f"     Lookback: {best_params['lookback_period']}")
                print(f"     Entry: {best_params['entry_threshold']}")
                print(f"     Exit: {best_params['exit_threshold']}")
                print(f"     Stop Loss: {best_params['stop_loss_threshold']}")
                print(f"     Expected Sharpe: {best_params['sharpe']:.2f}")
        
        self.optimal_configs['parameters'] = optimization_results
        
        return optimization_results
    
    def step4_cross_validation(
        self,
        num_windows: int = 3
    ) -> Dict:
        """
        Step 4: Cross-validation to ensure robustness
        
        Tests optimal configuration across multiple time windows
        """
        print("\n" + "="*80)
        print("🔄 STEP 4: CROSS-VALIDATION")
        print("="*80)
        
        # Define validation windows
        windows = [
            {
                'name': 'Q1 2024',
                'train': {'years': [2023], 'months': [10, 11, 12]},
                'test': {'years': [2024], 'months': [1, 2, 3]}
            },
            {
                'name': 'Q2 2024',
                'train': {'years': [2024], 'months': [1, 2, 3]},
                'test': {'years': [2024], 'months': [4, 5, 6]}
            },
            {
                'name': 'Q3 2024',
                'train': {'years': [2024], 'months': [4, 5, 6]},
                'test': {'years': [2024], 'months': [7, 8, 9]}
            }
        ][:num_windows]
        
        cv_results = {}
        
        for tf in self.cointegration_results.keys():
            print(f"\n{'='*60}")
            print(f"Cross-validating {tf}")
            print(f"{'='*60}")
            
            if tf not in self.optimal_configs.get('parameters', {}):
                print(f"⚠️ No optimized parameters for {tf}, skipping")
                continue
            
            best_params = self.optimal_configs['parameters'][tf]['best_params']
            window_results = []
            
            for window in windows:
                print(f"\n  Testing {window['name']}:")
                
                # Get pairs for this timeframe
                pairs = self.cointegration_results[tf]['data'].get('cointegrated_pairs', [])
                if len(pairs) < 10:
                    print(f"    Insufficient pairs, skipping")
                    continue
                
                # Use top pairs
                filter_obj = CointegrationFilter("moderate")
                for pair in pairs:
                    pair['quality_score'] = filter_obj.calculate_quality_score(pair)
                
                sorted_pairs = sorted(pairs, key=lambda x: x.get('quality_score', 0), reverse=True)
                test_pairs = sorted_pairs[:20]
                pair_list = [(p['symbol1'], p['symbol2']) for p in test_pairs]
                
                # Test with best parameters
                try:
                    results_df = run_fixed_parameters_backtest(
                        fixed_params={
                            'lookback_period': best_params['lookback_period'],
                            'entry_threshold': best_params['entry_threshold'],
                            'exit_threshold': best_params['exit_threshold'],
                            'stop_loss_threshold': best_params['stop_loss_threshold']
                        },
                        specific_pairs=pair_list[:10],
                        test_years=window['test']['years'],
                        test_months=window['test']['months'],
                        save_results=False,
                        save_plots=False,
                    )
                    
                    successful = results_df[results_df['success'] == True]
                    if len(successful) > 0:
                        window_results.append({
                            'window': window['name'],
                            'sharpe': successful['sharpe_ratio'].mean(),
                            'return': successful['total_return'].mean(),
                            'win_rate': successful['win_rate'].mean()
                        })
                        
                        print(f"    Sharpe: {successful['sharpe_ratio'].mean():.2f}")
                        print(f"    Return: {successful['total_return'].mean():.2%}")
                except Exception as e:
                    print(f"    Error: {e}")
            
            if window_results:
                # Calculate consistency
                sharpes = [w['sharpe'] for w in window_results]
                returns = [w['return'] for w in window_results]
                
                cv_results[tf] = {
                    'mean_sharpe': np.mean(sharpes),
                    'std_sharpe': np.std(sharpes),
                    'mean_return': np.mean(returns),
                    'std_return': np.std(returns),
                    'consistency_score': np.mean(sharpes) / (np.std(sharpes) + 0.01),
                    'window_results': window_results
                }
                
                print(f"\n  ✅ Cross-validation for {tf}:")
                print(f"     Mean Sharpe: {cv_results[tf]['mean_sharpe']:.2f} ± {cv_results[tf]['std_sharpe']:.2f}")
                print(f"     Consistency: {cv_results[tf]['consistency_score']:.2f}")
        
        self.optimal_configs['cross_validation'] = cv_results
        
        return cv_results
    
    def step5_final_selection(self) -> Dict:
        """
        Step 5: Make final selection of best configuration
        
        Selects the best timeframe, parameters, and pairs based on all testing
        """
        print("\n" + "="*80)
        print("🏆 STEP 5: FINAL SELECTION")
        print("="*80)
        
        # Score each timeframe configuration
        timeframe_scores = {}
        
        for tf in self.cointegration_results.keys():
            score = 0
            details = {}
            
            # Factor 1: Number of quality pairs
            pairs = self.cointegration_results[tf]['data'].get('cointegrated_pairs', [])
            filter_obj = CointegrationFilter("moderate")
            quality_pairs = [p for p in pairs if filter_obj.calculate_quality_score(p) >= 70]
            score += min(len(quality_pairs), 100) * 0.5
            details['quality_pairs'] = len(quality_pairs)
            
            # Factor 2: Parameter optimization results
            if tf in self.optimal_configs.get('parameters', {}):
                best_params = self.optimal_configs['parameters'][tf]['best_params']
                score += best_params.get('sharpe', 0) * 20
                score += best_params.get('return', 0) * 50
                details['optimized_sharpe'] = best_params.get('sharpe', 0)
                details['optimized_return'] = best_params.get('return', 0)
            
            # Factor 3: Cross-validation consistency
            if tf in self.optimal_configs.get('cross_validation', {}):
                cv = self.optimal_configs['cross_validation'][tf]
                score += cv['consistency_score'] * 10
                details['consistency'] = cv['consistency_score']
            
            timeframe_scores[tf] = {
                'score': score,
                'details': details
            }
        
        # Select best configuration
        if timeframe_scores:
            best_tf = max(timeframe_scores, key=lambda x: timeframe_scores[x]['score'])
            
            print(f"\n📊 Timeframe Scores:")
            for tf, info in sorted(timeframe_scores.items(), key=lambda x: x[1]['score'], reverse=True):
                print(f"  {tf}: Score={info['score']:.1f}")
                for key, value in info['details'].items():
                    if isinstance(value, float):
                        print(f"    - {key}: {value:.2f}")
                    else:
                        print(f"    - {key}: {value}")
            
            print(f"\n✅ BEST CONFIGURATION: {best_tf}")
            
            # Get final configuration details
            final_config = {
                'timeframe': best_tf,
                'parameters': self.optimal_configs.get('parameters', {}).get(best_tf, {}).get('best_params', {}),
                'threshold': self.optimal_configs.get('thresholds', {}).get(best_tf, {}).get('best_threshold', {}),
                'expected_performance': {
                    'sharpe': self.optimal_configs.get('cross_validation', {}).get(best_tf, {}).get('mean_sharpe', 0),
                    'return': self.optimal_configs.get('cross_validation', {}).get(best_tf, {}).get('mean_return', 0),
                },
                'score': timeframe_scores[best_tf]['score']
            }
            
            # Select final pairs
            pairs = self.cointegration_results[best_tf]['data'].get('cointegrated_pairs', [])
            filter_obj = CointegrationFilter("moderate")
            for pair in pairs:
                pair['quality_score'] = filter_obj.calculate_quality_score(pair)
            
            sorted_pairs = sorted(pairs, key=lambda x: x.get('quality_score', 0), reverse=True)
            
            # Apply threshold
            if 'num_pairs' in final_config.get('threshold', {}):
                final_pairs = sorted_pairs[:final_config['threshold']['num_pairs']]
            else:
                final_pairs = sorted_pairs[:10]  # Default to top 10
            
            final_config['selected_pairs'] = [
                {
                    'symbol1': p['symbol1'],
                    'symbol2': p['symbol2'],
                    'hedge_ratio': p['hedge_ratio'],
                    'quality_score': p.get('quality_score', 0),
                    'p_value': p['p_value']
                }
                for p in final_pairs
            ]
            
            self.final_selection = final_config
            
            # Generate trading configuration
            self._generate_trading_config(final_config)
            
            return final_config
        else:
            print("❌ No valid configurations found")
            return {}
    
    def _generate_trading_config(self, config: Dict) -> None:
        """Generate ready-to-use trading configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        trading_config = {
            'created_at': timestamp,
            'timeframe': config['timeframe'],
            'parameters': {
                'lookback_period': config['parameters'].get('lookback_period', 40),
                'entry_threshold': config['parameters'].get('entry_threshold', 2.0),
                'exit_threshold': config['parameters'].get('exit_threshold', 0.5),
                'stop_loss_threshold': config['parameters'].get('stop_loss_threshold', 3.5)
            },
            'expected_performance': config['expected_performance'],
            'pairs': config['selected_pairs']
        }
        
        # Save configuration
        config_file = f'trading_config_complete_{timestamp}.json'
        with open(config_file, 'w') as f:
            json.dump(trading_config, f, indent=2)
        
        print(f"\n💾 Trading configuration saved to: {config_file}")
        
        # Save detailed report
        report = {
            'timestamp': timestamp,
            'workflow_summary': {
                'timeframes_tested': list(self.cointegration_results.keys()),
                'best_timeframe': config['timeframe'],
                'optimization_results': self.optimal_configs,
                'final_configuration': config
            }
        }
        
        report_file = f'workflow_report_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Detailed report saved to: {report_file}")
    
    def run_complete_workflow(
        self,
        timeframes: List[str] = None,
        quick_mode: bool = False
    ) -> Dict:
        """
        Run the complete enhanced workflow
        """
        print("="*80)
        print("🔬 ENHANCED COMPLETE WORKFLOW")
        print("="*80)
        
        if quick_mode:
            print("\n⚡ Running in QUICK MODE (reduced testing)")
            if timeframes is None:
                timeframes = ["1H", "2H"]  # Just 2 timeframes
        else:
            print("\n🎯 Running in FULL MODE (comprehensive testing)")
            if timeframes is None:
                timeframes = ["30T", "1H", "2H", "4H"]  # All timeframes
        
        print(f"\nWorkflow will:")
        print(f"1. Generate cointegration for {len(timeframes)} timeframes")
        print(f"2. Discover optimal thresholds for pair selection")
        print(f"3. Optimize parameters for each timeframe")
        print(f"4. Cross-validate across time windows")
        print(f"5. Select best configuration and pairs")
        
        estimated_time = len(timeframes) * (15 if quick_mode else 30)
        print(f"\n⏱️ Estimated time: {estimated_time}-{estimated_time*2} minutes")
        
        proceed = input("\nProceed? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Cancelled")
            return {}
        
        start_time = datetime.now()
        
        # Step 1: Multi-timeframe cointegration
        self.step1_generate_multi_timeframe_cointegration(
            timeframes=timeframes,
            max_symbols=50 if quick_mode else None
        )
        
        # Step 2: Threshold discovery
        self.step2_discover_optimal_thresholds(
            sample_size=30 if quick_mode else 100
        )
        
        # Step 3: Parameter optimization
        if quick_mode:
            # Reduced parameter grid for quick mode
            param_grid = {
                'lookback_period': [30, 40, 50],
                'entry_threshold': [1.75, 2.0, 2.25],
                'exit_threshold': [0.25, 0.5],
                'stop_loss_threshold': [3.0, 3.5]
            }
        else:
            param_grid = None  # Use default comprehensive grid
        
        self.step3_optimize_parameters(
            parameter_grid=param_grid,
            test_sample_size=20 if quick_mode else 30
        )
        
        # Step 4: Cross-validation
        self.step4_cross_validation(
            num_windows=2 if quick_mode else 3
        )
        
        # Step 5: Final selection
        final_config = self.step5_final_selection()
        
        # Summary
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "="*80)
        print("✅ WORKFLOW COMPLETE!")
        print("="*80)
        print(f"\n📊 FINAL RESULTS:")
        print(f"  Best Timeframe: {final_config.get('timeframe', 'N/A')}")
        print(f"  Best Parameters:")
        if 'parameters' in final_config:
            for key, value in final_config['parameters'].items():
                if key in ['lookback_period', 'entry_threshold', 'exit_threshold', 'stop_loss_threshold']:
                    print(f"    - {key}: {value}")
        print(f"  Expected Performance:")
        if 'expected_performance' in final_config:
            print(f"    - Sharpe Ratio: {final_config['expected_performance']['sharpe']:.2f}")
            print(f"    - Return: {final_config['expected_performance']['return']:.2%}")
        print(f"  Selected Pairs: {len(final_config.get('selected_pairs', []))}")
        
        print(f"\n⏱️ Total runtime: {runtime:.1f} minutes")
        
        return final_config


def main():
    """Main entry point"""
    print("="*80)
    print("🚀 ENHANCED COMPLETE WORKFLOW")
    print("="*80)
    print("\nThis workflow systematically tests:")
    print("• Multiple timeframes for cointegration")
    print("• Different quality thresholds for pair selection")
    print("• Various parameter combinations")
    print("• Cross-validation across time periods")
    
    print("\n📋 OPTIONS:")
    print("1. Quick Mode (30-60 minutes) - 2 timeframes, reduced testing")
    print("2. Full Mode (2-4 hours) - 4 timeframes, comprehensive testing")
    print("3. Custom - Specify your own settings")
    
    choice = input("\nSelect mode (1-3, default=1): ").strip()
    
    workflow = EnhancedCompleteWorkflow()
    
    if choice == "2":
        # Full mode
        config = workflow.run_complete_workflow(quick_mode=False)
    elif choice == "3":
        # Custom mode
        print("\n🔧 CUSTOM SETTINGS:")
        tf_input = input("Timeframes (comma-separated, e.g., 1H,2H,4H): ").strip()
        if tf_input:
            timeframes = [tf.strip().upper() for tf in tf_input.split(',')]
        else:
            timeframes = ["1H", "2H"]
        
        config = workflow.run_complete_workflow(
            timeframes=timeframes,
            quick_mode=False
        )
    else:
        # Quick mode (default)
        config = workflow.run_complete_workflow(quick_mode=True)
    
    if config:
        print("\n🎯 Next Steps:")
        print("1. Review the trading configuration file")
        print("2. Implement the strategy with selected pairs")
        print("3. Monitor performance and compare to expectations")
        print("4. Rerun monthly to adapt to market changes")
    
    return config


if __name__ == "__main__":
    final_config = main()
