"""
Apply Optimal Filters to Cointegration Results
This script applies the recommended filtering criteria to select high-quality pairs
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob
from datetime import datetime


class CointegrationFilter:
    """
    Apply optimal filtering criteria to cointegration results
    """
    
    def __init__(self, strategy: str = "moderate"):
        """
        Initialize with filtering strategy
        
        Args:
            strategy: 'conservative', 'moderate', or 'aggressive'
        """
        self.strategy = strategy
        self.filters = self._get_filter_criteria(strategy)
        
    def _get_filter_criteria(self, strategy: str) -> Dict:
        """
        Get filtering criteria based on strategy
        """
        if strategy == "conservative":
            return {
                'max_p_value': 0.001,
                'min_correlation': 0.80,
                'max_correlation': 0.92,
                'min_half_life': 10,
                'max_half_life': 30,
                'min_stability_ratio': 0.7,
                'require_stationary': True,
                'min_quality_score': 80,  # High quality threshold
                'target_pairs': 5,  # Expected number of pairs
                'max_pairs': 10,
            }
        elif strategy == "aggressive":
            return {
                'max_p_value': 0.05,
                'min_correlation': 0.60,
                'max_correlation': 0.98,
                'min_half_life': 5,
                'max_half_life': 100,
                'min_stability_ratio': 0.3,
                'require_stationary': False,
                'min_quality_score': 50,
                'target_pairs': 30,
                'max_pairs': 50,
            }
        else:  # moderate
            return {
                'max_p_value': 0.01,
                'min_correlation': 0.70,
                'max_correlation': 0.95,
                'min_half_life': 10,
                'max_half_life': 50,
                'min_stability_ratio': 0.5,
                'require_stationary': True,
                'min_quality_score': 65,
                'target_pairs': 15,
                'max_pairs': 20,
            }
    
    def load_cointegration_results(self, filepath: str) -> List[Dict]:
        """
        Load cointegration results from file
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'cointegrated_pairs' in data:
                return data['cointegrated_pairs']
            return data
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'cointegrated_pairs' in data:
                return data['cointegrated_pairs']
            return data
        elif filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
            return df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def calculate_quality_score(self, pair: Dict) -> float:
        """
        Calculate quality score for a pair (same as in statistical_filter_discovery.py)
        """
        score = 0.0
        
        # P-value score (30 points max)
        p_value = pair.get('p_value', 1.0)
        if p_value <= 0.001:
            score += 30
        elif p_value <= 0.01:
            score += 24
        elif p_value <= 0.05:
            score += 15
        else:
            score += 5
        
        # Correlation score (20 points max)
        correlation = abs(pair.get('correlation', 0))
        if 0.75 <= correlation <= 0.90:
            score += 20
        elif 0.70 <= correlation <= 0.95:
            score += 15
        elif 0.60 <= correlation <= 0.98:
            score += 10
        else:
            score += 5
        
        # Half-life score (25 points max)
        half_life = None
        if 'spread_properties' in pair:
            props = pair['spread_properties']
            if isinstance(props, dict):
                half_life = props.get('half_life_ou') or props.get('half_life')
        
        if half_life:
            if 10 <= half_life <= 30:  # Optimal
                score += 25
            elif 5 <= half_life <= 50:  # Good
                score += 18
            elif 2 <= half_life <= 100:  # Acceptable
                score += 10
            else:
                score += 5
        
        # Stationarity score (15 points max)
        if 'spread_properties' in pair:
            props = pair['spread_properties']
            if isinstance(props, dict):
                if props.get('spread_is_stationary', False):
                    score += 15
                elif props.get('spread_stationarity_pvalue', 1.0) < 0.1:
                    score += 8
        
        # Stability score (10 points max)
        if 'rolling_stability' in pair:
            stability = pair['rolling_stability']
            if isinstance(stability, dict):
                stability_ratio = stability.get('stability_ratio', 0)
                score += stability_ratio * 10
        
        return score
    
    def apply_filters(self, pairs: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Apply filtering criteria to pairs
        
        Returns:
            Tuple of (filtered_pairs, filter_statistics)
        """
        print(f"\n🔍 APPLYING {self.strategy.upper()} FILTERS")
        print("=" * 60)
        
        # Calculate quality scores
        for pair in pairs:
            pair['quality_score'] = self.calculate_quality_score(pair)
        
        # Track filtering statistics
        stats = {
            'initial_pairs': len(pairs),
            'after_cointegration': 0,
            'after_p_value': 0,
            'after_correlation': 0,
            'after_half_life': 0,
            'after_stationarity': 0,
            'after_stability': 0,
            'after_quality': 0,
            'final_pairs': 0,
        }
        
        filtered = []
        
        for pair in pairs:
            # Step 1: Must be cointegrated
            if not pair.get('is_cointegrated', False):
                continue
            stats['after_cointegration'] += 1
            
            # Step 2: P-value filter
            if pair.get('p_value', 1.0) > self.filters['max_p_value']:
                continue
            stats['after_p_value'] += 1
            
            # Step 3: Correlation filter
            correlation = abs(pair.get('correlation', 0))
            if correlation < self.filters['min_correlation']:
                continue
            if correlation > self.filters['max_correlation']:
                continue
            stats['after_correlation'] += 1
            
            # Step 4: Half-life filter
            half_life = None
            if 'spread_properties' in pair:
                props = pair['spread_properties']
                if isinstance(props, dict):
                    half_life = props.get('half_life_ou') or props.get('half_life')
            
            if half_life:
                if half_life < self.filters['min_half_life']:
                    continue
                if half_life > self.filters['max_half_life']:
                    continue
            stats['after_half_life'] += 1
            
            # Step 5: Stationarity filter (if required)
            if self.filters['require_stationary']:
                is_stationary = False
                if 'spread_properties' in pair:
                    props = pair['spread_properties']
                    if isinstance(props, dict):
                        is_stationary = props.get('spread_is_stationary', False)
                
                if not is_stationary:
                    # Check p-value as alternative
                    if 'spread_properties' in pair:
                        props = pair['spread_properties']
                        if isinstance(props, dict):
                            stat_pvalue = props.get('spread_stationarity_pvalue', 1.0)
                            if stat_pvalue >= 0.05:  # Not stationary
                                continue
            stats['after_stationarity'] += 1
            
            # Step 6: Stability filter
            if 'rolling_stability' in pair:
                stability = pair['rolling_stability']
                if isinstance(stability, dict):
                    stability_ratio = stability.get('stability_ratio', 0)
                    if stability_ratio < self.filters['min_stability_ratio']:
                        continue
            stats['after_stability'] += 1
            
            # Step 7: Quality score filter
            if pair.get('quality_score', 0) < self.filters['min_quality_score']:
                continue
            stats['after_quality'] += 1
            
            # Pair passed all filters
            filtered.append(pair)
        
        stats['final_pairs'] = len(filtered)
        
        # Sort by quality score
        filtered = sorted(filtered, key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # Limit to max pairs if specified
        if len(filtered) > self.filters['max_pairs']:
            filtered = filtered[:self.filters['max_pairs']]
            stats['final_pairs'] = len(filtered)
        
        return filtered, stats
    
    def print_filter_statistics(self, stats: Dict) -> None:
        """
        Print filtering statistics
        """
        print(f"\n📊 FILTERING STATISTICS:")
        print(f"   Initial pairs: {stats['initial_pairs']}")
        print(f"   After cointegration check: {stats['after_cointegration']} "
              f"(removed {stats['initial_pairs'] - stats['after_cointegration']})")
        print(f"   After p-value ≤ {self.filters['max_p_value']}: {stats['after_p_value']} "
              f"(removed {stats['after_cointegration'] - stats['after_p_value']})")
        print(f"   After correlation [{self.filters['min_correlation']:.2f}-{self.filters['max_correlation']:.2f}]: "
              f"{stats['after_correlation']} (removed {stats['after_p_value'] - stats['after_correlation']})")
        print(f"   After half-life [{self.filters['min_half_life']}-{self.filters['max_half_life']}]: "
              f"{stats['after_half_life']} (removed {stats['after_correlation'] - stats['after_half_life']})")
        
        if self.filters['require_stationary']:
            print(f"   After stationarity check: {stats['after_stationarity']} "
                  f"(removed {stats['after_half_life'] - stats['after_stationarity']})")
        
        print(f"   After stability ≥ {self.filters['min_stability_ratio']}: {stats['after_stability']} "
              f"(removed {stats['after_stationarity'] - stats['after_stability']})")
        print(f"   After quality score ≥ {self.filters['min_quality_score']}: {stats['after_quality']} "
              f"(removed {stats['after_stability'] - stats['after_quality']})")
        print(f"\n   ✅ FINAL PAIRS: {stats['final_pairs']}")
        print(f"   📎 Target was: {self.filters['target_pairs']} pairs")
    
    def print_selected_pairs(self, filtered_pairs: List[Dict]) -> None:
        """
        Print details of selected pairs
        """
        print(f"\n🏆 SELECTED PAIRS ({len(filtered_pairs)} pairs):")
        print("=" * 80)
        print(f"{'Rank':<5} {'Symbol1':<10} {'Symbol2':<10} {'Quality':<8} {'P-value':<10} "
              f"{'Corr':<8} {'Half-Life':<10} {'Stable':<8}")
        print("-" * 80)
        
        for i, pair in enumerate(filtered_pairs, 1):
            # Extract half-life
            half_life = "N/A"
            if 'spread_properties' in pair:
                props = pair['spread_properties']
                if isinstance(props, dict):
                    hl = props.get('half_life_ou') or props.get('half_life')
                    if hl:
                        half_life = f"{hl:.1f}"
            
            # Extract stability
            stability = "N/A"
            if 'rolling_stability' in pair:
                stab = pair['rolling_stability']
                if isinstance(stab, dict):
                    ratio = stab.get('stability_ratio', 0)
                    stability = f"{ratio:.1%}"
            
            print(f"{i:<5} {pair['symbol1'][:8]:<10} {pair['symbol2'][:8]:<10} "
                  f"{pair.get('quality_score', 0):<8.1f} {pair.get('p_value', 1):<10.4f} "
                  f"{abs(pair.get('correlation', 0)):<8.3f} {half_life:<10} {stability:<8}")
    
    def save_filtered_pairs(self, filtered_pairs: List[Dict], 
                           output_dir: str = "filtered_pairs") -> None:
        """
        Save filtered pairs to multiple formats
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_file = Path(output_dir) / f"filtered_pairs_{self.strategy}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(filtered_pairs, f, indent=2, default=str)
        print(f"\n💾 Saved JSON: {json_file}")
        
        # Save as CSV
        df = pd.DataFrame(filtered_pairs)
        
        # Flatten nested dictionaries for CSV
        if 'spread_properties' in df.columns:
            # Extract key spread properties
            df['half_life'] = df['spread_properties'].apply(
                lambda x: (x.get('half_life_ou') or x.get('half_life')) if isinstance(x, dict) else None
            )
            df['spread_is_stationary'] = df['spread_properties'].apply(
                lambda x: x.get('spread_is_stationary') if isinstance(x, dict) else None
            )
            df = df.drop('spread_properties', axis=1)
        
        if 'rolling_stability' in df.columns:
            df['stability_ratio'] = df['rolling_stability'].apply(
                lambda x: x.get('stability_ratio') if isinstance(x, dict) else None
            )
            df = df.drop('rolling_stability', axis=1)
        
        # Remove other complex columns for CSV
        complex_cols = ['regression_stats', 'stationarity', 'volume_metrics', 'funding_analysis']
        for col in complex_cols:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        csv_file = Path(output_dir) / f"filtered_pairs_{self.strategy}_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"💾 Saved CSV: {csv_file}")
        
        # Save summary report
        report_file = Path(output_dir) / f"filter_report_{self.strategy}_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(f"FILTERING REPORT - {self.strategy.upper()} STRATEGY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("FILTER CRITERIA:\n")
            f.write("-" * 40 + "\n")
            for key, value in self.filters.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nSELECTED PAIRS ({len(filtered_pairs)}):\n")
            f.write("-" * 40 + "\n")
            for i, pair in enumerate(filtered_pairs, 1):
                f.write(f"{i}. {pair['symbol1']} - {pair['symbol2']}\n")
                f.write(f"   Quality Score: {pair.get('quality_score', 0):.1f}\n")
                f.write(f"   P-value: {pair.get('p_value', 1):.6f}\n")
                f.write(f"   Correlation: {pair.get('correlation', 0):.3f}\n")
        
        print(f"💾 Saved Report: {report_file}")


def interactive_filter():
    """
    Interactive filtering with user input
    """
    print("=" * 80)
    print("🔍 COINTEGRATION PAIR FILTERING")
    print("=" * 80)
    
    # Find cointegration results files
    coint_files = glob.glob("cointegration_results*/cointegration_results_*.json")
    coint_files.extend(glob.glob("cointegration_results*/cointegration_results_*.pkl"))
    coint_files.extend(glob.glob("cointegration_results*/cointegrated_pairs_*.parquet"))
    
    if not coint_files:
        print("❌ No cointegration results found!")
        print("   Please run enhanced_cointegration_finder_v2.py first")
        return
    
    # Select file
    print("\n📁 Available cointegration results:")
    for i, file in enumerate(coint_files, 1):
        print(f"   {i}. {file}")
    
    if len(coint_files) == 1:
        selected_file = coint_files[0]
        print(f"\n✅ Auto-selected: {selected_file}")
    else:
        choice = input(f"\nSelect file (1-{len(coint_files)}): ").strip()
        try:
            selected_file = coint_files[int(choice) - 1]
        except:
            selected_file = coint_files[0]
            print(f"✅ Using default: {selected_file}")
    
    # Select strategy
    print("\n📊 FILTERING STRATEGIES:")
    print("1. Conservative (5-10 pairs, highest quality)")
    print("2. Moderate (10-20 pairs, balanced)")
    print("3. Aggressive (20-50 pairs, more opportunities)")
    print("4. Custom (specify your own criteria)")
    
    strategy_choice = input("\nSelect strategy (1-4, default=2): ").strip()
    
    if strategy_choice == "1":
        strategy = "conservative"
    elif strategy_choice == "3":
        strategy = "aggressive"
    elif strategy_choice == "4":
        # Custom strategy
        print("\n🔧 CUSTOM FILTER CRITERIA:")
        strategy = "custom"
        
        # Get custom values
        max_p = input("Max p-value (default=0.01): ").strip()
        max_p = float(max_p) if max_p else 0.01
        
        min_corr = input("Min correlation (default=0.7): ").strip()
        min_corr = float(min_corr) if min_corr else 0.7
        
        max_corr = input("Max correlation (default=0.95): ").strip()
        max_corr = float(max_corr) if max_corr else 0.95
        
        min_hl = input("Min half-life (default=10): ").strip()
        min_hl = float(min_hl) if min_hl else 10
        
        max_hl = input("Max half-life (default=50): ").strip()
        max_hl = float(max_hl) if max_hl else 50
        
        # Create custom filter
        filter = CointegrationFilter("moderate")
        filter.filters = {
            'max_p_value': max_p,
            'min_correlation': min_corr,
            'max_correlation': max_corr,
            'min_half_life': min_hl,
            'max_half_life': max_hl,
            'min_stability_ratio': 0.5,
            'require_stationary': True,
            'min_quality_score': 60,
            'target_pairs': 15,
            'max_pairs': 30,
        }
        filter.strategy = "custom"
    else:
        strategy = "moderate"
    
    if strategy != "custom":
        filter = CointegrationFilter(strategy)
    
    # Load and filter pairs
    print(f"\n📂 Loading cointegration results...")
    pairs = filter.load_cointegration_results(selected_file)
    print(f"✅ Loaded {len(pairs)} pairs")
    
    # Apply filters
    filtered_pairs, stats = filter.apply_filters(pairs)
    
    # Print results
    filter.print_filter_statistics(stats)
    filter.print_selected_pairs(filtered_pairs)
    
    # Save results
    save_choice = input("\n💾 Save filtered results? (y/n, default=y): ").strip().lower()
    if save_choice != 'n':
        filter.save_filtered_pairs(filtered_pairs)
    
    return filtered_pairs


def batch_filter_all_strategies():
    """
    Apply all three strategies and compare results
    """
    print("=" * 80)
    print("🔬 BATCH FILTERING - ALL STRATEGIES")
    print("=" * 80)
    
    # Find latest cointegration results
    coint_files = glob.glob("cointegration_results*/cointegration_results_*.json")
    coint_files.extend(glob.glob("cointegration_results*/cointegration_results_*.pkl"))
    
    if not coint_files:
        print("❌ No cointegration results found!")
        return
    
    latest_file = max(coint_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"📁 Using: {latest_file}")
    
    strategies = ["conservative", "moderate", "aggressive"]
    results = {}
    
    for strategy in strategies:
        print(f"\n{'=' * 60}")
        filter = CointegrationFilter(strategy)
        pairs = filter.load_cointegration_results(latest_file)
        
        filtered_pairs, stats = filter.apply_filters(pairs)
        
        results[strategy] = {
            'pairs': filtered_pairs,
            'stats': stats,
            'count': len(filtered_pairs)
        }
        
        print(f"✅ {strategy.upper()}: {len(filtered_pairs)} pairs selected")
    
    # Compare strategies
    print(f"\n📊 STRATEGY COMPARISON:")
    print("=" * 60)
    print(f"{'Strategy':<15} {'Pairs':<8} {'Avg Quality':<12} {'Avg P-value':<12} {'Avg Corr':<10}")
    print("-" * 60)
    
    for strategy in strategies:
        filtered = results[strategy]['pairs']
        if filtered:
            avg_quality = np.mean([p.get('quality_score', 0) for p in filtered])
            avg_pvalue = np.mean([p.get('p_value', 1) for p in filtered])
            avg_corr = np.mean([abs(p.get('correlation', 0)) for p in filtered])
            
            print(f"{strategy.capitalize():<15} {len(filtered):<8} "
                  f"{avg_quality:<12.1f} {avg_pvalue:<12.6f} {avg_corr:<10.3f}")
    
    # Find overlapping pairs
    print(f"\n🔄 PAIR OVERLAP ANALYSIS:")
    conservative_pairs = set(f"{p['symbol1']}-{p['symbol2']}" for p in results['conservative']['pairs'])
    moderate_pairs = set(f"{p['symbol1']}-{p['symbol2']}" for p in results['moderate']['pairs'])
    aggressive_pairs = set(f"{p['symbol1']}-{p['symbol2']}" for p in results['aggressive']['pairs'])
    
    core_pairs = conservative_pairs & moderate_pairs & aggressive_pairs
    print(f"   Core pairs (in all strategies): {len(core_pairs)}")
    
    if core_pairs:
        print("   Top core pairs:")
        for i, pair in enumerate(list(core_pairs)[:5], 1):
            print(f"      {i}. {pair}")
    
    # Save comparison
    comparison_df = pd.DataFrame({
        'Strategy': strategies,
        'Pairs_Count': [results[s]['count'] for s in strategies],
        'Initial_Pairs': [results[s]['stats']['initial_pairs'] for s in strategies],
        'After_P_Value': [results[s]['stats']['after_p_value'] for s in strategies],
        'After_Correlation': [results[s]['stats']['after_correlation'] for s in strategies],
        'Final_Pairs': [results[s]['stats']['final_pairs'] for s in strategies],
    })
    
    comparison_df.to_csv('strategy_comparison.csv', index=False)
    print(f"\n💾 Comparison saved to strategy_comparison.csv")
    
    return results


def main():
    """
    Main function with menu
    """
    print("=" * 80)
    print("🚀 COINTEGRATION PAIR FILTERING")
    print("=" * 80)
    print("\nThis tool applies optimal filtering criteria to select high-quality")
    print("cointegrated pairs based on statistical properties.")
    
    print("\n📋 OPTIONS:")
    print("1. Interactive filtering (choose strategy)")
    print("2. Batch filter (all strategies)")
    print("3. Quick filter (moderate strategy)")
    
    choice = input("\nSelect option (1-3, default=1): ").strip()
    
    if choice == "2":
        batch_filter_all_strategies()
    elif choice == "3":
        # Quick filter with moderate strategy
        coint_files = glob.glob("cointegration_results*/cointegration_results_*.json")
        coint_files.extend(glob.glob("cointegration_results*/cointegration_results_*.pkl"))
        
        if coint_files:
            latest_file = max(coint_files, key=lambda x: Path(x).stat().st_mtime)
            filter = CointegrationFilter("moderate")
            pairs = filter.load_cointegration_results(latest_file)
            filtered_pairs, stats = filter.apply_filters(pairs)
            filter.print_filter_statistics(stats)
            filter.print_selected_pairs(filtered_pairs)
            filter.save_filtered_pairs(filtered_pairs)
    else:
        interactive_filter()
    
    print("\n✅ Filtering complete!")


if __name__ == "__main__":
    main()
