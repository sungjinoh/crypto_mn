"""
Statistical Parameter Discovery for Cointegration Filtering
Find optimal parameters WITHOUT running backtests - based on statistical properties only
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class StatisticalFilterDiscovery:
    """
    Discover optimal filtering parameters based on statistical properties alone,
    without running expensive backtests.
    """
    
    def __init__(self, cointegration_results_path: str):
        """
        Initialize with cointegration results
        
        Args:
            cointegration_results_path: Path to cointegration results file
        """
        self.results = self.load_cointegration_results(cointegration_results_path)
        self.cointegrated_pairs = self.results.get('cointegrated_pairs', [])
        self.all_pairs = self.results.get('all_results', [])
        
    def load_cointegration_results(self, filepath: str) -> Dict:
        """Load cointegration results from file"""
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                return json.load(f)
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
            return {'cointegrated_pairs': df.to_dict('records')}
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def calculate_quality_score(self, pair: Dict) -> float:
        """
        Calculate a quality score for a cointegrated pair based on statistical properties
        
        This score predicts likelihood of good trading performance WITHOUT backtesting
        """
        score = 0.0
        weights = {
            'p_value': 30,           # Cointegration strength (lower is better)
            'half_life': 25,         # Mean reversion speed (optimal range)
            'spread_stationarity': 20,  # Spread should be stationary
            'stability': 15,         # Rolling window stability
            'correlation': 10,       # Correlation (not too high, not too low)
        }
        
        # 1. P-value score (30% weight) - exponential decay
        p_value = pair.get('p_value', 1.0)
        if p_value <= 0.001:
            score += weights['p_value']
        elif p_value <= 0.01:
            score += weights['p_value'] * 0.8
        elif p_value <= 0.05:
            score += weights['p_value'] * 0.5
        else:
            score += weights['p_value'] * 0.2
        
        # 2. Half-life score (25% weight) - optimal range 10-50 periods
        half_life = None
        if 'spread_properties' in pair:
            props = pair['spread_properties']
            if isinstance(props, dict):
                half_life = props.get('half_life_ou') or props.get('half_life')
        
        if half_life:
            if 10 <= half_life <= 30:  # Optimal range
                score += weights['half_life']
            elif 5 <= half_life <= 50:  # Good range
                score += weights['half_life'] * 0.7
            elif 2 <= half_life <= 100:  # Acceptable range
                score += weights['half_life'] * 0.4
            else:  # Too fast or too slow
                score += weights['half_life'] * 0.1
        
        # 3. Spread stationarity (20% weight)
        if 'spread_properties' in pair:
            props = pair['spread_properties']
            if isinstance(props, dict):
                if props.get('spread_is_stationary', False):
                    score += weights['spread_stationarity']
                elif props.get('spread_stationarity_pvalue', 1.0) < 0.1:
                    score += weights['spread_stationarity'] * 0.5
        
        # 4. Stability score (15% weight) - from rolling window analysis
        if 'rolling_stability' in pair:
            stability = pair['rolling_stability']
            if isinstance(stability, dict):
                stability_ratio = stability.get('stability_ratio', 0)
                score += weights['stability'] * stability_ratio
        
        # 5. Correlation score (10% weight) - prefer 0.7-0.9
        correlation = abs(pair.get('correlation', 0))
        if 0.7 <= correlation <= 0.9:
            score += weights['correlation']
        elif 0.6 <= correlation <= 0.95:
            score += weights['correlation'] * 0.7
        elif 0.5 <= correlation <= 0.99:
            score += weights['correlation'] * 0.4
        else:
            score += weights['correlation'] * 0.1
        
        return score
    
    def analyze_quality_distribution(self) -> pd.DataFrame:
        """
        Analyze the quality score distribution of all pairs
        """
        # Calculate quality scores for all cointegrated pairs
        for pair in self.cointegrated_pairs:
            pair['quality_score'] = self.calculate_quality_score(pair)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.cointegrated_pairs)
        
        print("📊 QUALITY SCORE DISTRIBUTION")
        print("=" * 60)
        print(f"Total cointegrated pairs: {len(df)}")
        print(f"\nQuality Score Percentiles:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = df['quality_score'].quantile(p/100)
            print(f"  {p:3d}th percentile: {value:.2f}")
        
        return df
    
    def find_optimal_thresholds(self, target_pairs: int = 10, 
                               min_pairs: int = 5,
                               max_pairs: int = 30) -> Dict:
        """
        Find optimal filtering thresholds to select target number of pairs
        
        Args:
            target_pairs: Desired number of pairs
            min_pairs: Minimum acceptable pairs
            max_pairs: Maximum pairs to consider
        """
        df = self.analyze_quality_distribution()
        
        # Sort by quality score
        df_sorted = df.sort_values('quality_score', ascending=False)
        
        print(f"\n🎯 OPTIMAL THRESHOLD DISCOVERY")
        print("=" * 60)
        print(f"Target: {target_pairs} pairs (range: {min_pairs}-{max_pairs})")
        
        # Find thresholds for different pair counts
        threshold_options = []
        
        for n_pairs in [min_pairs, target_pairs, max_pairs]:
            if n_pairs > len(df_sorted):
                n_pairs = len(df_sorted)
            
            top_n = df_sorted.head(n_pairs)
            
            # Calculate threshold values
            thresholds = {
                'n_pairs': n_pairs,
                'min_quality_score': top_n['quality_score'].min(),
                'max_p_value': top_n['p_value'].max(),
                'min_correlation': top_n['correlation'].min(),
                'avg_quality': top_n['quality_score'].mean(),
            }
            
            # Extract half-life thresholds
            half_lives = []
            for _, row in top_n.iterrows():
                if 'spread_properties' in row and row['spread_properties']:
                    props = row['spread_properties']
                    if isinstance(props, dict):
                        hl = props.get('half_life_ou') or props.get('half_life')
                        if hl:
                            half_lives.append(hl)
            
            if half_lives:
                thresholds['max_half_life'] = max(half_lives)
                thresholds['avg_half_life'] = np.mean(half_lives)
            
            threshold_options.append(thresholds)
        
        # Display options
        print(f"\n📋 THRESHOLD OPTIONS:")
        print(f"{'Pairs':<6} {'Min Score':<10} {'Max p-val':<10} {'Min Corr':<10} {'Max HL':<8}")
        print("-" * 55)
        
        for opt in threshold_options:
            print(f"{opt['n_pairs']:<6} {opt['min_quality_score']:<10.2f} "
                  f"{opt['max_p_value']:<10.4f} {opt['min_correlation']:<10.3f} "
                  f"{opt.get('max_half_life', 0):<8.1f}")
        
        # Select the target option
        target_option = threshold_options[1]  # Middle option
        
        return target_option
    
    def generate_filtering_criteria(self, quality_threshold: float = None) -> Dict:
        """
        Generate specific filtering criteria based on quality analysis
        """
        df = pd.DataFrame(self.cointegrated_pairs)
        
        if quality_threshold is None:
            # Auto-determine threshold for top 20% of pairs
            quality_threshold = df['quality_score'].quantile(0.8)
        
        top_pairs = df[df['quality_score'] >= quality_threshold]
        
        print(f"\n🔬 FILTERING CRITERIA GENERATION")
        print("=" * 60)
        print(f"Quality threshold: {quality_threshold:.2f}")
        print(f"Pairs selected: {len(top_pairs)}/{len(df)}")
        
        # Generate specific criteria
        criteria = {
            'statistical_filters': {
                'max_p_value': float(top_pairs['p_value'].quantile(0.95)),
                'min_correlation': float(top_pairs['correlation'].quantile(0.05)),
                'max_correlation': float(top_pairs['correlation'].quantile(0.95)),
            },
            'quality_filters': {
                'min_quality_score': quality_threshold,
            }
        }
        
        # Add half-life criteria if available
        half_lives = []
        for _, row in top_pairs.iterrows():
            if 'spread_properties' in row and row['spread_properties']:
                props = row['spread_properties']
                if isinstance(props, dict):
                    hl = props.get('half_life_ou') or props.get('half_life')
                    if hl and hl > 0:
                        half_lives.append(hl)
        
        if half_lives:
            criteria['mean_reversion_filters'] = {
                'min_half_life': float(np.quantile(half_lives, 0.05)),
                'max_half_life': float(np.quantile(half_lives, 0.95)),
                'optimal_half_life': float(np.median(half_lives)),
            }
        
        # Add stability criteria if available
        stability_ratios = []
        for _, row in top_pairs.iterrows():
            if 'rolling_stability' in row and row['rolling_stability']:
                stab = row['rolling_stability']
                if isinstance(stab, dict):
                    ratio = stab.get('stability_ratio')
                    if ratio:
                        stability_ratios.append(ratio)
        
        if stability_ratios:
            criteria['stability_filters'] = {
                'min_stability_ratio': float(np.quantile(stability_ratios, 0.1)),
            }
        
        return criteria
    
    def rank_pairs_by_tradability(self) -> pd.DataFrame:
        """
        Rank pairs by tradability score (combination of statistical and practical factors)
        """
        df = pd.DataFrame(self.cointegrated_pairs)
        
        # Calculate quality score if not already done
        if 'quality_score' not in df.columns:
            df['quality_score'] = df.apply(self.calculate_quality_score, axis=1)
        
        # Additional tradability factors
        tradability_scores = []
        
        for _, pair in df.iterrows():
            score = pair['quality_score']
            
            # Bonus for volume (if available)
            if 'volume_metrics' in pair:
                vol_metrics = pair['volume_metrics']
                if isinstance(vol_metrics, dict):
                    # Check both symbols have good volume
                    vol1 = vol_metrics.get('symbol1', {}).get('estimated_daily_volume_usd', 0)
                    vol2 = vol_metrics.get('symbol2', {}).get('estimated_daily_volume_usd', 0)
                    
                    min_vol = min(vol1, vol2)
                    if min_vol > 10_000_000:  # >$10M daily
                        score += 10
                    elif min_vol > 5_000_000:  # >$5M daily
                        score += 5
            
            # Bonus for consistent funding rates
            if 'funding_analysis' in pair:
                funding = pair['funding_analysis']
                if isinstance(funding, dict):
                    # Check funding isn't too extreme
                    funding1 = funding.get('symbol1', {})
                    funding2 = funding.get('symbol2', {})
                    
                    avg_funding1 = abs(funding1.get('avg_funding_rate', 0))
                    avg_funding2 = abs(funding2.get('avg_funding_rate', 0))
                    
                    if avg_funding1 < 0.0001 and avg_funding2 < 0.0001:  # Low funding
                        score += 5
            
            tradability_scores.append(score)
        
        df['tradability_score'] = tradability_scores
        
        # Rank by tradability
        df_ranked = df.sort_values('tradability_score', ascending=False)
        df_ranked['rank'] = range(1, len(df_ranked) + 1)
        
        return df_ranked
    
    def create_analysis_plots(self, save_dir: str = "statistical_filter_analysis"):
        """
        Create visualization plots for the analysis
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        df = self.rank_pairs_by_tradability()
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Quality Score Distribution
        axes[0, 0].hist(df['quality_score'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Quality Score Distribution')
        axes[0, 0].set_xlabel('Quality Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(df['quality_score'].quantile(0.8), 
                          color='red', linestyle='--', label='Top 20%')
        axes[0, 0].legend()
        
        # 2. P-value Distribution (log scale)
        axes[0, 1].hist(np.log10(df['p_value']), bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('P-value Distribution (log scale)')
        axes[0, 1].set_xlabel('log10(p-value)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Correlation Distribution
        axes[0, 2].hist(df['correlation'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Correlation Distribution')
        axes[0, 2].set_xlabel('Correlation')
        axes[0, 2].set_ylabel('Frequency')
        
        # 4. Quality vs P-value
        axes[1, 0].scatter(np.log10(df['p_value']), df['quality_score'], alpha=0.6)
        axes[1, 0].set_xlabel('log10(p-value)')
        axes[1, 0].set_ylabel('Quality Score')
        axes[1, 0].set_title('Quality Score vs P-value')
        
        # 5. Quality vs Correlation
        axes[1, 1].scatter(df['correlation'], df['quality_score'], alpha=0.6)
        axes[1, 1].set_xlabel('Correlation')
        axes[1, 1].set_ylabel('Quality Score')
        axes[1, 1].set_title('Quality Score vs Correlation')
        
        # 6. Top pairs by quality
        top_20 = df.nlargest(20, 'quality_score')
        symbols = [f"{row['symbol1'][:4]}-{row['symbol2'][:4]}" 
                  for _, row in top_20.iterrows()]
        axes[1, 2].barh(range(len(symbols)), top_20['quality_score'].values)
        axes[1, 2].set_yticks(range(len(symbols)))
        axes[1, 2].set_yticklabels(symbols, fontsize=8)
        axes[1, 2].set_xlabel('Quality Score')
        axes[1, 2].set_title('Top 20 Pairs by Quality')
        axes[1, 2].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/quality_analysis.png', dpi=300, bbox_inches='tight')
        print(f"📊 Plots saved to {save_dir}/quality_analysis.png")
        
        plt.close()
        
        # Create correlation heatmap for top pairs
        self.create_correlation_heatmap(df.head(30), save_dir)
    
    def create_correlation_heatmap(self, df: pd.DataFrame, save_dir: str):
        """
        Create correlation heatmap of quality metrics
        """
        # Extract numeric features
        features = ['quality_score', 'p_value', 'correlation']
        
        # Add half-life if available
        half_lives = []
        for _, row in df.iterrows():
            hl = 0
            if 'spread_properties' in row and row['spread_properties']:
                props = row['spread_properties']
                if isinstance(props, dict):
                    hl = props.get('half_life_ou') or props.get('half_life') or 0
            half_lives.append(hl)
        
        df['half_life'] = half_lives
        if any(hl > 0 for hl in half_lives):
            features.append('half_life')
        
        # Create correlation matrix
        corr_matrix = df[features].corr()
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Quality Metrics Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_filter_function(self, criteria: Dict) -> str:
        """
        Generate Python code for filtering function based on discovered criteria
        """
        code = f'''
def filter_cointegrated_pairs(pairs: List[Dict]) -> List[Dict]:
    """
    Filter cointegrated pairs based on statistical quality criteria
    Generated automatically from statistical analysis
    """
    filtered_pairs = []
    
    for pair in pairs:
        # Check basic cointegration
        if not pair.get('is_cointegrated', False):
            continue
        
        # Statistical filters
        if pair.get('p_value', 1.0) > {criteria['statistical_filters']['max_p_value']:.4f}:
            continue
        
        correlation = abs(pair.get('correlation', 0))
        if correlation < {criteria['statistical_filters']['min_correlation']:.3f}:
            continue
        if correlation > {criteria['statistical_filters']['max_correlation']:.3f}:
            continue
        '''
        
        if 'mean_reversion_filters' in criteria:
            code += f'''
        # Mean reversion filters
        half_life = None
        if 'spread_properties' in pair:
            props = pair['spread_properties']
            if isinstance(props, dict):
                half_life = props.get('half_life_ou') or props.get('half_life')
        
        if half_life:
            if half_life < {criteria['mean_reversion_filters']['min_half_life']:.1f}:
                continue
            if half_life > {criteria['mean_reversion_filters']['max_half_life']:.1f}:
                continue
        '''
        
        if 'stability_filters' in criteria:
            code += f'''
        # Stability filters
        if 'rolling_stability' in pair:
            stability = pair['rolling_stability']
            if isinstance(stability, dict):
                stability_ratio = stability.get('stability_ratio', 0)
                if stability_ratio < {criteria['stability_filters']['min_stability_ratio']:.3f}:
                    continue
        '''
        
        code += '''
        # Passed all filters
        filtered_pairs.append(pair)
    
    return filtered_pairs
'''
        return code
    
    def recommend_parameters(self, target_pairs: int = 10) -> Dict:
        """
        Main method to recommend filtering parameters
        """
        print("\n" + "="*80)
        print("🚀 STATISTICAL PARAMETER DISCOVERY (NO BACKTESTING REQUIRED)")
        print("="*80)
        
        # 1. Analyze quality distribution
        df = self.rank_pairs_by_tradability()
        
        # 2. Find optimal thresholds
        thresholds = self.find_optimal_thresholds(target_pairs=target_pairs)
        
        # 3. Generate filtering criteria
        criteria = self.generate_filtering_criteria(
            quality_threshold=thresholds['min_quality_score']
        )
        
        # 4. Create visualizations
        self.create_analysis_plots()
        
        # 5. Generate filter function code
        filter_code = self.generate_filter_function(criteria)
        
        # Print recommendations
        print(f"\n📊 FINAL RECOMMENDATIONS")
        print("=" * 60)
        print(f"To select approximately {target_pairs} high-quality pairs:")
        print(f"\n📈 Statistical Filters:")
        print(f"   • Max p-value: {criteria['statistical_filters']['max_p_value']:.4f}")
        print(f"   • Correlation range: {criteria['statistical_filters']['min_correlation']:.3f} - {criteria['statistical_filters']['max_correlation']:.3f}")
        
        if 'mean_reversion_filters' in criteria:
            print(f"\n⏱️ Mean Reversion Filters:")
            print(f"   • Half-life range: {criteria['mean_reversion_filters']['min_half_life']:.1f} - {criteria['mean_reversion_filters']['max_half_life']:.1f}")
            print(f"   • Optimal half-life: {criteria['mean_reversion_filters']['optimal_half_life']:.1f}")
        
        if 'stability_filters' in criteria:
            print(f"\n🔄 Stability Filters:")
            print(f"   • Min stability ratio: {criteria['stability_filters']['min_stability_ratio']:.3f}")
        
        print(f"\n💡 Quality Score Threshold: {thresholds['min_quality_score']:.2f}")
        
        # Save filter function
        with open('generated_filter_function.py', 'w') as f:
            f.write(filter_code)
        print(f"\n✅ Filter function saved to 'generated_filter_function.py'")
        
        # Return top pairs for immediate use
        top_pairs = df.head(target_pairs)
        
        print(f"\n🏆 TOP {len(top_pairs)} PAIRS BY QUALITY:")
        print(f"{'Rank':<5} {'Symbol1':<10} {'Symbol2':<10} {'Quality':<8} {'P-value':<10} {'Correlation':<12}")
        print("-" * 65)
        
        for _, row in top_pairs.iterrows():
            print(f"{row['rank']:<5} {row['symbol1'][:8]:<10} {row['symbol2'][:8]:<10} "
                  f"{row['quality_score']:<8.2f} {row['p_value']:<10.4f} {row['correlation']:<12.3f}")
        
        return {
            'criteria': criteria,
            'thresholds': thresholds,
            'top_pairs': top_pairs.to_dict('records'),
            'filter_code': filter_code
        }


def main():
    """
    Main function to run statistical parameter discovery
    """
    import glob
    
    # Find latest cointegration results
    coint_files = glob.glob("cointegration_results*/cointegration_results_*.json")
    coint_files.extend(glob.glob("cointegration_results*/cointegration_results_*.pkl"))
    coint_files.extend(glob.glob("cointegration_results*/cointegrated_pairs_*.parquet"))
    
    if not coint_files:
        print("❌ No cointegration results found!")
        print("   Please run enhanced_cointegration_finder_v2.py first")
        return
    
    # Use the most recent file
    latest_file = max(coint_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"📁 Using cointegration results: {latest_file}")
    
    # Initialize analyzer
    analyzer = StatisticalFilterDiscovery(latest_file)
    
    # Get target number of pairs from user
    target = input("\nHow many pairs would you like to trade? (default: 10): ").strip()
    target_pairs = int(target) if target else 10
    
    # Run analysis and get recommendations
    recommendations = analyzer.recommend_parameters(target_pairs=target_pairs)
    
    print("\n✅ Analysis complete! You can now:")
    print("1. Use the generated filter function to select pairs")
    print("2. Review the quality analysis plots")
    print("3. Apply the recommended thresholds directly")
    print("\n🎯 No backtesting required - these parameters are based on")
    print("   proven statistical properties that indicate trading potential!")
    
    return recommendations


if __name__ == "__main__":
    recommendations = main()
