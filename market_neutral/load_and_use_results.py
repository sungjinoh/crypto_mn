"""
Example script showing how to load and use the saved cointegration results
"""

import json
import pandas as pd
from pathlib import Path
import pickle


def load_cointegration_results(filepath):
    """
    Load cointegration results from a saved file.
    
    Args:
        filepath: Path to the results file (json, csv, pickle, or parquet)
    
    Returns:
        Dictionary or DataFrame with results
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif filepath.suffix == '.csv':
        return pd.read_csv(filepath)
    elif filepath.suffix == '.parquet':
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def filter_pairs_by_criteria(results, 
                           max_p_value=0.01, 
                           min_correlation=0.7,
                           max_half_life=50):
    """
    Filter cointegrated pairs based on specific criteria.
    
    Args:
        results: Results dictionary or DataFrame
        max_p_value: Maximum p-value for cointegration test
        min_correlation: Minimum correlation between pairs
        max_half_life: Maximum half-life for mean reversion
    
    Returns:
        Filtered list of pairs
    """
    if isinstance(results, dict):
        pairs = results.get('cointegrated_pairs', [])
    elif isinstance(results, pd.DataFrame):
        pairs = results.to_dict('records')
    else:
        pairs = results
    
    filtered_pairs = []
    
    for pair in pairs:
        # Check p-value
        if pair['p_value'] > max_p_value:
            continue
        
        # Check correlation
        if pair['correlation'] < min_correlation:
            continue
        
        # Check half-life if available
        if 'spread_properties' in pair and pair['spread_properties']:
            half_life = pair['spread_properties'].get('half_life')
            if half_life and half_life > max_half_life:
                continue
        
        filtered_pairs.append(pair)
    
    return filtered_pairs


def get_top_pairs_for_trading(results, n=10):
    """
    Get the top N pairs for trading based on a composite score.
    
    Args:
        results: Results dictionary or DataFrame
        n: Number of top pairs to return
    
    Returns:
        List of top pairs with scores
    """
    if isinstance(results, dict):
        pairs = results.get('cointegrated_pairs', [])
    else:
        pairs = results.to_dict('records') if isinstance(results, pd.DataFrame) else results
    
    # Calculate composite score for each pair
    scored_pairs = []
    
    for pair in pairs:
        # Lower p-value is better (more significant)
        p_value_score = 1 - pair['p_value']
        
        # Higher correlation is better
        correlation_score = abs(pair['correlation'])
        
        # Lower half-life is better (faster mean reversion)
        half_life_score = 0.5  # Default if not available
        if 'spread_properties' in pair and pair['spread_properties']:
            half_life = pair['spread_properties'].get('half_life')
            if half_life and half_life > 0:
                # Normalize half-life score (assuming 1-100 range)
                half_life_score = max(0, 1 - (half_life / 100))
        
        # Composite score (weighted average)
        composite_score = (
            0.4 * p_value_score +
            0.3 * correlation_score +
            0.3 * half_life_score
        )
        
        pair_with_score = pair.copy()
        pair_with_score['composite_score'] = composite_score
        scored_pairs.append(pair_with_score)
    
    # Sort by composite score
    scored_pairs.sort(key=lambda x: x['composite_score'], reverse=True)
    
    return scored_pairs[:n]


def format_pair_for_strategy(pair_info):
    """
    Format pair information for use in trading strategy.
    
    Args:
        pair_info: Dictionary with pair information
    
    Returns:
        Dictionary formatted for strategy use
    """
    strategy_config = {
        'symbol1': pair_info['symbol1'],
        'symbol2': pair_info['symbol2'],
        'hedge_ratio': pair_info['hedge_ratio'],
        'p_value': pair_info['p_value'],
        'correlation': pair_info['correlation'],
        'data_points': pair_info['data_points'],
        'date_range': {
            'start': pair_info['start_date'],
            'end': pair_info['end_date']
        }
    }
    
    # Add spread properties if available
    if 'spread_properties' in pair_info and pair_info['spread_properties']:
        strategy_config['spread_params'] = {
            'mean': pair_info['spread_properties'].get('mean', 0),
            'std': pair_info['spread_properties'].get('std', 1),
            'half_life': pair_info['spread_properties'].get('half_life'),
            'hurst_exponent': pair_info['spread_properties'].get('hurst_exponent')
        }
    
    # Add trading parameters (can be customized)
    strategy_config['trading_params'] = {
        'entry_threshold': 2.0,  # Enter when spread is 2 std devs away
        'exit_threshold': 0.5,   # Exit when spread returns to 0.5 std devs
        'stop_loss': 3.0,        # Stop loss at 3 std devs
        'lookback_period': 100,  # Lookback for calculating spread statistics
        'position_size': 0.1     # Position size as fraction of capital
    }
    
    return strategy_config


def export_for_backtesting(results, output_file='strategy_configs.json', top_n=20):
    """
    Export top pairs in a format ready for backtesting.
    
    Args:
        results: Results dictionary or DataFrame
        output_file: Output filename
        top_n: Number of top pairs to export
    """
    # Get top pairs
    top_pairs = get_top_pairs_for_trading(results, n=top_n)
    
    # Format for strategy
    strategy_configs = []
    for pair in top_pairs:
        config = format_pair_for_strategy(pair)
        strategy_configs.append(config)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(strategy_configs, f, indent=2)
    
    print(f"Exported {len(strategy_configs)} pair configurations to {output_file}")
    
    return strategy_configs


def main():
    """Example usage of the results."""
    
    # Find the most recent results file
    results_dir = Path("cointegration_results")
    
    if not results_dir.exists():
        print("No results directory found. Please run the cointegration finder first.")
        return
    
    # Get the most recent JSON file
    json_files = list(results_dir.glob("cointegration_results_*.json"))
    if not json_files:
        print("No results files found.")
        return
    
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    # Load results
    results = load_cointegration_results(latest_file)
    
    # Print summary
    print("\n" + "=" * 80)
    print("COINTEGRATION RESULTS SUMMARY")
    print("=" * 80)
    
    if isinstance(results, dict) and 'metadata' in results:
        meta = results['metadata']
        print(f"Analysis Date: {meta['analysis_date']}")
        print(f"Data Period: {meta['data_year']}, Months: {meta['data_months']}")
        print(f"Total Pairs Tested: {meta['total_pairs_tested']}")
        print(f"Cointegrated Pairs Found: {meta['cointegrated_pairs_found']}")
    
    # Filter pairs with strict criteria
    print("\n" + "-" * 40)
    print("FILTERING PAIRS")
    print("-" * 40)
    
    filtered_pairs = filter_pairs_by_criteria(
        results,
        max_p_value=0.01,      # Very strong cointegration
        min_correlation=0.8,    # High correlation
        max_half_life=30        # Fast mean reversion
    )
    
    print(f"Pairs meeting strict criteria: {len(filtered_pairs)}")
    
    if filtered_pairs:
        print("\nTop filtered pairs:")
        for i, pair in enumerate(filtered_pairs[:5], 1):
            print(f"{i}. {pair['symbol1']} - {pair['symbol2']} | "
                  f"p-value: {pair['p_value']:.6f} | "
                  f"correlation: {pair['correlation']:.4f}")
    
    # Get top pairs for trading
    print("\n" + "-" * 40)
    print("TOP PAIRS FOR TRADING")
    print("-" * 40)
    
    top_pairs = get_top_pairs_for_trading(results, n=10)
    
    for i, pair in enumerate(top_pairs, 1):
        print(f"{i:2d}. {pair['symbol1']:10s} - {pair['symbol2']:10s} | "
              f"Score: {pair['composite_score']:.4f} | "
              f"p-value: {pair['p_value']:.6f} | "
              f"hedge_ratio: {pair['hedge_ratio']:.4f}")
    
    # Export for backtesting
    print("\n" + "-" * 40)
    print("EXPORTING FOR BACKTESTING")
    print("-" * 40)
    
    strategy_configs = export_for_backtesting(results, 'strategy_configs.json', top_n=20)
    
    # Show example configuration
    if strategy_configs:
        print("\nExample strategy configuration:")
        print(json.dumps(strategy_configs[0], indent=2))
    
    print("\n✅ Results loaded and processed successfully!")


if __name__ == "__main__":
    main()
