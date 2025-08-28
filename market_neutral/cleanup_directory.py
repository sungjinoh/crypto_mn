#!/usr/bin/env python3
"""
Cleanup and organize the market_neutral directory
This script archives old/demo files and organizes the structure
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def cleanup_market_neutral_directory():
    """
    Organize and clean up the market_neutral directory
    """
    
    # Define file categories
    CORE_WORKFLOW = [
        'comprehensive_workflow.py',
        'hybrid_pair_selector.py', 
        'complete_workflow.py',
        'apply_optimal_filters.py'
    ]
    
    ANALYSIS_SCRIPTS = [
        'enhanced_cointegration_finder_v2.py',
        'enhanced_threshold_discovery.py',
        'statistical_filter_discovery.py'
    ]
    
    BACKTESTING_SCRIPTS = [
        'run_fixed_parameters.py',
        'mean_reversion_backtest.py',
        'mean_reversion_strategy.py'
    ]
    
    UTILITY_SCRIPTS = [
        'binance_futures_data_collector.py',
        'data_resampling_utils.py',
        'load_and_use_results.py'
    ]
    
    DOCUMENTATION = [
        'README.md',
        'QUICK_START.md',
        'cleanup_directory.py'  # This script
    ]
    
    # Files to archive
    ARCHIVE_PATTERNS = [
        'demo_*.py',
        'test_*.py',
        'fix_*.py',
        'clean_*.py',
        'analyze_trade_details.py',
        'enhanced_cointegration_finder.py',  # Old version
        'threshold_discovery_analysis.py',   # Old version
        'run_cointegration_finder.py',
        'run_multi_year_*.py',
        'run_mean_reversion_backtest.py',
        'optimal_backtesting_workflow.py',
        'quick_start_optimized.py',
        'strategy_examples.py',
        'test_*.py'
    ]
    
    print("="*60)
    print("🧹 CLEANING UP MARKET_NEUTRAL DIRECTORY")
    print("="*60)
    
    # Create organized structure
    directories = {
        'archive': Path('archive'),
        'archive/demo': Path('archive/demo'),
        'archive/test': Path('archive/test'),
        'archive/legacy': Path('archive/legacy'),
        'results': Path('results'),
        'results/cointegration': Path('results/cointegration'),
        'results/backtests': Path('results/backtests'),
        'results/reports': Path('results/reports'),
        'results/filtered_pairs': Path('results/filtered_pairs')
    }
    
    # Create directories
    for name, path in directories.items():
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {path}")
    
    # Move files to archive
    archived_count = 0
    current_dir = Path('.')
    
    for file_path in current_dir.glob('*.py'):
        filename = file_path.name
        
        # Skip if it's a core file
        if (filename in CORE_WORKFLOW or 
            filename in ANALYSIS_SCRIPTS or 
            filename in BACKTESTING_SCRIPTS or
            filename in UTILITY_SCRIPTS or
            filename in DOCUMENTATION):
            continue
        
        # Determine archive location
        if filename.startswith('demo_'):
            dest = directories['archive/demo'] / filename
        elif filename.startswith('test_'):
            dest = directories['archive/test'] / filename
        elif filename.startswith('fix_') or filename.startswith('clean_'):
            dest = directories['archive/test'] / filename
        else:
            # Check if it's a legacy file
            if any(filename == pattern.replace('*', '') or 
                   filename.startswith(pattern.replace('*.py', '')) 
                   for pattern in ARCHIVE_PATTERNS):
                dest = directories['archive/legacy'] / filename
            else:
                continue  # Skip unknown files
        
        # Move file
        try:
            shutil.move(str(file_path), str(dest))
            print(f"📦 Archived: {filename} → {dest.parent.name}/")
            archived_count += 1
        except Exception as e:
            print(f"⚠️  Could not move {filename}: {e}")
    
    # Clean up __pycache__
    pycache_path = Path('__pycache__')
    if pycache_path.exists():
        try:
            shutil.rmtree(pycache_path)
            print("✅ Removed __pycache__")
        except Exception as e:
            print(f"⚠️  Could not remove __pycache__: {e}")
    
    # Move result files to organized folders
    result_patterns = [
        ('*cointegration_results*.json', 'results/cointegration'),
        ('*cointegration_results*.csv', 'results/cointegration'),
        ('*cointegration_results*.pkl', 'results/cointegration'),
        ('*backtest_results*.csv', 'results/backtests'),
        ('*parameter_optimization*.csv', 'results/reports'),
        ('*cross_validation*.csv', 'results/reports'),
        ('*comprehensive_report*.json', 'results/reports'),
        ('*trading_config*.json', 'results/reports'),
        ('filtered_pairs*.json', 'results/filtered_pairs'),
        ('filtered_pairs*.csv', 'results/filtered_pairs'),
    ]
    
    moved_results = 0
    for pattern, dest_dir in result_patterns:
        for file_path in current_dir.glob(pattern):
            dest = Path(dest_dir) / file_path.name
            try:
                shutil.move(str(file_path), str(dest))
                print(f"📊 Moved result: {file_path.name} → {dest_dir}/")
                moved_results += 1
            except Exception as e:
                if "already exists" not in str(e):
                    print(f"⚠️  Could not move {file_path.name}: {e}")
    
    # Create a file organization report
    print("\n" + "="*60)
    print("📁 FINAL STRUCTURE")
    print("="*60)
    
    # Count files in each category
    core_count = len([f for f in current_dir.glob('*.py') 
                     if f.name in CORE_WORKFLOW])
    analysis_count = len([f for f in current_dir.glob('*.py') 
                         if f.name in ANALYSIS_SCRIPTS])
    backtest_count = len([f for f in current_dir.glob('*.py') 
                         if f.name in BACKTESTING_SCRIPTS])
    utility_count = len([f for f in current_dir.glob('*.py') 
                        if f.name in UTILITY_SCRIPTS])
    
    print(f"📂 Core Workflow Scripts: {core_count}")
    for script in CORE_WORKFLOW:
        if Path(script).exists():
            print(f"   ✅ {script}")
    
    print(f"\n📂 Analysis Scripts: {analysis_count}")
    for script in ANALYSIS_SCRIPTS:
        if Path(script).exists():
            print(f"   ✅ {script}")
    
    print(f"\n📂 Backtesting Scripts: {backtest_count}")
    for script in BACKTESTING_SCRIPTS:
        if Path(script).exists():
            print(f"   ✅ {script}")
    
    print(f"\n📂 Utility Scripts: {utility_count}")
    for script in UTILITY_SCRIPTS:
        if Path(script).exists():
            print(f"   ✅ {script}")
    
    print(f"\n📦 Archived Files: {archived_count}")
    print(f"📊 Moved Result Files: {moved_results}")
    
    # Create timestamp file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('archive/cleanup_log.txt', 'w') as f:
        f.write(f"Directory cleaned on: {timestamp}\n")
        f.write(f"Files archived: {archived_count}\n")
        f.write(f"Results organized: {moved_results}\n")
    
    print("\n✅ CLEANUP COMPLETE!")
    print("\n💡 Next steps:")
    print("   1. Review archived files in 'archive/' folder")
    print("   2. Delete archive folder if not needed: rm -rf archive/")
    print("   3. Check README.md for workflow documentation")
    print("   4. Run 'python complete_workflow.py' to start trading")

def list_current_files():
    """List all Python files in current directory"""
    print("\n📋 Current Python files:")
    print("-"*40)
    for file in sorted(Path('.').glob('*.py')):
        size = file.stat().st_size / 1024  # Size in KB
        print(f"  {file.name:<40} {size:>8.1f} KB")

if __name__ == "__main__":
    print("This will organize and clean up the market_neutral directory.")
    print("Old/demo/test files will be moved to 'archive/' folder.")
    print("Core scripts will remain in place.")
    
    # Show current files
    list_current_files()
    
    proceed = input("\nProceed with cleanup? (y/n): ").strip().lower()
    
    if proceed == 'y':
        cleanup_market_neutral_directory()
    else:
        print("Cleanup cancelled.")
        print("\nTo see what would be organized, check README.md")
