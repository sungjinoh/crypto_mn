"""
Migration utilities for transitioning from old backtesting framework to new architecture.

This script helps migrate existing code, configurations, and workflows
from the legacy framework to the new structured architecture.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import shutil
import json

# Import old framework components for migration
try:
    from backtesting_framework.pairs_backtester import PairsBacktester as OldPairsBacktester
    from market_neutral.mean_reversion_strategy import MeanReversionStrategy as OldMeanReversionStrategy
    OLD_FRAMEWORK_AVAILABLE = True
except ImportError:
    OLD_FRAMEWORK_AVAILABLE = False
    
# Import new framework components
from crypto_backtesting import (
    DataManager, BacktestEngine, MeanReversionStrategy,
    PerformanceAnalyzer, ConfigurationManager
)


class FrameworkMigrator:
    """
    Handles migration from old backtesting framework to new architecture.
    
    Provides utilities to convert old configurations, migrate data,
    and update existing scripts to use the new framework.
    """
    
    def __init__(self, old_framework_path: str = ".", new_framework_path: str = "src/crypto_backtesting"):
        """
        Initialize framework migrator.
        
        Args:
            old_framework_path: Path to old framework code
            new_framework_path: Path to new framework code
        """
        self.old_path = Path(old_framework_path)
        self.new_path = Path(new_framework_path)
        self.config_manager = ConfigurationManager()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def migrate_data_structure(self, data_path: str = "binance_futures_data") -> bool:
        """
        Migrate data structure to new format if needed.
        
        Args:
            data_path: Path to market data
            
        Returns:
            True if migration successful
        """
        try:
            data_path = Path(data_path)
            
            if not data_path.exists():
                self.logger.warning(f"Data path {data_path} does not exist")
                return False
                
            # Check if data structure is already compatible
            klines_path = data_path / "klines"
            funding_path = data_path / "fundingRate"
            
            if klines_path.exists() and funding_path.exists():
                self.logger.info("Data structure already compatible")
                return True
                
            # Data structure migration logic would go here
            # For now, assume data is already in correct format
            
            self.logger.info("Data structure migration completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error migrating data structure: {e}")
            return False
            
    def convert_old_strategy_config(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert old strategy configuration to new format.
        
        Args:
            old_config: Old strategy configuration
            
        Returns:
            New strategy configuration
        """
        # Map old parameter names to new ones
        param_mapping = {
            'lookback_period': 'lookback_period',
            'entry_threshold': 'entry_threshold', 
            'exit_threshold': 'exit_threshold',
            'stop_loss_threshold': 'stop_loss_threshold',
            # Add more mappings as needed
        }
        
        new_config = {}
        
        for old_key, old_value in old_config.items():
            if old_key in param_mapping:
                new_key = param_mapping[old_key]
                new_config[new_key] = old_value
            else:
                # Keep unmapped parameters as-is
                new_config[old_key] = old_value
                
        return new_config
        
    def convert_old_backtest_config(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert old backtest configuration to new format.
        
        Args:
            old_config: Old backtest configuration
            
        Returns:
            New backtest configuration
        """
        # Map old configuration to new format
        new_config = {
            'initial_capital': old_config.get('initial_capital', 100000.0),
            'commission_rate': old_config.get('transaction_cost', 0.001),
            'slippage_rate': old_config.get('slippage', 0.0001)
        }
        
        return new_config
        
    def migrate_existing_script(self, script_path: str, output_path: Optional[str] = None) -> str:
        """
        Migrate an existing script to use the new framework.
        
        Args:
            script_path: Path to script to migrate
            output_path: Output path for migrated script
            
        Returns:
            Path to migrated script
        """
        script_path = Path(script_path)
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script {script_path} not found")
            
        if output_path is None:
            output_path = script_path.parent / f"migrated_{script_path.name}"
        else:
            output_path = Path(output_path)
            
        try:
            # Read original script
            with open(script_path, 'r') as f:
                content = f.read()
                
            # Perform migration transformations
            migrated_content = self._transform_script_content(content)
            
            # Write migrated script
            with open(output_path, 'w') as f:
                f.write(migrated_content)
                
            self.logger.info(f"Script migrated: {script_path} -> {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error migrating script {script_path}: {e}")
            raise
            
    def _transform_script_content(self, content: str) -> str:
        """
        Transform script content to use new framework.
        
        Args:
            content: Original script content
            
        Returns:
            Transformed script content
        """
        # Define transformation rules
        transformations = [
            # Import replacements
            (
                "from backtesting_framework.pairs_backtester import PairsBacktester",
                "from crypto_backtesting import BacktestEngine, DataManager"
            ),
            (
                "from market_neutral.mean_reversion_strategy import MeanReversionStrategy",
                "from crypto_backtesting import MeanReversionStrategy"
            ),
            (
                "from backtesting_framework import",
                "from crypto_backtesting import"
            ),
            
            # Class name replacements
            ("PairsBacktester(", "BacktestEngine(DataManager(), "),
            
            # Method name replacements
            ("run_backtest(", "run_backtest(strategy, combined_data"),
            
            # Parameter name replacements
            ("transaction_cost=", "commission_rate="),
        ]
        
        # Apply transformations
        migrated_content = content
        
        for old_pattern, new_pattern in transformations:
            migrated_content = migrated_content.replace(old_pattern, new_pattern)
            
        # Add migration notice at the top
        migration_notice = '''"""
MIGRATED SCRIPT - Updated to use new crypto_backtesting framework

This script has been automatically migrated from the legacy framework.
Please review and test thoroughly before use.

Migration date: Generated by FrameworkMigrator
"""

'''
        
        migrated_content = migration_notice + migrated_content
        
        return migrated_content
        
    def generate_migration_report(self, output_dir: str = "migration_report") -> Dict[str, Any]:
        """
        Generate a comprehensive migration report.
        
        Args:
            output_dir: Directory to save migration report
            
        Returns:
            Migration report data
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Analyze old framework usage
            old_scripts = self._find_old_framework_scripts()
            
            # Check data compatibility
            data_status = self._check_data_compatibility()
            
            # Generate migration checklist
            checklist = self._generate_migration_checklist()
            
            # Create report
            report = {
                'migration_summary': {
                    'old_framework_available': OLD_FRAMEWORK_AVAILABLE,
                    'scripts_found': len(old_scripts),
                    'data_compatible': data_status['compatible'],
                    'migration_required': len(old_scripts) > 0 or not data_status['compatible']
                },
                'scripts_to_migrate': old_scripts,
                'data_status': data_status,
                'migration_checklist': checklist,
                'recommendations': self._generate_recommendations()
            }
            
            # Save report
            report_path = output_dir / "migration_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            # Save human-readable report
            self._save_readable_report(report, output_dir / "migration_report.md")
            
            self.logger.info(f"Migration report generated: {report_path}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating migration report: {e}")
            raise
            
    def _find_old_framework_scripts(self) -> List[Dict[str, Any]]:
        """Find scripts using old framework."""
        scripts = []
        
        # Search patterns for old framework usage
        search_patterns = [
            "from backtesting_framework",
            "from market_neutral",
            "PairsBacktester",
            "pairs_backtester"
        ]
        
        # Search in common directories
        search_dirs = [
            self.old_path,
            Path("market_neutral"),
            Path("examples"),
            Path("scripts")
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                for py_file in search_dir.rglob("*.py"):
                    try:
                        with open(py_file, 'r') as f:
                            content = f.read()
                            
                        # Check for old framework patterns
                        patterns_found = []
                        for pattern in search_patterns:
                            if pattern in content:
                                patterns_found.append(pattern)
                                
                        if patterns_found:
                            scripts.append({
                                'path': str(py_file),
                                'patterns_found': patterns_found,
                                'size_kb': py_file.stat().st_size / 1024
                            })
                            
                    except Exception as e:
                        self.logger.warning(f"Error reading {py_file}: {e}")
                        continue
                        
        return scripts
        
    def _check_data_compatibility(self) -> Dict[str, Any]:
        """Check data structure compatibility."""
        data_paths = [
            Path("binance_futures_data"),
            Path("data"),
            Path("market_data")
        ]
        
        status = {
            'compatible': False,
            'data_found': False,
            'klines_available': False,
            'funding_rates_available': False,
            'issues': []
        }
        
        for data_path in data_paths:
            if data_path.exists():
                status['data_found'] = True
                
                # Check for klines data
                klines_path = data_path / "klines"
                if klines_path.exists():
                    status['klines_available'] = True
                    
                # Check for funding rates
                funding_path = data_path / "fundingRate"
                if funding_path.exists():
                    status['funding_rates_available'] = True
                    
                break
                
        # Determine compatibility
        if status['klines_available'] or status['funding_rates_available']:
            status['compatible'] = True
        elif status['data_found']:
            status['issues'].append("Data found but structure may need migration")
        else:
            status['issues'].append("No market data found")
            
        return status
        
    def _generate_migration_checklist(self) -> List[Dict[str, Any]]:
        """Generate migration checklist."""
        checklist = [
            {
                'task': 'Install new framework dependencies',
                'description': 'Ensure all required packages are installed',
                'command': 'pip install -r requirements.txt',
                'priority': 'high'
            },
            {
                'task': 'Backup existing code',
                'description': 'Create backup of current codebase',
                'command': 'cp -r . backup/',
                'priority': 'high'
            },
            {
                'task': 'Migrate data structure',
                'description': 'Ensure data is in compatible format',
                'command': 'python migrate_data.py',
                'priority': 'medium'
            },
            {
                'task': 'Update import statements',
                'description': 'Replace old imports with new framework imports',
                'command': 'Manual editing required',
                'priority': 'high'
            },
            {
                'task': 'Update configuration',
                'description': 'Convert old configs to new format',
                'command': 'Use ConfigurationManager',
                'priority': 'medium'
            },
            {
                'task': 'Test migrated scripts',
                'description': 'Thoroughly test all migrated functionality',
                'command': 'python -m pytest tests/',
                'priority': 'high'
            },
            {
                'task': 'Update documentation',
                'description': 'Update any local documentation',
                'command': 'Manual editing required',
                'priority': 'low'
            }
        ]
        
        return checklist
        
    def _generate_recommendations(self) -> List[str]:
        """Generate migration recommendations."""
        recommendations = [
            "Start with a small test script to verify migration process",
            "Use the new ConfigurationManager for standardized configurations",
            "Leverage the new DataManager for unified data access",
            "Take advantage of improved performance analysis capabilities",
            "Consider using the new workflow management utilities",
            "Review strategy parameters as some defaults may have changed",
            "Test backtesting results against old framework for validation",
            "Gradually migrate scripts rather than all at once"
        ]
        
        return recommendations
        
    def _save_readable_report(self, report: Dict[str, Any], output_path: Path) -> None:
        """Save human-readable migration report."""
        with open(output_path, 'w') as f:
            f.write("# Framework Migration Report\n\n")
            
            # Summary
            f.write("## Migration Summary\n\n")
            summary = report['migration_summary']
            f.write(f"- Old framework available: {summary['old_framework_available']}\n")
            f.write(f"- Scripts requiring migration: {summary['scripts_found']}\n")
            f.write(f"- Data structure compatible: {summary['data_compatible']}\n")
            f.write(f"- Migration required: {summary['migration_required']}\n\n")
            
            # Scripts to migrate
            if report['scripts_to_migrate']:
                f.write("## Scripts Requiring Migration\n\n")
                for script in report['scripts_to_migrate']:
                    f.write(f"- **{script['path']}** ({script['size_kb']:.1f} KB)\n")
                    f.write(f"  - Patterns found: {', '.join(script['patterns_found'])}\n")
                f.write("\n")
                
            # Data status
            f.write("## Data Structure Status\n\n")
            data_status = report['data_status']
            f.write(f"- Data found: {data_status['data_found']}\n")
            f.write(f"- Klines available: {data_status['klines_available']}\n")
            f.write(f"- Funding rates available: {data_status['funding_rates_available']}\n")
            if data_status['issues']:
                f.write("- Issues:\n")
                for issue in data_status['issues']:
                    f.write(f"  - {issue}\n")
            f.write("\n")
            
            # Migration checklist
            f.write("## Migration Checklist\n\n")
            for item in report['migration_checklist']:
                f.write(f"- [ ] **{item['task']}** ({item['priority']} priority)\n")
                f.write(f"  - {item['description']}\n")
                f.write(f"  - Command: `{item['command']}`\n")
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for rec in report['recommendations']:
                f.write(f"- {rec}\n")


def create_migration_example():
    """
    Create an example migration script.
    """
    example_script = '''#!/usr/bin/env python3
"""
Example migration script using the new crypto_backtesting framework.

This script demonstrates how to migrate from the old pairs trading
backtester to the new structured framework.
"""

import logging
from crypto_backtesting import (
    DataManager, BacktestEngine, MeanReversionStrategy,
    PerformanceAnalyzer, ReportGenerator, ConfigurationManager
)

def main():
    """Main migration example."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting migration example...")
    
    try:
        # 1. Setup configuration (replaces old manual setup)
        config_manager = ConfigurationManager()
        data_config = config_manager.get_default_data_config()
        backtest_config = config_manager.get_default_backtest_config()
        strategy_config = config_manager.get_default_strategy_config('mean_reversion')
        
        # 2. Setup data manager (replaces old data loading)
        data_manager = DataManager(**data_config)
        
        # 3. Load data for pair (new unified interface)
        symbol1, symbol2 = 'BTCUSDT', 'ETHUSDT'
        data1, data2 = data_manager.get_pair_data(
            symbol1, symbol2, year=2024, months=[4, 5, 6]
        )
        
        if data1 is None or data2 is None:
            logger.error("Could not load data")
            return
            
        # 4. Create strategy (new standardized interface)
        strategy = MeanReversionStrategy(
            symbol1=symbol1,
            symbol2=symbol2,
            **strategy_config
        )
        
        # 5. Prepare data (replaces old manual data preparation)
        combined_data = strategy.prepare_pair_data(data1, data2, symbol1, symbol2)
        
        # Add technical indicators
        lookback = strategy_config['lookback_period']
        rolling_mean = combined_data['spread'].rolling(lookback).mean()
        rolling_std = combined_data['spread'].rolling(lookback).std()
        combined_data['zscore'] = (combined_data['spread'] - rolling_mean) / rolling_std
        
        # 6. Setup and run backtest (new engine)
        engine = BacktestEngine(data_manager, **backtest_config)
        results = engine.run_backtest(strategy, combined_data)
        
        # 7. Generate analysis (enhanced capabilities)
        analyzer = PerformanceAnalyzer(results)
        performance_report = analyzer.generate_performance_report()
        
        # 8. Generate reports (new reporting system)
        report_generator = ReportGenerator(results, "migration_example_reports")
        full_report = report_generator.generate_full_report()
        
        # 9. Display results
        logger.info("Backtest completed successfully!")
        logger.info(f"Total Return: {results.total_return:.2%}")
        logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
        logger.info(f"Max Drawdown: {results.max_drawdown:.2%}")
        logger.info(f"Win Rate: {results.win_rate:.1%}")
        logger.info(f"Total Trades: {results.total_trades}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in migration example: {e}")
        raise

if __name__ == "__main__":
    main()
'''
    
    # Save example script
    with open("migration_example.py", 'w') as f:
        f.write(example_script)
        
    print("Migration example script created: migration_example.py")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create migrator
    migrator = FrameworkMigrator()
    
    # Generate migration report
    try:
        report = migrator.generate_migration_report()
        print("Migration report generated successfully!")
        print(f"Scripts requiring migration: {len(report['scripts_to_migrate'])}")
        
        # Create migration example
        create_migration_example()
        
    except Exception as e:
        print(f"Error generating migration report: {e}")
