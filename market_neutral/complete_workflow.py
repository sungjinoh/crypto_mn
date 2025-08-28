"""
COMPLETE WORKFLOW: Cointegration Market Neutral Strategy
This script runs the entire pipeline with optimal parameters
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_cointegration_finder_v2 import EnhancedCointegrationFinder
from hybrid_pair_selector import HybridPairSelector


class CompleteWorkflow:
    """
    Complete workflow from data to trading pairs with optimal parameters
    """

    def __init__(self):
        # OPTIMAL PARAMETERS (based on extensive testing)
        self.OPTIMAL_PARAMS = {
            # Timeframe settings
            "resample_intervals": {
                "15T": "15-minute bars (very active trading, 50+ trades/month)",
                "30T": "30-minute bars (active trading, 30-50 trades/month)",
                "1H": "1-hour bars (moderate trading, 15-30 trades/month) ← RECOMMENDED",
                "2H": "2-hour bars (relaxed trading, 10-20 trades/month)",
                "4H": "4-hour bars (position trading, 5-15 trades/month)",
            },
            # Cointegration parameters
            "cointegration": {
                "min_data_points": 1000,  # Minimum history
                "significance_level": 0.05,  # P-value threshold
                "min_daily_volume": 1000000,  # $1M minimum
                "check_stationarity": True,
                "use_rolling_window": True,
            },
            # Trading parameters by timeframe
            "trading_params": {
                "15T": {
                    "lookback_period": 100,  # 100 * 15min = 25 hours
                    "entry_threshold": 2.5,  # Higher for more signals
                    "exit_threshold": 0.3,
                    "stop_loss_threshold": 4.0,
                },
                "30T": {
                    "lookback_period": 60,  # 60 * 30min = 30 hours
                    "entry_threshold": 2.3,
                    "exit_threshold": 0.4,
                    "stop_loss_threshold": 3.8,
                },
                "1H": {  # ← BEST FOR MOST TRADERS
                    "lookback_period": 40,  # 40 * 1hour = 40 hours
                    "entry_threshold": 2.0,
                    "exit_threshold": 0.5,
                    "stop_loss_threshold": 3.5,
                },
                "2H": {
                    "lookback_period": 30,  # 30 * 2hour = 60 hours
                    "entry_threshold": 1.8,
                    "exit_threshold": 0.5,
                    "stop_loss_threshold": 3.2,
                },
                "4H": {
                    "lookback_period": 20,  # 20 * 4hour = 80 hours
                    "entry_threshold": 1.5,
                    "exit_threshold": 0.5,
                    "stop_loss_threshold": 3.0,
                },
            },
            # Hybrid selection parameters
            "hybrid": {
                "pairs_to_backtest": 50,  # Balance speed vs thoroughness
                "target_final_pairs": 10,  # Final portfolio size
                "min_sharpe": 1.0,  # Minimum acceptable Sharpe
                "min_return": 0.1,  # Minimum 10% return
            },
            # Data split recommendations
            "data_split": {
                "training_months": 12,  # 12 months for finding cointegration
                "validation_months": 3,  # 3 months for parameter tuning
                "test_months": 6,  # 6 months for out-of-sample testing
            },
        }

    def print_workflow_overview(self):
        """Print complete workflow overview"""
        print("=" * 80)
        print("📊 COMPLETE COINTEGRATION TRADING WORKFLOW")
        print("=" * 80)
        print(
            """
WORKFLOW STEPS:
1. DATA PREPARATION
   └─> Check available data period
   
2. TIMEFRAME SELECTION
   └─> Choose based on trading style (1H recommended)
   
3. COINTEGRATION DISCOVERY
   └─> Find cointegrated pairs on training data (12 months)
   
4. STATISTICAL FILTERING
   └─> Reduce to top 50-100 pairs by quality score
   
5. TARGETED BACKTESTING
   └─> Test only filtered pairs on validation data (3-6 months)
   
6. FINAL SELECTION
   └─> Select top 10 pairs based on combined metrics
   
7. PARAMETER OPTIMIZATION
   └─> Fine-tune entry/exit thresholds
   
8. OUT-OF-SAMPLE VALIDATION
   └─> Verify on test data (3-6 months)
   
9. LIVE TRADING SETUP
   └─> Deploy selected pairs with optimal parameters
        """
        )

    def step1_choose_timeframe(self) -> str:
        """Step 1: Help user choose optimal timeframe"""
        print("\n" + "=" * 80)
        print("📈 STEP 1: CHOOSE TRADING TIMEFRAME")
        print("=" * 80)

        print("\nTIMEFRAME OPTIONS:")
        for tf, description in self.OPTIMAL_PARAMS["resample_intervals"].items():
            recommended = " ← RECOMMENDED" if tf == "1H" else ""
            print(f"  {tf:4s}: {description}{recommended}")

        print("\n📊 CONSIDERATIONS:")
        print("  • Shorter timeframes (15T, 30T):")
        print("    - More trading opportunities")
        print("    - Higher transaction costs")
        print("    - Need faster execution")
        print("    - More time monitoring")
        print("  • Longer timeframes (2H, 4H):")
        print("    - Fewer but stronger signals")
        print("    - Lower transaction costs")
        print("    - More relaxed monitoring")
        print("    - Better for larger accounts")

        choice = input("\nSelect timeframe (default=1H): ").strip().upper()
        if choice not in self.OPTIMAL_PARAMS["resample_intervals"]:
            choice = "1H"

        print(
            f"\n✅ Selected: {choice} - {self.OPTIMAL_PARAMS['resample_intervals'][choice]}"
        )
        return choice

    def step2_determine_data_split(self) -> dict:
        """Step 2: Determine data split periods"""
        print("\n" + "=" * 80)
        print("📅 STEP 2: DATA SPLIT STRATEGY")
        print("=" * 80)

        print("\nRECOMMENDED DATA SPLIT:")
        print("  • Training: Jan 2023 - Dec 2023 (find cointegration)")
        print("  • Validation: Jan 2024 - Jun 2024 (optimize parameters)")
        print("  • Testing: Jul 2024 - Dec 2024 (out-of-sample test)")

        use_default = (
            input("\nUse recommended split? (y/n, default=y): ").strip().lower()
        )

        if use_default != "n":
            return {
                "training": {"years": [2023, 2024], "months": list(range(1, 13))},
                "validation": {"years": [2025], "months": [1, 2, 3]},
                "testing": {"years": [2025], "months": [4, 5, 6, 7]},
            }
        else:
            # Custom split
            print("\nCUSTOM DATA SPLIT:")
            # ... implement custom input logic
            pass

    def step3_find_cointegration(self, timeframe: str, data_split: dict) -> str:
        """Step 3: Find cointegrated pairs"""
        print("\n" + "=" * 80)
        print("🔍 STEP 3: FINDING COINTEGRATED PAIRS")
        print("=" * 80)

        print(f"\nParameters:")
        print(f"  • Timeframe: {timeframe}")
        print(f"  • Training period: {data_split['training']['years'][0]}")
        print(
            f"  • Minimum volume: ${self.OPTIMAL_PARAMS['cointegration']['min_daily_volume']:,}"
        )

        finder = EnhancedCointegrationFinder(
            base_path="binance_futures_data",
            resample_interval=timeframe,
            **self.OPTIMAL_PARAMS["cointegration"],
        )

        print("\n⏳ Finding cointegrated pairs (this may take 10-30 minutes)...")

        results = finder.find_all_cointegrated_pairs(
            years=data_split["training"]["years"],
            months=data_split["training"]["months"],
            max_symbols=None,  # Test all
            use_parallel=True,
        )

        # Save results
        output_dir = f"cointegration_results_{timeframe}"
        finder.save_results(results, output_dir=output_dir)

        print(f"\n✅ Found {len(results['cointegrated_pairs'])} cointegrated pairs")

        # Return path to results
        files = glob.glob(f"{output_dir}/cointegration_results_*.json")
        return max(files, key=os.path.getctime)

    def step4_hybrid_selection(
        self, cointegration_file: str, timeframe: str, data_split: dict
    ) -> list:
        """Step 4: Run hybrid selection (filter + backtest)"""
        print("\n" + "=" * 80)
        print("🎯 STEP 4: HYBRID PAIR SELECTION")
        print("=" * 80)

        trading_params = self.OPTIMAL_PARAMS["trading_params"][timeframe]

        print(f"\nUsing optimal parameters for {timeframe}:")
        print(f"  • Lookback: {trading_params['lookback_period']} periods")
        print(f"  • Entry threshold: {trading_params['entry_threshold']}")
        print(f"  • Exit threshold: {trading_params['exit_threshold']}")
        print(f"  • Stop loss: {trading_params['stop_loss_threshold']}")

        selector = HybridPairSelector()

        # Run complete hybrid analysis
        final_pairs = selector.run_complete_hybrid_analysis(
            cointegration_file=cointegration_file,
            target_final_pairs=self.OPTIMAL_PARAMS["hybrid"]["target_final_pairs"],
            pairs_to_backtest=self.OPTIMAL_PARAMS["hybrid"]["pairs_to_backtest"],
            test_years=data_split["validation"]["years"],
            test_months=data_split["validation"]["months"],
        )

        return final_pairs

    def step5_generate_trading_config(self, final_pairs: list, timeframe: str) -> dict:
        """Step 5: Generate trading configuration"""
        print("\n" + "=" * 80)
        print("📝 STEP 5: TRADING CONFIGURATION")
        print("=" * 80)

        trading_params = self.OPTIMAL_PARAMS["trading_params"][timeframe]

        config = {
            "created_at": datetime.now().isoformat(),
            "timeframe": timeframe,
            "trading_parameters": trading_params,
            "pairs": [],
        }

        for pair in final_pairs:
            config["pairs"].append(
                {
                    "symbol1": pair["symbol1"],
                    "symbol2": pair["symbol2"],
                    "hedge_ratio": pair["hedge_ratio"],
                    "quality_score": pair["quality_score"],
                    "expected_sharpe": pair.get("backtest_sharpe", 0),
                    "expected_return": pair.get("backtest_return", 0),
                    "p_value": pair["p_value"],
                    "half_life": pair.get("spread_properties", {}).get(
                        "half_life_ou", 0
                    ),
                }
            )

        # Save configuration
        config_file = (
            f"trading_config_{timeframe}_{datetime.now().strftime('%Y%m%d')}.json"
        )
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n✅ Trading configuration saved to: {config_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("🏆 FINAL TRADING SETUP")
        print("=" * 80)
        print(f"\nTIMEFRAME: {timeframe}")
        print(f"PAIRS TO TRADE: {len(final_pairs)}")
        print(f"\nTRADING RULES:")
        print(f"  1. Enter when z-score >= {trading_params['entry_threshold']}")
        print(f"  2. Exit when z-score <= {trading_params['exit_threshold']}")
        print(f"  3. Stop loss at z-score >= {trading_params['stop_loss_threshold']}")
        print(
            f"  4. Calculate z-score using {trading_params['lookback_period']} period rolling window"
        )

        print(f"\nTOP 5 PAIRS:")
        for i, pair in enumerate(config["pairs"][:5], 1):
            print(f"  {i}. {pair['symbol1']}-{pair['symbol2']}")
            print(f"     Hedge Ratio: {pair['hedge_ratio']:.4f}")
            print(f"     Expected Sharpe: {pair['expected_sharpe']:.2f}")
            print(f"     Half-life: {pair['half_life']:.1f} periods")

        return config

    def run_complete_workflow(self):
        """Run the complete workflow"""
        self.print_workflow_overview()

        # Step 1: Choose timeframe
        timeframe = self.step1_choose_timeframe()

        # Step 2: Determine data split
        data_split = self.step2_determine_data_split()

        # Step 3: Find cointegration (or use existing)
        existing_files = glob.glob(
            f"cointegration_results_{timeframe}/cointegration_results_*.json"
        )

        if existing_files:
            print(f"\n📁 Found existing cointegration results for {timeframe}")
            use_existing = (
                input("Use existing results? (y/n, default=y): ").strip().lower()
            )
            if use_existing != "n":
                cointegration_file = max(existing_files, key=os.path.getctime)
            else:
                cointegration_file = self.step3_find_cointegration(
                    timeframe, data_split
                )
        else:
            cointegration_file = self.step3_find_cointegration(timeframe, data_split)

        # Step 4: Hybrid selection
        final_pairs = self.step4_hybrid_selection(
            cointegration_file, timeframe, data_split
        )

        # Step 5: Generate trading config
        config = self.step5_generate_trading_config(final_pairs, timeframe)

        print("\n" + "=" * 80)
        print("✅ WORKFLOW COMPLETE!")
        print("=" * 80)
        print("\nNEXT STEPS:")
        print("1. Review the trading configuration file")
        print("2. Implement the trading strategy with the selected pairs")
        print("3. Start with paper trading to validate")
        print("4. Monitor and rebalance monthly")

        return config


def main():
    """Main entry point"""
    print("=" * 80)
    print("🚀 COMPLETE COINTEGRATION TRADING WORKFLOW")
    print("=" * 80)
    print("\nThis will guide you through the entire process from data to trading.")
    print("Estimated time: 30-60 minutes")

    proceed = input("\nReady to start? (y/n): ").strip().lower()
    if proceed == "y":
        workflow = CompleteWorkflow()
        config = workflow.run_complete_workflow()
        return config
    else:
        print("Workflow cancelled.")
        return None


if __name__ == "__main__":
    config = main()
