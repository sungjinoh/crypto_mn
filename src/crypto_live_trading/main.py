"""
Live Mean Reversion Trading System
Entry point for live trading loop and orchestration.
"""

import os
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from strategies.mean_reversion import LiveMeanReversionStrategy
from execution.trade_executor import TradeExecutor
from state.position_tracker import PositionTracker
from state.asset_position_tracker import AssetPositionTracker, PendingOrder
from utils.live_data_manager import LiveDataManager
from utils.telegram_util import TelegramUtil

# Load monthly cointegration pairs (update this path as needed)
COINTEGRATION_PAIRS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "crypto_live_trading",
    "state",
)

# Strategy parameters (can be loaded from config)
FIXED_PARAMS = {
    "lookback_period": 60,
    "entry_threshold": 2.5,
    "exit_threshold": 0.75,
    "stop_loss_threshold": 2.6,
}

# Trading configuration
TRADING_CONFIG = {
    "leverage": 5,  # 5x leverage for futures trading
    "max_position_size_usdt": 100,  # Maximum $200 USDT per leg
    "portfolio_value": 1000,  # Portfolio value for risk calculations
}


def monitor_existing_positions(tracker, executor, telegram):
    """
    Monitor existing positions for margin levels and health.
    This can be called frequently without running the full trading cycle.
    """
    print(f"\n🔍 MONITORING: Checking existing positions...")
    open_positions = tracker.get_all_open_positions()

    if not open_positions:
        print("   ℹ️  No open positions to monitor")
        return

    for pair_key, position in open_positions.items():
        if position.status == "open":
            symbol1, symbol2 = position.symbol1, position.symbol2
            print(f"\n[MONITOR] {symbol1}-{symbol2}")
            print(
                f"   ⏸️  Holding {position.side} position (opened: {position.entry_time[:19]})"
            )

            # Monitor margin levels for existing position
            try:
                margin_status = executor.monitor_position_margin(
                    symbol1, symbol2, telegram
                )

                if margin_status.get("overall_status") == "attention_needed":
                    print(f"   ⚠️  Margin attention needed for {symbol1}-{symbol2}")
                    for symbol_key in ["symbol1", "symbol2"]:
                        if symbol_key in margin_status:
                            status = margin_status[symbol_key]
                            if status.get("action") != "none":
                                print(
                                    f"      {status.get('message', 'Margin action taken')}"
                                )
                else:
                    print(f"   💰 Margin levels healthy for {symbol1}-{symbol2}")

            except Exception as e:
                print(f"   ⚠️  Error monitoring margin: {e}")
                try:
                    telegram.notify_error(
                        f"Margin monitoring failed: {e}", symbol1, symbol2
                    )
                except:
                    pass

    print(f"\n✅ Monitoring complete - {len(open_positions)} positions checked")


def run_full_trading_cycle(
    data_manager, strategy, executor, tracker, asset_tracker, telegram, pairs
):
    """
    Run the full trading cycle: signal generation, position netting, execution, and monitoring.
    This should be called less frequently (e.g., every 4 hours).
    """
    # Update asset tracker with current positions
    open_positions = tracker.get_all_open_positions()
    asset_tracker.update_asset_positions(open_positions)

    # Step 1: Collect all signals from all pairs
    print(f"\n🔍 STEP 1: Collecting signals from {len(pairs)} pairs...")
    pair_signals = []
    signals_generated = 0

    for i, pair in enumerate(pairs, 1):
        symbol1, symbol2 = pair["symbol1"], pair["symbol2"]
        print(f"\n[{i:2d}/{len(pairs)}] Processing {symbol1}-{symbol2}")

        try:
            # Load latest data
            df1, df2 = data_manager.get_latest_pair_data(symbol1, symbol2)

            if df1 is None or df2 is None:
                print(f"   ❌ Failed to fetch data")
                continue

            print(f"   📊 Data: {len(df1)} bars, latest: {df1.index[-1]}")

            # Generate signal
            signal = strategy.generate_signal(df1, df2, pair_info=pair)
            signals_generated += 1

            print(f"   🎯 Signal: {signal.reason}")

            # Check position state
            position = tracker.get_position(symbol1, symbol2)

            # Store signal info for conflict analysis
            if signal.is_entry or signal.is_exit:
                pair_signals.append(
                    {
                        "pair": pair,
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                        "signal": signal,
                        "position": position,
                        "pair_key": f"{symbol1}-{symbol2}",
                    }
                )

        except Exception as e:
            error_msg = f"Error processing {symbol1}-{symbol2}: {e}"
            print(f"   ❌ {error_msg}")
            try:
                telegram.notify_error(str(e), symbol1, symbol2)
            except:
                pass

    # Step 2: Create pending orders and check for conflicts
    print(
        f"\n🔍 STEP 2: Analyzing {len(pair_signals)} actionable signals for conflicts..."
    )
    pending_orders = []

    for signal_info in pair_signals:
        signal = signal_info["signal"]
        position = signal_info["position"]
        symbol1, symbol2 = signal_info["symbol1"], signal_info["symbol2"]

        # Create pending orders for this signal
        if signal.is_entry and not position:
            # Entry orders - create orders for both assets
            pending_orders.append(
                PendingOrder(
                    pair_key=signal_info["pair_key"],
                    asset=symbol1,
                    quantity=signal.symbol1_qty,
                    signal_strength=abs(signal.z_score),
                    signal_type="entry",
                )
            )
            pending_orders.append(
                PendingOrder(
                    pair_key=signal_info["pair_key"],
                    asset=symbol2,
                    quantity=signal.symbol2_qty,
                    signal_strength=abs(signal.z_score),
                    signal_type="entry",
                )
            )
        elif signal.is_exit and position:
            # Exit orders - reverse the position
            pending_orders.append(
                PendingOrder(
                    pair_key=signal_info["pair_key"],
                    asset=symbol1,
                    quantity=-signal.symbol1_qty,  # Reverse for exit
                    signal_strength=abs(signal.z_score),
                    signal_type="exit",
                )
            )
            pending_orders.append(
                PendingOrder(
                    pair_key=signal_info["pair_key"],
                    asset=symbol2,
                    quantity=-signal.symbol2_qty,  # Reverse for exit
                    signal_strength=abs(signal.z_score),
                    signal_type="exit",
                )
            )

    # Analyze for position netting opportunities
    netting_analysis = asset_tracker.check_pending_orders_for_conflicts(pending_orders)

    if netting_analysis["has_netting_opportunities"]:
        print(f"\n💡 POSITION NETTING OPPORTUNITIES DETECTED!")
        for summary in netting_analysis["netting_summary"]:
            print(f"   🔄 {summary}")
    else:
        print(f"   ℹ️  No significant netting opportunities")

    # Generate optimized execution plan using position netting
    execution_plan = asset_tracker.generate_optimized_execution_plan(pending_orders)

    print(f"\n🎯 OPTIMIZED EXECUTION PLAN:")
    print(f"   📊 Original exposure: ${execution_plan['original_exposure']:.2f}")
    print(f"   📈 Optimized exposure: ${execution_plan['optimized_exposure']:.2f}")
    print(
        f"   💰 Capital savings: ${execution_plan['capital_savings']:.2f} ({execution_plan['savings_ratio']:.1%})"
    )

    # Show execution plan details
    for pair_key, plan in execution_plan["execution_plan"].items():
        if plan["status"] == "execute_netted":
            print(f"   ✅ {pair_key}: Execute with netting")
            for order in plan["orders"]:
                direction = "LONG" if order["net_quantity"] > 0 else "SHORT"
                netting_info = (
                    f"({order['netting_ratio']:.1%} netted)"
                    if order["netting_ratio"] > 0.1
                    else ""
                )
                print(
                    f"      • {order['asset']}: {direction} {abs(order['net_quantity']):.4f} {netting_info}"
                )
        else:
            print(f"   🚫 {pair_key}: {plan['reason']}")

    # Step 3: Execute optimized orders using position netting
    print(f"\n🔍 STEP 3: Executing optimized orders...")
    positions_opened = 0
    positions_closed = 0

    # Execute based on the optimized plan
    for pair_key, plan in execution_plan["execution_plan"].items():
        if plan["status"] != "execute_netted":
            print(f"\n[SKIP] {pair_key}: {plan['reason']}")
            continue

        print(f"\n[EXEC] {pair_key} - Netted Position Execution")

        # Find the corresponding signal info
        signal_info = next(
            (si for si in pair_signals if si["pair_key"] == pair_key), None
        )
        if not signal_info:
            print(f"   ❌ Signal info not found for {pair_key}")
            continue

        symbol1, symbol2 = signal_info["symbol1"], signal_info["symbol2"]
        signal = signal_info["signal"]
        position = signal_info["position"]

        try:
            # Create modified signal with net quantities
            net_signal = signal

            # Update quantities based on netting
            for net_order in plan["orders"]:
                if net_order["asset"] == symbol1:
                    net_signal.symbol1_qty = net_order["net_quantity"]
                elif net_order["asset"] == symbol2:
                    net_signal.symbol2_qty = net_order["net_quantity"]

            # Execute the netted position
            if signal.is_entry and not position:
                print(f"   🟢 Opening {signal.side} position (netted quantities)")
                result = executor.open_position(symbol1, symbol2, net_signal)
                if result:
                    # Record with ACTUAL execution details for proper tracking
                    tracker.record_open(
                        symbol1, symbol2, net_signal, execution_result=result
                    )
                    positions_opened += 1

                    # Send Slack notification for position open
                    try:
                        position_info = {
                            "total_notional": f"{TRADING_CONFIG['max_position_size_usdt']*2:.2f}",
                            "margin_required": f"{TRADING_CONFIG['max_position_size_usdt']*2/TRADING_CONFIG['leverage']:.2f}",
                        }
                        telegram.notify_position_open(
                            symbol1, symbol2, net_signal, position_info
                        )
                    except Exception as e:
                        print(f"   Warning: Telegram notification failed: {e}")

            elif signal.is_exit and position:
                print(
                    f"   🔴 Closing position (was {position.side}) - using actual executed quantities"
                )
                # Pass the actual position object so we can close exactly what was opened
                result = executor.close_position(symbol1, symbol2, net_signal, position)
                if result:
                    # Calculate position duration before closing
                    try:
                        entry_time = datetime.fromisoformat(position.entry_time)
                        duration = datetime.now() - entry_time
                        duration_str = f"{duration.total_seconds()/3600:.1f} hours"
                    except:
                        duration_str = "Unknown"

                    tracker.record_close(
                        symbol1, symbol2, signal, execution_result=result
                    )
                    positions_closed += 1

                    # Send Telegram notification for position close
                    try:
                        position_info = {"duration": duration_str}
                        telegram.notify_position_close(
                            symbol1, symbol2, net_signal, position_info
                        )
                    except Exception as e:
                        print(f"   Warning: Telegram notification failed: {e}")

        except Exception as e:
            print(f"   ❌ Error executing {pair_key}: {e}")
            try:
                telegram.notify_error(str(e), symbol1, symbol2)
            except:
                pass

    # Step 4: Monitor existing positions (same as monitoring mode)
    monitor_existing_positions(tracker, executor, telegram)

    # Final summary
    updated_positions = tracker.get_all_open_positions()
    asset_summary = asset_tracker.get_asset_summary()

    # Count execution results
    executed_pairs = sum(
        1
        for plan in execution_plan["execution_plan"].values()
        if plan["status"] == "execute_netted"
    )
    netted_out_pairs = sum(
        1
        for plan in execution_plan["execution_plan"].values()
        if plan["status"] == "skip_netted_out"
    )

    print(f"\n{'='*60}")
    print(f"TRADING CYCLE SUMMARY")
    print(f"{'='*60}")
    print(f"📊 Pairs processed: {len(pairs)}")
    print(f"🎯 Signals generated: {signals_generated}")
    print(
        f"💡 Position netting applied: {'Yes' if netting_analysis.get('has_netting_opportunities') else 'No'}"
    )
    print(f"✅ Pairs executed: {executed_pairs}")
    print(f"🔄 Pairs netted out: {netted_out_pairs}")
    print(
        f"💰 Capital savings: ${execution_plan['capital_savings']:.2f} ({execution_plan['savings_ratio']:.1%})"
    )
    print(f"🟢 Positions opened: {positions_opened}")
    print(f"🔴 Positions closed: {positions_closed}")
    print(f"📈 Active positions: {len(updated_positions)}")

    if updated_positions:
        print(f"\nActive Pair Positions:")
        for pair_key, pos in updated_positions.items():
            print(
                f"  • {pos.symbol1}-{pos.symbol2}: {pos.side} (opened: {pos.entry_time[:19]})"
            )

    if asset_summary:
        print(f"\nAsset-Level Net Positions:")
        for asset, info in asset_summary.items():
            if info["net_quantity"] != 0:
                print(
                    f"  • {asset}: {info['direction']} {abs(info['net_quantity']):.6f} (from {info['pairs_count']} pairs)"
                )

    # Show final net positions from execution plan
    final_net_positions = {
        k: v for k, v in execution_plan["net_positions"].items() if abs(v) > 0.001
    }
    if final_net_positions:
        print(f"\nFinal Asset Net Positions (after netting):")
        for asset, qty in final_net_positions.items():
            direction = "LONG" if qty > 0 else "SHORT"
            print(f"  • {asset}: {direction} {abs(qty):.6f}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Live Mean Reversion Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode signal     # Run full trading cycle (default)
  python main.py --mode monitor    # Monitor existing positions only
  python main.py -m signal         # Short form
  python main.py -m monitor        # Short form
        """,
    )

    parser.add_argument(
        "--mode",
        "-m",
        choices=["signal", "monitor"],
        default="signal",
        help="Trading mode: 'signal' for full trading cycle, 'monitor' for position monitoring only (default: signal)",
    )

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    print("=" * 80)
    print("LIVE MEAN REVERSION TRADING SYSTEM")
    print(f"MODE: {args.mode.upper()}")
    print("=" * 80)

    # Check if API keys are loaded
    api_key_loaded = bool(os.getenv("ZZT_BINANCE_KEY"))
    api_secret_loaded = bool(os.getenv("ZZT_BINANCE_SECRET"))
    print(f"🔑 API Key loaded: {'✅' if api_key_loaded else '❌'}")
    print(f"🔐 API Secret loaded: {'✅' if api_secret_loaded else '❌'}")

    # Initialize core components (needed for both modes)
    executor = TradeExecutor(
        sandbox=False,
        portfolio_value=TRADING_CONFIG["portfolio_value"],
        leverage=TRADING_CONFIG["leverage"],
        max_position_size_usdt=TRADING_CONFIG["max_position_size_usdt"],
    )
    tracker = PositionTracker()
    telegram = TelegramUtil()

    print(f"\n{'='*60}")
    print(f"Time: {datetime.now()}")
    print(f"{'='*60}")

    if args.mode == "monitor":
        # MONITORING MODE: Only check existing positions
        print("🔍 MONITORING MODE: Checking existing positions only...")
        monitor_existing_positions(tracker, executor, telegram)
        print(f"\n⏰ Monitoring complete. Run again as needed.")

    else:
        # SIGNAL MODE: Full trading cycle
        print("📊 SIGNAL MODE: Running full trading cycle...")

        # Initialize additional components needed for signal processing
        data_manager = LiveDataManager(timeframe="4h", lookback_periods=100)
        strategy = LiveMeanReversionStrategy(**FIXED_PARAMS)
        asset_tracker = AssetPositionTracker(conflict_threshold=0.1)

        # Load cointegrated pairs
        coint_results = tracker.load_cointegrated_pairs(COINTEGRATION_PAIRS_PATH)

        pairs = tracker.filter_top_pairs(
            coint_results,
            n_pairs=100,
            max_p_value=0.0276,
            min_correlation=0.884,
            max_half_life=23.4,
        )

        if not pairs:
            print(
                "❌ No cointegrated pairs found! Please run generate_cointegrated_pairs.py first."
            )
            return

        print(f"✅ Loaded {len(pairs)} cointegrated pairs for trading.")
        print(f"📊 Strategy Parameters:")
        print(f"   • Lookback Period: {FIXED_PARAMS['lookback_period']}")
        print(f"   • Entry Threshold: {FIXED_PARAMS['entry_threshold']}")
        print(f"   • Exit Threshold: {FIXED_PARAMS['exit_threshold']}")
        print(f"   • Stop Loss Threshold: {FIXED_PARAMS['stop_loss_threshold']}")
        print(f"💰 Trading Configuration:")
        print(f"   • Leverage: {TRADING_CONFIG['leverage']}x")
        print(
            f"   • Max Position Size: ${TRADING_CONFIG['max_position_size_usdt']} USDT per leg"
        )
        print(f"   • Portfolio Value: ${TRADING_CONFIG['portfolio_value']} USDT")
        print(
            f"   • Max Margin per Trade: ${TRADING_CONFIG['max_position_size_usdt']*2/TRADING_CONFIG['leverage']:.2f} USDT"
        )

        # Send system start notification
        system_config = {
            "Mode": "SIGNAL",
            "Pairs": f"{len(pairs)} loaded",
            "Leverage": f"{TRADING_CONFIG['leverage']}x",
            "Max Position": f"${TRADING_CONFIG['max_position_size_usdt']} USDT per leg",
            "Portfolio": f"${TRADING_CONFIG['portfolio_value']} USDT",
            "Lookback": f"{FIXED_PARAMS['lookback_period']} periods",
            "Entry Threshold": f"{FIXED_PARAMS['entry_threshold']}",
            "Exit Threshold": f"{FIXED_PARAMS['exit_threshold']}",
        }
        telegram.notify_system_start(system_config)

        # Run the full trading cycle
        run_full_trading_cycle(
            data_manager, strategy, executor, tracker, asset_tracker, telegram, pairs
        )

        print(f"\n⏰ Waiting for next 4-hour cycle...")


if __name__ == "__main__":
    main()
