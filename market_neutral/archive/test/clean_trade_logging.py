#!/usr/bin/env python3
"""
Clean trade logging implementation
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def save_trade_details_clean(
    trades, symbol1, symbol2, strategy, output_dir="trade_logs"
):
    """
    Clean implementation of trade details saving with funding support
    """
    if not trades:
        print(f"   📝 No trades to save for {symbol1}-{symbol2}")
        return None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert trades to DataFrame
    trade_details = []

    for i, trade in enumerate(trades, 1):
        try:
            # Safe duration calculation
            duration = 0.0
            if trade.entry_time and trade.exit_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600

            # Safe PnL handling
            pnl = trade.pnl if trade.pnl is not None else 0.0

            # Safe return calculation
            entry_price1 = trade.entry_price1 if trade.entry_price1 is not None else 0.0
            entry_price2 = trade.entry_price2 if trade.entry_price2 is not None else 0.0
            entry_value = abs(entry_price1) + abs(entry_price2)

            return_pct = 0.0
            if entry_value > 0 and pnl != 0:
                return_pct = pnl / entry_value * 100

            # Handle funding costs safely
            funding_cost1 = getattr(trade, "funding_cost1", None) or 0.0
            funding_cost2 = getattr(trade, "funding_cost2", None) or 0.0
            total_funding_cost = getattr(trade, "total_funding_cost", None) or 0.0
            net_pnl = getattr(trade, "net_pnl", None) or pnl

            # Calculate funding impact
            funding_impact_pct = 0.0
            if abs(pnl) > 0:
                funding_impact_pct = (total_funding_cost / abs(pnl)) * 100

            trade_detail = {
                # Basic info
                "trade_id": i,
                "symbol1": symbol1,
                "symbol2": symbol2,
                "pair": f"{symbol1}-{symbol2}",
                # Strategy parameters
                "lookback_period": getattr(strategy, "lookback_period", 60),
                "entry_threshold": getattr(strategy, "entry_threshold", 2.0),
                "exit_threshold": getattr(strategy, "exit_threshold", 0.0),
                "stop_loss_threshold": getattr(strategy, "stop_loss_threshold", 3.0),
                # Entry details
                "entry_time": trade.entry_time,
                "entry_price1": entry_price1,
                "entry_price2": entry_price2,
                "entry_spread": getattr(trade, "entry_spread", 0.0),
                "entry_zscore": getattr(trade, "entry_zscore", 0.0),
                "position1": getattr(trade, "position1", 0),
                "position2": getattr(trade, "position2", 0),
                # Exit details
                "exit_time": trade.exit_time,
                "exit_price1": getattr(trade, "exit_price1", entry_price1),
                "exit_price2": getattr(trade, "exit_price2", entry_price2),
                "exit_spread": getattr(trade, "exit_spread", 0.0),
                "exit_zscore": getattr(trade, "exit_zscore", 0.0),
                # Performance
                "duration_hours": duration,
                "pnl": pnl,
                "return_pct": return_pct,
                "is_profitable": pnl > 0,
                "is_closed": getattr(trade, "is_closed", True),
                # Funding information
                "funding_cost1": funding_cost1,
                "funding_cost2": funding_cost2,
                "total_funding_cost": total_funding_cost,
                "net_pnl": net_pnl,
                "net_is_profitable": net_pnl > 0,
                "funding_impact_pct": funding_impact_pct,
                "funding_payments_count": len(getattr(trade, "funding_payments", [])),
                # Analysis
                "spread_change": (
                    getattr(trade, "exit_spread", 0.0)
                    - getattr(trade, "entry_spread", 0.0)
                ),
                "zscore_change": (
                    getattr(trade, "exit_zscore", 0.0)
                    - getattr(trade, "entry_zscore", 0.0)
                ),
                "price1_change": (
                    getattr(trade, "exit_price1", entry_price1) - entry_price1
                ),
                "price2_change": (
                    getattr(trade, "exit_price2", entry_price2) - entry_price2
                ),
            }

            trade_details.append(trade_detail)

        except Exception as e:
            print(f"   ⚠️ Error processing trade {i}: {e}")
            # Add minimal record
            trade_details.append(
                {
                    "trade_id": i,
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "pair": f"{symbol1}-{symbol2}",
                    "pnl": 0.0,
                    "is_profitable": False,
                    "error": str(e),
                }
            )

    # Create DataFrame
    trades_df = pd.DataFrame(trade_details)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trades_{symbol1}_{symbol2}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    # Save to CSV
    trades_df.to_csv(filepath, index=False)

    # Print summary
    total_trades = len(trades_df)
    profitable_trades = trades_df["is_profitable"].sum()
    net_profitable_trades = trades_df["net_is_profitable"].sum()
    total_pnl = trades_df["pnl"].sum()
    total_funding_cost = trades_df["total_funding_cost"].sum()
    net_pnl = trades_df["net_pnl"].sum()
    avg_duration = trades_df["duration_hours"].mean()

    print(f"   📝 Trade details saved: {filepath}")
    print(f"      • Total trades: {total_trades}")
    print(
        f"      • Profitable (gross): {profitable_trades} ({profitable_trades/total_trades:.1%})"
    )
    print(
        f"      • Profitable (net): {net_profitable_trades} ({net_profitable_trades/total_trades:.1%})"
    )
    print(f"      • Gross PnL: ${total_pnl:.2f}")
    print(f"      • Funding Cost: ${total_funding_cost:.2f}")
    print(f"      • Net PnL: ${net_pnl:.2f}")
    print(f"      • Avg duration: {avg_duration:.1f} hours")

    if abs(total_pnl) > 0:
        funding_impact = (total_funding_cost / abs(total_pnl)) * 100
        print(f"      • Funding impact: {funding_impact:.1f}% of gross PnL")

    return filepath


if __name__ == "__main__":
    print("Clean trade logging implementation ready")
