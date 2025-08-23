"""
Position Tracker - Manages open positions and trade state
"""

import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path
import pickle


@dataclass
class Position:
    """Position data class"""

    symbol1: str
    symbol2: str
    side: str  # 'long' or 'short'
    symbol1_qty: float  # Actual executed quantity
    symbol2_qty: float  # Actual executed quantity
    hedge_ratio: float
    entry_time: str
    entry_spread: float
    entry_z_score: float
    status: str = "open"  # 'open' or 'closed'
    # Store actual executed order details
    symbol1_executed_size: float = 0.0
    symbol2_executed_size: float = 0.0
    symbol1_executed_side: str = ""  # 'buy' or 'sell'
    symbol2_executed_side: str = ""  # 'buy' or 'sell'
    symbol1_entry_price: float = 0.0
    symbol2_entry_price: float = 0.0


class PositionTracker:
    def __init__(self, positions_file="state/positions.json"):
        self.positions_file = positions_file
        self.positions: Dict[str, Position] = {}
        self.load_positions()

    def load_cointegrated_pairs(self, path):
        """Load cointegrated pairs from JSON file"""
        path = Path(path)
        json_files = list(path.glob("cointegration_results_*.json"))
        pkl_files = list(path.glob("cointegration_results_*.pkl"))

        all_files = json_files + pkl_files
        if not all_files:
            raise FileNotFoundError(f"No results files found in {self.results_dir}")

        filepath = max(all_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading most recent results from: {filepath}")

        # Load based on file type
        if filepath.suffix == ".json":
            with open(filepath, "r") as f:
                return json.load(f)
        elif filepath.suffix == ".pkl":
            with open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def get_position(self, symbol1, symbol2):
        """Check if there's an open position for this pair"""
        pair_key = self._get_pair_key(symbol1, symbol2)
        position = self.positions.get(pair_key)
        return position if position and position.status == "open" else None

    def record_open(self, symbol1, symbol2, signal, execution_result=None):
        """Record a new open position with actual execution details"""
        pair_key = self._get_pair_key(symbol1, symbol2)

        # Extract actual executed details if available
        if execution_result:
            symbol1_executed_size = execution_result.get("symbol1_order", {}).get("size", signal.symbol1_qty)
            symbol2_executed_size = execution_result.get("symbol2_order", {}).get("size", signal.symbol2_qty)
            symbol1_executed_side = execution_result.get("symbol1_order", {}).get("side", "")
            symbol2_executed_side = execution_result.get("symbol2_order", {}).get("side", "")
            symbol1_entry_price = execution_result.get("symbol1_order", {}).get("price", 0.0)
            symbol2_entry_price = execution_result.get("symbol2_order", {}).get("price", 0.0)
            
            # Use actual executed quantities, taking into account the side
            actual_symbol1_qty = symbol1_executed_size if symbol1_executed_side == "buy" else -symbol1_executed_size
            actual_symbol2_qty = symbol2_executed_size if symbol2_executed_side == "buy" else -symbol2_executed_size
        else:
            # Fallback to signal quantities if no execution result
            symbol1_executed_size = abs(signal.symbol1_qty)
            symbol2_executed_size = abs(signal.symbol2_qty)
            symbol1_executed_side = "buy" if signal.symbol1_qty > 0 else "sell"
            symbol2_executed_side = "buy" if signal.symbol2_qty > 0 else "sell"
            symbol1_entry_price = 0.0
            symbol2_entry_price = 0.0
            actual_symbol1_qty = signal.symbol1_qty
            actual_symbol2_qty = signal.symbol2_qty

        position = Position(
            symbol1=symbol1,
            symbol2=symbol2,
            side=signal.side,
            symbol1_qty=actual_symbol1_qty,  # Store actual executed quantity
            symbol2_qty=actual_symbol2_qty,  # Store actual executed quantity
            hedge_ratio=signal.hedge_ratio,
            entry_time=datetime.now().isoformat(),
            entry_spread=signal.spread_value,
            entry_z_score=signal.z_score,
            status="open",
            symbol1_executed_size=symbol1_executed_size,
            symbol2_executed_size=symbol2_executed_size,
            symbol1_executed_side=symbol1_executed_side,
            symbol2_executed_side=symbol2_executed_side,
            symbol1_entry_price=symbol1_entry_price,
            symbol2_entry_price=symbol2_entry_price,
        )

        self.positions[pair_key] = position
        self._save_positions()

        print(f"Recorded open position: {symbol1}-{symbol2} {signal.side}")
        print(f"  Executed: {symbol1} {symbol1_executed_side} {symbol1_executed_size:.6f} @ ${symbol1_entry_price:.2f}")
        print(f"  Executed: {symbol2} {symbol2_executed_side} {symbol2_executed_size:.6f} @ ${symbol2_entry_price:.2f}")

    def record_close(self, symbol1, symbol2, signal, execution_result=None):
        """Record position closure with actual execution details"""
        pair_key = self._get_pair_key(symbol1, symbol2)

        if pair_key in self.positions:
            closed_position = self.positions[pair_key]
            closed_position.status = "closed"
            
            # Log the actual close execution details
            if execution_result:
                symbol1_close_size = execution_result.get("symbol1_order", {}).get("size", 0.0)
                symbol2_close_size = execution_result.get("symbol2_order", {}).get("size", 0.0)
                symbol1_close_side = execution_result.get("symbol1_order", {}).get("side", "")
                symbol2_close_side = execution_result.get("symbol2_order", {}).get("side", "")
                
                print(f"Recorded close position: {symbol1}-{symbol2}")
                print(f"  Close executed: {symbol1} {symbol1_close_side} {symbol1_close_size:.6f}")
                print(f"  Close executed: {symbol2} {symbol2_close_side} {symbol2_close_size:.6f}")
                
                # Verify close quantities match open quantities
                if abs(symbol1_close_size - closed_position.symbol1_executed_size) > 0.000001:
                    print(f"  ⚠️  WARNING: Close quantity mismatch for {symbol1}!")
                    print(f"      Opened: {closed_position.symbol1_executed_size:.6f}, Closed: {symbol1_close_size:.6f}")
                    
                if abs(symbol2_close_size - closed_position.symbol2_executed_size) > 0.000001:
                    print(f"  ⚠️  WARNING: Close quantity mismatch for {symbol2}!")
                    print(f"      Opened: {closed_position.symbol2_executed_size:.6f}, Closed: {symbol2_close_size:.6f}")
            else:
                print(f"Recorded close position: {symbol1}-{symbol2} (no execution details)")
            
            self._save_positions()

            # Archive closed position (remove from active positions)
            self.positions.pop(pair_key)
            self._archive_position(closed_position, signal, execution_result)

        else:
            print(f"Warning: Tried to close non-existent position {symbol1}-{symbol2}")

    def get_all_open_positions(self):
        """Get all open positions"""
        return {k: v for k, v in self.positions.items() if v.status == "open"}

    def load_positions(self):
        """Load positions from file"""
        try:
            if os.path.exists(self.positions_file):
                with open(self.positions_file, "r") as f:
                    data = json.load(f)

                self.positions = {}
                for pair_key, pos_data in data.items():
                    self.positions[pair_key] = Position(**pos_data)

                print(
                    f"Loaded {len(self.positions)} positions from {self.positions_file}"
                )
            else:
                self.positions = {}
                print(f"No existing positions file found")

        except Exception as e:
            print(f"Error loading positions: {e}")
            self.positions = {}

    def filter_top_pairs(
        self,
        results,
        n_pairs=10,
        max_p_value=0.05,
        min_correlation=0.6,
        max_half_life=100,
    ):
        """
        Filter and rank pairs based on multiple criteria.

        Args:
            pairs: List of cointegrated pair dicts
            n_pairs: Number of top pairs to return
            max_p_value: Maximum p-value threshold
            min_correlation: Minimum correlation threshold
            max_half_life: Maximum half-life threshold

        Returns:
            List of top pairs with scoring
        """

        pairs = results.get("cointegrated_pairs", [])

        if not pairs:
            print("No cointegrated pairs found in input list")
            return []

        # Filter pairs
        filtered_pairs = []
        for pair in pairs:
            # Check p-value
            if pair.get("p_value", 1) > max_p_value:
                continue

            # Check correlation
            if abs(pair.get("correlation", 0)) < min_correlation:
                continue

            # Check half-life if available
            if (
                max_half_life
                and "spread_properties" in pair
                and pair["spread_properties"]
            ):
                half_life = pair["spread_properties"].get("half_life")
                if half_life and half_life > max_half_life:
                    continue

            filtered_pairs.append(pair)

        # Score and rank pairs
        scored_pairs = []
        for pair in filtered_pairs:
            # Calculate composite score
            p_value_score = 1 - pair.get("p_value", 1)  # Lower is better
            correlation_score = abs(pair.get("correlation", 0))  # Higher is better

            # Half-life score (lower is better for faster mean reversion)
            half_life_score = 0.5  # Default
            if "spread_properties" in pair and pair["spread_properties"]:
                half_life = pair["spread_properties"].get("half_life")
                if half_life and half_life > 0:
                    half_life_score = max(0, 1 - (half_life / 100))

            # Weighted composite score
            composite_score = (
                0.3 * p_value_score + 0.3 * correlation_score + 0.4 * half_life_score
            )

            pair_with_score = pair.copy()
            pair_with_score["composite_score"] = composite_score
            scored_pairs.append(pair_with_score)

        # Sort by composite score
        scored_pairs.sort(key=lambda x: x["composite_score"], reverse=True)

        # Return top N pairs
        return scored_pairs[:n_pairs]

    def _save_positions(self):
        """Save positions to file"""
        try:
            os.makedirs(os.path.dirname(self.positions_file), exist_ok=True)

            # Convert positions to dict for JSON serialization
            data = {}
            for pair_key, position in self.positions.items():
                data[pair_key] = asdict(position)

            with open(self.positions_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Error saving positions: {e}")

    def _archive_position(self, position, signal, execution_result=None):
        """Archive closed position to trades history"""
        try:
            archive_file = "state/trades_history.json"

            # Load existing history
            history = []
            if os.path.exists(archive_file):
                with open(archive_file, "r") as f:
                    history = json.load(f)

            # Add closed position with exit info
            trade_record = asdict(position)
            trade_record.update(
                {
                    "exit_time": datetime.now().isoformat(),
                    "exit_spread": signal.spread_value,
                    "exit_z_score": signal.z_score,
                    "exit_reason": signal.reason,
                }
            )
            
            # Add exit execution details if available
            if execution_result:
                trade_record.update({
                    "exit_symbol1_size": execution_result.get("symbol1_order", {}).get("size", 0.0),
                    "exit_symbol2_size": execution_result.get("symbol2_order", {}).get("size", 0.0),
                    "exit_symbol1_side": execution_result.get("symbol1_order", {}).get("side", ""),
                    "exit_symbol2_side": execution_result.get("symbol2_order", {}).get("side", ""),
                    "exit_symbol1_price": execution_result.get("symbol1_order", {}).get("price", 0.0),
                    "exit_symbol2_price": execution_result.get("symbol2_order", {}).get("price", 0.0),
                })

            history.append(trade_record)

            # Save updated history
            os.makedirs(os.path.dirname(archive_file), exist_ok=True)
            with open(archive_file, "w") as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            print(f"Error archiving position: {e}")

    def _get_pair_key(self, symbol1, symbol2):
        """Generate consistent key for symbol pair"""
        return f"{symbol1}-{symbol2}"
