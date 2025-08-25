"""
Asset Position Tracker - Manages asset-level positions and detects conflicts
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import os


@dataclass
class AssetPosition:
    """Track net position for a single asset across all pairs"""

    asset: str
    net_quantity: float = 0.0
    pairs: List[str] = None  # Which pairs contribute to this position

    def __post_init__(self):
        if self.pairs is None:
            self.pairs = []


@dataclass
class PendingOrder:
    """Represents a pending order for asset conflict analysis"""

    pair_key: str
    asset: str
    quantity: float  # Positive = long, negative = short
    signal_strength: float  # abs(z_score) for prioritization
    signal_type: str  # 'entry' or 'exit'


class AssetPositionTracker:
    """
    Tracks asset-level positions and detects conflicts across pairs
    """

    def __init__(self, conflict_threshold: float = 0.1):
        """
        Args:
            conflict_threshold: Minimum net position ratio to consider as conflict
                               (0.1 = 10% of total position)
        """
        self.asset_positions: Dict[str, AssetPosition] = {}
        self.conflict_threshold = conflict_threshold

    def update_asset_positions(self, pair_positions: Dict[str, any]):
        """
        Update asset positions based on current pair positions

        Args:
            pair_positions: Dict of pair_key -> Position from PositionTracker
        """
        # Reset asset positions
        self.asset_positions.clear()

        for pair_key, position in pair_positions.items():
            if position.status != "open":
                continue

            # Extract assets from position
            symbol1, symbol2 = position.symbol1, position.symbol2

            # Add to asset positions
            self._add_asset_position(symbol1, position.symbol1_qty, pair_key)
            self._add_asset_position(symbol2, position.symbol2_qty, pair_key)

    def check_pending_orders_for_conflicts(
        self, pending_orders: List[PendingOrder]
    ) -> Dict:
        """
        Analyze pending orders and generate optimized execution plan with position netting

        Args:
            pending_orders: List of pending orders to analyze

        Returns:
            Dict with netting analysis and optimized execution plan
        """
        # Calculate net positions if all pending orders were executed
        projected_positions = defaultdict(float)
        order_contributions = defaultdict(list)
        pair_contributions = defaultdict(list)

        # Start with current positions
        for asset, position in self.asset_positions.items():
            projected_positions[asset] = position.net_quantity

        # Add pending orders
        for order in pending_orders:
            projected_positions[order.asset] += order.quantity
            order_contributions[order.asset].append(order)
            if order.pair_key not in pair_contributions[order.asset]:
                pair_contributions[order.asset].append(order.pair_key)

        # Analyze for position netting opportunities
        netting_opportunities = {}
        netting_summary = []

        for asset, net_position in projected_positions.items():
            contributing_orders = order_contributions.get(asset, [])

            if len(contributing_orders) <= 1:
                continue  # No netting opportunity with single order

            # Calculate total gross exposure
            total_gross_exposure = sum(
                abs(order.quantity) for order in contributing_orders
            )
            net_exposure = abs(net_position)

            # Check if significant netting is possible
            netting_ratio = (
                1 - (net_exposure / total_gross_exposure)
                if total_gross_exposure > 0
                else 0
            )

            if netting_ratio > 0.1:  # More than 10% netting possible
                netting_opportunities[asset] = {
                    "net_position": net_position,
                    "gross_exposure": total_gross_exposure,
                    "net_exposure": net_exposure,
                    "netting_ratio": netting_ratio,
                    "capital_saved": total_gross_exposure - net_exposure,
                    "contributing_orders": contributing_orders,
                    "contributing_pairs": pair_contributions[asset],
                    "severity": (
                        "high_netting" if netting_ratio > 0.7 else "medium_netting"
                    ),
                }

                netting_summary.append(
                    f"{asset}: {total_gross_exposure:.2f} gross → {net_exposure:.2f} net "
                    f"({netting_ratio:.1%} netting, ${total_gross_exposure - net_exposure:.2f} saved)"
                )

        return {
            "has_netting_opportunities": len(netting_opportunities) > 0,
            "netting_opportunities": netting_opportunities,
            "netting_summary": netting_summary,
            "projected_positions": dict(projected_positions),
            "optimization_approach": "position_netting",
        }

    def generate_optimized_execution_plan(
        self, pending_orders: List[PendingOrder]
    ) -> Dict:
        """
        Generate an optimized execution plan using position netting

        Args:
            pending_orders: List of all pending orders

        Returns:
            Dict with optimized execution plan
        """
        # Calculate net positions per asset
        net_positions = defaultdict(float)
        pair_orders = defaultdict(list)
        asset_to_pairs = defaultdict(set)

        # Start with current positions
        for asset, position in self.asset_positions.items():
            net_positions[asset] = position.net_quantity

        # Add pending orders and track pair associations
        for order in pending_orders:
            net_positions[order.asset] += order.quantity
            pair_orders[order.pair_key].append(order)
            asset_to_pairs[order.asset].add(order.pair_key)

        # CRITICAL FIX: Calculate proportional allocation to avoid double execution
        # First, calculate each asset's contribution by pair
        asset_contributions = defaultdict(list)  # asset -> [(pair_key, quantity), ...]

        for order in pending_orders:
            asset_contributions[order.asset].append((order.pair_key, order.quantity))

        # Generate optimized orders with proportional allocation
        optimized_orders = []
        pair_execution_status = {}
        processed_pairs = set()

        for pair_key, orders in pair_orders.items():
            if pair_key in processed_pairs:
                continue

            # Group orders by asset for this pair
            pair_assets = {}
            for order in orders:
                pair_assets[order.asset] = order

            # Check if both assets in this pair need execution
            execute_pair = False
            pair_net_orders = []

            for asset, order in pair_assets.items():
                total_net_qty = net_positions[asset]

                # Skip if net position is too small
                if abs(total_net_qty) <= abs(order.quantity) * 0.01:
                    continue

                # Calculate this pair's proportional share of the net quantity
                asset_pairs = asset_contributions[asset]
                total_original_qty = sum(abs(qty) for pair, qty in asset_pairs)
                this_pair_qty = abs(order.quantity)

                if total_original_qty > 0:
                    allocation_ratio = this_pair_qty / total_original_qty
                    allocated_qty = total_net_qty * allocation_ratio
                else:
                    allocation_ratio = 1.0  # Full allocation when no other pairs
                    allocated_qty = total_net_qty  # Fallback if calculation fails

                # Only include if allocated quantity is significant
                if abs(allocated_qty) > abs(order.quantity) * 0.01:
                    pair_net_orders.append(
                        {
                            "asset": asset,
                            "original_quantity": order.quantity,
                            "net_quantity": allocated_qty,
                            "netting_ratio": (
                                1 - (abs(allocated_qty) / abs(order.quantity))
                                if order.quantity != 0
                                else 0
                            ),
                            "pair_key": pair_key,
                            "signal_type": order.signal_type,
                            "total_net": total_net_qty,
                            "allocation_ratio": allocation_ratio,
                            "allocated": allocated_qty,
                        }
                    )
                    execute_pair = True

            if execute_pair:
                pair_execution_status[pair_key] = {
                    "status": "execute_netted",
                    "orders": pair_net_orders,
                    "reason": "Executing with proportional allocation",
                }
                optimized_orders.extend(pair_net_orders)
            else:
                pair_execution_status[pair_key] = {
                    "status": "skip_netted_out",
                    "orders": [],
                    "reason": "Net position too small or zero",
                }

            processed_pairs.add(pair_key)

        # Verify proportional allocation adds up correctly
        allocation_check = {}
        for asset, total_net in net_positions.items():
            allocated_orders = [o for o in optimized_orders if o["asset"] == asset]
            total_allocated = sum(o["net_quantity"] for o in allocated_orders)
            allocation_check[asset] = {
                "total_net": total_net,
                "total_allocated": total_allocated,
                "allocation_error": abs(total_net - total_allocated),
                "pairs_allocated": len(allocated_orders),
            }

        # Calculate savings
        original_total_exposure = sum(abs(order.quantity) for order in pending_orders)
        optimized_total_exposure = sum(
            abs(order["net_quantity"]) for order in optimized_orders
        )
        capital_savings = original_total_exposure - optimized_total_exposure

        return {
            "execution_plan": pair_execution_status,
            "optimized_orders": optimized_orders,
            "original_exposure": original_total_exposure,
            "optimized_exposure": optimized_total_exposure,
            "capital_savings": capital_savings,
            "savings_ratio": (
                capital_savings / original_total_exposure
                if original_total_exposure > 0
                else 0
            ),
            "net_positions": dict(net_positions),
            "allocation_check": allocation_check,  # For debugging double allocation
        }

    def _add_asset_position(self, asset: str, quantity: float, pair_key: str):
        """Add quantity to asset position"""
        if asset not in self.asset_positions:
            self.asset_positions[asset] = AssetPosition(asset=asset)

        self.asset_positions[asset].net_quantity += quantity
        if pair_key not in self.asset_positions[asset].pairs:
            self.asset_positions[asset].pairs.append(pair_key)

    def get_asset_summary(self) -> Dict:
        """Get summary of current asset positions"""
        summary = {}

        for asset, position in self.asset_positions.items():
            summary[asset] = {
                "net_quantity": position.net_quantity,
                "direction": (
                    "LONG"
                    if position.net_quantity > 0
                    else "SHORT" if position.net_quantity < 0 else "NEUTRAL"
                ),
                "pairs_count": len(position.pairs),
                "pairs": position.pairs,
            }

        return summary

    def save_state(self, filepath: str = "state/asset_positions.json"):
        """Save current asset positions to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Convert to serializable format
            data = {}
            for asset, position in self.asset_positions.items():
                data[asset] = {
                    "asset": position.asset,
                    "net_quantity": position.net_quantity,
                    "pairs": position.pairs,
                }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Error saving asset positions: {e}")

    def load_state(self, filepath: str = "state/asset_positions.json"):
        """Load asset positions from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    data = json.load(f)

                self.asset_positions.clear()
                for asset, pos_data in data.items():
                    self.asset_positions[asset] = AssetPosition(
                        asset=pos_data["asset"],
                        net_quantity=pos_data["net_quantity"],
                        pairs=pos_data["pairs"],
                    )

                print(
                    f"Loaded {len(self.asset_positions)} asset positions from {filepath}"
                )
            else:
                print("No existing asset positions file found")

        except Exception as e:
            print(f"Error loading asset positions: {e}")
            self.asset_positions.clear()
