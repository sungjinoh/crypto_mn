"""
Live Mean Reversion Strategy - Based on backtesting framework
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from dataclasses import dataclass


@dataclass
class Signal:
    """Trading signal data class"""

    is_entry: bool = False
    is_exit: bool = False
    side: str = None  # 'long' or 'short'
    symbol1_qty: float = 0.0
    symbol2_qty: float = 0.0
    hedge_ratio: float = 0.0
    spread_value: float = 0.0
    z_score: float = 0.0
    reason: str = ""


class LiveMeanReversionStrategy:
    def __init__(
        self, lookback_period, entry_threshold, exit_threshold, stop_loss_threshold
    ):
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold

    def generate_signal(self, df1, df2, pair_info=None):
        """
        Generate trading signal based on mean reversion strategy

        Args:
            df1, df2: Price DataFrames for the pair
            pair_info: Cointegration info (hedge_ratio, etc.)

        Returns:
            Signal object with trading decision
        """
        if df1 is None or df2 is None or len(df1) < self.lookback_period:
            return Signal(reason="Insufficient data")

        try:
            # Use close prices for calculation
            prices1 = df1["close"].values
            prices2 = df2["close"].values

            # Get hedge ratio from pair info or calculate dynamically
            if pair_info and "hedge_ratio" in pair_info:
                hedge_ratio = pair_info["hedge_ratio"]
            else:
                # Calculate hedge ratio using cointegration
                _, p_value, hedge_ratio = coint(prices1, prices2)
                hedge_ratio = -hedge_ratio  # Adjust sign

            # Calculate spread
            spread = prices1 + hedge_ratio * prices2

            # Get lookback window
            spread_window = spread[-self.lookback_period :]

            # Calculate z-score
            spread_mean = np.mean(spread_window)
            spread_std = np.std(spread_window)

            if spread_std == 0:
                return Signal(reason="Zero spread volatility")

            current_spread = spread[-1]
            z_score = (current_spread - spread_mean) / spread_std

            # Generate signals based on z-score thresholds
            signal = Signal(
                hedge_ratio=hedge_ratio, spread_value=current_spread, z_score=z_score
            )

            # Entry signals
            if abs(z_score) >= self.entry_threshold:
                signal.is_entry = True

                if z_score > 0:  # Spread is high, expect reversion down
                    signal.side = "short"  # Short the spread
                    signal.symbol1_qty = -1.0  # Short symbol1 (unit quantity)
                    signal.symbol2_qty = hedge_ratio  # Long symbol2 (hedge_ratio units)
                    signal.reason = f"Short spread entry, z-score: {z_score:.3f}"
                else:  # Spread is low, expect reversion up
                    signal.side = "long"  # Long the spread
                    signal.symbol1_qty = 1.0  # Long symbol1 (unit quantity) 
                    signal.symbol2_qty = -hedge_ratio  # Short symbol2 (hedge_ratio units)
                    signal.reason = f"Long spread entry, z-score: {z_score:.3f}"

            # Exit signals (for existing positions)
            elif abs(z_score) <= self.exit_threshold:
                signal.is_exit = True
                signal.reason = f"Mean reversion exit, z-score: {z_score:.3f}"

            # Stop loss signals
            elif abs(z_score) >= self.stop_loss_threshold:
                signal.is_exit = True
                signal.reason = f"Stop loss triggered, z-score: {z_score:.3f}"

            else:
                signal.reason = f"No action, z-score: {z_score:.3f}"

            return signal

        except Exception as e:
            return Signal(reason=f"Error calculating signal: {e}")

    # Position sizing is now handled by TradeExecutor._calculate_position_sizes()
    # This method is removed to avoid confusion and duplication
