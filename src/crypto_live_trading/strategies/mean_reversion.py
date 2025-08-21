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
                    signal.symbol1_qty = -1.0  # Short symbol1
                    signal.symbol2_qty = hedge_ratio  # Long symbol2
                    signal.reason = f"Short spread entry, z-score: {z_score:.3f}"
                else:  # Spread is low, expect reversion up
                    signal.side = "long"  # Long the spread
                    signal.symbol1_qty = 1.0  # Long symbol1
                    signal.symbol2_qty = -hedge_ratio  # Short symbol2
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

    def calculate_position_size(
        self,
        signal,
        portfolio_value,
        risk_per_trade=0.02,
        leverage=1,
        max_position_usdt=None,
    ):
        """
        Calculate position sizes based on portfolio value, leverage, and risk management

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            risk_per_trade: Risk per trade as fraction of portfolio
            leverage: Leverage multiplier for futures trading
            max_position_usdt: Maximum position size in USDT (overrides risk-based sizing)

        Returns:
            symbol1_size, symbol2_size: Position sizes in USDT notional value
        """
        if not signal.is_entry:
            return 0.0, 0.0

        if max_position_usdt:
            # Use fixed maximum position size
            symbol1_notional = max_position_usdt
            symbol2_notional = max_position_usdt * abs(signal.hedge_ratio)
        else:
            # Use risk-based position sizing
            risk_amount = portfolio_value * risk_per_trade

            # With leverage, we can take larger positions with same risk
            # But let's be conservative and not automatically scale up
            base_notional = risk_amount / 2  # Split between two legs

            symbol1_notional = base_notional
            symbol2_notional = base_notional * abs(signal.hedge_ratio)

        # Calculate margin requirement (actual capital needed)
        total_notional = symbol1_notional + symbol2_notional
        margin_required = total_notional / leverage

        print(f"  Position sizing:")
        print(f"    Symbol1 notional: ${symbol1_notional:.2f}")
        print(f"    Symbol2 notional: ${symbol2_notional:.2f}")
        print(f"    Total notional: ${total_notional:.2f}")
        print(f"    Margin required ({leverage}x leverage): ${margin_required:.2f}")

        return symbol1_notional, symbol2_notional
