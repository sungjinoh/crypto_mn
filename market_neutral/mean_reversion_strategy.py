"""
Mean Reversion Statistical Arbitrage Strategy for Pairs Trading
"""

import pandas as pd
import numpy as np
from typing import List
from backtesting_framework.pairs_backtester import PairsStrategy


class MeanReversionStrategy(PairsStrategy):
    """Mean reversion statistical arbitrage strategy"""

    def __init__(
        self,
        lookback_period: int = 60,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        stop_loss_threshold: float = 3.0,
        **params
    ):
        super().__init__(**params)
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold

    def get_required_indicators(self) -> List[str]:
        return ["spread", "spread_mean", "spread_std", "zscore"]

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on z-score thresholds

        Strategy Logic:
        - Entry: Z-score > +2.0 (short outperformer, long underperformer)
                 or < -2.0 (long outperformer, short underperformer)
        - Exit: Z-score crosses 0 (mean reversion)
        - Stop Loss: |Z-score| > 3.0

        Returns:
            DataFrame with columns: signal, position1, position2
            signal: 1 for entry, -1 for exit, 0 for hold
            position1/position2: +1 for long, -1 for short, 0 for flat
        """
        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0
        signals["position1"] = 0
        signals["position2"] = 0

        # Use consistent z-score if available, otherwise fallback to regular zscore
        if "consistent_zscore" in data.columns:
            zscore = data["consistent_zscore"]
            # print(f"   Using consistent z-score from cointegration analysis")
        else:
            zscore = data["zscore"]
            # print(f"   Using standard z-score calculation")

        # Entry signals
        long_entry = (
            zscore < -self.entry_threshold
        )  # spread too low, expect reversion up
        short_entry = (
            zscore > self.entry_threshold
        )  # spread too high, expect reversion down

        # Exit signals
        exit_condition = abs(zscore) < self.exit_threshold

        # Stop loss signals
        stop_loss = abs(zscore) > self.stop_loss_threshold

        # Generate signals
        signals.loc[long_entry, "signal"] = 1
        signals.loc[long_entry, "position1"] = 1  # long symbol1
        signals.loc[long_entry, "position2"] = -1  # short symbol2

        signals.loc[short_entry, "signal"] = 1
        signals.loc[short_entry, "position1"] = -1  # short symbol1
        signals.loc[short_entry, "position2"] = 1  # long symbol2

        signals.loc[exit_condition | stop_loss, "signal"] = -1  # exit signal

        return signals


class AdaptiveMeanReversionStrategy(PairsStrategy):
    """
    Adaptive mean reversion strategy with dynamic thresholds
    """

    def __init__(
        self,
        lookback_period: int = 60,
        threshold_lookback: int = 252,  # 1 year for threshold calculation
        entry_percentile: float = 95,  # Top/bottom 5% for entry
        exit_percentile: float = 50,  # Median for exit
        stop_percentile: float = 99,  # Top/bottom 1% for stop loss
        **params
    ):
        super().__init__(**params)
        self.lookback_period = lookback_period
        self.threshold_lookback = threshold_lookback
        self.entry_percentile = entry_percentile
        self.exit_percentile = exit_percentile
        self.stop_percentile = stop_percentile

    def get_required_indicators(self) -> List[str]:
        return ["spread", "spread_mean", "spread_std", "zscore"]

    def calculate_dynamic_thresholds(self, zscore: pd.Series, idx: int) -> dict:
        """Calculate dynamic thresholds based on historical z-score distribution"""
        # Get historical window
        start_idx = max(0, idx - self.threshold_lookback)
        historical_zscore = zscore.iloc[start_idx : idx + 1].dropna()

        if len(historical_zscore) < 50:
            # Fallback to static thresholds
            return {
                "entry_upper": 2.0,
                "entry_lower": -2.0,
                "exit_threshold": 0.0,
                "stop_upper": 3.0,
                "stop_lower": -3.0,
            }

        # Calculate percentile-based thresholds
        entry_upper = np.percentile(historical_zscore, self.entry_percentile)
        entry_lower = np.percentile(historical_zscore, 100 - self.entry_percentile)
        exit_threshold = np.percentile(historical_zscore, self.exit_percentile)
        stop_upper = np.percentile(historical_zscore, self.stop_percentile)
        stop_lower = np.percentile(historical_zscore, 100 - self.stop_percentile)

        return {
            "entry_upper": entry_upper,
            "entry_lower": entry_lower,
            "exit_threshold": exit_threshold,
            "stop_upper": stop_upper,
            "stop_lower": stop_lower,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals with adaptive thresholds"""
        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0
        signals["position1"] = 0
        signals["position2"] = 0

        zscore = data["zscore"]

        for i, (idx, row) in enumerate(data.iterrows()):
            if pd.isna(row["zscore"]):
                continue

            # Calculate dynamic thresholds for this point
            thresholds = self.calculate_dynamic_thresholds(zscore, i)

            current_zscore = row["zscore"]

            # Entry signals
            if current_zscore < thresholds["entry_lower"]:  # Long entry
                signals.loc[idx, "signal"] = 1
                signals.loc[idx, "position1"] = 1
                signals.loc[idx, "position2"] = -1
            elif current_zscore > thresholds["entry_upper"]:  # Short entry
                signals.loc[idx, "signal"] = 1
                signals.loc[idx, "position1"] = -1
                signals.loc[idx, "position2"] = 1

            # Exit signals
            elif (
                abs(current_zscore) < abs(thresholds["exit_threshold"])
                or current_zscore > thresholds["stop_upper"]
                or current_zscore < thresholds["stop_lower"]
            ):
                signals.loc[idx, "signal"] = -1

        return signals


class VolatilityAdjustedStrategy(PairsStrategy):
    """
    Mean reversion strategy with volatility-adjusted position sizing
    """

    def __init__(
        self,
        lookback_period: int = 60,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        stop_loss_threshold: float = 3.0,
        volatility_lookback: int = 20,
        vol_target: float = 0.15,  # Target 15% annualized volatility
        **params
    ):
        super().__init__(**params)
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.volatility_lookback = volatility_lookback
        self.vol_target = vol_target

    def get_required_indicators(self) -> List[str]:
        return ["spread", "spread_mean", "spread_std", "zscore", "spread_volatility"]

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals with volatility adjustment"""
        # First calculate spread volatility
        data_copy = data.copy()
        spread_returns = data_copy["spread"].pct_change()
        data_copy["spread_volatility"] = spread_returns.rolling(
            window=self.volatility_lookback
        ).std() * np.sqrt(
            252
        )  # Annualized

        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0
        signals["position1"] = 0
        signals["position2"] = 0
        signals["vol_adjustment"] = 1.0  # Volatility adjustment factor

        zscore = data_copy["zscore"]
        spread_vol = data_copy["spread_volatility"]

        # Entry signals with volatility adjustment
        long_entry = zscore < -self.entry_threshold
        short_entry = zscore > self.entry_threshold

        # Exit signals
        exit_condition = abs(zscore) < self.exit_threshold
        stop_loss = abs(zscore) > self.stop_loss_threshold

        # Apply signals
        signals.loc[long_entry, "signal"] = 1
        signals.loc[long_entry, "position1"] = 1
        signals.loc[long_entry, "position2"] = -1

        signals.loc[short_entry, "signal"] = 1
        signals.loc[short_entry, "position1"] = -1
        signals.loc[short_entry, "position2"] = 1

        signals.loc[exit_condition | stop_loss, "signal"] = -1

        # Calculate volatility adjustment
        vol_adjustment = np.where(
            spread_vol > 0,
            np.clip(self.vol_target / spread_vol, 0.5, 2.0),  # Cap adjustment
            1.0,
        )
        signals["vol_adjustment"] = vol_adjustment

        return signals


class MomentumMeanReversionStrategy(PairsStrategy):
    """
    Hybrid strategy combining momentum and mean reversion signals
    """

    def __init__(
        self,
        lookback_period: int = 60,
        momentum_period: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        stop_loss_threshold: float = 3.0,
        momentum_weight: float = 0.3,  # Weight for momentum component
        **params
    ):
        super().__init__(**params)
        self.lookback_period = lookback_period
        self.momentum_period = momentum_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.momentum_weight = momentum_weight

    def get_required_indicators(self) -> List[str]:
        return ["spread", "spread_mean", "spread_std", "zscore"]

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate hybrid momentum-mean reversion signals"""
        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0
        signals["position1"] = 0
        signals["position2"] = 0

        # Calculate momentum indicator
        spread_momentum = (
            data["spread"]
            .rolling(window=self.momentum_period)
            .apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
            )
        )

        zscore = data["zscore"]

        # Combine mean reversion and momentum signals
        mean_reversion_signal = np.where(
            zscore > self.entry_threshold,
            -1,
            np.where(zscore < -self.entry_threshold, 1, 0),
        )

        momentum_signal = np.where(
            spread_momentum > 0, 1, np.where(spread_momentum < 0, -1, 0)
        )

        # Weighted combination
        combined_signal = (
            1 - self.momentum_weight
        ) * mean_reversion_signal + self.momentum_weight * momentum_signal

        # Entry signals
        long_entry = combined_signal > 0.5
        short_entry = combined_signal < -0.5

        # Exit signals
        exit_condition = abs(zscore) < self.exit_threshold
        stop_loss = abs(zscore) > self.stop_loss_threshold

        # Apply signals
        signals.loc[long_entry, "signal"] = 1
        signals.loc[long_entry, "position1"] = 1
        signals.loc[long_entry, "position2"] = -1

        signals.loc[short_entry, "signal"] = 1
        signals.loc[short_entry, "position1"] = -1
        signals.loc[short_entry, "position2"] = 1

        signals.loc[exit_condition | stop_loss, "signal"] = -1

        return signals
