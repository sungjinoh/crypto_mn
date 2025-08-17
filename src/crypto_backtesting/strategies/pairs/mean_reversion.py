"""
Mean reversion strategy for pairs trading        # Create configuration
        config = PairsStrategyConfig(
            name="MeanReversion",
            parameters={
                'lookback_period': lookback_period,
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'stop_loss_threshold': stop_loss_threshold,
                **kwargs
            },
            optimization_params={
                'lookback_period': (20, 120),
                'entry_threshold': (1.5, 3.0),
                'exit_threshold': (0.0, 1.0),
                'stop_loss_threshold': (2.5, 4.0)
            },
            symbol1=symbol1,
            symbol2=symbol2
        )ements various mean reversion strategies for statistical
arbitrage trading of cointegrated pairs.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from ..base import PairsStrategy, PairsStrategyConfig, Signal, SignalType


class MeanReversionStrategy(PairsStrategy):
    """
    Mean reversion pairs trading strategy.
    
    This strategy identifies when the spread between two cointegrated
    assets deviates from its historical mean and trades on the expectation
    that the spread will revert to the mean.
    """
    
    def __init__(
        self,
        symbol1: str,
        symbol2: str,
        lookback_period: int = 60,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        stop_loss_threshold: float = 3.0,
        **kwargs
    ):
        """
        Initialize mean reversion strategy.
        
        Args:
            symbol1: First symbol in the pair
            symbol2: Second symbol in the pair
            lookback_period: Period for calculating rolling statistics
            entry_threshold: Z-score threshold for entry signals
            exit_threshold: Z-score threshold for exit signals
            stop_loss_threshold: Z-score threshold for stop loss
            **kwargs: Additional parameters
        """
        # Create configuration
        config = PairsStrategyConfig(
            name="MeanReversion",
            symbol1=symbol1,
            symbol2=symbol2,
            parameters={
                'lookback_period': lookback_period,
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'stop_loss_threshold': stop_loss_threshold,
                **kwargs
            },
            optimization_params={
                'lookback_period': (20, 120),
                'entry_threshold': (1.5, 3.0),
                'exit_threshold': (-0.5, 0.5),
                'stop_loss_threshold': (2.5, 4.0)
            }
        )
        
        super().__init__(config)
        
        # Strategy-specific attributes
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        
        # State tracking
        self.current_position = 0  # 0: no position, 1: long spread, -1: short spread
        self.entry_zscore = None
        self.spread_mean = None
        self.spread_std = None
        
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Initialize strategy with market data.
        
        Args:
            data: Historical market data
        """
        # Validate required columns
        required_cols = [f'{self.pairs_config.symbol1}_close', 
                        f'{self.pairs_config.symbol2}_close']
        
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column {col} not found in data")
                
        # Calculate initial spread statistics
        if 'spread' in data.columns:
            self.spread_mean = data['spread'].mean()
            self.spread_std = data['spread'].std()
            
        self.is_initialized = True
        
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        
        Returns:
            True if parameters are valid
        """
        checks = [
            self.lookback_period > 0,
            self.entry_threshold > 0,
            self.stop_loss_threshold > self.entry_threshold,
            self.lookback_period <= 500  # Reasonable upper bound
        ]
        
        return all(checks)
        
    def validate_pair(
        self, 
        data1: pd.DataFrame, 
        data2: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate that the pair is suitable for mean reversion trading.
        
        Args:
            data1: Price data for first symbol
            data2: Price data for second symbol
            
        Returns:
            Dict with validation results
        """
        # Test cointegration
        close1 = data1['close'] if 'close' in data1.columns else data1.iloc[:, 3]
        close2 = data2['close'] if 'close' in data2.columns else data2.iloc[:, 3]
        
        cointegration_result = self.test_cointegration(close1, close2)
        
        # Calculate spread statistics
        spread = self.calculate_spread(data1, data2)
        
        validation_result = {
            'is_valid': cointegration_result['is_cointegrated'],
            'cointegration': cointegration_result,
            'spread_stats': {
                'mean': spread.mean(),
                'std': spread.std(),
                'min': spread.min(),
                'max': spread.max(),
                'adf_pvalue': None  # Could add ADF test here
            },
            'correlation': close1.corr(close2),
            'data_quality': {
                'data1_length': len(data1),
                'data2_length': len(data2),
                'common_periods': len(close1.dropna().index.intersection(close2.dropna().index))
            }
        }
        
        return validation_result
        
    def calculate_spread(
        self, 
        data1: pd.DataFrame, 
        data2: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate spread between two assets.
        
        Args:
            data1: Price data for first symbol
            data2: Price data for second symbol
            
        Returns:
            Spread time series
        """
        close1 = data1['close'] if 'close' in data1.columns else data1.iloc[:, 3]
        close2 = data2['close'] if 'close' in data2.columns else data2.iloc[:, 3]
        
        # Calculate hedge ratio if not set
        if self.hedge_ratio is None:
            self.hedge_ratio = self.calculate_hedge_ratio(close1, close2)
            
        # Calculate spread: price1 - hedge_ratio * price2
        spread = close1 - self.hedge_ratio * close2
        
        return spread
        
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on spread mean reversion.
        
        Args:
            data: Market data with spread and indicators
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if not self.is_initialized:
            return signals
            
        # Get the latest data point
        if len(data) == 0:
            return signals
            
        latest_idx = data.index[-1]
        latest_data = data.loc[latest_idx]
        
        # Calculate rolling z-score if not already present
        if 'zscore' not in data.columns:
            data = self._add_zscore_indicator(data)
            latest_data = data.loc[latest_idx]
            
        if pd.isna(latest_data.get('zscore')):
            return signals
            
        current_zscore = latest_data['zscore']
        
        # Generate signals based on current position and z-score
        signal = self._evaluate_signal_conditions(current_zscore, latest_idx)
        
        if signal is not None:
            signals.append(signal)
            
        return signals
        
    def _add_zscore_indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add z-score indicator to data.
        
        Args:
            data: Market data
            
        Returns:
            Data with z-score column added
        """
        data = data.copy()
        
        # Calculate rolling mean and std of spread
        rolling_mean = data['spread'].rolling(
            window=self.lookback_period, min_periods=self.lookback_period//2
        ).mean()
        
        rolling_std = data['spread'].rolling(
            window=self.lookback_period, min_periods=self.lookback_period//2
        ).std()
        
        # Calculate z-score
        data['zscore'] = (data['spread'] - rolling_mean) / rolling_std
        
        # Store current statistics
        if not rolling_mean.empty:
            self.spread_mean = rolling_mean.iloc[-1]
        if not rolling_std.empty:
            self.spread_std = rolling_std.iloc[-1]
            
        return data
        
    def _evaluate_signal_conditions(
        self, 
        current_zscore: float, 
        timestamp: pd.Timestamp
    ) -> Optional[Signal]:
        """
        Evaluate current conditions and generate signal if appropriate.
        
        Args:
            current_zscore: Current z-score of spread
            timestamp: Current timestamp
            
        Returns:
            Signal if conditions are met, None otherwise
        """
        signal_metadata = {
            'zscore': current_zscore,
            'position_before': self.current_position,
            'entry_zscore': self.entry_zscore
        }
        
        # No position - look for entry signals
        if self.current_position == 0:
            if current_zscore > self.entry_threshold:
                # Spread is high - short the spread (long symbol2, short symbol1)
                self.current_position = -1
                self.entry_zscore = current_zscore
                
                signal_metadata.update({
                    'action': 'short_spread',
                    'position1': -1,  # Short symbol1
                    'position2': 1    # Long symbol2
                })
                
                return Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.ENTRY,
                    symbol=f"{self.pairs_config.symbol1}-{self.pairs_config.symbol2}",
                    position_size=1.0,
                    confidence=min(abs(current_zscore) / self.entry_threshold, 1.0),
                    metadata=signal_metadata
                )
                
            elif current_zscore < -self.entry_threshold:
                # Spread is low - long the spread (long symbol1, short symbol2)
                self.current_position = 1
                self.entry_zscore = current_zscore
                
                signal_metadata.update({
                    'action': 'long_spread',
                    'position1': 1,   # Long symbol1
                    'position2': -1   # Short symbol2
                })
                
                return Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.ENTRY,
                    symbol=f"{self.pairs_config.symbol1}-{self.pairs_config.symbol2}",
                    position_size=1.0,
                    confidence=min(abs(current_zscore) / self.entry_threshold, 1.0),
                    metadata=signal_metadata
                )
                
        # Have position - look for exit signals
        else:
            # Check for stop loss
            if self._should_stop_loss(current_zscore):
                previous_position = self.current_position
                self.current_position = 0
                self.entry_zscore = None
                
                signal_metadata.update({
                    'action': 'stop_loss',
                    'exit_reason': 'stop_loss'
                })
                
                return Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.EXIT,
                    symbol=f"{self.pairs_config.symbol1}-{self.pairs_config.symbol2}",
                    position_size=0.0,
                    confidence=1.0,
                    metadata=signal_metadata
                )
                
            # Check for normal exit
            elif self._should_exit(current_zscore):
                previous_position = self.current_position
                self.current_position = 0
                self.entry_zscore = None
                
                signal_metadata.update({
                    'action': 'normal_exit',
                    'exit_reason': 'mean_reversion'
                })
                
                return Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.EXIT,
                    symbol=f"{self.pairs_config.symbol1}-{self.pairs_config.symbol2}",
                    position_size=0.0,
                    confidence=1.0,
                    metadata=signal_metadata
                )
                
        return None
        
    def _should_exit(self, current_zscore: float) -> bool:
        """
        Check if position should be exited based on mean reversion.
        
        Args:
            current_zscore: Current z-score
            
        Returns:
            True if should exit
        """
        if self.current_position == 0:
            return False
            
        # Exit when z-score crosses back towards exit threshold
        if self.current_position == 1:  # Long spread position
            return current_zscore >= self.exit_threshold
        else:  # Short spread position
            return current_zscore <= self.exit_threshold
            
    def _should_stop_loss(self, current_zscore: float) -> bool:
        """
        Check if position should be stopped out.
        
        Args:
            current_zscore: Current z-score
            
        Returns:
            True if should stop loss
        """
        if self.current_position == 0:
            return False
            
        # Stop loss when z-score moves further against us
        if self.current_position == 1:  # Long spread position
            return current_zscore < -self.stop_loss_threshold
        else:  # Short spread position
            return current_zscore > self.stop_loss_threshold
            
    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self.current_position = 0
        self.entry_zscore = None
        self.spread_mean = None
        self.spread_std = None
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get current strategy status.
        
        Returns:
            Dict with current strategy state
        """
        return {
            'name': self.name,
            'position': self.current_position,
            'entry_zscore': self.entry_zscore,
            'spread_mean': self.spread_mean,
            'spread_std': self.spread_std,
            'parameters': self.parameters,
            'is_initialized': self.is_initialized
        }
