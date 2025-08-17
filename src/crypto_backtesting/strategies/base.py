"""
Base strategy classes for cryptocurrency trading strategies.

This module provides the abstract base classes that all trading strategies
must inherit from to ensure consistent interfaces and behavior.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Types of trading signals."""
    ENTRY = 1
    EXIT = -1
    HOLD = 0


@dataclass
class Signal:
    """
    Represents a trading signal.
    
    Attributes:
        timestamp: When the signal was generated
        signal_type: Type of signal (entry, exit, hold)
        symbol: Trading symbol
        position_size: Suggested position size (0-1)
        confidence: Signal confidence (0-1)
        metadata: Additional signal metadata
    """
    timestamp: pd.Timestamp
    signal_type: SignalType
    symbol: str
    position_size: float = 1.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StrategyConfig:
    """
    Base configuration for trading strategies.
    
    Attributes:
        name: Strategy name
        parameters: Strategy-specific parameters
        risk_management: Risk management settings
        optimization_params: Parameters available for optimization
    """
    name: str
    parameters: Dict[str, Any]
    risk_management: Dict[str, Any] = None
    optimization_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.risk_management is None:
            self.risk_management = {}
        if self.optimization_params is None:
            self.optimization_params = {}


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class defines the interface that all strategies must implement
    to work with the backtesting framework.
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize the strategy.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.name = config.name
        self.parameters = config.parameters
        
        # Strategy state
        self.is_initialized = False
        self.last_signal_time = None
        
    @abstractmethod
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Initialize the strategy with market data.
        
        This method is called once before backtesting begins.
        Use it to set up indicators, validate parameters, etc.
        
        Args:
            data: Historical market data for initialization
        """
        pass
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on market data.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            List of trading signals
        """
        pass
        
    @abstractmethod
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        pass
        
    def get_required_data(self) -> List[str]:
        """
        Get list of required data columns for this strategy.
        
        Returns:
            List of required column names
        """
        return ['open', 'high', 'low', 'close', 'volume']
        
    def get_lookback_period(self) -> int:
        """
        Get the minimum lookback period required by this strategy.
        
        Returns:
            Minimum number of periods needed for signal generation
        """
        return self.parameters.get('lookback_period', 1)
        
    def get_optimization_bounds(self) -> Dict[str, tuple]:
        """
        Get parameter bounds for optimization.
        
        Returns:
            Dict mapping parameter names to (min, max) tuples
        """
        return self.config.optimization_params
        
    def update_parameters(self, new_parameters: Dict[str, Any]) -> None:
        """
        Update strategy parameters.
        
        Args:
            new_parameters: Dictionary of parameter updates
        """
        self.parameters.update(new_parameters)
        
    def reset(self) -> None:
        """Reset strategy state for new backtest."""
        self.is_initialized = False
        self.last_signal_time = None
        
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.name}({self.parameters})"
        
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"


@dataclass
class PairsStrategyConfig(StrategyConfig):
    """
    Configuration specific to pairs trading strategies.
    
    Attributes:
        symbol1: First symbol in the pair
        symbol2: Second symbol in the pair
        hedge_ratio: Hedge ratio for the pair (optional)
        cointegration_params: Cointegration testing parameters
    """
    symbol1: str = None
    symbol2: str = None
    hedge_ratio: Optional[float] = None
    cointegration_params: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.cointegration_params is None:
            self.cointegration_params = {
                'significance_level': 0.05,
                'max_lags': 1
            }


class PairsStrategy(BaseStrategy):
    """
    Base class for pairs trading strategies.
    
    Provides common functionality for strategies that trade
    pairs of assets based on statistical relationships.
    """
    
    def __init__(self, config: PairsStrategyConfig):
        """
        Initialize pairs strategy.
        
        Args:
            config: Pairs strategy configuration
        """
        super().__init__(config)
        self.pairs_config = config
        
        # Pairs-specific state
        self.hedge_ratio = config.hedge_ratio
        self.spread_stats = {}
        self.is_pair_valid = False
        
    @abstractmethod
    def validate_pair(
        self, 
        data1: pd.DataFrame, 
        data2: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate that the pair is suitable for trading.
        
        Args:
            data1: Price data for first symbol
            data2: Price data for second symbol
            
        Returns:
            Dict with validation results and statistics
        """
        pass
        
    @abstractmethod
    def calculate_spread(
        self, 
        data1: pd.DataFrame, 
        data2: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate the spread between two assets.
        
        Args:
            data1: Price data for first symbol
            data2: Price data for second symbol
            
        Returns:
            Time series of spread values
        """
        pass
        
    def calculate_hedge_ratio(
        self, 
        prices1: pd.Series, 
        prices2: pd.Series
    ) -> float:
        """
        Calculate the hedge ratio between two price series.
        
        Args:
            prices1: Price series for first symbol
            prices2: Price series for second symbol
            
        Returns:
            Hedge ratio (beta coefficient)
        """
        # Use simple linear regression to find hedge ratio
        from scipy import stats
        
        # Remove any NaN values
        valid_data = pd.concat([prices1, prices2], axis=1).dropna()
        if len(valid_data) < 2:
            return 1.0
            
        x = valid_data.iloc[:, 1].values  # prices2
        y = valid_data.iloc[:, 0].values  # prices1
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return slope
        
    def test_cointegration(
        self, 
        prices1: pd.Series, 
        prices2: pd.Series
    ) -> Dict[str, Any]:
        """
        Test for cointegration between two price series.
        
        Args:
            prices1: Price series for first symbol
            prices2: Price series for second symbol
            
        Returns:
            Dict with cointegration test results
        """
        from statsmodels.tsa.stattools import coint
        
        try:
            # Align the series
            aligned_data = pd.concat([prices1, prices2], axis=1).dropna()
            
            if len(aligned_data) < 10:
                return {
                    'is_cointegrated': False,
                    'p_value': 1.0,
                    'test_statistic': None,
                    'critical_values': None,
                    'error': 'Insufficient data'
                }
                
            y = aligned_data.iloc[:, 0].values
            x = aligned_data.iloc[:, 1].values
            
            # Perform Engle-Granger cointegration test
            test_stat, p_value, critical_values = coint(y, x)
            
            significance_level = self.pairs_config.cointegration_params.get(
                'significance_level', 0.05
            )
            
            is_cointegrated = p_value < significance_level
            
            return {
                'is_cointegrated': is_cointegrated,
                'p_value': p_value,
                'test_statistic': test_stat,
                'critical_values': critical_values,
                'hedge_ratio': self.calculate_hedge_ratio(prices1, prices2)
            }
            
        except Exception as e:
            return {
                'is_cointegrated': False,
                'p_value': 1.0,
                'test_statistic': None,
                'critical_values': None,
                'error': str(e)
            }
            
    def get_required_data(self) -> List[str]:
        """
        Get required data columns for pairs trading.
        
        Returns:
            List of required column names
        """
        base_cols = super().get_required_data()
        # Add spread and ratio columns that will be calculated
        return base_cols + ['spread', 'ratio', 'zscore']
        
    def prepare_pair_data(
        self, 
        data1: pd.DataFrame, 
        data2: pd.DataFrame,
        symbol1: str,
        symbol2: str
    ) -> pd.DataFrame:
        """
        Prepare synchronized data for pairs trading.
        
        Args:
            data1: Data for first symbol
            data2: Data for second symbol
            symbol1: Name of first symbol
            symbol2: Name of second symbol
            
        Returns:
            Combined DataFrame with pair data
        """
        # Align timestamps
        common_index = data1.index.intersection(data2.index)
        
        if len(common_index) == 0:
            raise ValueError("No common timestamps between the two datasets")
            
        # Create combined dataset
        combined_data = pd.DataFrame(index=common_index)
        
        # Add price data with symbol prefixes
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data1.columns:
                combined_data[f'{symbol1}_{col}'] = data1.loc[common_index, col]
            if col in data2.columns:
                combined_data[f'{symbol2}_{col}'] = data2.loc[common_index, col]
                
        # Calculate spread and ratio
        close1 = combined_data[f'{symbol1}_close']
        close2 = combined_data[f'{symbol2}_close']
        
        # Calculate hedge ratio if not provided
        if self.hedge_ratio is None:
            self.hedge_ratio = self.calculate_hedge_ratio(close1, close2)
            
        # Calculate spread (price1 - hedge_ratio * price2)
        combined_data['spread'] = close1 - self.hedge_ratio * close2
        combined_data['ratio'] = close1 / close2
        combined_data['log_ratio'] = np.log(combined_data['ratio'])
        
        return combined_data
