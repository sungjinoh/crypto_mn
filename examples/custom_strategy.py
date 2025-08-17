"""
Custom Strategy Example
======================

This example demonstrates how to create custom trading strategies using the framework.
It shows how to:
1. Implement a custom strategy class
2. Define custom signal generation logic
3. Add risk management features
4. Test and validate the strategy

Requirements:
- Binance futures data in the correct format
- Framework installed (pip install -e .)
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto_backtesting import DataManager, BacktestEngine, PerformanceAnalyzer
from crypto_backtesting.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig


@dataclass
class CustomStrategyConfig:
    """Configuration for custom strategy."""
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    sma_short: int = 10
    sma_long: int = 20
    volume_threshold: float = 1.5
    max_position_size: float = 0.1
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04


class RSIMomentumStrategy(BaseStrategy):
    """
    Custom RSI Momentum Strategy
    
    This strategy combines RSI oversold/overbought conditions with moving average
    crossovers and volume confirmation for entry signals.
    
    Entry Rules:
    - Long: RSI < oversold AND price > SMA_long AND volume > threshold
    - Short: RSI > overbought AND price < SMA_long AND volume > threshold
    
    Exit Rules:
    - Stop loss: 2% adverse move
    - Take profit: 4% favorable move
    - RSI reversal: RSI crosses back to neutral zone
    """
    
    def __init__(self, symbol: str, config: CustomStrategyConfig):
        strategy_config = StrategyConfig(
            name="RSIMomentumStrategy",
            parameters={
                'rsi_period': config.rsi_period,
                'rsi_oversold': config.rsi_oversold,
                'rsi_overbought': config.rsi_overbought,
                'sma_short': config.sma_short,
                'sma_long': config.sma_long,
                'volume_threshold': config.volume_threshold,
                'max_position_size': config.max_position_size,
                'stop_loss_pct': config.stop_loss_pct,
                'take_profit_pct': config.take_profit_pct
            }
        )
        super().__init__(strategy_config)
        self.symbol = symbol
        self.config = config
        self.current_position = 0.0
        self.entry_price = 0.0
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_sma(self, prices, period):
        """Calculate Simple Moving Average."""
        return prices.rolling(window=period).mean()
    
    def initialize(self, data):
        """Initialize strategy with technical indicators."""
        print(f"Initializing RSI Momentum Strategy for {self.symbol}")
        
        # Calculate technical indicators
        data['rsi'] = self.calculate_rsi(data['close'], self.config.rsi_period)
        data['sma_short'] = self.calculate_sma(data['close'], self.config.sma_short)
        data['sma_long'] = self.calculate_sma(data['close'], self.config.sma_long)
        
        # Calculate volume moving average for threshold
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        print(f"Calculated indicators for {len(data)} data points")
        return data
    
    def generate_signals(self, data) -> List[Signal]:
        """Generate trading signals based on strategy rules."""
        signals = []
        
        for i in range(max(self.config.rsi_period, self.config.sma_long), len(data)):
            current_data = data.iloc[i]
            timestamp = data.index[i]
            
            # Get current values
            rsi = current_data['rsi']
            price = current_data['close']
            sma_long = current_data['sma_long']
            volume_ratio = current_data['volume_ratio']
            
            # Skip if we don't have valid indicators
            if any(pd.isna(val) for val in [rsi, sma_long, volume_ratio]):
                continue
            
            # Check for entry signals
            if self.current_position == 0:
                # Long entry condition
                if (rsi < self.config.rsi_oversold and 
                    price > sma_long and 
                    volume_ratio > self.config.volume_threshold):
                    
                    signal = Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.LONG_ENTRY,
                        symbol=self.symbol,
                        price=price,
                        quantity=self.config.max_position_size,
                        confidence=min(1.0, (self.config.rsi_oversold - rsi) / 10 + 0.5)
                    )
                    signals.append(signal)
                    self.current_position = signal.quantity
                    self.entry_price = price
                    
                # Short entry condition
                elif (rsi > self.config.rsi_overbought and 
                      price < sma_long and 
                      volume_ratio > self.config.volume_threshold):
                    
                    signal = Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.SHORT_ENTRY,
                        symbol=self.symbol,
                        price=price,
                        quantity=self.config.max_position_size,
                        confidence=min(1.0, (rsi - self.config.rsi_overbought) / 10 + 0.5)
                    )
                    signals.append(signal)
                    self.current_position = -signal.quantity
                    self.entry_price = price
            
            # Check for exit signals
            elif self.current_position != 0:
                should_exit = False
                exit_reason = ""
                
                # Stop loss check
                if self.current_position > 0:  # Long position
                    pnl_pct = (price - self.entry_price) / self.entry_price
                    if pnl_pct <= -self.config.stop_loss_pct:
                        should_exit = True
                        exit_reason = "Stop Loss"
                    elif pnl_pct >= self.config.take_profit_pct:
                        should_exit = True
                        exit_reason = "Take Profit"
                    elif rsi > 50:  # RSI reversal
                        should_exit = True
                        exit_reason = "RSI Reversal"
                        
                else:  # Short position
                    pnl_pct = (self.entry_price - price) / self.entry_price
                    if pnl_pct <= -self.config.stop_loss_pct:
                        should_exit = True
                        exit_reason = "Stop Loss"
                    elif pnl_pct >= self.config.take_profit_pct:
                        should_exit = True
                        exit_reason = "Take Profit"
                    elif rsi < 50:  # RSI reversal
                        should_exit = True
                        exit_reason = "RSI Reversal"
                
                if should_exit:
                    signal_type = SignalType.LONG_EXIT if self.current_position > 0 else SignalType.SHORT_EXIT
                    signal = Signal(
                        timestamp=timestamp,
                        signal_type=signal_type,
                        symbol=self.symbol,
                        price=price,
                        quantity=abs(self.current_position),
                        confidence=0.8,
                        metadata={'exit_reason': exit_reason}
                    )
                    signals.append(signal)
                    self.current_position = 0.0
                    self.entry_price = 0.0
        
        return signals
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        checks = [
            self.config.rsi_period > 0,
            0 < self.config.rsi_oversold < self.config.rsi_overbought < 100,
            self.config.sma_short < self.config.sma_long,
            self.config.volume_threshold > 0,
            0 < self.config.max_position_size <= 1,
            self.config.stop_loss_pct > 0,
            self.config.take_profit_pct > 0
        ]
        return all(checks)


class BollingerBandMeanReversion(BaseStrategy):
    """
    Custom Bollinger Band Mean Reversion Strategy
    
    This strategy uses Bollinger Bands to identify overbought/oversold conditions
    and trades mean reversion back to the middle band.
    """
    
    def __init__(self, symbol: str, bb_period: int = 20, bb_std: float = 2.0, 
                 position_size: float = 0.1):
        strategy_config = StrategyConfig(
            name="BollingerBandMeanReversion",
            parameters={
                'bb_period': bb_period,
                'bb_std': bb_std,
                'position_size': position_size
            }
        )
        super().__init__(strategy_config)
        self.symbol = symbol
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.position_size = position_size
        self.current_position = 0.0
    
    def calculate_bollinger_bands(self, prices, period, std_mult):
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_mult)
        lower_band = sma - (std * std_mult)
        
        return sma, upper_band, lower_band
    
    def initialize(self, data):
        """Initialize strategy with Bollinger Bands."""
        middle, upper, lower = self.calculate_bollinger_bands(
            data['close'], self.bb_period, self.bb_std
        )
        
        data['bb_middle'] = middle
        data['bb_upper'] = upper
        data['bb_lower'] = lower
        data['bb_position'] = (data['close'] - lower) / (upper - lower)
        
        return data
    
    def generate_signals(self, data) -> List[Signal]:
        """Generate mean reversion signals."""
        signals = []
        
        for i in range(self.bb_period, len(data)):
            current_data = data.iloc[i]
            timestamp = data.index[i]
            
            price = current_data['close']
            bb_upper = current_data['bb_upper']
            bb_lower = current_data['bb_lower']
            bb_middle = current_data['bb_middle']
            
            if any(pd.isna(val) for val in [bb_upper, bb_lower, bb_middle]):
                continue
            
            # Entry signals
            if self.current_position == 0:
                # Long entry when price touches lower band
                if price <= bb_lower:
                    signal = Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.LONG_ENTRY,
                        symbol=self.symbol,
                        price=price,
                        quantity=self.position_size,
                        confidence=0.8
                    )
                    signals.append(signal)
                    self.current_position = self.position_size
                
                # Short entry when price touches upper band
                elif price >= bb_upper:
                    signal = Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.SHORT_ENTRY,
                        symbol=self.symbol,
                        price=price,
                        quantity=self.position_size,
                        confidence=0.8
                    )
                    signals.append(signal)
                    self.current_position = -self.position_size
            
            # Exit signals - when price returns to middle band
            elif self.current_position != 0:
                if ((self.current_position > 0 and price >= bb_middle) or
                    (self.current_position < 0 and price <= bb_middle)):
                    
                    signal_type = SignalType.LONG_EXIT if self.current_position > 0 else SignalType.SHORT_EXIT
                    signal = Signal(
                        timestamp=timestamp,
                        signal_type=signal_type,
                        symbol=self.symbol,
                        price=price,
                        quantity=abs(self.current_position),
                        confidence=0.7
                    )
                    signals.append(signal)
                    self.current_position = 0.0
        
        return signals
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        return (self.bb_period > 0 and 
                self.bb_std > 0 and 
                0 < self.position_size <= 1)


def test_custom_strategy():
    """Test the custom RSI momentum strategy."""
    
    print("🎯 Testing Custom RSI Momentum Strategy")
    print("=" * 45)
    
    # Setup data manager
    data_path = Path(__file__).parent.parent / "binance_futures_data"
    data_manager = DataManager(data_path=str(data_path))
    
    # Load data
    print("\n📊 Loading data for BTCUSDT...")
    try:
        btc_data = data_manager.get_klines_data('BTCUSDT', 2024, [4, 5, 6])
        print(f"Loaded {len(btc_data)} candles")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create custom strategy
    config = CustomStrategyConfig(
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        sma_short=10,
        sma_long=20,
        volume_threshold=1.2,
        max_position_size=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )
    
    strategy = RSIMomentumStrategy('BTCUSDT', config)
    
    # Validate parameters
    if not strategy.validate_parameters():
        print("❌ Strategy parameters are invalid")
        return
    
    print("✅ Strategy parameters validated")
    
    # Initialize strategy
    prepared_data = strategy.initialize(btc_data)
    
    # Generate signals
    print("\n⚡ Generating trading signals...")
    signals = strategy.generate_signals(prepared_data)
    print(f"Generated {len(signals)} signals")
    
    # Analyze signals
    if signals:
        entry_signals = [s for s in signals if 'ENTRY' in s.signal_type.value]
        exit_signals = [s for s in signals if 'EXIT' in s.signal_type.value]
        long_signals = [s for s in signals if 'LONG' in s.signal_type.value]
        short_signals = [s for s in signals if 'SHORT' in s.signal_type.value]
        
        print(f"Entry signals: {len(entry_signals)}")
        print(f"Exit signals: {len(exit_signals)}")
        print(f"Long signals: {len(long_signals)}")
        print(f"Short signals: {len(short_signals)}")
        
        # Show first few signals
        print("\n📋 First 5 signals:")
        for i, signal in enumerate(signals[:5]):
            print(f"  {i+1}. {signal.timestamp}: {signal.signal_type.value} at ${signal.price:.2f}")
    
    # Run backtest
    print("\n🚀 Running backtest...")
    engine = BacktestEngine(
        data_manager,
        initial_capital=100000,
        commission_rate=0.001
    )
    
    try:
        results = engine.run_backtest(strategy, prepared_data)
        print("✅ Backtest completed!")
        
        # Display results
        print(f"\nTotal Return: {results.total_return:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Number of Trades: {len(results.trades)}")
        print(f"Win Rate: {results.win_rate:.1%}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return None


def test_bollinger_strategy():
    """Test the Bollinger Band mean reversion strategy."""
    
    print("\n\n🎯 Testing Bollinger Band Mean Reversion Strategy")
    print("=" * 55)
    
    # Setup
    data_path = Path(__file__).parent.parent / "binance_futures_data"
    data_manager = DataManager(data_path=str(data_path))
    
    # Load data
    try:
        eth_data = data_manager.get_klines_data('ETHUSDT', 2024, [4, 5])
        print(f"Loaded {len(eth_data)} candles for ETHUSDT")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create strategy
    strategy = BollingerBandMeanReversion(
        symbol='ETHUSDT',
        bb_period=20,
        bb_std=2.0,
        position_size=0.1
    )
    
    # Validate and run
    if strategy.validate_parameters():
        print("✅ Strategy parameters validated")
        
        prepared_data = strategy.initialize(eth_data)
        signals = strategy.generate_signals(prepared_data)
        
        print(f"Generated {len(signals)} signals")
        
        # Run backtest
        engine = BacktestEngine(data_manager, initial_capital=100000)
        results = engine.run_backtest(strategy, prepared_data)
        
        print(f"Bollinger Strategy Results:")
        print(f"  Total Return: {results.total_return:.2%}")
        print(f"  Sharpe Ratio: {results.sharpe_ratio:.3f}")
        print(f"  Max Drawdown: {results.max_drawdown:.2%}")
        
        return results


def main():
    """Run custom strategy examples."""
    
    print("🚀 Cryptocurrency Backtesting Framework - Custom Strategies")
    print("=" * 65)
    
    # Test RSI momentum strategy
    rsi_results = test_custom_strategy()
    
    # Test Bollinger Band strategy
    bb_results = test_bollinger_strategy()
    
    # Compare strategies
    if rsi_results and bb_results:
        print("\n📊 Strategy Comparison:")
        print("-" * 30)
        print(f"RSI Strategy: {rsi_results.total_return:.2%} return, {rsi_results.sharpe_ratio:.3f} Sharpe")
        print(f"BB Strategy:  {bb_results.total_return:.2%} return, {bb_results.sharpe_ratio:.3f} Sharpe")
    
    print("\n🎉 Custom strategy examples completed!")
    print("\nKey Principles for Custom Strategies:")
    print("1. Always validate parameters before running")
    print("2. Implement proper risk management (stop loss, position sizing)")
    print("3. Use multiple indicators for confirmation")
    print("4. Test on different market conditions")
    print("5. Consider transaction costs in strategy design")


if __name__ == "__main__":
    main()
