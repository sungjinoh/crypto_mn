"""
Core backtesting engine for cryptocurrency trading strategies.

This module provides the main backtesting engine that orchestrates
strategy execution, portfolio management, and result generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..strategies.base import BaseStrategy, Signal, SignalType
from ..data.manager import DataManager


@dataclass
class Trade:
    """
    Represents a completed trade.
    
    Attributes:
        entry_time: When the trade was entered
        exit_time: When the trade was exited
        symbol: Trading symbol or pair
        entry_price: Price at entry
        exit_price: Price at exit
        quantity: Position size
        side: Trade direction (1 for long, -1 for short)
        pnl: Profit/loss of the trade
        commission: Commission paid
        metadata: Additional trade information
    """
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    symbol: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: int  # 1 for long, -1 for short
    pnl: Optional[float] = None
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_time is None
        
    @property
    def duration(self) -> Optional[pd.Timedelta]:
        """Get trade duration."""
        if self.exit_time is None:
            return None
        return self.exit_time - self.entry_time


@dataclass
class Position:
    """
    Represents a current position.
    
    Attributes:
        symbol: Trading symbol
        quantity: Position size (positive for long, negative for short)
        entry_price: Average entry price
        market_value: Current market value
        unrealized_pnl: Unrealized profit/loss
        entry_time: When position was first established
    """
    symbol: str
    quantity: float
    entry_price: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    entry_time: Optional[pd.Timestamp] = None


@dataclass
class BacktestResults:
    """
    Container for backtest results and performance metrics.
    """
    # Raw data
    trades: List[Trade] = field(default_factory=list)
    portfolio_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    position_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Additional metrics
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    var_95: float = 0.0
    
    # Metadata
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    strategy_name: str = ""
    initial_capital: float = 0.0
    final_capital: float = 0.0


class PortfolioManager:
    """
    Manages portfolio state during backtesting.
    
    Handles position tracking, risk management, and PnL calculation.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0001
    ):
        """
        Initialize portfolio manager.
        
        Args:
            initial_capital: Starting capital
            commission_rate: Commission rate per trade
            slippage_rate: Slippage rate per trade
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.open_trades: Dict[str, Trade] = {}
        
        # History tracking
        self.portfolio_history = []
        self.trade_history = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def enter_position(
        self,
        signal: Signal,
        current_prices: Dict[str, float],
        timestamp: pd.Timestamp
    ) -> Optional[Trade]:
        """
        Enter a new position based on signal.
        
        Args:
            signal: Trading signal
            current_prices: Current market prices
            timestamp: Current timestamp
            
        Returns:
            Trade object if position was entered, None otherwise
        """
        symbol = signal.symbol
        
        # Check if this is a pairs trading signal
        if '-' in symbol and ('position1' in signal.metadata and 'position2' in signal.metadata):
            # Handle pairs trade directly
            return self._enter_pairs_position(signal, current_prices, timestamp)
        
        # Single asset trade
        if symbol not in current_prices:
            self.logger.warning(f"No price data for symbol {symbol}")
            return None
            
        current_price = current_prices[symbol]
        
        # Calculate position size based on signal and available capital
        position_value = self.cash * signal.position_size
        
        # Apply slippage
        execution_price = current_price * (1 + self.slippage_rate * np.sign(1))
        
        # Calculate quantity (considering direction from metadata)
        if 'position1' in signal.metadata and 'position2' in signal.metadata:
            # Pairs trade - handle both legs
            return self._enter_pairs_position(signal, current_prices, timestamp)
        else:
            # Single asset trade
            quantity = position_value / execution_price
            
            # Calculate commission
            commission = position_value * self.commission_rate
            
            if commission > self.cash:
                self.logger.warning("Insufficient funds for trade")
                return None
                
            # Create trade
            trade = Trade(
                entry_time=timestamp,
                exit_time=None,
                symbol=symbol,
                entry_price=execution_price,
                exit_price=None,
                quantity=quantity,
                side=1,  # Assume long for now
                commission=commission,
                metadata=signal.metadata.copy()
            )
            
            # Update portfolio
            self.cash -= (position_value + commission)
            
            # Update position
            if symbol in self.positions:
                # Add to existing position
                pos = self.positions[symbol]
                new_quantity = pos.quantity + quantity
                new_value = pos.quantity * pos.entry_price + position_value
                new_entry_price = new_value / new_quantity if new_quantity != 0 else 0
                
                pos.quantity = new_quantity
                pos.entry_price = new_entry_price
            else:
                # Create new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=execution_price,
                    entry_time=timestamp
                )
                
            # Track open trade
            self.open_trades[symbol] = trade
            
            return trade
            
    def _enter_pairs_position(
        self,
        signal: Signal,
        current_prices: Dict[str, float],
        timestamp: pd.Timestamp
    ) -> Optional[Trade]:
        """
        Enter a pairs trading position.
        
        Args:
            signal: Pairs trading signal
            current_prices: Current prices for both symbols
            timestamp: Current timestamp
            
        Returns:
            Trade object representing the pairs position
        """
        # Extract symbol names from signal
        symbols = signal.symbol.split('-')
        if len(symbols) != 2:
            self.logger.error(f"Invalid pairs symbol format: {signal.symbol}")
            return None
            
        symbol1, symbol2 = symbols
        
        # Get positions from signal metadata
        pos1 = signal.metadata.get('position1', 0)
        pos2 = signal.metadata.get('position2', 0)
        
        if symbol1 not in current_prices or symbol2 not in current_prices:
            self.logger.warning(f"Missing price data for pairs {symbol1}/{symbol2}")
            return None
            
        price1 = current_prices[symbol1]
        price2 = current_prices[symbol2]
        
        # Calculate position sizes for equal dollar exposure
        total_value = self.cash * signal.position_size
        half_value = total_value / 2
        
        # Apply slippage
        exec_price1 = price1 * (1 + self.slippage_rate * np.sign(pos1))
        exec_price2 = price2 * (1 + self.slippage_rate * np.sign(pos2))
        
        # Calculate quantities
        quantity1 = (half_value / exec_price1) * pos1
        quantity2 = (half_value / exec_price2) * pos2
        
        # Calculate total commission
        commission1 = abs(quantity1 * exec_price1) * self.commission_rate
        commission2 = abs(quantity2 * exec_price2) * self.commission_rate
        total_commission = commission1 + commission2
        
        if total_commission > self.cash:
            self.logger.warning("Insufficient funds for pairs trade")
            return None
            
        # Create pairs trade
        trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            symbol=signal.symbol,
            entry_price=(exec_price1 + exec_price2) / 2,  # Average price
            exit_price=None,
            quantity=total_value,  # Use total value as quantity for pairs
            side=1 if pos1 > 0 else -1,  # Direction based on symbol1
            commission=total_commission,
            metadata={
                **signal.metadata,
                'symbol1': symbol1,
                'symbol2': symbol2,
                'price1': exec_price1,
                'price2': exec_price2,
                'quantity1': quantity1,
                'quantity2': quantity2
            }
        )
        
        # Update cash
        self.cash -= (total_value + total_commission)
        
        # Update positions
        for symbol, quantity, price in [(symbol1, quantity1, exec_price1), 
                                       (symbol2, quantity2, exec_price2)]:
            if symbol in self.positions:
                pos = self.positions[symbol]
                new_quantity = pos.quantity + quantity
                if new_quantity != 0:
                    new_value = pos.quantity * pos.entry_price + quantity * price
                    pos.entry_price = new_value / new_quantity
                    pos.quantity = new_quantity
                else:
                    del self.positions[symbol]
            else:
                if quantity != 0:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        entry_price=price,
                        entry_time=timestamp
                    )
                    
        # Track open trade
        self.open_trades[signal.symbol] = trade
        
        return trade
        
    def exit_position(
        self,
        signal: Signal,
        current_prices: Dict[str, float],
        timestamp: pd.Timestamp
    ) -> Optional[Trade]:
        """
        Exit an existing position.
        
        Args:
            signal: Exit signal
            current_prices: Current market prices
            timestamp: Current timestamp
            
        Returns:
            Completed trade object if position was exited
        """
        symbol = signal.symbol
        
        if symbol not in self.open_trades:
            self.logger.warning(f"No open trade found for {symbol}")
            return None
            
        trade = self.open_trades[symbol]
        
        # Handle pairs exit
        if '-' in symbol and 'symbol1' in trade.metadata:
            return self._exit_pairs_position(trade, current_prices, timestamp)
        
        # Single asset exit
        if symbol not in current_prices:
            self.logger.warning(f"No price data for symbol {symbol}")
            return None
            
        current_price = current_prices[symbol]
        
        # Apply slippage (opposite direction for exit)
        execution_price = current_price * (1 - self.slippage_rate * np.sign(trade.side))
        
        # Calculate PnL
        pnl = trade.quantity * (execution_price - trade.entry_price) * trade.side
        
        # Calculate exit commission
        exit_value = abs(trade.quantity * execution_price)
        exit_commission = exit_value * self.commission_rate
        
        # Complete the trade
        trade.exit_time = timestamp
        trade.exit_price = execution_price
        trade.pnl = pnl - trade.commission - exit_commission
        
        # Update cash
        self.cash += (exit_value - exit_commission)
        
        # Update position
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.quantity -= trade.quantity
            if abs(pos.quantity) < 1e-8:  # Close to zero
                del self.positions[symbol]
                
        # Move to completed trades
        del self.open_trades[symbol]
        self.trade_history.append(trade)
        
        return trade
        
    def _exit_pairs_position(
        self,
        trade: Trade,
        current_prices: Dict[str, float],
        timestamp: pd.Timestamp
    ) -> Trade:
        """
        Exit a pairs trading position.
        
        Args:
            trade: Open pairs trade
            current_prices: Current market prices
            timestamp: Current timestamp
            
        Returns:
            Completed trade object
        """
        symbol1 = trade.metadata['symbol1']
        symbol2 = trade.metadata['symbol2']
        
        if symbol1 not in current_prices or symbol2 not in current_prices:
            self.logger.warning(f"Missing price data for pairs exit {symbol1}/{symbol2}")
            return trade
            
        price1 = current_prices[symbol1]
        price2 = current_prices[symbol2]
        
        # Get original quantities
        quantity1 = trade.metadata['quantity1']
        quantity2 = trade.metadata['quantity2']
        
        # Apply slippage for exit
        exec_price1 = price1 * (1 - self.slippage_rate * np.sign(quantity1))
        exec_price2 = price2 * (1 - self.slippage_rate * np.sign(quantity2))
        
        # Calculate PnL for each leg
        entry_price1 = trade.metadata['price1']
        entry_price2 = trade.metadata['price2']
        
        pnl1 = quantity1 * (exec_price1 - entry_price1)
        pnl2 = quantity2 * (exec_price2 - entry_price2)
        total_pnl = pnl1 + pnl2
        
        # Calculate exit commissions
        exit_value1 = abs(quantity1 * exec_price1)
        exit_value2 = abs(quantity2 * exec_price2)
        exit_commission = (exit_value1 + exit_value2) * self.commission_rate
        
        # Complete the trade
        trade.exit_time = timestamp
        trade.exit_price = (exec_price1 + exec_price2) / 2
        trade.pnl = total_pnl - trade.commission - exit_commission
        
        # Update cash
        self.cash += (exit_value1 + exit_value2 - exit_commission)
        
        # Update positions
        for symbol, quantity in [(symbol1, quantity1), (symbol2, quantity2)]:
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.quantity -= quantity
                if abs(pos.quantity) < 1e-8:
                    del self.positions[symbol]
                    
        # Move to completed trades
        del self.open_trades[trade.symbol]
        self.trade_history.append(trade)
        
        return trade
        
    def update_portfolio_value(
        self,
        current_prices: Dict[str, float],
        timestamp: pd.Timestamp
    ) -> None:
        """
        Update portfolio value with current market prices.
        
        Args:
            current_prices: Current market prices
            timestamp: Current timestamp
        """
        # Calculate market value of positions
        total_position_value = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                market_value = position.quantity * current_prices[symbol]
                position.market_value = market_value
                position.unrealized_pnl = market_value - (position.quantity * position.entry_price)
                total_position_value += market_value
                
        # Total portfolio value
        total_value = self.cash + total_position_value
        
        # Record portfolio snapshot
        self.portfolio_history.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'positions_value': total_position_value,
            'total_value': total_value,
            'num_positions': len(self.positions),
            'num_open_trades': len(self.open_trades)
        })
        
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get current portfolio summary.
        
        Returns:
            Dict with portfolio statistics
        """
        total_position_value = sum(pos.market_value for pos in self.positions.values())
        total_value = self.cash + total_position_value
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': total_position_value,
            'cash_percentage': self.cash / total_value if total_value > 0 else 0,
            'num_positions': len(self.positions),
            'num_open_trades': len(self.open_trades),
            'total_return': (total_value - self.initial_capital) / self.initial_capital,
            'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values())
        }


class BacktestEngine:
    """
    Main backtesting engine that orchestrates strategy execution and result generation.
    """
    
    def __init__(
        self,
        data_manager: DataManager,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0001
    ):
        """
        Initialize backtest engine.
        
        Args:
            data_manager: Data manager for market data
            initial_capital: Starting capital
            commission_rate: Commission rate per trade
            slippage_rate: Slippage rate per trade
        """
        self.data_manager = data_manager
        self.portfolio_manager = PortfolioManager(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate
        )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def run_backtest(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> BacktestResults:
        """
        Run a complete backtest of the strategy.
        
        Args:
            strategy: Trading strategy to test
            data: Market data for backtesting
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Backtest results with performance metrics
        """
        self.logger.info(f"Starting backtest for strategy: {strategy.name}")
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        if len(data) == 0:
            raise ValueError("No data available for the specified date range")
            
        # Initialize strategy
        strategy.initialize(data)
        
        # Reset portfolio state
        self.portfolio_manager = PortfolioManager(
            initial_capital=self.portfolio_manager.initial_capital,
            commission_rate=self.portfolio_manager.commission_rate,
            slippage_rate=self.portfolio_manager.slippage_rate
        )
        
        # Run backtest
        for timestamp, row in data.iterrows():
            self._process_timestamp(strategy, timestamp, row, data.loc[:timestamp])
            
        # Generate results
        results = self._generate_results(strategy, data)
        
        self.logger.info(f"Backtest completed. Total return: {results.total_return:.2%}")
        
        return results
        
    def _process_timestamp(
        self,
        strategy: BaseStrategy,
        timestamp: pd.Timestamp,
        row: pd.Series,
        historical_data: pd.DataFrame
    ) -> None:
        """
        Process a single timestamp during backtesting.
        
        Args:
            strategy: Trading strategy
            timestamp: Current timestamp
            row: Current market data row
            historical_data: Historical data up to current timestamp
        """
        # Extract current prices
        current_prices = self._extract_current_prices(row)
        
        # Update portfolio value
        self.portfolio_manager.update_portfolio_value(current_prices, timestamp)
        
        # Generate signals from strategy
        signals = strategy.generate_signals(historical_data)
        
        # Process each signal
        for signal in signals:
            if signal.signal_type == SignalType.ENTRY:
                trade = self.portfolio_manager.enter_position(
                    signal, current_prices, timestamp
                )
                if trade:
                    self.logger.debug(f"Entered position: {trade.symbol} at {trade.entry_price}")
                    
            elif signal.signal_type == SignalType.EXIT:
                trade = self.portfolio_manager.exit_position(
                    signal, current_prices, timestamp
                )
                if trade:
                    self.logger.debug(f"Exited position: {trade.symbol}, PnL: {trade.pnl:.2f}")
                    
    def _extract_current_prices(self, row: pd.Series) -> Dict[str, float]:
        """
        Extract current prices from market data row.
        
        Args:
            row: Market data row
            
        Returns:
            Dict mapping symbols to prices
        """
        prices = {}
        
        # Handle different data formats
        for col in row.index:
            if col.endswith('_close'):
                symbol = col.replace('_close', '')
                prices[symbol] = row[col]
            elif col == 'close':
                # Single asset data
                prices['asset'] = row[col]
                
        return prices
        
    def _generate_results(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame
    ) -> BacktestResults:
        """
        Generate comprehensive backtest results.
        
        Args:
            strategy: Trading strategy
            data: Market data used in backtest
            
        Returns:
            BacktestResults object with all metrics
        """
        # Convert portfolio history to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_manager.portfolio_history)
        if not portfolio_df.empty:
            portfolio_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        returns = portfolio_df['total_value'].pct_change().dropna() if not portfolio_df.empty else pd.Series()
        
        # Calculate basic metrics
        total_return = (
            (portfolio_df['total_value'].iloc[-1] / self.portfolio_manager.initial_capital - 1)
            if not portfolio_df.empty else 0.0
        )
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(returns, portfolio_df)
        
        # Analyze trades
        trade_stats = self._analyze_trades(self.portfolio_manager.trade_history)
        
        # Calculate final capital
        final_capital = (
            portfolio_df['total_value'].iloc[-1] 
            if not portfolio_df.empty else self.portfolio_manager.initial_capital
        )
        
        # Create results object
        results = BacktestResults(
            trades=self.portfolio_manager.trade_history.copy(),
            portfolio_history=portfolio_df,
            **metrics,
            **trade_stats,
            start_date=data.index[0] if len(data) > 0 else None,
            end_date=data.index[-1] if len(data) > 0 else None,
            strategy_name=strategy.name,
            initial_capital=self.portfolio_manager.initial_capital,
            final_capital=final_capital
        )
        
        return results
        
    def _calculate_performance_metrics(
        self,
        returns: pd.Series,
        portfolio_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from returns.
        
        Args:
            returns: Series of portfolio returns
            portfolio_df: Portfolio value history
            
        Returns:
            Dict with performance metrics
        """
        if len(returns) == 0:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0,
                'sortino_ratio': 0.0,
                'var_95': 0.0
            }
            
        # Basic metrics
        total_return = (portfolio_df['total_value'].iloc[-1] / portfolio_df['total_value'].iloc[0] - 1)
        
        # Annualized return (assuming daily returns)
        periods_per_year = 365.25 * 24 * 60  # Assuming minute data
        if len(returns) > 0:
            periods = len(returns)
            years = periods / periods_per_year
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        else:
            annualized_return = 0
            
        # Volatility
        volatility = returns.std() * np.sqrt(periods_per_year) if len(returns) > 1 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(periods_per_year) if len(negative_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Value at Risk (95%)
        var_95 = returns.quantile(0.05) if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'var_95': var_95
        }
        
    def _analyze_trades(self, trades: List[Trade]) -> Dict[str, Any]:
        """
        Analyze completed trades for statistics.
        
        Args:
            trades: List of completed trades
            
        Returns:
            Dict with trade statistics
        """
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
            
        # Separate winning and losing trades
        pnls = [trade.pnl for trade in trades if trade.pnl is not None]
        winning_pnls = [pnl for pnl in pnls if pnl > 0]
        losing_pnls = [pnl for pnl in pnls if pnl < 0]
        
        total_trades = len(trades)
        winning_trades = len(winning_pnls)
        losing_trades = len(losing_pnls)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        total_wins = sum(winning_pnls) if winning_pnls else 0
        total_losses = abs(sum(losing_pnls)) if losing_pnls else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Average wins/losses
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        
        # Largest wins/losses
        largest_win = max(winning_pnls) if winning_pnls else 0
        largest_loss = min(losing_pnls) if losing_pnls else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }
