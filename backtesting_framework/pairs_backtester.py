"""
Pairs Trading Backtesting Framework
Supports mean reversion statistical arbitrage strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS

warnings.filterwarnings("ignore")


@dataclass
class Trade:
    """Represents a single trade in the pairs strategy"""

    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    symbol1: str
    symbol2: str
    position1: float  # +1 for long, -1 for short
    position2: float  # +1 for long, -1 for short
    entry_price1: float
    entry_price2: float
    exit_price1: Optional[float]
    exit_price2: Optional[float]
    entry_spread: float
    exit_spread: Optional[float]
    entry_zscore: float
    exit_zscore: Optional[float]
    pnl: Optional[float] = None
    is_closed: bool = False


@dataclass
class BacktestResults:
    """Container for backtest results"""

    trades: List[Trade]
    portfolio_value: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    metrics: Dict[str, float]
    pair_stats: Dict[str, Any]


class PairsStrategy(ABC):
    """Abstract base class for pairs trading strategies"""

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on the data"""
        pass

    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """Return list of required indicators for this strategy"""
        pass


class PairsBacktester:
    """Main backtesting engine for pairs trading strategies"""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        position_size: float = 0.5,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.position_size = position_size
        self.reset()

    def reset(self):
        """Reset backtester state"""
        self.trades = []
        self.current_trade = None
        self.portfolio_value = []
        self.positions = []
        self.cash = self.initial_capital

    def prepare_pair_data(
        self,
        symbol1_data: pd.DataFrame,
        symbol2_data: pd.DataFrame,
        symbol1: str,
        symbol2: str,
    ) -> pd.DataFrame:
        """
        Prepare synchronized pair data with indicators and determine transformation method

        Args:
            symbol1_data: OHLCV data for first symbol
            symbol2_data: OHLCV data for second symbol
            symbol1: Symbol name for first asset
            symbol2: Symbol name for second asset

        Returns:
            DataFrame with synchronized data, indicators, and transformation flags
        """
        # Merge data on timestamp
        data = pd.merge(
            symbol1_data[["timestamp", "close"]].rename(
                columns={"close": f"{symbol1}_close"}
            ),
            symbol2_data[["timestamp", "close"]].rename(
                columns={"close": f"{symbol2}_close"}
            ),
            on="timestamp",
            how="inner",
        )

        # Convert timestamp to datetime
        data["datetime"] = pd.to_datetime(data["timestamp"], unit="ms")
        data.set_index("datetime", inplace=True)

        # Store original prices for later use
        data["price1"] = data[f"{symbol1}_close"]
        data["price2"] = data[f"{symbol2}_close"]

        # Determine if we should use log prices based on price level differences
        price1_mean = data["price1"].mean()
        price2_mean = data["price2"].mean()
        price_ratio = max(price1_mean, price2_mean) / min(price1_mean, price2_mean)
        use_log_prices = price_ratio > 10  # Use log if prices differ by more than 10x

        # Store transformation metadata in the dataframe for later use
        data.attrs["use_log_prices"] = use_log_prices
        data.attrs["price_ratio"] = price_ratio
        data.attrs["price1_mean"] = price1_mean
        data.attrs["price2_mean"] = price2_mean

        # Calculate basic spread and ratio (for backward compatibility)
        data["spread"] = data["price1"] - data["price2"]
        data["ratio"] = data["price1"] / data["price2"]
        data["log_ratio"] = np.log(data["ratio"])

        # Add transformation info as columns for easy access
        data["use_log_prices"] = use_log_prices
        data["price_level_ratio"] = price_ratio

        print(f"📊 Price analysis for {symbol1}-{symbol2}:")
        print(f"   {symbol1} mean price: ${price1_mean:,.2f}")
        print(f"   {symbol2} mean price: ${price2_mean:,.2f}")
        print(f"   Price ratio: {price_ratio:.2f}")
        print(f"   Use log transformation: {use_log_prices}")

        return data

    def calculate_spread_and_signals(
        self,
        data: pd.DataFrame,
        hedge_ratio: float,
        intercept: float = 0,
        use_log_prices: bool = None,
        lookback_period: int = 60,
    ) -> pd.DataFrame:
        """
        Calculate spread and z-score for trading signals using consistent methodology

        Args:
            data: DataFrame with price1 and price2 columns
            hedge_ratio: Hedge ratio from cointegration analysis
            intercept: Intercept from cointegration analysis
            use_log_prices: Whether to use log transformation (if None, use from data.attrs)
            lookback_period: Rolling window for z-score calculation
        """
        # Use flag from data preparation if not explicitly provided
        if use_log_prices is None:
            use_log_prices = data.attrs.get("use_log_prices", False)

        if use_log_prices:
            # Log spread - consistent with cointegration analysis
            spread = (
                np.log(data["price1"])
                - hedge_ratio * np.log(data["price2"])
                - intercept
            )
            print(
                f"   Using log spread: ln(P1) - {hedge_ratio:.4f} × ln(P2) - {intercept:.4f}"
            )
        else:
            # Regular spread - consistent with cointegration analysis
            spread = data["price1"] - hedge_ratio * data["price2"] - intercept
            print(
                f"   Using normal spread: P1 - {hedge_ratio:.4f} × P2 - {intercept:.4f}"
            )

        # Calculate z-score using rolling statistics
        spread_mean = spread.rolling(window=lookback_period).mean()
        spread_std = spread.rolling(window=lookback_period).std()
        z_score = (spread - spread_mean) / spread_std

        # Add to dataframe
        data["consistent_spread"] = spread
        data["consistent_zscore"] = z_score

        # Store the parameters used for reference
        data.attrs["hedge_ratio"] = hedge_ratio
        data.attrs["intercept"] = intercept
        data.attrs["lookback_period"] = lookback_period

        return data

    def add_indicators(
        self, data: pd.DataFrame, lookback_period: int = 60
    ) -> pd.DataFrame:
        """Add technical indicators for pairs trading"""
        # Rolling statistics for spread
        data["spread_mean"] = data["spread"].rolling(window=lookback_period).mean()
        data["spread_std"] = data["spread"].rolling(window=lookback_period).std()

        # Z-score
        data["zscore"] = (data["spread"] - data["spread_mean"]) / data["spread_std"]

        # Rolling statistics for log ratio
        data["log_ratio_mean"] = (
            data["log_ratio"].rolling(window=lookback_period).mean()
        )
        data["log_ratio_std"] = data["log_ratio"].rolling(window=lookback_period).std()
        data["log_ratio_zscore"] = (data["log_ratio"] - data["log_ratio_mean"]) / data[
            "log_ratio_std"
        ]

        # Add additional indicators useful for different timeframes
        # Bollinger Bands for spread
        data["spread_upper_band"] = data["spread_mean"] + (2 * data["spread_std"])
        data["spread_lower_band"] = data["spread_mean"] - (2 * data["spread_std"])

        # Spread momentum (useful for higher timeframes)
        data["spread_momentum"] = data["spread"].pct_change(
            periods=min(5, lookback_period // 4)
        )

        return data

    def check_cointegration(
        self,
        symbol1_prices: pd.Series,
        symbol2_prices: pd.Series,
        significance_level: float = 0.05,
        use_log_prices: bool = None,
    ) -> Dict[str, Any]:
        """
        Test for cointegration between two price series

        Args:
            symbol1_prices: First symbol price series
            symbol2_prices: Second symbol price series
            significance_level: P-value threshold for cointegration
            use_log_prices: Whether to use log transformation (if None, auto-determine)

        Returns:
            Dictionary with cointegration test results
        """
        # Ensure we have aligned data
        aligned_data = pd.DataFrame(
            {"symbol1": symbol1_prices, "symbol2": symbol2_prices}
        ).dropna()

        if len(aligned_data) < 50:  # Need sufficient data
            return {
                "is_cointegrated": False,
                "p_value": 1.0,
                "critical_values": None,
                "test_statistic": None,
                "hedge_ratio": None,
            }

        # Cointegration test
        try:
            test_stat, p_value, critical_values = coint(
                aligned_data["symbol1"], aligned_data["symbol2"]
            )

            # Determine if we should use log prices (if not provided)
            if use_log_prices is None:
                price1_mean = aligned_data["symbol1"].mean()
                price2_mean = aligned_data["symbol2"].mean()
                price_ratio = max(price1_mean, price2_mean) / min(
                    price1_mean, price2_mean
                )
                use_log_prices = price_ratio > 10
            else:
                price_ratio = max(
                    aligned_data["symbol1"].mean(), aligned_data["symbol2"].mean()
                ) / min(aligned_data["symbol1"].mean(), aligned_data["symbol2"].mean())

            # Transform prices if needed
            if use_log_prices:
                y = np.log(aligned_data["symbol1"])
                x = np.log(aligned_data["symbol2"])
            else:
                y = aligned_data["symbol1"]
                x = aligned_data["symbol2"]

            # Calculate hedge ratio using OLS WITH CONSTANT
            import statsmodels.api as sm

            X = sm.add_constant(x)  # Add constant term
            model = OLS(y, X).fit()

            # Extract parameters
            intercept = model.params[0]  # Constant term (alpha)
            hedge_ratio = model.params[1]  # Slope (beta) - this is the hedge ratio

            # If using log prices, calculate position ratio for actual trading
            if use_log_prices:
                # Convert log hedge ratio to position ratio
                avg_price_ratio = (
                    aligned_data["symbol1"].mean() / aligned_data["symbol2"].mean()
                )
                position_ratio = hedge_ratio * avg_price_ratio
            else:
                position_ratio = hedge_ratio

            is_cointegrated = p_value < significance_level

            # Extract only serializable regression statistics
            regression_stats = {
                "r_squared": float(model.rsquared),
                "r_squared_adj": float(model.rsquared_adj),
                "aic": float(model.aic),
                "bic": float(model.bic),
                "f_statistic": (
                    float(model.fvalue) if hasattr(model, "fvalue") else None
                ),
                "f_pvalue": (
                    float(model.f_pvalue) if hasattr(model, "f_pvalue") else None
                ),
                "durbin_watson": (
                    float(model.durbin_watson())
                    if hasattr(model, "durbin_watson")
                    else None
                ),
                "params": [float(p) for p in model.params],
                "pvalues": [float(p) for p in model.pvalues],
                "std_errors": (
                    [float(se) for se in model.bse] if hasattr(model, "bse") else None
                ),
            }

            return {
                "is_cointegrated": is_cointegrated,
                "p_value": p_value,
                "critical_values": critical_values,
                "test_statistic": test_stat,
                "hedge_ratio": hedge_ratio,
                "position_ratio": position_ratio,
                "intercept": intercept,
                "use_log_prices": use_log_prices,
                "price_ratio": price_ratio,
                "regression_stats": regression_stats,
            }

        except Exception as e:
            print(f"Error in cointegration test: {e}")
            return {
                "is_cointegrated": False,
                "p_value": 1.0,
                "critical_values": None,
                "test_statistic": None,
                "hedge_ratio": None,
            }

    def calculate_position_size(
        self, price1: float, price2: float
    ) -> Tuple[float, float]:
        """Calculate position sizes for the pair"""
        total_value = self.cash + self.get_portfolio_value()
        position_value = (
            total_value * self.position_size / 2
        )  # Split between two assets

        size1 = position_value / price1
        size2 = position_value / price2

        return size1, size2

    def get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        if self.current_trade is None:
            return 0.0

        # This would need current prices to calculate unrealized PnL
        # For now, return 0 as placeholder
        return 0.0

    def execute_trade(
        self,
        data_row: pd.Series,
        signal: int,
        position1: int,
        position2: int,
        symbol1: str,
        symbol2: str,
    ) -> None:
        """Execute a trade based on the signal"""
        current_time = data_row.name
        price1 = data_row[f"{symbol1}_close"]
        price2 = data_row[f"{symbol2}_close"]
        spread = data_row["spread"]
        zscore = data_row["zscore"]

        if signal == 1 and self.current_trade is None:  # Entry signal
            size1, size2 = self.calculate_position_size(price1, price2)

            # Apply position direction
            size1 *= position1
            size2 *= position2

            # Calculate transaction costs
            cost1 = abs(size1 * price1 * self.transaction_cost)
            cost2 = abs(size2 * price2 * self.transaction_cost)
            total_cost = cost1 + cost2

            # Create new trade
            self.current_trade = Trade(
                entry_time=current_time,
                exit_time=None,
                symbol1=symbol1,
                symbol2=symbol2,
                position1=position1,
                position2=position2,
                entry_price1=price1,
                entry_price2=price2,
                exit_price1=None,
                exit_price2=None,
                entry_spread=spread,
                exit_spread=None,
                entry_zscore=zscore,
                exit_zscore=None,
            )

            # Update cash
            self.cash -= total_cost

        elif signal == -1 and self.current_trade is not None:  # Exit signal
            # Close current trade
            self.current_trade.exit_time = current_time
            self.current_trade.exit_price1 = price1
            self.current_trade.exit_price2 = price2
            self.current_trade.exit_spread = spread
            self.current_trade.exit_zscore = zscore

            # Calculate PnL
            pnl1 = self.current_trade.position1 * (
                price1 - self.current_trade.entry_price1
            )
            pnl2 = self.current_trade.position2 * (
                price2 - self.current_trade.entry_price2
            )

            # Calculate position sizes (assuming same as entry for simplicity)
            size1, size2 = self.calculate_position_size(
                self.current_trade.entry_price1, self.current_trade.entry_price2
            )

            total_pnl = (pnl1 * size1) + (pnl2 * size2)

            # Transaction costs for exit
            cost1 = abs(size1 * price1 * self.transaction_cost)
            cost2 = abs(size2 * price2 * self.transaction_cost)
            total_cost = cost1 + cost2

            self.current_trade.pnl = total_pnl - total_cost
            self.current_trade.is_closed = True

            # Update cash
            self.cash += total_pnl - total_cost

            # Add to completed trades
            self.trades.append(self.current_trade)
            self.current_trade = None

    def run_backtest(
        self,
        symbol1_data: pd.DataFrame,
        symbol2_data: pd.DataFrame,
        symbol1: str,
        symbol2: str,
        strategy: PairsStrategy,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResults:
        """
        Run the pairs trading backtest

        Args:
            symbol1_data: OHLCV data for first symbol
            symbol2_data: OHLCV data for second symbol
            symbol1: Symbol name for first asset
            symbol2: Symbol name for second asset
            strategy: Trading strategy instance
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)

        Returns:
            BacktestResults object with all results
        """
        self.reset()

        # Prepare data
        data = self.prepare_pair_data(symbol1_data, symbol2_data, symbol1, symbol2)

        # Filter by date range if specified
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        # Add indicators
        data = self.add_indicators(data, strategy.lookback_period)

        # Check cointegration using the transformation flag from prepared data
        use_log_flag = data.attrs.get("use_log_prices", None)
        cointegration_results = self.check_cointegration(
            data[f"{symbol1}_close"],
            data[f"{symbol2}_close"],
            use_log_prices=use_log_flag,
        )

        # If cointegrated, calculate consistent spread and signals
        if cointegration_results.get("is_cointegrated", False):
            # Use the hedge ratio and other parameters from cointegration analysis
            data = self.calculate_spread_and_signals(
                data,
                hedge_ratio=cointegration_results["hedge_ratio"],
                intercept=cointegration_results.get("intercept", 0),
                use_log_prices=cointegration_results.get("use_log_prices", False),
                lookback_period=getattr(strategy, "lookback_period", 60),
            )

        # Generate signals
        signals = strategy.generate_signals(data)

        # Combine data and signals
        combined_data = pd.concat([data, signals], axis=1)

        # Run backtest
        portfolio_values = []
        positions_data = []

        for idx, row in combined_data.iterrows():
            if pd.isna(row["zscore"]):  # Skip rows without indicators
                continue

            # Execute trade if signal present
            if row["signal"] != 0:
                self.execute_trade(
                    row,
                    row["signal"],
                    row["position1"],
                    row["position2"],
                    symbol1,
                    symbol2,
                )

            # Record portfolio value
            current_portfolio_value = self.cash + self.get_portfolio_value()
            portfolio_values.append(
                {
                    "datetime": idx,
                    "portfolio_value": current_portfolio_value,
                    "cash": self.cash,
                }
            )

            # Record positions
            if self.current_trade:
                positions_data.append(
                    {
                        "datetime": idx,
                        "symbol1_position": self.current_trade.position1,
                        "symbol2_position": self.current_trade.position2,
                        "in_trade": True,
                    }
                )
            else:
                positions_data.append(
                    {
                        "datetime": idx,
                        "symbol1_position": 0,
                        "symbol2_position": 0,
                        "in_trade": False,
                    }
                )

        # Create results DataFrames
        portfolio_df = pd.DataFrame(portfolio_values).set_index("datetime")
        positions_df = pd.DataFrame(positions_data).set_index("datetime")

        # Calculate returns
        returns = portfolio_df["portfolio_value"].pct_change().dropna()

        # Calculate metrics
        metrics = self.calculate_metrics(portfolio_df, returns)

        # Prepare pair statistics
        pair_stats = {
            "cointegration": cointegration_results,
            "correlation": data[f"{symbol1}_close"].corr(data[f"{symbol2}_close"]),
            "spread_stats": {
                "mean": data["spread"].mean(),
                "std": data["spread"].std(),
                "min": data["spread"].min(),
                "max": data["spread"].max(),
            },
        }

        return BacktestResults(
            trades=self.trades,
            portfolio_value=portfolio_df["portfolio_value"],
            returns=returns,
            positions=positions_df,
            metrics=metrics,
            pair_stats=pair_stats,
        )

    def calculate_metrics(
        self, portfolio_df: pd.DataFrame, returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics"""
        if len(returns) == 0:
            return {}

        total_return = (
            portfolio_df["portfolio_value"].iloc[-1] / self.initial_capital
        ) - 1

        # Check if quantstats is available
        try:
            import quantstats as qs

            has_quantstats = True
        except ImportError:
            has_quantstats = False

        # Use quantstats for comprehensive metrics if available
        if has_quantstats:
            try:
                metrics = {
                    "Total Return": total_return,
                    "Annualized Return": qs.stats.cagr(returns),
                    "Volatility": qs.stats.volatility(returns),
                    "Sharpe Ratio": qs.stats.sharpe(returns),
                    "Max Drawdown": qs.stats.max_drawdown(returns),
                    "Calmar Ratio": qs.stats.calmar(returns),
                    "Win Rate": (
                        len([t for t in self.trades if t.pnl > 0]) / len(self.trades)
                        if self.trades
                        else 0
                    ),
                    "Total Trades": len(self.trades),
                    "Avg Trade PnL": (
                        np.mean([t.pnl for t in self.trades]) if self.trades else 0
                    ),
                }
            except:
                has_quantstats = False  # Fallback if quantstats fails

        if not has_quantstats:
            # Fallback to basic metrics if quantstats fails
            metrics = {
                "Total Return": total_return,
                "Volatility": returns.std() * np.sqrt(252),
                "Sharpe Ratio": (
                    returns.mean() / returns.std() * np.sqrt(252)
                    if returns.std() > 0
                    else 0
                ),
                "Max Drawdown": (returns.cumsum().cummax() - returns.cumsum()).max(),
                "Win Rate": (
                    len([t for t in self.trades if t.pnl > 0]) / len(self.trades)
                    if self.trades
                    else 0
                ),
                "Total Trades": len(self.trades),
                "Avg Trade PnL": (
                    np.mean([t.pnl for t in self.trades]) if self.trades else 0
                ),
            }

        return metrics


def plot_backtest_results(results: BacktestResults, symbol1: str, symbol2: str):
    """Plot comprehensive backtest results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Portfolio value
    axes[0, 0].plot(results.portfolio_value.index, results.portfolio_value.values)
    axes[0, 0].set_title("Portfolio Value Over Time")
    axes[0, 0].set_ylabel("Portfolio Value")
    axes[0, 0].grid(True)

    # Returns distribution
    axes[0, 1].hist(results.returns.dropna(), bins=50, alpha=0.7)
    axes[0, 1].set_title("Returns Distribution")
    axes[0, 1].set_xlabel("Returns")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True)

    # Cumulative returns
    cumulative_returns = (1 + results.returns).cumprod()
    axes[1, 0].plot(cumulative_returns.index, cumulative_returns.values)
    axes[1, 0].set_title("Cumulative Returns")
    axes[1, 0].set_ylabel("Cumulative Returns")
    axes[1, 0].grid(True)

    # Trade PnL
    if results.trades:
        trade_pnls = [t.pnl for t in results.trades if t.pnl is not None]
        axes[1, 1].bar(range(len(trade_pnls)), trade_pnls)
        axes[1, 1].set_title("Individual Trade PnL")
        axes[1, 1].set_xlabel("Trade Number")
        axes[1, 1].set_ylabel("PnL")
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # Print metrics
    print(f"\n=== Backtest Results for {symbol1}/{symbol2} ===")
    for metric, value in results.metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    print(f"\n=== Pair Statistics ===")
    print(f"Cointegrated: {results.pair_stats['cointegration']['is_cointegrated']}")
    print(
        f"Cointegration p-value: {results.pair_stats['cointegration']['p_value']:.4f}"
    )
    print(f"Correlation: {results.pair_stats['correlation']:.4f}")
