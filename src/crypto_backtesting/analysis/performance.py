"""
Analysis and reporting framework for backtesting results.

This module provides comprehensive analysis tools for evaluating
strategy performance and generating detailed reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

from ..backtesting.engine import BacktestResults, Trade

warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for backtesting results.
    
    Provides detailed metrics, statistical analysis, and risk assessment
    for trading strategy evaluation.
    """
    
    def __init__(self, results: BacktestResults):
        """
        Initialize performance analyzer.
        
        Args:
            results: Backtest results to analyze
        """
        self.results = results
        self.returns = self._calculate_returns()
        
    def _calculate_returns(self) -> pd.Series:
        """Calculate portfolio returns from results."""
        if self.results.portfolio_history.empty:
            return pd.Series()
            
        return self.results.portfolio_history['total_value'].pct_change().dropna()
        
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dict with all performance metrics and analysis
        """
        report = {
            'summary_metrics': self._calculate_summary_metrics(),
            'risk_metrics': self._calculate_risk_metrics(),
            'trade_analysis': self._analyze_trades(),
            'drawdown_analysis': self._analyze_drawdowns(),
            'monthly_returns': self._calculate_monthly_returns(),
            'rolling_metrics': self._calculate_rolling_metrics(),
            'benchmark_comparison': self._benchmark_analysis()
        }
        
        return report
        
    def _calculate_summary_metrics(self) -> Dict[str, float]:
        """Calculate summary performance metrics."""
        if len(self.returns) == 0:
            return {}
            
        total_return = self.results.total_return
        annualized_return = self.results.annualized_return
        volatility = self.results.volatility
        sharpe_ratio = self.results.sharpe_ratio
        
        # Additional metrics
        skewness = self.returns.skew()
        kurtosis = self.returns.kurtosis()
        
        # Information ratio (assuming benchmark is 0)
        excess_returns = self.returns - 0  # No benchmark
        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': self.results.calmar_ratio,
            'sortino_ratio': self.results.sortino_ratio,
            'information_ratio': information_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': self.results.var_95,
            'cvar_95': self.returns[self.returns <= self.results.var_95].mean() if len(self.returns) > 0 else 0
        }
        
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk-specific metrics."""
        if len(self.returns) == 0:
            return {}
            
        # Tail risk metrics
        var_99 = self.returns.quantile(0.01)
        cvar_99 = self.returns[self.returns <= var_99].mean()
        
        # Maximum consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for ret in self.returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
                
        # Beta (assuming market return is normal distribution)
        market_returns = np.random.normal(0.001, 0.02, len(self.returns))  # Placeholder
        beta = np.cov(self.returns, market_returns)[0, 1] / np.var(market_returns)
        
        return {
            'var_99': var_99,
            'cvar_99': cvar_99,
            'max_consecutive_losses': max_consecutive_losses,
            'beta': beta,
            'downside_capture': self._calculate_downside_capture(),
            'upside_capture': self._calculate_upside_capture()
        }
        
    def _calculate_downside_capture(self) -> float:
        """Calculate downside capture ratio."""
        # Placeholder implementation
        negative_returns = self.returns[self.returns < 0]
        if len(negative_returns) == 0:
            return 0.0
        return negative_returns.mean() / -0.01  # Relative to -1% benchmark
        
    def _calculate_upside_capture(self) -> float:
        """Calculate upside capture ratio."""
        # Placeholder implementation
        positive_returns = self.returns[self.returns > 0]
        if len(positive_returns) == 0:
            return 0.0
        return positive_returns.mean() / 0.01  # Relative to 1% benchmark
        
    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze trade-level statistics."""
        trades = self.results.trades
        
        if not trades:
            return {}
            
        # Trade duration analysis
        durations = []
        for trade in trades:
            if trade.duration:
                durations.append(trade.duration.total_seconds() / 3600)  # Hours
                
        # PnL analysis
        pnls = [trade.pnl for trade in trades if trade.pnl is not None]
        
        # Win/loss streaks
        win_streak, loss_streak = self._calculate_streaks(trades)
        
        return {
            'total_trades': len(trades),
            'profitable_trades': len([t for t in trades if t.pnl and t.pnl > 0]),
            'losing_trades': len([t for t in trades if t.pnl and t.pnl < 0]),
            'win_rate': self.results.win_rate,
            'profit_factor': self.results.profit_factor,
            'avg_trade_pnl': np.mean(pnls) if pnls else 0,
            'median_trade_pnl': np.median(pnls) if pnls else 0,
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0,
            'avg_duration_hours': np.mean(durations) if durations else 0,
            'median_duration_hours': np.median(durations) if durations else 0,
            'max_win_streak': win_streak,
            'max_loss_streak': loss_streak
        }
        
    def _calculate_streaks(self, trades: List[Trade]) -> Tuple[int, int]:
        """Calculate maximum winning and losing streaks."""
        if not trades:
            return 0, 0
            
        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in trades:
            if trade.pnl and trade.pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif trade.pnl and trade.pnl < 0:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
                
        return max_win_streak, max_loss_streak
        
    def _analyze_drawdowns(self) -> Dict[str, Any]:
        """Analyze drawdown characteristics."""
        if self.results.portfolio_history.empty:
            return {}
            
        portfolio_values = self.results.portfolio_history['total_value']
        
        # Calculate drawdown series
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []
        
        start_idx = None
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                end_idx = i - 1
                dd_period = {
                    'start': portfolio_values.index[start_idx],
                    'end': portfolio_values.index[end_idx],
                    'duration': end_idx - start_idx + 1,
                    'depth': drawdown.iloc[start_idx:end_idx+1].min()
                }
                drawdown_periods.append(dd_period)
                start_idx = None
                
        # Handle ongoing drawdown
        if start_idx is not None:
            dd_period = {
                'start': portfolio_values.index[start_idx],
                'end': portfolio_values.index[-1],
                'duration': len(portfolio_values) - start_idx,
                'depth': drawdown.iloc[start_idx:].min()
            }
            drawdown_periods.append(dd_period)
            
        # Calculate statistics
        if drawdown_periods:
            avg_duration = np.mean([dd['duration'] for dd in drawdown_periods])
            avg_depth = np.mean([dd['depth'] for dd in drawdown_periods])
            max_duration = max([dd['duration'] for dd in drawdown_periods])
        else:
            avg_duration = avg_depth = max_duration = 0
            
        return {
            'max_drawdown': self.results.max_drawdown,
            'num_drawdown_periods': len(drawdown_periods),
            'avg_drawdown_duration': avg_duration,
            'max_drawdown_duration': max_duration,
            'avg_drawdown_depth': avg_depth,
            'current_drawdown': drawdown.iloc[-1] if len(drawdown) > 0 else 0,
            'drawdown_periods': drawdown_periods
        }
        
    def _calculate_monthly_returns(self) -> pd.DataFrame:
        """Calculate monthly return statistics."""
        if self.results.portfolio_history.empty:
            return pd.DataFrame()
            
        portfolio_values = self.results.portfolio_history['total_value']
        monthly_values = portfolio_values.resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna()
        
        # Create summary DataFrame
        monthly_stats = pd.DataFrame({
            'Monthly Return': monthly_returns,
            'Cumulative Return': (1 + monthly_returns).cumprod() - 1
        })
        
        return monthly_stats
        
    def _calculate_rolling_metrics(self, window_days: int = 30) -> pd.DataFrame:
        """Calculate rolling performance metrics."""
        if len(self.returns) < window_days:
            return pd.DataFrame()
            
        rolling_data = pd.DataFrame(index=self.returns.index)
        
        # Rolling returns
        rolling_data['Rolling Return'] = (1 + self.returns).rolling(window_days).apply(
            lambda x: x.prod() - 1
        )
        
        # Rolling Sharpe ratio
        rolling_data['Rolling Sharpe'] = (
            self.returns.rolling(window_days).mean() / 
            self.returns.rolling(window_days).std()
        ) * np.sqrt(252)  # Annualized
        
        # Rolling volatility
        rolling_data['Rolling Volatility'] = (
            self.returns.rolling(window_days).std() * np.sqrt(252)
        )
        
        return rolling_data
        
    def _benchmark_analysis(self) -> Dict[str, Any]:
        """Compare performance against benchmark."""
        # For now, use a simple buy-and-hold benchmark
        # In practice, this would use actual benchmark data
        
        if self.results.portfolio_history.empty:
            return {}
            
        # Simulate benchmark returns (placeholder)
        benchmark_return = 0.1  # 10% annual return
        
        excess_return = self.results.annualized_return - benchmark_return
        
        return {
            'benchmark_return': benchmark_return,
            'excess_return': excess_return,
            'tracking_error': self.returns.std() * np.sqrt(252),
            'information_ratio': excess_return / (self.returns.std() * np.sqrt(252)) if self.returns.std() > 0 else 0
        }


class ReportGenerator:
    """
    Generate comprehensive reports and visualizations for backtesting results.
    """
    
    def __init__(self, results: BacktestResults, output_dir: str = "reports"):
        """
        Initialize report generator.
        
        Args:
            results: Backtest results to report on
            output_dir: Directory to save reports
        """
        self.results = results
        self.analyzer = PerformanceAnalyzer(results)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_full_report(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Generate complete performance report with visualizations.
        
        Args:
            save_plots: Whether to save plot files
            
        Returns:
            Complete performance analysis
        """
        # Generate analysis
        performance_report = self.analyzer.generate_performance_report()
        
        # Create visualizations
        if save_plots:
            self._create_performance_plots()
            self._create_trade_analysis_plots()
            self._create_risk_analysis_plots()
            
        # Save text report
        self._save_text_report(performance_report)
        
        return performance_report
        
    def _create_performance_plots(self) -> None:
        """Create performance-related plots."""
        if self.results.portfolio_history.empty:
            return
            
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # Portfolio value over time
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value
        self.results.portfolio_history['total_value'].plot(
            ax=axes[0, 0], title='Portfolio Value Over Time'
        )
        axes[0, 0].set_ylabel('Portfolio Value')
        
        # Drawdown
        portfolio_values = self.results.portfolio_history['total_value']
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        
        drawdown.plot(ax=axes[0, 1], title='Drawdown Over Time', color='red')
        axes[0, 1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].set_ylabel('Drawdown')
        
        # Returns distribution
        returns = self.analyzer.returns
        if len(returns) > 0:
            returns.hist(bins=50, ax=axes[1, 0], alpha=0.7)
            axes[1, 0].set_title('Returns Distribution')
            axes[1, 0].set_xlabel('Returns')
            axes[1, 0].set_ylabel('Frequency')
            
        # Cumulative returns
        if len(returns) > 0:
            (1 + returns).cumprod().plot(ax=axes[1, 1], title='Cumulative Returns')
            axes[1, 1].set_ylabel('Cumulative Return')
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_trade_analysis_plots(self) -> None:
        """Create trade analysis plots."""
        if not self.results.trades:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Trade PnL
        pnls = [trade.pnl for trade in self.results.trades if trade.pnl is not None]
        if pnls:
            axes[0, 0].hist(pnls, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Trade PnL Distribution')
            axes[0, 0].set_xlabel('PnL')
            axes[0, 0].set_ylabel('Frequency')
            
        # Trade duration
        durations = []
        for trade in self.results.trades:
            if trade.duration:
                durations.append(trade.duration.total_seconds() / 3600)
                
        if durations:
            axes[0, 1].hist(durations, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Trade Duration Distribution')
            axes[0, 1].set_xlabel('Duration (hours)')
            axes[0, 1].set_ylabel('Frequency')
            
        # Cumulative PnL
        if pnls:
            cumulative_pnl = np.cumsum(pnls)
            axes[1, 0].plot(range(len(cumulative_pnl)), cumulative_pnl)
            axes[1, 0].set_title('Cumulative PnL by Trade')
            axes[1, 0].set_xlabel('Trade Number')
            axes[1, 0].set_ylabel('Cumulative PnL')
            
        # Win/Loss by month
        trade_dates = [trade.entry_time for trade in self.results.trades if trade.entry_time]
        if trade_dates and pnls:
            monthly_pnl = pd.DataFrame({
                'date': trade_dates,
                'pnl': pnls
            })
            monthly_pnl['month'] = pd.to_datetime(monthly_pnl['date']).dt.to_period('M')
            monthly_summary = monthly_pnl.groupby('month')['pnl'].sum()
            
            colors = ['green' if x > 0 else 'red' for x in monthly_summary.values]
            monthly_summary.plot(kind='bar', ax=axes[1, 1], color=colors)
            axes[1, 1].set_title('Monthly PnL')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('PnL')
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'trade_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_risk_analysis_plots(self) -> None:
        """Create risk analysis plots."""
        returns = self.analyzer.returns
        
        if len(returns) == 0:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rolling Sharpe ratio
        rolling_sharpe = (
            returns.rolling(30).mean() / returns.rolling(30).std()
        ) * np.sqrt(252)
        
        rolling_sharpe.plot(ax=axes[0, 0], title='Rolling Sharpe Ratio (30-day)')
        axes[0, 0].axhline(y=1, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_ylabel('Sharpe Ratio')
        
        # Rolling volatility
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)
        rolling_vol.plot(ax=axes[0, 1], title='Rolling Volatility (30-day)')
        axes[0, 1].set_ylabel('Volatility')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
        
        # Autocorrelation
        from statsmodels.tsa.stattools import acf
        
        lags = min(40, len(returns) // 4)
        autocorr = acf(returns, nlags=lags, fft=True)
        
        axes[1, 1].bar(range(len(autocorr)), autocorr)
        axes[1, 1].axhline(y=0, color='black', linewidth=0.8)
        axes[1, 1].set_title('Autocorrelation of Returns')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _save_text_report(self, performance_report: Dict[str, Any]) -> None:
        """Save detailed text report."""
        report_path = self.output_dir / 'performance_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("BACKTESTING PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Strategy: {self.results.strategy_name}\n")
            f.write(f"Period: {self.results.start_date} to {self.results.end_date}\n")
            f.write(f"Initial Capital: ${self.results.initial_capital:,.2f}\n\n")
            
            # Summary metrics
            f.write("SUMMARY METRICS\n")
            f.write("-" * 20 + "\n")
            if 'summary_metrics' in performance_report:
                for key, value in performance_report['summary_metrics'].items():
                    if isinstance(value, float):
                        if 'ratio' in key.lower() or 'return' in key.lower():
                            f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
                        else:
                            f.write(f"{key.replace('_', ' ').title()}: {value:.6f}\n")
                    else:
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
            
            # Trade analysis
            f.write("TRADE ANALYSIS\n")
            f.write("-" * 20 + "\n")
            if 'trade_analysis' in performance_report:
                for key, value in performance_report['trade_analysis'].items():
                    if isinstance(value, float):
                        f.write(f"{key.replace('_', ' ').title()}: {value:.2f}\n")
                    else:
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
            
            # Risk metrics
            f.write("RISK METRICS\n")
            f.write("-" * 20 + "\n")
            if 'risk_metrics' in performance_report:
                for key, value in performance_report['risk_metrics'].items():
                    if isinstance(value, float):
                        f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
                    else:
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")


def compare_strategies(
    results_list: List[BacktestResults],
    strategy_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare performance of multiple strategies.
    
    Args:
        results_list: List of backtest results to compare
        strategy_names: Optional list of strategy names
        
    Returns:
        DataFrame with comparison metrics
    """
    if not results_list:
        return pd.DataFrame()
        
    if strategy_names is None:
        strategy_names = [f"Strategy {i+1}" for i in range(len(results_list))]
        
    comparison_data = []
    
    for i, results in enumerate(results_list):
        analyzer = PerformanceAnalyzer(results)
        report = analyzer.generate_performance_report()
        
        row_data = {
            'Strategy': strategy_names[i],
            'Total Return': results.total_return,
            'Annualized Return': results.annualized_return,
            'Volatility': results.volatility,
            'Sharpe Ratio': results.sharpe_ratio,
            'Max Drawdown': results.max_drawdown,
            'Calmar Ratio': results.calmar_ratio,
            'Win Rate': results.win_rate,
            'Profit Factor': results.profit_factor,
            'Total Trades': results.total_trades
        }
        
        comparison_data.append(row_data)
        
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.set_index('Strategy', inplace=True)
    
    return comparison_df
