#!/usr/bin/env python3
"""
Trade Details Analysis Script
Analyze detailed trading transactions saved by the backtesting system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class TradeAnalyzer:
    """
    Analyze detailed trade transaction logs
    """

    def __init__(self, trades_dir: str = "fixed_params_trades"):
        self.trades_dir = Path(trades_dir)
        self.all_trades = None

    def load_all_trade_files(self) -> pd.DataFrame:
        """
        Load all trade CSV files from the directory

        Returns:
            Combined DataFrame with all trades
        """
        if not self.trades_dir.exists():
            print(f"❌ Trade directory not found: {self.trades_dir}")
            return pd.DataFrame()

        # Find all trade CSV files
        trade_files = list(self.trades_dir.glob("trades_*.csv"))

        if not trade_files:
            print(f"❌ No trade files found in {self.trades_dir}")
            return pd.DataFrame()

        print(f"📁 Found {len(trade_files)} trade files:")

        all_trades = []
        for file_path in trade_files:
            try:
                df = pd.read_csv(file_path)
                df["source_file"] = file_path.name
                all_trades.append(df)
                print(f"   • {file_path.name}: {len(df)} trades")
            except Exception as e:
                print(f"   ❌ Error loading {file_path.name}: {e}")

        if all_trades:
            combined_df = pd.concat(all_trades, ignore_index=True)

            # Convert datetime columns
            datetime_cols = ["entry_time", "exit_time"]
            for col in datetime_cols:
                if col in combined_df.columns:
                    combined_df[col] = pd.to_datetime(combined_df[col])

            self.all_trades = combined_df
            print(
                f"✅ Loaded {len(combined_df)} total trades from {len(trade_files)} pairs"
            )
            return combined_df
        else:
            return pd.DataFrame()

    def analyze_overall_performance(self) -> dict:
        """
        Analyze overall trading performance across all pairs

        Returns:
            Dictionary with performance metrics
        """
        if self.all_trades is None or self.all_trades.empty:
            print("❌ No trade data loaded")
            return {}

        df = self.all_trades

        print("=" * 80)
        print("OVERALL TRADING PERFORMANCE ANALYSIS")
        print("=" * 80)

        # Basic statistics
        total_trades = len(df)
        unique_pairs = df["pair"].nunique()
        profitable_trades = (df["pnl"] > 0).sum()
        losing_trades = (df["pnl"] < 0).sum()
        breakeven_trades = (df["pnl"] == 0).sum()

        win_rate = profitable_trades / total_trades if total_trades > 0 else 0

        print(f"📊 Trade Summary:")
        print(f"   • Total Trades: {total_trades:,}")
        print(f"   • Unique Pairs: {unique_pairs}")
        print(f"   • Profitable: {profitable_trades:,} ({win_rate:.1%})")
        print(f"   • Losing: {losing_trades:,} ({losing_trades/total_trades:.1%})")
        print(f"   • Breakeven: {breakeven_trades:,}")

        # PnL Analysis
        total_pnl = df["pnl"].sum()
        avg_pnl = df["pnl"].mean()
        median_pnl = df["pnl"].median()
        best_trade = df["pnl"].max()
        worst_trade = df["pnl"].min()

        profitable_pnl = df[df["pnl"] > 0]["pnl"].sum()
        losing_pnl = df[df["pnl"] < 0]["pnl"].sum()

        print(f"\n💰 PnL Analysis:")
        print(f"   • Total PnL: ${total_pnl:,.2f}")
        print(f"   • Average PnL per Trade: ${avg_pnl:.2f}")
        print(f"   • Median PnL per Trade: ${median_pnl:.2f}")
        print(f"   • Best Trade: ${best_trade:.2f}")
        print(f"   • Worst Trade: ${worst_trade:.2f}")
        print(f"   • Gross Profit: ${profitable_pnl:.2f}")
        print(f"   • Gross Loss: ${losing_pnl:.2f}")
        print(
            f"   • Profit Factor: {abs(profitable_pnl/losing_pnl):.2f}"
            if losing_pnl != 0
            else "   • Profit Factor: ∞"
        )

        # Duration Analysis
        avg_duration = df["duration_hours"].mean()
        median_duration = df["duration_hours"].median()
        max_duration = df["duration_hours"].max()
        min_duration = df["duration_hours"].min()

        print(f"\n⏱️ Duration Analysis:")
        print(f"   • Average Duration: {avg_duration:.1f} hours")
        print(f"   • Median Duration: {median_duration:.1f} hours")
        print(f"   • Longest Trade: {max_duration:.1f} hours")
        print(f"   • Shortest Trade: {min_duration:.1f} hours")

        # Return Analysis
        avg_return = df["return_pct"].mean()
        median_return = df["return_pct"].median()
        best_return = df["return_pct"].max()
        worst_return = df["return_pct"].min()

        print(f"\n📈 Return Analysis:")
        print(f"   • Average Return: {avg_return:.2f}%")
        print(f"   • Median Return: {median_return:.2f}%")
        print(f"   • Best Return: {best_return:.2f}%")
        print(f"   • Worst Return: {worst_return:.2f}%")

        # Funding Rate Analysis (if available)
        if "total_funding_cost" in df.columns:
            total_funding_cost = df["total_funding_cost"].sum()
            avg_funding_cost = df["total_funding_cost"].mean()
            trades_with_funding = (df["funding_payments_count"] > 0).sum()

            print(f"\n💰 Funding Rate Analysis:")
            print(f"   • Total Funding Cost: ${total_funding_cost:.2f}")
            print(f"   • Average Funding Cost per Trade: ${avg_funding_cost:.2f}")
            print(
                f"   • Trades with Funding Data: {trades_with_funding}/{total_trades}"
            )

            if "net_pnl" in df.columns:
                net_total_pnl = df["net_pnl"].sum()
                net_profitable = (
                    (df["net_is_profitable"] == True).sum()
                    if "net_is_profitable" in df.columns
                    else 0
                )

                print(f"   • Net PnL (after funding): ${net_total_pnl:.2f}")
                print(
                    f"   • Net Profitable Trades: {net_profitable}/{total_trades} ({net_profitable/total_trades:.1%})"
                )

                if total_pnl != 0:
                    funding_impact = (total_funding_cost / abs(total_pnl)) * 100
                    print(f"   • Funding Impact: {funding_impact:.1f}% of gross PnL")

                    if funding_impact > 10:
                        print(
                            f"   • ⚠️ High funding impact - consider shorter holding periods"
                        )
                    elif funding_impact < -5:
                        print(
                            f"   • ✅ Net funding income - strategy benefits from funding rates"
                        )
                    else:
                        print(
                            f"   • ✅ Low funding impact - strategy is funding-efficient"
                        )

        print(f"\n📊 Additional Analysis:")
        print(f"   • Average Return: {avg_return:.2f}%")
        print(f"   • Median Return: {median_return:.2f}%")
        print(f"   • Best Return: {best_return:.2f}%")
        print(f"   • Worst Return: {worst_return:.2f}%")

        # Z-score Analysis
        avg_entry_zscore = abs(df["entry_zscore"]).mean()
        avg_exit_zscore = abs(df["exit_zscore"]).mean()

        print(f"\n📊 Z-score Analysis:")
        print(f"   • Average Entry |Z-score|: {avg_entry_zscore:.2f}")
        print(f"   • Average Exit |Z-score|: {avg_exit_zscore:.2f}")

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "profit_factor": (
                abs(profitable_pnl / losing_pnl) if losing_pnl != 0 else float("inf")
            ),
            "avg_duration": avg_duration,
            "avg_return": avg_return,
        }

    def analyze_by_pair(self) -> pd.DataFrame:
        """
        Analyze performance by trading pair

        Returns:
            DataFrame with pair-level statistics
        """
        if self.all_trades is None or self.all_trades.empty:
            return pd.DataFrame()

        print(f"\n{'=' * 80}")
        print("PAIR-LEVEL PERFORMANCE ANALYSIS")
        print("=" * 80)

        pair_stats = []

        for pair in self.all_trades["pair"].unique():
            pair_trades = self.all_trades[self.all_trades["pair"] == pair]

            total_trades = len(pair_trades)
            profitable_trades = (pair_trades["pnl"] > 0).sum()
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0

            total_pnl = pair_trades["pnl"].sum()
            avg_pnl = pair_trades["pnl"].mean()
            avg_duration = pair_trades["duration_hours"].mean()
            avg_return = pair_trades["return_pct"].mean()

            best_trade = pair_trades["pnl"].max()
            worst_trade = pair_trades["pnl"].min()

            pair_stats.append(
                {
                    "pair": pair,
                    "total_trades": total_trades,
                    "profitable_trades": profitable_trades,
                    "win_rate": win_rate,
                    "total_pnl": total_pnl,
                    "avg_pnl": avg_pnl,
                    "avg_duration_hours": avg_duration,
                    "avg_return_pct": avg_return,
                    "best_trade": best_trade,
                    "worst_trade": worst_trade,
                }
            )

        pair_df = pd.DataFrame(pair_stats)
        pair_df = pair_df.sort_values("total_pnl", ascending=False)

        print(f"📊 Top 10 Pairs by Total PnL:")
        print(
            f"{'Pair':<15} {'Trades':<7} {'Win%':<6} {'Total PnL':<10} {'Avg PnL':<8} {'Avg Ret%':<8}"
        )
        print("-" * 70)

        for _, row in pair_df.head(10).iterrows():
            print(
                f"{row['pair']:<15} {row['total_trades']:<7} {row['win_rate']:<6.1%} "
                f"${row['total_pnl']:<9.2f} ${row['avg_pnl']:<7.2f} {row['avg_return_pct']:<7.2f}%"
            )

        return pair_df

    def analyze_trade_timing(self):
        """
        Analyze trade timing patterns
        """
        if self.all_trades is None or self.all_trades.empty:
            return

        print(f"\n{'=' * 80}")
        print("TRADE TIMING ANALYSIS")
        print("=" * 80)

        df = self.all_trades.copy()

        # Extract time components
        df["entry_hour"] = df["entry_time"].dt.hour
        df["entry_day"] = df["entry_time"].dt.day_name()
        df["entry_month"] = df["entry_time"].dt.month

        # Hourly analysis
        hourly_stats = (
            df.groupby("entry_hour")
            .agg(
                {
                    "pnl": ["count", "mean", "sum"],
                    "return_pct": "mean",
                    "is_profitable": "mean",
                }
            )
            .round(3)
        )

        print(f"⏰ Best Trading Hours (by average PnL):")
        hourly_pnl = df.groupby("entry_hour")["pnl"].mean().sort_values(ascending=False)
        for hour, avg_pnl in hourly_pnl.head(5).items():
            trade_count = df[df["entry_hour"] == hour].shape[0]
            print(
                f"   • {hour:02d}:00 - Avg PnL: ${avg_pnl:.2f} ({trade_count} trades)"
            )

        # Daily analysis
        daily_stats = (
            df.groupby("entry_day")["pnl"].agg(["count", "mean", "sum"]).round(2)
        )
        print(f"\n📅 Best Trading Days (by average PnL):")
        daily_pnl = df.groupby("entry_day")["pnl"].mean().sort_values(ascending=False)
        for day, avg_pnl in daily_pnl.items():
            trade_count = df[df["entry_day"] == day].shape[0]
            print(f"   • {day}: Avg PnL: ${avg_pnl:.2f} ({trade_count} trades)")

    def create_visualizations(self, save_plots: bool = True):
        """
        Create comprehensive visualizations of trade data

        Args:
            save_plots: Whether to save plots to files
        """
        if self.all_trades is None or self.all_trades.empty:
            print("❌ No trade data for visualization")
            return

        print(f"\n{'=' * 80}")
        print("CREATING TRADE VISUALIZATIONS")
        print("=" * 80)

        df = self.all_trades

        # Set up the plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Trading Performance Analysis", fontsize=16, fontweight="bold")

        # 1. PnL Distribution
        axes[0, 0].hist(df["pnl"], bins=50, alpha=0.7, edgecolor="black")
        axes[0, 0].axvline(
            df["pnl"].mean(),
            color="red",
            linestyle="--",
            label=f'Mean: ${df["pnl"].mean():.2f}',
        )
        axes[0, 0].axvline(0, color="black", linestyle="-", alpha=0.5)
        axes[0, 0].set_title("PnL Distribution")
        axes[0, 0].set_xlabel("PnL ($)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Return Distribution
        axes[0, 1].hist(
            df["return_pct"], bins=50, alpha=0.7, edgecolor="black", color="green"
        )
        axes[0, 1].axvline(
            df["return_pct"].mean(),
            color="red",
            linestyle="--",
            label=f'Mean: {df["return_pct"].mean():.2f}%',
        )
        axes[0, 1].axvline(0, color="black", linestyle="-", alpha=0.5)
        axes[0, 1].set_title("Return Distribution")
        axes[0, 1].set_xlabel("Return (%)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Duration vs PnL
        axes[0, 2].scatter(df["duration_hours"], df["pnl"], alpha=0.6)
        axes[0, 2].axhline(0, color="black", linestyle="-", alpha=0.5)
        axes[0, 2].set_title("Trade Duration vs PnL")
        axes[0, 2].set_xlabel("Duration (hours)")
        axes[0, 2].set_ylabel("PnL ($)")
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Entry Z-score vs PnL
        axes[1, 0].scatter(df["entry_zscore"], df["pnl"], alpha=0.6, color="orange")
        axes[1, 0].axhline(0, color="black", linestyle="-", alpha=0.5)
        axes[1, 0].axvline(0, color="black", linestyle="-", alpha=0.5)
        axes[1, 0].set_title("Entry Z-score vs PnL")
        axes[1, 0].set_xlabel("Entry Z-score")
        axes[1, 0].set_ylabel("PnL ($)")
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Hourly Performance
        hourly_pnl = df.groupby(df["entry_time"].dt.hour)["pnl"].mean()
        axes[1, 1].bar(hourly_pnl.index, hourly_pnl.values, alpha=0.7, color="purple")
        axes[1, 1].axhline(0, color="black", linestyle="-", alpha=0.5)
        axes[1, 1].set_title("Average PnL by Hour")
        axes[1, 1].set_xlabel("Hour of Day")
        axes[1, 1].set_ylabel("Average PnL ($)")
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Cumulative PnL
        df_sorted = df.sort_values("entry_time")
        cumulative_pnl = df_sorted["pnl"].cumsum()
        axes[1, 2].plot(
            range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color="darkblue"
        )
        axes[1, 2].axhline(0, color="black", linestyle="-", alpha=0.5)
        axes[1, 2].set_title("Cumulative PnL Over Time")
        axes[1, 2].set_xlabel("Trade Number")
        axes[1, 2].set_ylabel("Cumulative PnL ($)")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_analysis_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"📊 Visualization saved: {filename}")

        plt.show()

    def export_detailed_report(self, filename: str = None):
        """
        Export a detailed trade analysis report

        Args:
            filename: Output filename (optional)
        """
        if self.all_trades is None or self.all_trades.empty:
            print("❌ No trade data for report")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detailed_trade_report_{timestamp}.csv"

        # Add additional analysis columns
        df = self.all_trades.copy()

        # Trade quality metrics
        df["profit_per_hour"] = df["pnl"] / df["duration_hours"]
        df["zscore_efficiency"] = df["pnl"] / abs(df["entry_zscore"])
        df["spread_capture"] = df["spread_change"] / df["entry_spread"] * 100

        # Risk metrics
        df["max_adverse_excursion"] = df.apply(
            lambda row: (
                min(0, row["worst_trade"]) if "worst_trade" in df.columns else 0
            ),
            axis=1,
        )

        # Export enhanced dataset
        df.to_csv(filename, index=False)
        print(f"📄 Detailed report exported: {filename}")
        print(f"   • {len(df)} trades with {len(df.columns)} data columns")
        print(f"   • Includes performance metrics, timing analysis, and risk measures")

        return filename


def main():
    """
    Main function to run comprehensive trade analysis
    """
    print("🔍 COMPREHENSIVE TRADE ANALYSIS")
    print("=" * 80)

    # Initialize analyzer
    analyzer = TradeAnalyzer("fixed_params_trades")

    # Load all trade data
    trades_df = analyzer.load_all_trade_files()

    if trades_df.empty:
        print("❌ No trade data found. Run backtests first to generate trade logs.")
        return

    # Run comprehensive analysis
    print(f"\n🚀 Starting comprehensive analysis...")

    # 1. Overall performance analysis
    overall_stats = analyzer.analyze_overall_performance()

    # 2. Pair-level analysis
    pair_stats = analyzer.analyze_by_pair()

    # 3. Timing analysis
    analyzer.analyze_trade_timing()

    # 4. Create visualizations
    analyzer.create_visualizations(save_plots=True)

    # 5. Export detailed report
    report_file = analyzer.export_detailed_report()

    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"   • Overall statistics calculated")
    print(f"   • Pair-level performance analyzed")
    print(f"   • Timing patterns identified")
    print(f"   • Visualizations created and saved")
    print(f"   • Detailed report exported: {report_file}")

    # Show key insights
    if overall_stats:
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • Total Trades: {overall_stats['total_trades']:,}")
        print(f"   • Win Rate: {overall_stats['win_rate']:.1%}")
        print(f"   • Total PnL: ${overall_stats['total_pnl']:,.2f}")
        print(f"   • Average Trade Duration: {overall_stats['avg_duration']:.1f} hours")

        if overall_stats["total_pnl"] > 0:
            print(f"   • ✅ Overall strategy is profitable!")
        else:
            print(f"   • ⚠️ Overall strategy needs improvement")


if __name__ == "__main__":
    main()
