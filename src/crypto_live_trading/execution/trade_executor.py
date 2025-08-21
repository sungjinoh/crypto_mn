"""
Trade Executor - Handles order placement and execution via CCXT
"""

import os
import ccxt
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TradeExecutor:
    def __init__(
        self,
        api_key=None,
        secret=None,
        sandbox=True,
        portfolio_value=10000,
        leverage=10,
        max_position_size_usdt=100,
    ):
        """
        Initialize trade executor with exchange connection

        Args:
            api_key: Binance API key (if None, loads from ZZT_BINANCE_KEY env var)
            secret: Binance API secret (if None, loads from ZZT_BINANCE_SECRET env var)
            sandbox: Use testnet for testing
            portfolio_value: Portfolio value for position sizing
            leverage: Leverage multiplier for futures trading
            max_position_size_usdt: Maximum position size per leg in USDT
        """
        # Load API credentials from environment if not provided
        if api_key is None:
            api_key = os.getenv("ZZT_BINANCE_KEY", "")
        if secret is None:
            secret = os.getenv("ZZT_BINANCE_SECRET", "")

        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": secret,
                "sandbox": sandbox,
                "enableRateLimit": True,
            }
        )
        self.portfolio_value = portfolio_value
        self.leverage = leverage
        self.max_position_size_usdt = max_position_size_usdt
        self.risk_per_trade = 0.02  # 2% risk per trade

    def open_position(self, symbol1, symbol2, signal):
        """
        Open a pairs trading position

        Args:
            symbol1, symbol2: Trading symbols
            signal: Trading signal with position details
        """
        try:
            print(f"Opening position: {symbol1}-{symbol2} {signal.side}")
            print(f"  Hedge ratio: {signal.hedge_ratio:.4f}")
            print(f"  Z-score: {signal.z_score:.3f}")

            # Convert symbols to CCXT format
            symbol1_ccxt = self._format_symbol(symbol1)
            symbol2_ccxt = self._format_symbol(symbol2)

            # Calculate position sizes
            symbol1_size, symbol2_size, price1, price2 = self._calculate_position_sizes(
                symbol1_ccxt, symbol2_ccxt, signal
            )

            # Determine order sides
            symbol1_side = "buy" if signal.symbol1_qty > 0 else "sell"
            symbol2_side = "buy" if signal.symbol2_qty > 0 else "sell"

            # Place orders (paper trading for now)
            if self.exchange.sandbox or not self.exchange.apiKey:
                # Paper trading mode
                print(
                    f"  [PAPER] {symbol1_side.upper()} {symbol1_size:.6f} {symbol1_ccxt} (Leverage: {self.leverage}x)"
                )
                print(
                    f"  [PAPER] {symbol2_side.upper()} {symbol2_size:.6f} {symbol2_ccxt} (Leverage: {self.leverage}x)"
                )
                print(
                    f"  [PAPER] Position size: ${symbol1_size*price1:.2f} + ${symbol2_size*price2:.2f} USDT notional"
                )
                print(
                    f"  [PAPER] Margin required: ${(symbol1_size*price1 + symbol2_size*price2)/self.leverage:.2f} USDT"
                )

                # Simulate successful orders
                return {
                    "symbol1_order": {
                        "id": f"paper_{datetime.now().timestamp()}",
                        "status": "filled",
                        "size": symbol1_size,
                        "price": price1,
                        "leverage": self.leverage,
                    },
                    "symbol2_order": {
                        "id": f"paper_{datetime.now().timestamp()}",
                        "status": "filled",
                        "size": symbol2_size,
                        "price": price2,
                        "leverage": self.leverage,
                    },
                }
            else:
                # Real trading mode - set leverage first
                try:
                    self.exchange.set_leverage(self.leverage, symbol1_ccxt)
                    self.exchange.set_leverage(self.leverage, symbol2_ccxt)
                except Exception as e:
                    print(f"Warning: Could not set leverage: {e}")

                symbol1_order = self.exchange.create_market_order(
                    symbol1_ccxt, symbol1_side, symbol1_size
                )
                symbol2_order = self.exchange.create_market_order(
                    symbol2_ccxt, symbol2_side, symbol2_size
                )

                print(f"  Order 1: {symbol1_order['id']} - {symbol1_order['status']}")
                print(f"  Order 2: {symbol2_order['id']} - {symbol2_order['status']}")

                return {"symbol1_order": symbol1_order, "symbol2_order": symbol2_order}

        except Exception as e:
            print(f"Error opening position {symbol1}-{symbol2}: {e}")
            return None

    def close_position(self, symbol1, symbol2, signal):
        """
        Close a pairs trading position

        Args:
            symbol1, symbol2: Trading symbols
            signal: Exit signal
        """
        try:
            print(f"Closing position: {symbol1}-{symbol2}")
            print(f"  Exit reason: {signal.reason}")
            print(f"  Z-score: {signal.z_score:.3f}")

            # Convert symbols to CCXT format
            symbol1_ccxt = self._format_symbol(symbol1)
            symbol2_ccxt = self._format_symbol(symbol2)

            # For simplicity, we'll close by reversing the original position
            # In a real system, you'd track the exact position sizes

            # Calculate position sizes (reverse of opening)
            symbol1_size, symbol2_size, price1, price2 = self._calculate_position_sizes(
                symbol1_ccxt, symbol2_ccxt, signal
            )

            # Reverse the order sides for closing
            symbol1_side = "sell" if signal.symbol1_qty > 0 else "buy"
            symbol2_side = "sell" if signal.symbol2_qty > 0 else "buy"

            # Place closing orders (paper trading for now)
            if self.exchange.sandbox or not self.exchange.apiKey:
                # Paper trading mode
                print(
                    f"  [PAPER] {symbol1_side.upper()} {symbol1_size:.6f} {symbol1_ccxt}"
                )
                print(
                    f"  [PAPER] {symbol2_side.upper()} {symbol2_size:.6f} {symbol2_ccxt}"
                )

                return {
                    "symbol1_order": {
                        "id": f"paper_{datetime.now().timestamp()}",
                        "status": "filled",
                    },
                    "symbol2_order": {
                        "id": f"paper_{datetime.now().timestamp()}",
                        "status": "filled",
                    },
                }
            else:
                # Real trading mode
                symbol1_order = self.exchange.create_market_order(
                    symbol1_ccxt, symbol1_side, symbol1_size
                )
                symbol2_order = self.exchange.create_market_order(
                    symbol2_ccxt, symbol2_side, symbol2_size
                )

                print(
                    f"  Close Order 1: {symbol1_order['id']} - {symbol1_order['status']}"
                )
                print(
                    f"  Close Order 2: {symbol2_order['id']} - {symbol2_order['status']}"
                )

                return {"symbol1_order": symbol1_order, "symbol2_order": symbol2_order}

        except Exception as e:
            print(f"Error closing position {symbol1}-{symbol2}: {e}")
            return None

    def _format_symbol(self, symbol):
        """Convert symbol format to CCXT format"""
        if "/" not in symbol:
            # Assume it's a futures symbol like 'BTCUSDT'
            if symbol.endswith("USDT"):
                base = symbol[:-4]
                return f"{base}/USDT"
        return symbol

    def _calculate_position_sizes(self, symbol1, symbol2, signal):
        """
        Calculate position sizes based on portfolio value, leverage, and risk management

        Args:
            symbol1, symbol2: CCXT formatted symbols
            signal: Trading signal

        Returns:
            symbol1_size, symbol2_size, price1, price2: Position sizes and current prices
        """
        try:
            # Get current prices
            ticker1 = self.exchange.fetch_ticker(symbol1)
            ticker2 = self.exchange.fetch_ticker(symbol2)

            price1 = ticker1["last"]
            price2 = ticker2["last"]

            # Calculate notional values based on max position size
            # Use leverage to determine effective buying power
            max_notional_per_leg = self.max_position_size_usdt

            # Calculate position sizes for each leg
            symbol1_notional = min(max_notional_per_leg, self.max_position_size_usdt)
            symbol2_notional = min(
                max_notional_per_leg,
                self.max_position_size_usdt * abs(signal.hedge_ratio),
            )

            # Convert notional to actual position sizes
            symbol1_size = abs(symbol1_notional / price1)
            symbol2_size = abs(symbol2_notional / price2)

            # Apply leverage consideration (with leverage, we need less margin)
            # But position size remains the same for the notional exposure

            print(f"  Position calculation:")
            print(
                f"    Symbol1 ({symbol1}): ${symbol1_notional:.2f} notional = {symbol1_size:.6f} units @ ${price1:.2f}"
            )
            print(
                f"    Symbol2 ({symbol2}): ${symbol2_notional:.2f} notional = {symbol2_size:.6f} units @ ${price2:.2f}"
            )
            print(
                f"    Total notional: ${symbol1_notional + symbol2_notional:.2f} USDT"
            )
            print(
                f"    Margin required: ${(symbol1_notional + symbol2_notional)/self.leverage:.2f} USDT"
            )

            return symbol1_size, symbol2_size, price1, price2

        except Exception as e:
            print(f"Error calculating position sizes: {e}")
            # Return conservative default sizes with dummy prices
            return 0.001, 0.001, 50000, 3000
