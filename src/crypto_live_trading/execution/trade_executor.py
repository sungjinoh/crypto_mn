"""
Trade Executor - Handles order placement and execution via CCXT
"""

import ccxt
from datetime import datetime


class TradeExecutor:
    def __init__(self, api_key="", secret="", sandbox=True, portfolio_value=10000):
        """
        Initialize trade executor with exchange connection

        Args:
            api_key: Binance API key
            secret: Binance API secret
            sandbox: Use testnet for testing
            portfolio_value: Portfolio value for position sizing
        """
        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": secret,
                "sandbox": sandbox,
                "enableRateLimit": True,
            }
        )
        self.portfolio_value = portfolio_value
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
            symbol1_size, symbol2_size = self._calculate_position_sizes(
                symbol1_ccxt, symbol2_ccxt, signal
            )

            # Determine order sides
            symbol1_side = "buy" if signal.symbol1_qty > 0 else "sell"
            symbol2_side = "buy" if signal.symbol2_qty > 0 else "sell"

            # Place orders (paper trading for now)
            if self.exchange.sandbox or not self.exchange.apiKey:
                # Paper trading mode
                print(
                    f"  [PAPER] {symbol1_side.upper()} {symbol1_size:.6f} {symbol1_ccxt}"
                )
                print(
                    f"  [PAPER] {symbol2_side.upper()} {symbol2_size:.6f} {symbol2_ccxt}"
                )

                # Simulate successful orders
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
            symbol1_size, symbol2_size = self._calculate_position_sizes(
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
        Calculate position sizes based on portfolio value and risk management

        Args:
            symbol1, symbol2: CCXT formatted symbols
            signal: Trading signal

        Returns:
            symbol1_size, symbol2_size: Position sizes in base currency
        """
        try:
            # Get current prices
            ticker1 = self.exchange.fetch_ticker(symbol1)
            ticker2 = self.exchange.fetch_ticker(symbol2)

            price1 = ticker1["last"]
            price2 = ticker2["last"]

            # Calculate risk-adjusted position size
            risk_amount = self.portfolio_value * self.risk_per_trade

            # Simple position sizing - can be enhanced with volatility adjustments
            base_notional = risk_amount / 2  # Split risk between two legs

            symbol1_size = abs(base_notional / price1)
            symbol2_size = abs(base_notional / price2 * abs(signal.hedge_ratio))

            return symbol1_size, symbol2_size

        except Exception as e:
            print(f"Error calculating position sizes: {e}")
            # Return conservative default sizes
            return 0.001, 0.001
