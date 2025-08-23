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
                "options": {
                    "defaultType": "future",
                },
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
                        "side": symbol1_side,
                        "price": price1,
                        "leverage": self.leverage,
                    },
                    "symbol2_order": {
                        "id": f"paper_{datetime.now().timestamp()}",
                        "status": "filled",
                        "size": symbol2_size,
                        "side": symbol2_side,
                        "price": price2,
                        "leverage": self.leverage,
                    },
                }
            else:
                # Real trading mode - set leverage first
                try:
                    self._set_leverage_and_margin(symbol1_ccxt, self.leverage)
                    self._set_leverage_and_margin(symbol2_ccxt, self.leverage)
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

    def close_position(self, symbol1, symbol2, signal, position):
        """
        Close a pairs trading position using actual executed quantities
        
        Args:
            symbol1, symbol2: Trading symbols
            signal: Exit signal
            position: Actual position to close (from position tracker)
        """
        try:
            print(f"Closing position: {symbol1}-{symbol2}")
            print(f"  Exit reason: {signal.reason}")
            print(f"  Z-score: {signal.z_score:.3f}")
            print(f"  Original position: {position.side}")

            # Convert symbols to CCXT format  
            symbol1_ccxt = self._format_symbol(symbol1)
            symbol2_ccxt = self._format_symbol(symbol2)

            # Use ACTUAL executed quantities and reverse the sides
            symbol1_size = position.symbol1_executed_size
            symbol2_size = position.symbol2_executed_size
            
            # Reverse the original executed sides for closing
            symbol1_side = "sell" if position.symbol1_executed_side == "buy" else "buy"
            symbol2_side = "sell" if position.symbol2_executed_side == "buy" else "buy"
            
            print(f"  Closing: {symbol1} {position.symbol1_executed_side} {symbol1_size:.6f} → {symbol1_side} {symbol1_size:.6f}")
            print(f"  Closing: {symbol2} {position.symbol2_executed_side} {symbol2_size:.6f} → {symbol2_side} {symbol2_size:.6f}")

            # Place closing orders
            if self.exchange.sandbox or not self.exchange.apiKey:
                # Paper trading mode
                print(f"  [PAPER] {symbol1_side.upper()} {symbol1_size:.6f} {symbol1_ccxt}")
                print(f"  [PAPER] {symbol2_side.upper()} {symbol2_size:.6f} {symbol2_ccxt}")

                # Get current prices for reporting
                try:
                    ticker1 = self.exchange.fetch_ticker(symbol1_ccxt)
                    ticker2 = self.exchange.fetch_ticker(symbol2_ccxt)
                    price1 = ticker1["last"]
                    price2 = ticker2["last"]
                except:
                    price1, price2 = 0.0, 0.0

                return {
                    "symbol1_order": {
                        "id": f"paper_{datetime.now().timestamp()}",
                        "status": "filled",
                        "size": symbol1_size,
                        "side": symbol1_side,
                        "price": price1,
                    },
                    "symbol2_order": {
                        "id": f"paper_{datetime.now().timestamp()}",
                        "status": "filled",
                        "size": symbol2_size,
                        "side": symbol2_side,
                        "price": price2,
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

                print(f"  Close Order 1: {symbol1_order['id']} - {symbol1_order['status']}")
                print(f"  Close Order 2: {symbol2_order['id']} - {symbol2_order['status']}")

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
        Calculate position sizes based on max position size, leverage, and signal quantities

        Args:
            symbol1, symbol2: CCXT formatted symbols
            signal: Trading signal with symbol1_qty and symbol2_qty

        Returns:
            symbol1_size, symbol2_size, price1, price2: Position sizes and current prices
        """
        try:
            # Get current prices
            ticker1 = self.exchange.fetch_ticker(symbol1)
            ticker2 = self.exchange.fetch_ticker(symbol2)

            price1 = ticker1["last"]
            price2 = ticker2["last"]

            # The signal provides directional quantities (e.g., +1.0, -2.5)
            # We need to scale these to actual position sizes
            
            # Base approach: Use max_position_size_usdt as the reference for symbol1
            symbol1_base_notional = self.max_position_size_usdt
            
            # Calculate symbol1 actual size from base notional and signal quantity
            symbol1_base_size = symbol1_base_notional / price1
            symbol1_size = abs(signal.symbol1_qty) * symbol1_base_size
            symbol1_notional = symbol1_size * price1
            
            # For symbol2: scale proportionally based on signal ratio
            # If signal says symbol1_qty=1.0, symbol2_qty=-2.5, then symbol2 should be 2.5x symbol1 size
            if signal.symbol1_qty != 0:
                quantity_ratio = abs(signal.symbol2_qty / signal.symbol1_qty)
            else:
                quantity_ratio = abs(signal.symbol2_qty)
                
            symbol2_size = symbol1_base_size * quantity_ratio  # Same base unit size, scaled by ratio
            symbol2_notional = symbol2_size * price2

            # Verify we're not exceeding reasonable limits
            total_notional = symbol1_notional + symbol2_notional
            margin_required = total_notional / self.leverage
            
            # If too large, scale down proportionally
            if total_notional > self.max_position_size_usdt * 3:  # Allow up to 3x for pairs
                scale_factor = (self.max_position_size_usdt * 3) / total_notional
                symbol1_size *= scale_factor
                symbol2_size *= scale_factor
                symbol1_notional *= scale_factor
                symbol2_notional *= scale_factor
                margin_required *= scale_factor
                print(f"  ⚠️  Scaled down by {scale_factor:.3f} to stay within limits")

            print(f"  Position calculation:")
            print(f"    Signal quantities: {symbol1} {signal.symbol1_qty:+.3f}, {symbol2} {signal.symbol2_qty:+.3f}")
            print(f"    Quantity ratio: {quantity_ratio:.3f}")
            print(f"    {symbol1}: {symbol1_size:.6f} units = ${symbol1_notional:.2f} @ ${price1:.2f}")
            print(f"    {symbol2}: {symbol2_size:.6f} units = ${symbol2_notional:.2f} @ ${price2:.2f}")
            print(f"    Total notional: ${total_notional:.2f} USDT")
            print(f"    Margin required ({self.leverage}x leverage): ${margin_required:.2f} USDT")

            return symbol1_size, symbol2_size, price1, price2

        except Exception as e:
            print(f"Error calculating position sizes: {e}")
            # Return conservative default sizes with dummy prices
            return 0.001, 0.001, 50000, 3000

    def _set_leverage_and_margin(self, symbol, leverage, margin_type="ISOLATED"):
        """
        Set margin type and leverage for a symbol on Binance Futures.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            leverage: Desired leverage (e.g., 10)
            margin_type: Margin type ('ISOLATED' or 'CROSS'), default 'ISOLATED'
        """
        try:
            market = self.exchange.market(symbol)
            symbol_id = market["id"]

            # Set margin type first (ignore errors if already set)
            try:
                self.exchange.fapiPrivatePostMarginType(
                    {"symbol": symbol_id, "marginType": margin_type}
                )
                print(f"  Set margin type to {margin_type} for {symbol}")
            except Exception as e:
                # Ignore errors - margin type might already be set
                print(
                    f"  Margin type already set for {symbol} (or error: {str(e)[:50]})"
                )

            # Set leverage
            try:
                self.exchange.fapiPrivatePostLeverage(
                    {"symbol": symbol_id, "leverage": leverage}
                )
                print(f"  Set leverage to {leverage}x for {symbol}")
            except Exception as e:
                print(f"  Warning: Could not set leverage for {symbol}: {e}")

        except Exception as e:
            print(f"  Error setting leverage for {symbol}: {e}")
            # Don't raise exception - continue with trading even if leverage setting fails

    def monitor_position_margin(self, symbol1, symbol2, slack_util=None):
        """
        Monitor margin levels for both symbols in a pairs trade and adjust if needed.

        Args:
            symbol1, symbol2: Trading symbols
            slack_util: SlackUtil instance for notifications

        Returns:
            dict: Summary of margin status and actions taken
        """
        try:
            symbol1_ccxt = self._format_symbol(symbol1)
            symbol2_ccxt = self._format_symbol(symbol2)

            # Check margin for both symbols
            symbol1_status = self._check_and_adjust_margin(symbol1_ccxt, slack_util)
            symbol2_status = self._check_and_adjust_margin(symbol2_ccxt, slack_util)

            return {
                "symbol1": symbol1_status,
                "symbol2": symbol2_status,
                "overall_status": (
                    "healthy"
                    if symbol1_status["status"] == "healthy"
                    and symbol2_status["status"] == "healthy"
                    else "attention_needed"
                ),
            }

        except Exception as e:
            print(f"  Error monitoring margin for {symbol1}-{symbol2}: {e}")
            return {"error": str(e)}

    def _check_and_adjust_margin(self, symbol, slack_util=None):
        """
        Check and adjust margin for a single symbol.

        Args:
            symbol: CCXT formatted symbol (e.g., 'BTC/USDT')
            slack_util: SlackUtil instance for notifications

        Returns:
            dict: Margin status and actions taken
        """
        try:
            if self.exchange.sandbox or not self.exchange.apiKey:
                # Paper trading mode - simulate margin check
                print(f"  [PAPER] Margin check for {symbol} - OK (simulated)")
                return {
                    "status": "healthy",
                    "margin_balance": 100.0,  # Simulated
                    "action": "none",
                    "message": "Paper trading - margin OK",
                }

            # Fetch position information
            positions = self.exchange.fetch_positions([symbol])
            if not positions or len(positions) == 0:
                return {
                    "status": "no_position",
                    "action": "none",
                    "message": f"No position found for {symbol}",
                }

            position = positions[0]

            # Extract margin information
            margin_balance = float(position.get("info", {}).get("isolatedMargin", 0))
            position_amt = float(position.get("info", {}).get("positionAmt", 0))
            maintenance_margin = float(position.get("maintenanceMargin", 0))
            entry_price = float(position.get("info", {}).get("entryPrice", 0))
            liquidation_price = float(
                position.get("info", {}).get("liquidationPrice", 0)
            )
            mark_price = float(position.get("info", {}).get("markPrice", 0))
            initial_margin_pct = float(
                position.get("initialMarginPercentage", 0.1)
            )  # Default 10%

            # Skip if no position
            if position_amt == 0:
                return {
                    "status": "no_position",
                    "margin_balance": margin_balance,
                    "action": "none",
                    "message": f"No active position for {symbol}",
                }

            # Calculate margin thresholds
            initial_margin = entry_price * abs(position_amt) * initial_margin_pct
            upper_threshold = initial_margin * 2  # 200% of initial margin
            lower_threshold = max(
                initial_margin / 2, maintenance_margin * 2
            )  # 50% of initial or 2x maintenance

            print(f"  📊 Margin Status for {symbol}:")
            print(f"    Current Balance: ${margin_balance:.2f}")
            print(f"    Initial Margin: ${initial_margin:.2f}")
            print(f"    Lower Threshold: ${lower_threshold:.2f}")
            print(f"    Upper Threshold: ${upper_threshold:.2f}")
            print(
                f"    Liquidation Price: ${liquidation_price:.4f} (Current: ${mark_price:.4f})"
            )

            # Check if margin adjustment is needed
            if margin_balance <= lower_threshold:
                # Need to add margin
                new_margin_balance = int(lower_threshold * 2) + 1
                add_amount = new_margin_balance - margin_balance

                print(f"  🚨 LOW MARGIN ALERT for {symbol}!")
                print(f"    Adding ${add_amount:.2f} margin")

                try:
                    # Add margin
                    response = self.exchange.add_margin(symbol, add_amount)

                    message = f"✅ Added ${add_amount:.2f} margin to {symbol}. New balance: ${new_margin_balance:.2f}"
                    print(f"    {message}")

                    # Send Slack notification
                    if slack_util:
                        slack_util.send_msg_to_slack(
                            f"💰 MARGIN ADDED\n"
                            f"Symbol: {symbol}\n"
                            f"Amount: ${add_amount:.2f}\n"
                            f"Current: ${margin_balance:.2f} → New: ${new_margin_balance:.2f}\n"
                            f"Liquidation Price: ${liquidation_price:.4f}",
                            color="#ffaa00",
                        )

                    return {
                        "status": "margin_added",
                        "margin_balance": new_margin_balance,
                        "action": "add_margin",
                        "amount": add_amount,
                        "message": message,
                    }

                except Exception as e:
                    error_msg = f"Failed to add margin to {symbol}: {e}"
                    print(f"    ❌ {error_msg}")

                    if slack_util:
                        slack_util.notify_error(
                            f"Margin addition failed for {symbol}: {e}"
                        )

                    return {
                        "status": "error",
                        "margin_balance": margin_balance,
                        "action": "add_margin_failed",
                        "error": str(e),
                        "message": error_msg,
                    }

            elif margin_balance >= upper_threshold:
                # Can reduce margin to optimize capital
                reduce_amount = int(
                    (margin_balance - upper_threshold) * 0.8
                )  # Reduce 80% of excess

                if reduce_amount > 10:  # Only reduce if amount is significant
                    print(f"  💰 EXCESS MARGIN for {symbol}")
                    print(f"    Reducing ${reduce_amount:.2f} margin")

                    try:
                        # Reduce margin
                        response = self.exchange.reduce_margin(symbol, reduce_amount)

                        new_balance = margin_balance - reduce_amount
                        message = f"✅ Reduced ${reduce_amount:.2f} margin from {symbol}. New balance: ${new_balance:.2f}"
                        print(f"    {message}")

                        # Send Slack notification
                        if slack_util:
                            slack_util.send_msg_to_slack(
                                f"💸 MARGIN REDUCED\n"
                                f"Symbol: {symbol}\n"
                                f"Amount: ${reduce_amount:.2f}\n"
                                f"Current: ${margin_balance:.2f} → New: ${new_balance:.2f}\n"
                                f"Capital freed for other trades",
                                color="#00aa00",
                            )

                        return {
                            "status": "margin_reduced",
                            "margin_balance": new_balance,
                            "action": "reduce_margin",
                            "amount": reduce_amount,
                            "message": message,
                        }

                    except Exception as e:
                        error_msg = f"Failed to reduce margin for {symbol}: {e}"
                        print(f"    ⚠️ {error_msg}")

                        return {
                            "status": "reduce_failed",
                            "margin_balance": margin_balance,
                            "action": "reduce_margin_failed",
                            "error": str(e),
                            "message": error_msg,
                        }
            else:
                # Margin is within healthy range
                return {
                    "status": "healthy",
                    "margin_balance": margin_balance,
                    "action": "none",
                    "message": f"Margin healthy for {symbol}: ${margin_balance:.2f}",
                }

        except Exception as e:
            error_msg = f"Error checking margin for {symbol}: {e}"
            print(f"  ❌ {error_msg}")
            return {
                "status": "error",
                "action": "check_failed",
                "error": str(e),
                "message": error_msg,
            }
