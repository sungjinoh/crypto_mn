"""
Telegram Utility for sending notifications
"""

import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TelegramUtil:
    def __init__(self, bot_token=None, chat_id=None):
        """
        Initialize TelegramUtil with bot token and chat ID

        Args:
            bot_token: Telegram bot token (if None, loads from TELEGRAM_BOT_TOKEN env var)
            chat_id: Telegram chat ID (if None, loads from TELEGRAM_CHAT_ID env var)
        """
        if bot_token is None:
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if chat_id is None:
            chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_message(self, message, parse_mode="Markdown"):
        """
        Send a message to Telegram

        Args:
            message: Message to send
            parse_mode: Parse mode for formatting (Markdown or HTML)
        """
        if not self.bot_token or not self.chat_id:
            print(f"[TELEGRAM DISABLED] {message}")
            return

        print(message)  # Also print to console

        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }

            response = requests.post(url, json=payload)

            if response.status_code != 200:
                print(
                    f"Telegram notification failed: {response.status_code} - {response.text}"
                )
            else:
                result = response.json()
                if not result.get("ok"):
                    print(
                        f"Telegram API error: {result.get('description', 'Unknown error')}"
                    )

        except Exception as e:
            print(f"Error sending Telegram notification: {e}")

    def notify_position_open(self, symbol1, symbol2, signal, position_info=None):
        """
        Send notification when a position is opened
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg = f"🟢 *POSITION OPENED* 🟢\n"
        msg += f"📅 Time: `{timestamp}`\n"
        msg += f"📊 Pair: `{symbol1}` - `{symbol2}`\n"
        msg += f"📈 Side: `{signal.side.upper()}`\n"
        msg += f"⚖️ Hedge Ratio: `{signal.hedge_ratio:.4f}`\n"
        msg += f"📉 Z-Score: `{signal.z_score:.3f}`\n"
        msg += f"💰 Spread: `{signal.spread_value:.6f}`\n"

        if position_info:
            msg += (
                f"💵 Position Size: `${position_info.get('total_notional', 'N/A')}`\n"
            )
            msg += f"💸 Margin Used: `${position_info.get('margin_required', 'N/A')}`\n"

        msg += f"📝 Reason: {signal.reason}"

        self.send_message(msg)

    def notify_position_close(self, symbol1, symbol2, signal, position_info=None):
        """
        Send notification when a position is closed
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg = f"🔴 *POSITION CLOSED* 🔴\n"
        msg += f"📅 Time: `{timestamp}`\n"
        msg += f"📊 Pair: `{symbol1}` - `{symbol2}`\n"
        msg += f"📉 Z-Score: `{signal.z_score:.3f}`\n"
        msg += f"💰 Spread: `{signal.spread_value:.6f}`\n"
        msg += f"📝 Reason: {signal.reason}"

        if position_info:
            if "duration" in position_info:
                msg += f"\n⏱️ Duration: `{position_info['duration']}`"
            if "pnl" in position_info:
                pnl = position_info["pnl"]
                pnl_emoji = "💚" if pnl > 0 else "❤️"
                msg += f"\n{pnl_emoji} P&L: `${pnl:.2f}`"

        self.send_message(msg)

    def notify_error(self, error_msg, symbol1=None, symbol2=None):
        """
        Send error notification to Telegram
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg = f"❌ *ERROR* ❌\n"
        msg += f"📅 Time: `{timestamp}`\n"

        if symbol1 and symbol2:
            msg += f"📊 Pair: `{symbol1}` - `{symbol2}`\n"

        msg += f"🚨 Error: `{error_msg}`"

        self.send_message(msg)

    def notify_system_start(self, config_info=None):
        """
        Send notification when trading system starts
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg = f"🚀 *TRADING SYSTEM STARTED* 🚀\n"
        msg += f"📅 Time: `{timestamp}`\n"

        if config_info:
            msg += f"⚙️ Configuration:\n"
            for key, value in config_info.items():
                msg += f"  • {key}: `{value}`\n"

        self.send_message(msg)

    def notify_margin_warning(
        self, symbol, current_margin, liquidation_price, mark_price
    ):
        """
        Send margin warning notification
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        distance_to_liquidation = abs(mark_price - liquidation_price) / mark_price * 100

        msg = f"⚠️ *MARGIN WARNING* ⚠️\n"
        msg += f"📅 Time: `{timestamp}`\n"
        msg += f"📊 Symbol: `{symbol}`\n"
        msg += f"💰 Current Margin: `${current_margin:.2f}`\n"
        msg += f"💀 Liquidation Price: `${liquidation_price:.4f}`\n"
        msg += f"📈 Current Price: `${mark_price:.4f}`\n"
        msg += f"📏 Distance to Liquidation: `{distance_to_liquidation:.2f}%`\n"
        msg += f"🚨 Action: Consider adding margin or reducing position"

        self.send_message(msg)

    def notify_margin_healthy(self, summary):
        """
        Send margin health summary notification (optional, for periodic updates)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg = f"💚 *MARGIN HEALTH CHECK* 💚\n"
        msg += f"📅 Time: `{timestamp}`\n"
        msg += f"📊 Active Positions: `{summary.get('active_positions', 0)}`\n"
        msg += f"💰 Total Margin Used: `${summary.get('total_margin', 0):.2f}`\n"
        msg += f"🆓 Available Balance: `${summary.get('available_balance', 0):.2f}`\n"
        msg += f"✅ All positions have healthy margins"

        self.send_message(msg)

    def send_msg_to_telegram(self, msg):
        """
        Legacy method for compatibility - sends message to Telegram

        Args:
            msg: Message to send
        """
        self.send_message(msg)
