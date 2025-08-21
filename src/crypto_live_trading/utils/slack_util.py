"""
Slack Utility for sending notifications
"""

import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SlackUtil:
    def __init__(self, slack_webhook=None):
        """
        Initialize SlackUtil with webhook URL

        Args:
            slack_webhook: Slack webhook URL (if None, loads from SLACK_WEBHOOK env var)
        """
        if slack_webhook is None:
            slack_webhook = os.getenv("SLACK_WEBHOOK", "")

        self.slack_webhook = slack_webhook

    def send_msg_to_slack(self, msg, color="#36a64f"):
        """
        Send a message to Slack

        Args:
            msg: Message to send
            color: Color for the attachment (green by default)
        """
        if not self.slack_webhook:
            print(f"[SLACK DISABLED] {msg}")
            return

        text = f"{msg}"
        slack_data = {"attachments": [{"text": text, "color": color}]}

        print(msg)  # Also print to console

        try:
            response = requests.post(
                self.slack_webhook,
                data=json.dumps(slack_data),
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                print(
                    f"Slack notification failed: {response.status_code} - {response.text}"
                )
        except Exception as e:
            print(f"Error sending Slack notification: {e}")

    def notify_slack(self, msg, mention_user=True):
        """
        Send notification to Slack with optional user mention

        Args:
            msg: Message to send
            mention_user: Whether to mention the user
        """
        if mention_user:
            # Get user ID from env var or use default
            user_id = os.getenv("SLACK_USER_ID", "U8D92RR9S")
            msg = f"\n<@{user_id}> {msg}"
        self.send_msg_to_slack(msg)

    def notify_position_open(self, symbol1, symbol2, signal, position_info=None):
        """
        Send notification when a position is opened
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg = f"🟢 *POSITION OPENED* 🟢\n"
        msg += f"📅 Time: {timestamp}\n"
        msg += f"📊 Pair: {symbol1} - {symbol2}\n"
        msg += f"📈 Side: {signal.side.upper()}\n"
        msg += f"⚖️ Hedge Ratio: {signal.hedge_ratio:.4f}\n"
        msg += f"📉 Z-Score: {signal.z_score:.3f}\n"
        msg += f"💰 Spread: {signal.spread_value:.6f}\n"

        if position_info:
            msg += f"💵 Position Size: ${position_info.get('total_notional', 'N/A')}\n"
            msg += f"💸 Margin Used: ${position_info.get('margin_required', 'N/A')}\n"

        msg += f"📝 Reason: {signal.reason}"

        self.notify_slack(msg)

    def notify_position_close(self, symbol1, symbol2, signal, position_info=None):
        """
        Send notification when a position is closed
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg = f"🔴 *POSITION CLOSED* 🔴\n"
        msg += f"📅 Time: {timestamp}\n"
        msg += f"📊 Pair: {symbol1} - {symbol2}\n"
        msg += f"📉 Z-Score: {signal.z_score:.3f}\n"
        msg += f"💰 Spread: {signal.spread_value:.6f}\n"
        msg += f"📝 Reason: {signal.reason}"

        if position_info:
            if "duration" in position_info:
                msg += f"\n⏱️ Duration: {position_info['duration']}"
            if "pnl" in position_info:
                pnl = position_info["pnl"]
                pnl_emoji = "💚" if pnl > 0 else "❤️"
                msg += f"\n{pnl_emoji} P&L: ${pnl:.2f}"

        # Use red color for position close
        self.send_msg_to_slack(msg, color="#ff0000")

    def notify_error(self, error_msg, symbol1=None, symbol2=None):
        """
        Send error notification to Slack
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg = f"❌ *ERROR* ❌\n"
        msg += f"📅 Time: {timestamp}\n"

        if symbol1 and symbol2:
            msg += f"📊 Pair: {symbol1} - {symbol2}\n"

        msg += f"🚨 Error: {error_msg}"

        # Use orange color for errors
        self.send_msg_to_slack(msg, color="#ff9900")

    def notify_system_start(self, config_info=None):
        """
        Send notification when trading system starts
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg = f"🚀 *TRADING SYSTEM STARTED* 🚀\n"
        msg += f"📅 Time: {timestamp}\n"

        if config_info:
            msg += f"⚙️ Configuration:\n"
            for key, value in config_info.items():
                msg += f"  • {key}: {value}\n"

        # Use blue color for system events
        self.send_msg_to_slack(msg, color="#0099cc")
