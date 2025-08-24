#!/usr/bin/env python3
"""
Test script for Telegram notifications
Run this to verify your Telegram bot is working correctly.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.telegram_util import TelegramUtil


def test_telegram_setup():
    """Test Telegram bot setup and send sample notifications"""

    print("🔍 Testing Telegram Bot Setup...")
    print("=" * 50)

    # Check environment variables
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    print(f"Bot Token: {'✅ Set' if bot_token else '❌ Missing'}")
    print(f"Chat ID: {'✅ Set' if chat_id else '❌ Missing'}")

    if not bot_token or not chat_id:
        print("\n❌ Missing required environment variables!")
        print("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file")
        print("See TELEGRAM_SETUP.md for instructions")
        return False

    # Initialize Telegram utility
    telegram = TelegramUtil()

    print(f"\n📱 Sending test messages...")

    # Test 1: Basic message
    print("1. Testing basic message...")
    telegram.send_message("🧪 **Test Message**\nTelegram bot is working correctly!")

    # Test 2: System start notification
    print("2. Testing system start notification...")
    test_config = {
        "Mode": "TEST",
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Status": "Testing Telegram Integration",
    }
    telegram.notify_system_start(test_config)

    # Test 3: Mock position open notification
    print("3. Testing position open notification...")

    class MockSignal:
        def __init__(self):
            self.side = "long"
            self.hedge_ratio = 1.5
            self.z_score = 2.1
            self.spread_value = 0.0025
            self.reason = "Test position for Telegram integration"

    mock_signal = MockSignal()
    position_info = {"total_notional": "200.00", "margin_required": "40.00"}
    telegram.notify_position_open("BTCUSDT", "ETHUSDT", mock_signal, position_info)

    # Test 4: Error notification
    print("4. Testing error notification...")
    telegram.notify_error("This is a test error message", "BTCUSDT", "ETHUSDT")

    print("\n✅ Test completed!")
    print("Check your Telegram chat to see if you received all 4 test messages.")
    print("If you didn't receive messages, check your bot token and chat ID.")

    return True


if __name__ == "__main__":
    try:
        test_telegram_setup()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Please check your configuration and try again.")
