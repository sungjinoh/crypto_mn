#!/usr/bin/env python3
"""
Helper script to get your Telegram chat ID
This script helps you find your correct chat ID for Telegram notifications.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_chat_id():
    """Help user get their correct Telegram chat ID"""

    print("🔍 Telegram Chat ID Helper")
    print("=" * 40)

    # Check if bot token is set
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN not found in environment variables")
        print("Please add your bot token to the .env file first")
        print("Get your bot token from @BotFather in Telegram")
        return

    print(f"✅ Bot token found: {bot_token[:10]}...")
    print("\n📋 Instructions:")
    print("1. Open Telegram and find your bot")
    print("2. Send ANY message to your bot (e.g., 'hello')")
    print("3. Press Enter here to check for messages...")

    input("Press Enter after sending a message to your bot...")

    # Get updates from Telegram
    try:
        url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"❌ Failed to get updates: {response.status_code}")
            print(f"Response: {response.text}")
            return

        data = response.json()

        if not data.get("ok"):
            print(f"❌ Telegram API error: {data.get('description', 'Unknown error')}")
            return

        updates = data.get("result", [])

        if not updates:
            print("❌ No messages found!")
            print("Make sure you:")
            print("   1. Sent a message to YOUR bot (not another bot)")
            print("   2. Started the conversation with your bot")
            print("   3. Used the correct bot token")
            return

        print(f"\n✅ Found {len(updates)} message(s)!")
        print("\n📱 Available Chat IDs:")

        chat_ids = set()
        for update in updates:
            if "message" in update:
                chat = update["message"]["chat"]
                chat_id = chat["id"]
                chat_type = chat["type"]

                if chat_type == "private":
                    first_name = chat.get("first_name", "Unknown")
                    username = chat.get("username", "No username")
                    print(f"   👤 Personal chat: {chat_id}")
                    print(f"      Name: {first_name}")
                    print(f"      Username: @{username}")
                    chat_ids.add(chat_id)

                elif chat_type in ["group", "supergroup"]:
                    title = chat.get("title", "Unknown Group")
                    print(f"   👥 Group chat: {chat_id}")
                    print(f"      Title: {title}")
                    chat_ids.add(chat_id)

        if chat_ids:
            print(f"\n🎯 Recommended Configuration:")
            # Use the most recent chat ID (usually the personal one)
            recommended_id = list(chat_ids)[0]
            print(f"Add this to your .env file:")
            print(f"TELEGRAM_CHAT_ID={recommended_id}")

            print(f"\n🧪 Test your setup:")
            print(f"python test_telegram.py")

    except Exception as e:
        print(f"❌ Error getting chat ID: {e}")
        print("Please check your bot token and try again")


if __name__ == "__main__":
    get_chat_id()
