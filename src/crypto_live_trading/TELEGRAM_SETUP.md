# Telegram Bot Setup Guide

This guide will help you set up Telegram notifications for your crypto trading system.

## Step 1: Create a Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Start a chat with BotFather and send `/newbot`
3. Follow the instructions to create your bot:
   - Choose a name for your bot (e.g., "Crypto Trading Bot")
   - Choose a username for your bot (must end with 'bot', e.g., "crypto_trading_bot")
4. BotFather will give you a **Bot Token** - save this!

## Step 2: Get Your Chat ID (IMPORTANT!)

⚠️ **Common Mistake**: Don't use your bot's chat ID - bots can't send messages to other bots!

### Method 1: Get Your Personal Chat ID (Recommended)

1. Search for `@userinfobot` in Telegram
2. Start a chat with @userinfobot and send any message
3. The bot will reply with your user information including your **Chat ID**
4. Your personal chat ID will be a positive number (e.g., `123456789`)

### Method 2: Using Your Bot to Get Your Chat ID

1. **Start a chat with YOUR bot** (the one you just created, not another bot!)
2. Send any message to your bot (e.g., "Hello")
3. Open this URL in your browser (replace `YOUR_BOT_TOKEN` with your actual token):
   ```
   https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
   ```
4. Look for the "chat" object in the response and find the "id" field
5. This should be your personal user ID (positive number)

### Method 3: Create a Private Group (Alternative)

1. Create a new group in Telegram
2. Add your bot to the group
3. Make your bot an admin (optional but recommended)
4. Send a message in the group
5. Use the getUpdates URL method above to get the group chat ID
6. Group chat IDs are usually negative numbers (e.g., `-123456789`)

## Step 3: Configure Environment Variables

Add these variables to your `.env` file:

```bash
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

**Important Notes:**

- `TELEGRAM_BOT_TOKEN`: Your bot's token from BotFather
- `TELEGRAM_CHAT_ID`: YOUR personal user ID or group ID (NOT another bot's ID)

## Step 4: Test Your Setup

Run the test script to verify everything works:

```bash
cd src/crypto_live_trading
python test_telegram.py
```

## Common Issues & Solutions

### ❌ Error: "Forbidden: bots can't send messages to bots"

**Problem**: You're using another bot's chat ID instead of your personal user ID
**Solution**:

- Use Method 1 above to get your personal chat ID
- Make sure you're getting YOUR user ID, not your bot's ID
- Your personal chat ID should be a positive number

### ❌ Error: "Forbidden: bot was blocked by the user"

**Problem**: You blocked the bot or haven't started a conversation with it
**Solution**:

- Start a chat with your bot and send it a message first
- Make sure you haven't blocked the bot

### ❌ Error: "Bad Request: chat not found"

**Problem**: Invalid chat ID
**Solution**:

- Double-check your chat ID is correct
- Make sure there are no extra spaces or characters
- Try the alternative methods to get your chat ID

### ❌ Error: "Unauthorized"

**Problem**: Invalid bot token
**Solution**:

- Check your bot token is complete and correct
- Make sure there are no extra spaces in the token
- Get a fresh token from BotFather if needed

## Notification Types

The system will send the following types of notifications:

- 🟢 **Position Opened**: When a new trading position is opened
- 🔴 **Position Closed**: When a position is closed
- ❌ **Errors**: When trading errors occur
- 🚀 **System Start**: When the trading system starts
- ⚠️ **Margin Warnings**: When margin levels need attention
- 💰 **Margin Actions**: When margin is added or reduced

## Message Format

Messages use Telegram's Markdown formatting for better readability:

- `code blocks` for values
- **bold text** for headers
- Emojis for visual indicators

## Security Notes

- Keep your bot token secret - don't share it or commit it to version control
- Consider using a dedicated group chat for trading notifications
- You can disable notifications by removing the environment variables
- Your personal chat ID is not sensitive information, but keep your bot token private

## Quick Troubleshooting Checklist

1. ✅ Created bot with @BotFather
2. ✅ Got bot token from BotFather
3. ✅ Started conversation with YOUR bot (sent at least one message)
4. ✅ Got YOUR personal chat ID (not your bot's ID)
5. ✅ Added both variables to .env file
6. ✅ No extra spaces in token or chat ID
7. ✅ Ran test script successfully
