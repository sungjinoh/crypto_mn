"""
Scheduler - Handles timing for live trading system
"""

import time
from datetime import datetime, timedelta


def every_4h():
    """
    Generator that yields every 4 hours at specific times
    (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC)
    """
    while True:
        current_time = datetime.utcnow()

        # Calculate next 4-hour interval
        hour = current_time.hour
        next_hour = (hour // 4 + 1) * 4

        if next_hour >= 24:
            next_hour = 0
            next_day = current_time + timedelta(days=1)
            next_run = next_day.replace(
                hour=next_hour, minute=0, second=0, microsecond=0
            )
        else:
            next_run = current_time.replace(
                hour=next_hour, minute=0, second=0, microsecond=0
            )

        # Sleep until next run time
        sleep_seconds = (next_run - current_time).total_seconds()

        print(f"Next trading cycle at: {next_run} UTC")
        print(f"Sleeping for {sleep_seconds/3600:.2f} hours...")

        time.sleep(sleep_seconds)
        yield


def every_n_minutes(n=5):
    """
    Generator that yields every N minutes (for testing)

    Args:
        n: Minutes between runs
    """
    while True:
        current_time = datetime.utcnow()
        print(f"Trading cycle at: {current_time} UTC")

        # Sleep for N minutes
        time.sleep(n * 60)
        yield


def run_once():
    """
    Generator that yields once (for testing)
    """
    yield
    return
