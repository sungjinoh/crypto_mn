"""
Live Mean Reversion Strategy stub
"""


class LiveMeanReversionStrategy:
    def __init__(
        self, lookback_period, entry_threshold, exit_threshold, stop_loss_threshold
    ):
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold

    def generate_signal(self, df1, df2):
        # TODO: Implement live signal logic
        class Signal:
            is_entry = False
            is_exit = False

        return Signal()
