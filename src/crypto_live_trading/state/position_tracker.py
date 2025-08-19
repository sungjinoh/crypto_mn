"""
Position Tracker stub
"""


class PositionTracker:
    def load_cointegrated_pairs(self, path):
        # TODO: Load pairs from file
        return []

    def get_position(self, symbol1, symbol2):
        # TODO: Check if position is open
        return False

    def record_open(self, symbol1, symbol2, signal):
        # TODO: Record open position
        pass

    def record_close(self, symbol1, symbol2, signal):
        # TODO: Record close position
        pass
