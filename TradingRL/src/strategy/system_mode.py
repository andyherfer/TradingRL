from enum import Enum


class SystemMode(Enum):
    """System operation modes."""

    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    TRAIN = "train"
    TEST = "test"
