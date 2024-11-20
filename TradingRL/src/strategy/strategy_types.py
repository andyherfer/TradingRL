from enum import Enum


class SignalType(Enum):
    """Trading signal types."""

    LONG = "long"
    SHORT = "short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    NO_SIGNAL = "no_signal"


class StrategyState(Enum):
    """Strategy states."""

    INACTIVE = "inactive"
    ACTIVE = "active"
    WARMUP = "warmup"
    OPTIMIZING = "optimizing"
    ERROR = "error"
