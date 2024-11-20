from typing import Dict, Any
import logging
from datetime import datetime
import asyncio
from dataclasses import dataclass


@dataclass
class SafetyLimits:
    max_daily_loss: float = 100.0  # Maximum daily loss in USD
    max_position_size: float = 0.1  # Maximum position size as % of portfolio
    max_orders_per_minute: int = 5
    emergency_stop_loss: float = 0.05  # 5% max loss per position
    max_leverage: float = 1.0  # No leverage initially


class SafetyMonitor:
    """Critical safety monitoring for live trading."""

    def __init__(self, limits: SafetyLimits):
        self.limits = limits
        self.logger = logging.getLogger(__name__)
        self.daily_loss = 0.0
        self.order_count = 0
        self.last_order_time = datetime.now()
        self.active = True

    async def check_trade(self, order: Dict[str, Any]) -> bool:
        """Pre-trade safety checks."""
        try:
            # Check daily loss limit
            if (
                self.daily_loss + (order.get("potential_loss", 0) or 0)
                > self.limits.max_daily_loss
            ):
                self.logger.error(
                    f"Daily loss limit would be exceeded: {self.daily_loss}"
                )
                return False

            # Check position size
            if order.get("size", 0) > self.limits.max_position_size:
                self.logger.error(f"Position size exceeds limit: {order['size']}")
                return False

            # Check order rate
            now = datetime.now()
            if (now - self.last_order_time).seconds < 60:
                if self.order_count >= self.limits.max_orders_per_minute:
                    self.logger.error("Order rate limit exceeded")
                    return False

            # Verify stop loss
            if not order.get("stop_loss"):
                self.logger.error("No stop loss specified")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Safety check error: {e}")
            return False

    async def emergency_stop(self) -> None:
        """Emergency stop all trading."""
        self.active = False
        self.logger.critical("EMERGENCY STOP ACTIVATED")
        # Add code to close all positions
