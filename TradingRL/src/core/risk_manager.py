from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
import numpy as np
from datetime import datetime

from TradingRL.src.analysis.event_manager import (
    EventManager,
    Event,
    EventType,
    EventPriority,
)


@dataclass
class RiskConfig:
    """Risk management configuration."""

    max_position_size: float = 0.1
    max_drawdown: float = 0.2
    stop_loss: float = 0.02
    take_profit: float = 0.05
    max_leverage: float = 1.0
    position_sizing_method: str = "fixed"


class RiskManager:
    """Manages trading risk and position sizing."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize risk manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            "current_drawdown": 0.0,
            "max_drawdown": 0.0,
            "risk_exposure": 0.0,
            "position_sizes": {},
            "emergency_status": False,
        }

    async def update_risk_status(self, emergency: bool = False) -> None:
        """Update risk status and handle emergency situations."""
        try:
            self.metrics["emergency_status"] = emergency

            if emergency:
                self.logger.warning("Emergency risk status activated")
                # Emit risk alert event if we have event manager
                if hasattr(self, "event_manager"):
                    await self.event_manager.publish(
                        Event(
                            type=EventType.RISK_UPDATE,
                            data={
                                "status": "emergency",
                                "metrics": self.metrics,
                                "timestamp": datetime.now(),
                            },
                            priority=EventPriority.HIGH,
                        )
                    )
            else:
                self.logger.info("Risk status normalized")

        except Exception as e:
            self.logger.error(f"Error updating risk status: {e}")

    async def update_portfolio_value(self, total_value: float) -> None:
        """Update portfolio value and risk metrics."""
        try:
            # Calculate drawdown
            if not hasattr(self, "peak_value"):
                self.peak_value = total_value
            elif total_value > self.peak_value:
                self.peak_value = total_value

            current_drawdown = (self.peak_value - total_value) / self.peak_value
            self.metrics["current_drawdown"] = current_drawdown

            # Update max drawdown
            if current_drawdown > self.metrics["max_drawdown"]:
                self.metrics["max_drawdown"] = current_drawdown

            # Check for emergency conditions
            if current_drawdown > self.config["max_drawdown"]:
                await self.update_risk_status(emergency=True)

        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {e}")

    async def calculate_position_risk(self, position: Dict[str, Any]) -> float:
        """Calculate risk score for a position."""
        try:
            # Basic risk calculation based on position size and unrealized PnL
            position_size = position.get("quantity", 0) * position.get(
                "current_price", 0
            )
            unrealized_pnl = position.get("unrealized_pnl", 0)

            # Calculate risk factors
            size_risk = position_size / self.config.get("max_position_size", 1.0)
            pnl_risk = abs(unrealized_pnl) / position_size if position_size > 0 else 0

            # Combine risk factors (weighted average)
            risk_score = (size_risk * 0.6) + (pnl_risk * 0.4)

            return min(risk_score, 1.0)  # Normalize to 0-1

        except Exception as e:
            self.logger.error(f"Error calculating position risk: {e}")
            return 1.0  # Return max risk on error

    async def calculate_position_size(
        self, amount: float, risk_factor: float = 0.5
    ) -> float:
        """
        Calculate position size based on risk parameters.

        Args:
            amount: Base amount for position
            risk_factor: Risk factor between 0 and 1
        """
        try:
            # Get max position size from config
            max_size = self.config.get("max_position_size", 1.0)

            # Calculate position size based on risk factor
            position_size = amount * risk_factor * max_size

            # Ensure within limits
            position_size = min(position_size, max_size)
            position_size = max(position_size, 0.0)

            return position_size

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0

    async def check_risk_limits(
        self, position_size: float, current_drawdown: float
    ) -> bool:
        """
        Check if position size and drawdown are within risk limits.

        Args:
            position_size: Position size to check
            current_drawdown: Current drawdown percentage
        """
        try:
            # Get risk limits from config
            max_position = self.config.get("max_position_size", 1.0)
            max_drawdown = self.config.get("max_drawdown", 0.2)

            # Check limits
            if position_size > max_position:
                return False

            if current_drawdown > max_drawdown:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False

    def get_stop_loss_price(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price based on configuration."""
        try:
            if side == "buy":
                return entry_price * (1 - self.config.stop_loss)
            else:
                return entry_price * (1 + self.config.stop_loss)
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            return entry_price

    def get_take_profit_price(self, entry_price: float, side: str) -> float:
        """Calculate take profit price based on configuration."""
        try:
            if side == "buy":
                return entry_price * (1 + self.config.take_profit)
            else:
                return entry_price * (1 - self.config.take_profit)
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {e}")
            return entry_price
