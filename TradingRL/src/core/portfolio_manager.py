from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio
from collections import defaultdict
import numpy as np
import pandas as pd

from TradingRL.src.analysis.event_manager import (
    EventManager,
    Event,
    EventType,
    EventPriority,
)
from TradingRL.src.core.risk_manager import RiskManager
from TradingRL.src.core.order_manager import OrderManager


@dataclass
class Position:
    """Position information."""

    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime


class PortfolioManager:
    """Manages portfolio positions and performance tracking."""

    def __init__(
        self,
        event_manager: EventManager,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        initial_capital: float,
    ):
        """Initialize portfolio manager."""
        self.event_manager = event_manager
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self.initial_capital = initial_capital

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Portfolio tracking
        self.balance = initial_capital
        self.positions: Dict[str, Position] = {}
        self.position_history = []
        self.trade_history = []

        # Performance tracking
        self.metrics = defaultdict(float)
        self.daily_returns = []
        self.portfolio_values = []

        # Subscribe to events
        self._setup_event_subscriptions()

        self._running = False
        self._update_task = None

    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions."""
        self.event_manager.subscribe(
            self._handle_price_update,
            [EventType.PRICE_UPDATE],
        )
        self.event_manager.subscribe(
            self._handle_trade_update,
            [EventType.TRADE_UPDATE],
        )

    async def _handle_price_update(self, event: Event) -> None:
        """Handle price update events."""
        try:
            symbol = event.data["symbol"]
            price = event.data["price"]

            if symbol in self.positions:
                position = self.positions[symbol]
                position.current_price = price
                position.unrealized_pnl = (
                    price - position.entry_price
                ) * position.quantity

                # Update risk metrics
                self.risk_manager.update_portfolio_value(self.get_total_value())

        except Exception as e:
            self.logger.error(f"Error handling price update: {e}")

    async def _handle_trade_update(self, event: Event) -> None:
        """Handle trade update events."""
        try:
            trade_data = event.data
            symbol = trade_data["symbol"]
            quantity = trade_data["quantity"]
            price = trade_data["price"]
            side = trade_data["side"]

            if side == "buy":
                await self._open_position(symbol, quantity, price)
            else:
                await self._close_position(symbol, quantity, price)

        except Exception as e:
            self.logger.error(f"Error handling trade update: {e}")

    async def _open_position(self, symbol: str, quantity: float, price: float) -> None:
        """Open a new position."""
        try:
            position = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                timestamp=datetime.now(),
            )

            self.positions[symbol] = position
            self.position_history.append(position)
            self.balance -= quantity * price

            await self.event_manager.publish(
                Event(
                    type=EventType.POSITION_OPENED,
                    data={
                        "symbol": symbol,
                        "quantity": quantity,
                        "price": price,
                        "timestamp": datetime.now(),
                    },
                )
            )

        except Exception as e:
            self.logger.error(f"Error opening position: {e}")

    async def _close_position(self, symbol: str, quantity: float, price: float) -> None:
        """Close an existing position."""
        try:
            if symbol in self.positions:
                position = self.positions[symbol]
                pnl = (price - position.entry_price) * quantity
                self.balance += quantity * price + pnl

                # Update metrics
                self.metrics["total_pnl"] += pnl
                self.metrics["total_trades"] += 1
                if pnl > 0:
                    self.metrics["winning_trades"] += 1

                # Record trade
                self.trade_history.append(
                    {
                        "symbol": symbol,
                        "entry_price": position.entry_price,
                        "exit_price": price,
                        "quantity": quantity,
                        "pnl": pnl,
                        "timestamp": datetime.now(),
                    }
                )

                # Remove position if fully closed
                if quantity >= position.quantity:
                    del self.positions[symbol]

                await self.event_manager.publish(
                    Event(
                        type=EventType.POSITION_CLOSED,
                        data={
                            "symbol": symbol,
                            "quantity": quantity,
                            "price": price,
                            "pnl": pnl,
                            "timestamp": datetime.now(),
                        },
                    )
                )

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")

    def get_total_value(self) -> float:
        """Get total portfolio value including positions."""
        try:
            position_value = sum(
                pos.quantity * pos.current_price for pos in self.positions.values()
            )
            return self.balance + position_value
        except Exception as e:
            self.logger.error(f"Error calculating total value: {e}")
            return self.balance

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position information for a symbol."""
        return self.positions.get(symbol)

    async def close_all_positions(self) -> None:
        """Close all open positions."""
        try:
            for symbol, position in list(self.positions.items()):
                await self._close_position(
                    symbol, position.quantity, position.current_price
                )
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")

    async def handle_risk_event(self, event: Dict[str, Any]) -> None:
        """Handle risk-related events."""
        try:
            # Extract risk data
            drawdown = event["data"].get("drawdown", 0.0)
            positions = event["data"].get("positions", [])

            # Check if emergency action is needed
            if drawdown > self.risk_manager.config["max_drawdown"]:
                self.logger.warning(f"Emergency risk handling: Drawdown {drawdown:.2%}")
                await self._handle_emergency_risk(positions)

            # Update position risk metrics
            for position in positions:
                await self._update_position_risk(position)

        except Exception as e:
            self.logger.error(f"Error handling risk event: {e}")

    async def _handle_emergency_risk(self, positions: List[Dict[str, Any]]) -> None:
        """Handle emergency risk situations."""
        try:
            # Close all positions if drawdown exceeds limit
            for position in positions:
                await self.close_position(
                    symbol=position["symbol"],
                    reason="emergency_risk",
                    metadata={"drawdown": True},
                )

            # Update risk manager status
            await self.risk_manager.update_risk_status(emergency=True)

            # Emit emergency event
            event = Event(
                type=EventType.RISK_UPDATE,
                data={
                    "action": "emergency_closure",
                    "positions": positions,
                    "timestamp": datetime.now(),
                },
                priority=EventPriority.HIGH,
            )
            await self.event_manager.publish(event)

            self.logger.warning(
                f"Emergency risk handling completed. Closed {len(positions)} positions"
            )

        except Exception as e:
            self.logger.error(f"Error in emergency risk handling: {e}")

    async def _update_position_risk(self, position: Dict[str, Any]) -> None:
        """Update risk metrics for a position."""
        try:
            symbol = position["symbol"]
            if symbol in self.positions:
                # Update position risk metrics
                self.positions[symbol].update(
                    {
                        "risk_score": await self.risk_manager.calculate_position_risk(
                            position
                        ),
                        "last_risk_update": datetime.now(),
                    }
                )
        except Exception as e:
            self.logger.error(f"Error updating position risk: {e}")

    async def start(self) -> None:
        """Start portfolio manager."""
        if self._running:
            return
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        self.logger.info("Portfolio manager started")

    async def stop(self) -> None:
        """Stop portfolio manager."""
        if not self._running:
            return
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Portfolio manager stopped")

    async def _update_loop(self) -> None:
        """Background loop for portfolio updates."""
        while self._running:
            try:
                # Update portfolio metrics
                total_value = self.get_total_value()
                drawdown = (self.initial_capital - total_value) / self.initial_capital

                # Check risk limits
                if drawdown > self.risk_manager.config["max_drawdown"]:
                    await self.handle_risk_event(
                        {
                            "type": EventType.RISK_UPDATE,
                            "data": {
                                "drawdown": drawdown,
                                "positions": list(self.positions.values()),
                            },
                        }
                    )

                # Emit portfolio update event
                await self.event_manager.publish(
                    Event(
                        type=EventType.POSITION_UPDATE,
                        data={
                            "total_value": total_value,
                            "balance": self.balance,
                            "positions": {
                                symbol: vars(pos)
                                for symbol, pos in self.positions.items()
                            },
                            "metrics": self.metrics,
                            "timestamp": datetime.now(),
                        },
                    )
                )

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                self.logger.error(f"Error in portfolio update loop: {e}")
                await asyncio.sleep(5)  # Short delay on error

    async def close_position(
        self, symbol: str, reason: str = "", metadata: Dict = None
    ) -> None:
        """Close a position."""
        try:
            if symbol not in self.positions:
                self.logger.warning(f"No position found for {symbol}")
                return

            position = self.positions[symbol]

            # Create close order
            order = {
                "symbol": symbol,
                "side": "sell" if position["side"] == "long" else "buy",
                "type": "MARKET",
                "quantity": position["quantity"],
                "reduce_only": True,
                "metadata": {
                    "reason": reason,
                    **(metadata or {}),
                },
            }

            # Submit close order
            await self.order_manager.submit_order(order)

            # Record position closure
            self.position_history.append(
                {
                    **position,
                    "exit_price": position["current_price"],
                    "exit_time": datetime.now(),
                    "reason": reason,
                    "metadata": metadata,
                }
            )

            # Remove position
            del self.positions[symbol]

            # Emit position closed event
            await self.event_manager.publish(
                Event(
                    type=EventType.POSITION_CLOSED,
                    data={
                        "symbol": symbol,
                        "position": position,
                        "reason": reason,
                        "metadata": metadata,
                        "timestamp": datetime.now(),
                    },
                )
            )

            self.logger.info(f"Closed position for {symbol}: {position}")

        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            raise
