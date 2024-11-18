from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Callable, List
from datetime import datetime
import asyncio
import logging


class EventType(Enum):
    """Types of events in the system."""

    MARKET_DATA = "market_data"
    PRICE_UPDATE = "price_update"
    ORDER_UPDATE = "order_update"
    TRADE_UPDATE = "trade_update"
    POSITION_UPDATE = "position_update"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    MARKET_REGIME_CHANGE = "market_regime_change"
    RISK_UPDATE = "risk_update"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"
    LIQUIDITY_UPDATE = "liquidity_update"


class EventPriority(Enum):
    """Priority levels for event processing."""

    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class Event:
    """Event data structure."""

    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = None
    priority: EventPriority = EventPriority.NORMAL

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EventManager:
    """Manages event distribution and handling."""

    def __init__(self):
        """Initialize event manager."""
        self.subscribers = {}
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.event_queue = asyncio.Queue()

    def subscribe(
        self,
        callback: Callable,
        event_types: List[EventType],
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        """Subscribe to events."""
        for event_type in event_types:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append((callback, priority))

    async def publish(self, event: Event) -> None:
        """Publish an event."""
        try:
            await self.event_queue.put(event)
        except Exception as e:
            self.logger.error(f"Error publishing event: {e}")

    async def start(self) -> None:
        """Start event processing."""
        self.is_running = True
        try:
            while self.is_running:
                event = await self.event_queue.get()
                await self._process_event(event)
                self.event_queue.task_done()
        except Exception as e:
            self.logger.error(f"Error in event processing loop: {e}")
            raise

    async def stop(self) -> None:
        """Stop event processing."""
        self.is_running = False
        # Process remaining events
        while not self.event_queue.empty():
            event = await self.event_queue.get()
            await self._process_event(event)
            self.event_queue.task_done()

    async def _process_event(self, event: Event) -> None:
        """Process a single event."""
        try:
            if event.type in self.subscribers:
                # Sort subscribers by priority
                sorted_subscribers = sorted(
                    self.subscribers[event.type], key=lambda x: x[1].value
                )

                # Call each subscriber
                for callback, _ in sorted_subscribers:
                    try:
                        await callback(event)
                    except Exception as e:
                        self.logger.error(f"Error in event handler: {e}")

        except Exception as e:
            self.logger.error(f"Error processing event: {e}")
