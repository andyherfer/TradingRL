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
        self._processing_task = None

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
        if self.is_running:
            return

        self.is_running = True
        self._processing_task = asyncio.create_task(self._process_events())

        # Signal successful startup without waiting for events
        await asyncio.sleep(0)

    async def stop(self) -> None:
        """Stop event processing."""
        if not self.is_running:
            return

        self.is_running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None

    async def _process_events(self) -> None:
        """Process events from queue."""
        try:
            while self.is_running:
                try:
                    async with asyncio.timeout(1.0):  # 1 second timeout
                        event = await self.event_queue.get()
                        await self._process_event(event)
                        self.event_queue.task_done()
                except TimeoutError:
                    continue  # No events to process, continue loop
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error processing event: {e}")
        except Exception as e:
            self.logger.error(f"Error in event processing loop: {e}")
            self.is_running = False

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
