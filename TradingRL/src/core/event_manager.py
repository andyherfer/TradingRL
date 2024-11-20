from typing import Callable, Dict, List, Optional, Set, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum
from queue import PriorityQueue
import asyncio
from collections import defaultdict
import json
import uuid
import wandb
import time
import traceback


class EventPriority(Enum):
    """Priority levels for events."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class EventType(Enum):
    """Types of system events."""

    # Market Events
    PRICE_UPDATE = "price_update"
    MARKET_REGIME_CHANGE = "market_regime_change"
    LIQUIDITY_UPDATE = "liquidity_update"

    # Trading Events
    SIGNAL_GENERATED = "signal_generated"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"

    # Risk Events
    RISK_LIMIT_BREACH = "risk_limit_breach"
    DRAWDOWN_ALERT = "drawdown_alert"
    EXPOSURE_ALERT = "exposure_alert"
    VOLATILITY_ALERT = "volatility_alert"

    # System Events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    MODULE_ERROR = "module_error"
    HEALTH_CHECK = "health_check"
    PERFORMANCE_ALERT = "performance_alert"


@dataclass
class Event:
    """Event data structure."""

    type: EventType
    priority: EventPriority
    timestamp: datetime
    data: Dict[str, Any]
    id: str = None
    source: str = None

    def __post_init__(self):
        """Initialize event ID if not provided."""
        if self.id is None:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict:
        """Convert event to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
        }


class EventManager:
    """
    Manages system-wide events and inter-module communication.
    Implements a pub/sub pattern with priority queuing and async processing.
    """

    def __init__(
        self,
        max_queue_size: int = 1000,
        processing_interval: float = 0.1,
        use_wandb: bool = True,
    ):
        """
        Initialize the EventManager.

        Args:
            max_queue_size: Maximum size of event queue
            processing_interval: Interval between queue processing in seconds
            use_wandb: Whether to log to Weights & Biases
        """
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Event queue and subscribers
        self.event_queue = PriorityQueue(maxsize=max_queue_size)
        self.subscribers = defaultdict(set)
        self.global_subscribers = set()

        # Processing settings
        self.processing_interval = processing_interval
        self.is_running = False
        self.use_wandb = use_wandb

        # Metrics tracking
        self.metrics = {
            "events_processed": 0,
            "events_dropped": 0,
            "processing_times": [],
            "queue_sizes": [],
            "error_count": 0,
        }

        # Event history for debugging
        self.event_history = []
        self.max_history = 1000

        # Initialize asyncio event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    async def start(self):
        """Start the event processing loop."""
        self.is_running = True
        try:
            while self.is_running:
                await self.process_event_queue()
                await asyncio.sleep(self.processing_interval)
        except Exception as e:
            self.logger.error(f"Error in event processing loop: {e}")
            self.metrics["error_count"] += 1

    async def stop(self):
        """Stop the event processing loop."""
        self.is_running = False
        # Process remaining events
        while not self.event_queue.empty():
            await self.process_event_queue()

    def subscribe(
        self,
        callback: Callable[[Event], None],
        event_types: Optional[List[EventType]] = None,
        priority_threshold: EventPriority = EventPriority.LOW,
    ) -> None:
        """
        Subscribe to events.

        Args:
            callback: Function to call when event occurs
            event_types: List of event types to subscribe to (None for all)
            priority_threshold: Minimum priority level to receive
        """
        subscriber = {"callback": callback, "priority_threshold": priority_threshold}

        if event_types is None:
            self.global_subscribers.add(frozenset(subscriber.items()))
        else:
            for event_type in event_types:
                self.subscribers[event_type].add(frozenset(subscriber.items()))

        self.logger.info(f"New subscription added for {event_types or 'all events'}")

    def unsubscribe(
        self,
        callback: Callable[[Event], None],
        event_types: Optional[List[EventType]] = None,
    ) -> None:
        """
        Unsubscribe from events.

        Args:
            callback: Callback function to remove
            event_types: Event types to unsubscribe from (None for all)
        """
        if event_types is None:
            # Remove from global subscribers
            self.global_subscribers = {
                sub
                for sub in self.global_subscribers
                if dict(sub)["callback"] != callback
            }
        else:
            # Remove from specific event types
            for event_type in event_types:
                self.subscribers[event_type] = {
                    sub
                    for sub in self.subscribers[event_type]
                    if dict(sub)["callback"] != callback
                }

    async def publish(self, event: Event) -> None:
        """
        Publish an event to the queue.

        Args:
            event: Event to publish
        """
        try:
            # Add to queue with priority
            await asyncio.get_event_loop().run_in_executor(
                None, self.event_queue.put, (-event.priority.value, event)
            )

            # Update metrics
            self.metrics["queue_sizes"].append(self.event_queue.qsize())

            # Log to W&B if enabled
            if self.use_wandb and wandb.run is not None:
                wandb.log(
                    {
                        "event_manager/queue_size": self.event_queue.qsize(),
                        f"event_manager/event_type_{event.type.value}": 1,
                    }
                )

        except Exception as e:
            self.logger.error(f"Error publishing event: {e}")
            self.metrics["error_count"] += 1
            self.metrics["events_dropped"] += 1

    async def process_event_queue(self) -> None:
        """Process events in the queue."""
        if self.event_queue.empty():
            return

        start_time = time.time()

        try:
            # Get event from queue
            _, event = await asyncio.get_event_loop().run_in_executor(
                None, self.event_queue.get
            )

            # Store in history
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)

            # Get relevant subscribers
            subscribers = set().union(
                self.global_subscribers, self.subscribers[event.type]
            )

            # Notify subscribers
            for sub in subscribers:
                sub_dict = dict(sub)
                if event.priority.value >= sub_dict["priority_threshold"].value:
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, sub_dict["callback"], event
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Error in subscriber callback: {e}\n{traceback.format_exc()}"
                        )
                        self.metrics["error_count"] += 1

            # Update metrics
            self.metrics["events_processed"] += 1
            processing_time = time.time() - start_time
            self.metrics["processing_times"].append(processing_time)

            # Log to W&B if enabled
            if self.use_wandb and wandb.run is not None:
                wandb.log(
                    {
                        "event_manager/processing_time": processing_time,
                        "event_manager/events_processed": self.metrics[
                            "events_processed"
                        ],
                        "event_manager/error_count": self.metrics["error_count"],
                    }
                )

        except Exception as e:
            self.logger.error(f"Error processing event queue: {e}")
            self.metrics["error_count"] += 1

    def get_metrics(self) -> Dict:
        """Get current event manager metrics."""
        return {
            "events_processed": self.metrics["events_processed"],
            "events_dropped": self.metrics["events_dropped"],
            "average_processing_time": (
                sum(self.metrics["processing_times"])
                / len(self.metrics["processing_times"])
                if self.metrics["processing_times"]
                else 0
            ),
            "average_queue_size": (
                sum(self.metrics["queue_sizes"]) / len(self.metrics["queue_sizes"])
                if self.metrics["queue_sizes"]
                else 0
            ),
            "error_count": self.metrics["error_count"],
        }

    def get_recent_events(self, n: int = 10) -> List[Dict]:
        """Get the n most recent events."""
        return [event.to_dict() for event in self.event_history[-n:]]

    async def replay_events(
        self, events: List[Event], speed_multiplier: float = 1.0
    ) -> None:
        """
        Replay a list of events (useful for testing and debugging).

        Args:
            events: List of events to replay
            speed_multiplier: Speed up or slow down replay
        """
        events = sorted(events, key=lambda x: x.timestamp)
        last_time = events[0].timestamp

        for event in events:
            # Calculate delay
            delay = (event.timestamp - last_time).total_seconds() / speed_multiplier
            if delay > 0:
                await asyncio.sleep(delay)

            # Publish event
            await self.publish(event)
            last_time = event.timestamp
