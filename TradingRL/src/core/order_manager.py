from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio
from enum import Enum
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict
import time
import wandb
from binance.client import Client
from binance.exceptions import BinanceAPIException

from ..analysis.event_manager import EventManager, Event, EventType, EventPriority
from .risk_manager import RiskManager


class OrderStatus(Enum):
    """Order status enumeration."""

    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order type enumeration."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderConfig:
    """Order configuration parameters."""

    max_slippage: float = 0.001  # Maximum allowed slippage
    min_order_size: float = 0.001  # Minimum order size in BTC
    max_order_size: float = 1.0  # Maximum order size in BTC
    default_timeout: int = 30  # Default order timeout in seconds
    retry_attempts: int = 3  # Number of retry attempts for failed orders
    retry_delay: int = 1  # Delay between retries in seconds


@dataclass
class Order:
    """Order data structure."""

    id: str
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: float
    price: Optional[float]
    status: OrderStatus
    timestamp: datetime
    params: Dict[str, Any]
    fills: List[Dict[str, Any]]

    def __post_init__(self):
        """Initialize order ID if not provided."""
        if not hasattr(self, "id"):
            self.id = str(uuid.uuid4())
        if not hasattr(self, "fills"):
            self.fills = []


class OrderManager:
    """
    Manages order creation, execution, and lifecycle tracking.
    Integrates with exchange API for order execution.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        event_manager: EventManager,
        risk_manager: RiskManager,
        config: Optional[OrderConfig] = None,
        use_wandb: bool = True,
    ):
        """
        Initialize OrderManager.

        Args:
            api_key: Exchange API key
            api_secret: Exchange API secret
            event_manager: EventManager instance
            risk_manager: RiskManager instance
            config: Order configuration parameters
            use_wandb: Whether to use W&B logging
        """
        self.client = Client(api_key, api_secret)
        self.event_manager = event_manager
        self.risk_manager = risk_manager
        self.config = config or OrderConfig()
        self.use_wandb = use_wandb

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Order tracking
        self.active_orders = {}
        self.order_history = []
        self.filled_orders = []
        self.pending_orders = []

        # Performance tracking
        self.metrics = {
            "orders_placed": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "orders_rejected": 0,
            "total_volume": 0,
            "total_fees": 0,
            "average_slippage": [],
            "execution_times": [],
            "retry_counts": defaultdict(int),
        }

        # Subscribe to relevant events
        self._setup_event_subscriptions()

    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions."""
        self.event_manager.subscribe(
            self._handle_order_update,
            [EventType.ORDER_FILLED, EventType.ORDER_CANCELLED],
            EventPriority.HIGH,
        )

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None,
    ) -> Order:
        """
        Create a new order.

        Args:
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            order_type: Type of order
            quantity: Order quantity
            price: Order price (optional)
            params: Additional order parameters

        Returns:
            Created order object
        """
        try:
            # Validate order parameters
            self._validate_order_params(symbol, quantity, price)

            # Check risk limits
            risk_check = await self.risk_manager.check_risk_limits(
                {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price or await self._get_market_price(symbol),
                }
            )

            if not risk_check["approved"]:
                raise ValueError(f"Risk check failed: {risk_check['reason']}")

            # Create order object
            order = Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                order_type=order_type,
                side=side,
                quantity=quantity,
                price=price,
                status=OrderStatus.PENDING,
                timestamp=datetime.now(),
                params=params or {},
                fills=[],
            )

            # Add to tracking
            self.active_orders[order.id] = order
            self.pending_orders.append(order)

            # Log order creation
            self.logger.info(f"Created order: {order}")
            if self.use_wandb:
                wandb.log(
                    {
                        "order/created": 1,
                        "order/type": order_type.value,
                        "order/side": side.value,
                        "order/quantity": quantity,
                        "order/price": price,
                    }
                )

            return order

        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            raise

    async def execute_order(self, order: Order) -> Order:
        """
        Execute an order on the exchange.

        Args:
            order: Order to execute

        Returns:
            Updated order object
        """
        try:
            start_time = time.time()

            # Prepare order parameters
            exchange_params = self._prepare_exchange_params(order)

            # Execute order with retry logic
            for attempt in range(self.config.retry_attempts):
                try:
                    if order.order_type == OrderType.MARKET:
                        response = await self._execute_market_order(exchange_params)
                    else:
                        response = await self._execute_limit_order(exchange_params)

                    # Update order with response
                    order = self._update_order_from_response(order, response)
                    break

                except BinanceAPIException as e:
                    if attempt == self.config.retry_attempts - 1:
                        raise
                    self.metrics["retry_counts"][order.id] += 1
                    await asyncio.sleep(self.config.retry_delay)

            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["execution_times"].append(execution_time)
            self.metrics["orders_placed"] += 1

            # Log execution
            if self.use_wandb:
                wandb.log(
                    {
                        "execution/time": execution_time,
                        "execution/success": 1,
                        "execution/retries": self.metrics["retry_counts"][order.id],
                    }
                )

            return order

        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            order.status = OrderStatus.REJECTED
            self.metrics["orders_rejected"] += 1
            raise

    async def cancel_order(self, order_id: str) -> Order:
        """
        Cancel an active order.

        Args:
            order_id: ID of order to cancel

        Returns:
            Cancelled order object
        """
        try:
            order = self.active_orders.get(order_id)
            if not order:
                raise ValueError(f"Order not found: {order_id}")

            # Cancel on exchange
            response = await self._cancel_exchange_order(order)

            # Update order status
            order.status = OrderStatus.CANCELLED
            self.metrics["orders_cancelled"] += 1

            # Remove from active orders
            self.active_orders.pop(order_id)

            # Log cancellation
            if self.use_wandb:
                wandb.log(
                    {
                        "order/cancelled": 1,
                        "order/cancel_reason": response.get("reason", "manual"),
                    }
                )

            return order

        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            raise

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get list of open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        try:
            if symbol:
                return [
                    order
                    for order in self.active_orders.values()
                    if order.symbol == symbol
                    and order.status
                    in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]
                ]
            return [
                order
                for order in self.active_orders.values()
                if order.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]
            ]
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            raise

    async def _handle_order_update(self, event: Event) -> None:
        """Handle order update events."""
        try:
            order_id = event.data["order_id"]
            order = self.active_orders.get(order_id)

            if not order:
                return

            if event.type == EventType.ORDER_FILLED:
                await self._process_fill(order, event.data)
            elif event.type == EventType.ORDER_CANCELLED:
                await self._process_cancellation(order, event.data)

        except Exception as e:
            self.logger.error(f"Error handling order update: {e}")

    async def _process_fill(self, order: Order, fill_data: Dict) -> None:
        """Process an order fill."""
        try:
            # Update order fills
            order.fills.append(fill_data)

            # Calculate fill metrics
            fill_quantity = fill_data["quantity"]
            fill_price = fill_data["price"]

            # Update order status
            filled_quantity = sum(fill["quantity"] for fill in order.fills)
            if filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                self.metrics["orders_filled"] += 1
                self.filled_orders.append(order)
                self.active_orders.pop(order.id)
            else:
                order.status = OrderStatus.PARTIALLY_FILLED

            # Calculate and record slippage
            if order.order_type == OrderType.MARKET:
                expected_price = order.params.get("expected_price")
                if expected_price:
                    slippage = abs(fill_price - expected_price) / expected_price
                    self.metrics["average_slippage"].append(slippage)

            # Update metrics
            self.metrics["total_volume"] += fill_quantity * fill_price
            self.metrics["total_fees"] += fill_data.get("commission", 0)

            # Log fill
            if self.use_wandb:
                wandb.log(
                    {
                        "fill/quantity": fill_quantity,
                        "fill/price": fill_price,
                        "fill/slippage": slippage if "slippage" in locals() else 0,
                        "fill/fees": fill_data.get("commission", 0),
                    }
                )

        except Exception as e:
            self.logger.error(f"Error processing fill: {e}")
            raise

    async def _process_cancellation(self, order: Order, cancel_data: Dict) -> None:
        """Process an order cancellation."""
        try:
            order.status = OrderStatus.CANCELLED
            self.metrics["orders_cancelled"] += 1
            self.active_orders.pop(order.id)

            # Log cancellation
            if self.use_wandb:
                wandb.log(
                    {
                        "order/cancelled": 1,
                        "order/cancel_reason": cancel_data.get("reason", "unknown"),
                    }
                )

        except Exception as e:
            self.logger.error(f"Error processing cancellation: {e}")
            raise

    def get_order_metrics(self) -> Dict:
        """Get current order metrics."""
        try:
            return {
                "orders_placed": self.metrics["orders_placed"],
                "orders_filled": self.metrics["orders_filled"],
                "orders_cancelled": self.metrics["orders_cancelled"],
                "orders_rejected": self.metrics["orders_rejected"],
                "total_volume": self.metrics["total_volume"],
                "total_fees": self.metrics["total_fees"],
                "average_slippage": (
                    np.mean(self.metrics["average_slippage"])
                    if self.metrics["average_slippage"]
                    else 0
                ),
                "average_execution_time": (
                    np.mean(self.metrics["execution_times"])
                    if self.metrics["execution_times"]
                    else 0
                ),
                "active_orders_count": len(self.active_orders),
                "pending_orders_count": len(self.pending_orders),
            }
        except Exception as e:
            self.logger.error(f"Error getting order metrics: {e}")
            raise

    async def _get_market_price(self, symbol: str) -> float:
        """Get current market price for a symbol."""
        try:
            ticker = await self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:
            self.logger.error(f"Error getting market price: {e}")
            raise

    def _validate_order_params(
        self, symbol: str, quantity: float, price: Optional[float]
    ) -> None:
        """Validate order parameters."""
        if quantity < self.config.min_order_size:
            raise ValueError(f"Order size too small: {quantity}")
        if quantity > self.config.max_order_size:
            raise ValueError(f"Order size too large: {quantity}")
        if price and price <= 0:
            raise ValueError(f"Invalid price: {price}")
