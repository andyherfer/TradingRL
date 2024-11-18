from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import logging
import asyncio
import time
from collections import deque
import json
import wandb
from binance.client import Client
from binance.exceptions import BinanceAPIException

from ..analysis.event_manager import EventManager, Event, EventType, EventPriority
from .order_manager import OrderManager, OrderType, OrderSide, Order


class ExecutionAlgo(Enum):
    """Available execution algorithms."""

    MARKET = "market"
    TWAP = "twap"
    VWAP = "vwap"
    SMART = "smart"
    ADAPTIVE = "adaptive"
    ICEBERG = "iceberg"
    SNIPER = "sniper"


class ExecutionPriority(Enum):
    """Execution priority levels."""

    HIGH_URGENCY = "high_urgency"
    NORMAL = "normal"
    LOW_URGENCY = "low_urgency"


@dataclass
class ExecutionParams:
    """Execution algorithm parameters."""

    algo: ExecutionAlgo
    priority: ExecutionPriority
    start_time: datetime
    end_time: Optional[datetime] = None
    max_participation_rate: float = 0.1
    min_execution_size: float = 0.001
    urgency_factor: float = 0.5
    price_limit: Optional[float] = None
    iceberg_qty: Optional[float] = None


@dataclass
class ExecutionMetrics:
    """Execution performance metrics."""

    arrival_price: float
    average_execution_price: float
    implementation_shortfall: float
    market_impact: float
    timing_cost: float
    execution_time: float
    participation_rate: float
    num_child_orders: int
    slippage: float


class OrderExecutor:
    """
    Handles smart order execution and algorithm management.
    Implements various execution strategies to minimize market impact.
    """

    def __init__(
        self,
        client: Client,
        event_manager: EventManager,
        order_manager: OrderManager,
        use_wandb: bool = True,
    ):
        """
        Initialize OrderExecutor.

        Args:
            client: Binance client instance
            event_manager: EventManager instance
            order_manager: OrderManager instance
            use_wandb: Whether to use W&B logging
        """
        self.client = client
        self.event_manager = event_manager
        self.order_manager = order_manager
        self.use_wandb = use_wandb

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Execution tracking
        self.active_executions = {}
        self.execution_history = []
        self.market_data_cache = {}

        # Performance metrics
        self.metrics = defaultdict(list)

        # Initialize execution algorithms
        self.execution_algos = {
            ExecutionAlgo.MARKET: self._execute_market,
            ExecutionAlgo.TWAP: self._execute_twap,
            ExecutionAlgo.VWAP: self._execute_vwap,
            ExecutionAlgo.SMART: self._execute_smart,
            ExecutionAlgo.ADAPTIVE: self._execute_adaptive,
            ExecutionAlgo.ICEBERG: self._execute_iceberg,
            ExecutionAlgo.SNIPER: self._execute_sniper,
        }

        # Setup event subscriptions
        self._setup_event_subscriptions()

    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions."""
        self.event_manager.subscribe(
            self._handle_market_update,
            [EventType.PRICE_UPDATE, EventType.LIQUIDITY_UPDATE],
            EventPriority.HIGH,
        )

    async def execute_order(
        self, order: Order, params: ExecutionParams
    ) -> Tuple[bool, ExecutionMetrics]:
        """
        Execute an order using specified algorithm and parameters.

        Args:
            order: Order to execute
            params: Execution parameters

        Returns:
            Tuple of (success, execution_metrics)
        """
        try:
            # Validate execution parameters
            self._validate_execution_params(order, params)

            # Record arrival price
            arrival_price = await self._get_market_price(order.symbol)
            start_time = time.time()

            # Execute using specified algorithm
            execution_algo = self.execution_algos.get(params.algo)
            if not execution_algo:
                raise ValueError(f"Unknown execution algorithm: {params.algo}")

            # Execute order
            success, child_orders = await execution_algo(order, params)

            # Calculate execution metrics
            execution_time = time.time() - start_time
            execution_metrics = await self._calculate_execution_metrics(
                order, child_orders, arrival_price, execution_time
            )

            # Log execution
            await self._log_execution(order, params, execution_metrics)

            return success, execution_metrics

        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            raise

    async def _execute_market(
        self, order: Order, params: ExecutionParams
    ) -> Tuple[bool, List[Order]]:
        """Execute order using market execution."""
        try:
            child_orders = []
            remaining_qty = order.quantity

            # Split into smaller orders if size is large
            max_single_order = self._calculate_max_order_size(order.symbol)

            while remaining_qty > 0:
                # Calculate child order size
                child_qty = min(remaining_qty, max_single_order)

                # Create and execute child order
                child_order = await self.order_manager.create_order(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=OrderType.MARKET,
                    quantity=child_qty,
                )

                child_orders.append(child_order)
                remaining_qty -= child_qty

                # Add small delay between orders
                if remaining_qty > 0:
                    await asyncio.sleep(0.1)

            return True, child_orders

        except Exception as e:
            self.logger.error(f"Error in market execution: {e}")
            return False, child_orders

    async def _execute_twap(
        self, order: Order, params: ExecutionParams
    ) -> Tuple[bool, List[Order]]:
        """Execute order using Time-Weighted Average Price strategy."""
        try:
            child_orders = []
            remaining_qty = order.quantity

            # Calculate time parameters
            if not params.end_time:
                params.end_time = params.start_time + timedelta(hours=1)

            total_duration = (params.end_time - params.start_time).total_seconds()
            num_intervals = max(1, int(total_duration / 60))  # 1-minute intervals
            qty_per_interval = remaining_qty / num_intervals

            # Execute child orders
            for i in range(num_intervals):
                if remaining_qty <= 0:
                    break

                # Calculate child order size
                child_qty = min(remaining_qty, qty_per_interval)

                # Create and execute child order
                child_order = await self.order_manager.create_order(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=OrderType.MARKET,
                    quantity=child_qty,
                )

                child_orders.append(child_order)
                remaining_qty -= child_qty

                # Wait for next interval
                await asyncio.sleep(total_duration / num_intervals)

            return True, child_orders

        except Exception as e:
            self.logger.error(f"Error in TWAP execution: {e}")
            return False, child_orders

    async def _execute_vwap(
        self, order: Order, params: ExecutionParams
    ) -> Tuple[bool, List[Order]]:
        """Execute order using Volume-Weighted Average Price strategy."""
        try:
            child_orders = []
            remaining_qty = order.quantity

            # Get historical volume profile
            volume_profile = await self._get_volume_profile(
                order.symbol,
                params.start_time,
                params.end_time or params.start_time + timedelta(hours=1),
            )

            # Calculate volume-weighted intervals
            intervals = self._calculate_vwap_intervals(volume_profile, remaining_qty)

            # Execute child orders
            for interval in intervals:
                if remaining_qty <= 0:
                    break

                # Calculate child order size based on volume profile
                child_qty = min(remaining_qty, interval["quantity"])

                # Create and execute child order
                child_order = await self.order_manager.create_order(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=OrderType.MARKET,
                    quantity=child_qty,
                )

                child_orders.append(child_order)
                remaining_qty -= child_qty

                # Wait for next interval
                await asyncio.sleep(interval["duration"])

            return True, child_orders

        except Exception as e:
            self.logger.error(f"Error in VWAP execution: {e}")
            return False, child_orders

    async def _execute_smart(
        self, order: Order, params: ExecutionParams
    ) -> Tuple[bool, List[Order]]:
        """Execute order using Smart execution strategy."""
        try:
            child_orders = []
            remaining_qty = order.quantity

            # Get market state
            market_state = await self._analyze_market_state(order.symbol)

            # Choose optimal execution strategy based on market state
            if market_state["volatility"] > 0.02:  # High volatility
                execution_func = self._execute_adaptive
            elif market_state["liquidity"] < 0.5:  # Low liquidity
                execution_func = self._execute_iceberg
            elif market_state["spread"] > 0.0005:  # Wide spread
                execution_func = self._execute_sniper
            else:
                execution_func = self._execute_twap

            # Execute using chosen strategy
            success, orders = await execution_func(order, params)
            child_orders.extend(orders)

            return success, child_orders

        except Exception as e:
            self.logger.error(f"Error in smart execution: {e}")
            return False, child_orders

    async def _execute_adaptive(
        self, order: Order, params: ExecutionParams
    ) -> Tuple[bool, List[Order]]:
        """Execute order using Adaptive execution strategy."""
        try:
            child_orders = []
            remaining_qty = order.quantity

            while remaining_qty > 0:
                # Analyze current market conditions
                market_state = await self._analyze_market_state(order.symbol)

                # Calculate optimal order size based on market conditions
                optimal_size = self._calculate_optimal_size(
                    remaining_qty, market_state, params
                )

                # Adjust urgency based on progress and market conditions
                params.urgency_factor = self._adjust_urgency(
                    order, remaining_qty, market_state
                )

                # Create and execute child order
                child_order = await self.order_manager.create_order(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=OrderType.MARKET,
                    quantity=optimal_size,
                )

                child_orders.append(child_order)
                remaining_qty -= optimal_size

                # Dynamic sleep based on market conditions
                sleep_time = self._calculate_sleep_time(market_state)
                await asyncio.sleep(sleep_time)

            return True, child_orders

        except Exception as e:
            self.logger.error(f"Error in adaptive execution: {e}")
            return False, child_orders

    async def _execute_iceberg(
        self, order: Order, params: ExecutionParams
    ) -> Tuple[bool, List[Order]]:
        """Execute order using Iceberg execution strategy."""
        try:
            child_orders = []
            remaining_qty = order.quantity

            # Calculate iceberg order parameters
            visible_qty = params.iceberg_qty or (order.quantity * 0.1)

            while remaining_qty > 0:
                # Calculate child order size
                child_qty = min(remaining_qty, visible_qty)

                # Create and execute child order
                child_order = await self.order_manager.create_order(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=OrderType.LIMIT,
                    quantity=child_qty,
                    price=await self._get_optimal_limit_price(order.symbol, order.side),
                )

                child_orders.append(child_order)
                remaining_qty -= child_qty

                # Wait for order to fill
                await self._wait_for_fill(child_order)

            return True, child_orders

        except Exception as e:
            self.logger.error(f"Error in iceberg execution: {e}")
            return False, child_orders

    async def _execute_sniper(
        self, order: Order, params: ExecutionParams
    ) -> Tuple[bool, List[Order]]:
        """Execute order using Sniper execution strategy."""
        try:
            child_orders = []
            remaining_qty = order.quantity

            while remaining_qty > 0:
                # Wait for optimal execution opportunity
                await self._wait_for_opportunity(order.symbol, order.side, params)

                # Calculate optimal order size
                optimal_size = self._calculate_optimal_size(
                    remaining_qty,
                    await self._analyze_market_state(order.symbol),
                    params,
                )

                # Create and execute child order
                child_order = await self.order_manager.create_order(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=OrderType.MARKET,
                    quantity=optimal_size,
                )

                child_orders.append(child_order)
                remaining_qty -= optimal_size

                # Dynamic pause between executions
                await asyncio.sleep(self._calculate_sniper_pause(order.symbol))

            return True, child_orders

        except Exception as e:
            self.logger.error(f"Error in sniper execution: {e}")
            return False, child_orders

    async def _analyze_market_state(self, symbol: str) -> Dict[str, float]:
        """Analyze current market state for execution decisions."""
        try:
            # Get market data
            orderbook = await self._get_orderbook(symbol)
            trades = await self._get_recent_trades(symbol)

            # Calculate market metrics
            spread = (orderbook["asks"][0][0] - orderbook["bids"][0][0]) / orderbook[
                "bids"
            ][0][0]
            volatility = np.std([float(t["price"]) for t in trades]) / np.mean(
                [float(t["price"]) for t in trades]
            )

            # Calculate liquidity metrics
            bid_liquidity = sum(float(level[1]) for level in orderbook["bids"][:10])
            ask_liquidity = sum(float(level[1]) for level in orderbook["asks"][:10])
            total_liquidity = bid_liquidity + ask_liquidity

            # Calculate volume metrics
            recent_volume = sum(float(t["qty"]) for t in trades)
            avg_trade_size = np.mean([float(t["qty"]) for t in trades])

            # Calculate market impact estimate
            impact_estimate = self._estimate_market_impact(orderbook, recent_volume)

            state = {
                "spread": spread,
                "volatility": volatility,
                "liquidity": total_liquidity,
                "bid_liquidity": bid_liquidity,
                "ask_liquidity": ask_liquidity,
                "recent_volume": recent_volume,
                "avg_trade_size": avg_trade_size,
                "market_impact": impact_estimate,
                "timestamp": datetime.now(),
            }

            # Cache market state
            self.market_data_cache[symbol] = state

            # Log market state
            if self.use_wandb:
                wandb.log(
                    {
                        f"market_state/{symbol}/spread": spread,
                        f"market_state/{symbol}/volatility": volatility,
                        f"market_state/{symbol}/liquidity": total_liquidity,
                        f"market_state/{symbol}/market_impact": impact_estimate,
                    }
                )

            return state

        except Exception as e:
            self.logger.error(f"Error analyzing market state: {e}")
            raise

    def _calculate_optimal_size(
        self,
        remaining_qty: float,
        market_state: Dict[str, float],
        params: ExecutionParams,
    ) -> float:
        """Calculate optimal order size based on market conditions."""
        try:
            # Base size on participation rate
            base_size = market_state["recent_volume"] * params.max_participation_rate

            # Adjust for volatility
            volatility_factor = np.exp(-market_state["volatility"] * 10)
            size = base_size * volatility_factor

            # Adjust for spread
            spread_factor = np.exp(-market_state["spread"] * 100)
            size *= spread_factor

            # Adjust for urgency
            size *= params.urgency_factor

            # Ensure size is within bounds
            size = min(size, remaining_qty)
            size = max(size, params.min_execution_size)

            return size

        except Exception as e:
            self.logger.error(f"Error calculating optimal size: {e}")
            raise

    async def _get_optimal_limit_price(self, symbol: str, side: OrderSide) -> float:
        """Calculate optimal limit price based on orderbook."""
        try:
            orderbook = await self._get_orderbook(symbol)

            if side == OrderSide.BUY:
                # Place just above best bid
                best_bid = float(orderbook["bids"][0][0])
                spread = float(orderbook["asks"][0][0]) - best_bid
                return best_bid + (spread * 0.2)  # Place at 20% of spread
            else:
                # Place just below best ask
                best_ask = float(orderbook["asks"][0][0])
                spread = best_ask - float(orderbook["bids"][0][0])
                return best_ask - (spread * 0.2)

        except Exception as e:
            self.logger.error(f"Error calculating optimal limit price: {e}")
            raise

    async def _wait_for_opportunity(
        self, symbol: str, side: OrderSide, params: ExecutionParams
    ) -> None:
        """Wait for optimal execution opportunity."""
        try:
            while True:
                market_state = await self._analyze_market_state(symbol)

                # Define opportunity criteria
                good_spread = market_state["spread"] < 0.0003  # 3 bps spread
                good_liquidity = (
                    market_state["liquidity"] > market_state["recent_volume"] * 2
                )
                low_impact = market_state["market_impact"] < 0.0005  # 5 bps impact

                if good_spread and good_liquidity and low_impact:
                    break

                # Check timeout
                if params.end_time and datetime.now() > params.end_time:
                    raise TimeoutError("Execution opportunity timeout")

                await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Error waiting for opportunity: {e}")
            raise

    def _adjust_urgency(
        self, order: Order, remaining_qty: float, market_state: Dict[str, float]
    ) -> float:
        """Dynamically adjust execution urgency."""
        try:
            # Base urgency on remaining quantity
            progress = 1 - (remaining_qty / order.quantity)
            base_urgency = 1 - np.exp(-progress * 2)

            # Adjust for market conditions
            volatility_urgency = np.exp(-market_state["volatility"] * 5)
            spread_urgency = np.exp(-market_state["spread"] * 50)

            # Combine factors
            urgency = (base_urgency + volatility_urgency + spread_urgency) / 3

            return min(max(urgency, 0.1), 1.0)  # Bound between 0.1 and 1.0

        except Exception as e:
            self.logger.error(f"Error adjusting urgency: {e}")
            raise

    async def _calculate_execution_metrics(
        self,
        order: Order,
        child_orders: List[Order],
        arrival_price: float,
        execution_time: float,
    ) -> ExecutionMetrics:
        """Calculate execution performance metrics."""
        try:
            # Calculate average execution price
            total_qty = sum(co.quantity for co in child_orders)
            total_value = sum(co.quantity * co.price for co in child_orders)
            avg_price = total_value / total_qty if total_qty > 0 else arrival_price

            # Calculate implementation shortfall
            shortfall = (avg_price - arrival_price) / arrival_price
            if order.side == OrderSide.SELL:
                shortfall = -shortfall

            # Calculate market impact
            market_impact = await self._estimate_market_impact_cost(
                order.symbol, arrival_price, avg_price, total_qty
            )

            # Calculate timing cost
            timing_cost = shortfall - market_impact

            # Calculate participation rate
            market_volume = await self._get_market_volume(
                order.symbol,
                child_orders[0].timestamp if child_orders else order.timestamp,
                child_orders[-1].timestamp if child_orders else order.timestamp,
            )
            participation_rate = total_qty / market_volume if market_volume > 0 else 1.0

            metrics = ExecutionMetrics(
                arrival_price=arrival_price,
                average_execution_price=avg_price,
                implementation_shortfall=shortfall,
                market_impact=market_impact,
                timing_cost=timing_cost,
                execution_time=execution_time,
                participation_rate=participation_rate,
                num_child_orders=len(child_orders),
                slippage=(avg_price - arrival_price) / arrival_price,
            )

            # Log metrics
            if self.use_wandb:
                wandb.log(
                    {
                        "execution/implementation_shortfall": shortfall,
                        "execution/market_impact": market_impact,
                        "execution/timing_cost": timing_cost,
                        "execution/participation_rate": participation_rate,
                        "execution/num_child_orders": len(child_orders),
                        "execution/slippage": metrics.slippage,
                        "execution/time": execution_time,
                    }
                )

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating execution metrics: {e}")
            raise

    async def _get_market_volume(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> float:
        """Get market volume for a time period."""
        try:
            trades = await self._get_trades_in_period(symbol, start_time, end_time)
            return sum(float(trade["qty"]) for trade in trades)
        except Exception as e:
            self.logger.error(f"Error getting market volume: {e}")
            raise

    def _estimate_market_impact(self, orderbook: Dict, recent_volume: float) -> float:
        """Estimate market impact based on orderbook depth."""
        try:
            # Calculate total depth up to recent volume
            total_depth = 0
            weighted_price = 0

            for level in orderbook["asks"]:
                price, qty = float(level[0]), float(level[1])
                if total_depth + qty > recent_volume:
                    remaining = recent_volume - total_depth
                    weighted_price += price * remaining
                    break
                weighted_price += price * qty
                total_depth += qty
                if total_depth >= recent_volume:
                    break

            # Calculate average price movement
            avg_price = weighted_price / recent_volume if recent_volume > 0 else 0
            base_price = float(orderbook["asks"][0][0])

            return (avg_price - base_price) / base_price

        except Exception as e:
            self.logger.error(f"Error estimating market impact: {e}")
            raise

    async def _log_execution(
        self, order: Order, params: ExecutionParams, metrics: ExecutionMetrics
    ) -> None:
        """Log execution details and metrics."""
        try:
            execution_record = {
                "order_id": order.id,
                "symbol": order.symbol,
                "size": order.quantity,
                "side": order.side.value,
                "algo": params.algo.value,
                "priority": params.priority.value,
                "execution_time": metrics.execution_time,
                "arrival_price": metrics.arrival_price,
                "avg_price": metrics.average_execution_price,
                "shortfall": metrics.implementation_shortfall,
                "market_impact": metrics.market_impact,
                "timing_cost": metrics.timing_cost,
                "participation_rate": metrics.participation_rate,
                "child_orders": metrics.num_child_orders,
                "timestamp": datetime.now(),
            }

            # Store execution record
            self.execution_history.append(execution_record)

            # Log to W&B
            if self.use_wandb:
                wandb.log(
                    {
                        "execution_details": execution_record,
                        "execution_plot": self._create_execution_plot(order, metrics),
                    }
                )

        except Exception as e:
            self.logger.error(f"Error logging execution: {e}")
            raise

    def _create_execution_plot(
        self, order: Order, metrics: ExecutionMetrics
    ) -> wandb.Image:
        """Create execution analysis plot."""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Price evolution plot
            ax1.plot(metrics.price_trajectory, label="Market Price")
            ax1.axhline(
                y=metrics.arrival_price,
                color="g",
                linestyle="--",
                label="Arrival Price",
            )
            ax1.axhline(
                y=metrics.average_execution_price,
                color="r",
                linestyle="--",
                label="Avg Execution Price",
            )
            ax1.set_title("Execution Price Evolution")
            ax1.legend()

            # Execution analysis
            labels = ["Shortfall", "Market Impact", "Timing Cost", "Slippage"]
            values = [
                metrics.implementation_shortfall * 10000,
                metrics.market_impact * 10000,
                metrics.timing_cost * 10000,
                metrics.slippage * 10000,
            ]
            ax2.bar(labels, values)
            ax2.set_title("Execution Analysis (bps)")

            return wandb.Image(fig)

        except Exception as e:
            self.logger.error(f"Error creating execution plot: {e}")
            raise
