from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
import asyncio
import json
import wandb
from enum import Enum

from ..analysis.event_manager import EventManager, Event, EventType, EventPriority
from .risk_manager import RiskManager
from .order_manager import OrderManager, OrderSide, OrderType


class PositionStatus(Enum):
    """Position status enumeration."""

    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"
    REDUCING = "reducing"
    ERROR = "error"


@dataclass
class Position:
    """Position data structure."""

    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    status: PositionStatus
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class PortfolioConfig:
    """Portfolio configuration parameters."""

    max_position_size: float = 0.2  # Maximum position size as fraction of portfolio
    target_allocation: Dict[str, float] = None  # Target allocation per asset
    rebalance_threshold: float = 0.1  # Threshold for rebalancing
    max_positions: int = 10  # Maximum number of simultaneous positions
    min_cash_reserve: float = 0.05  # Minimum cash reserve as fraction of portfolio


class PortfolioManager:
    """
    Manages portfolio positions, allocations, and performance tracking.
    Coordinates with RiskManager and OrderManager for position management.
    """

    def __init__(
        self,
        event_manager: EventManager,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        initial_capital: float,
        config: Optional[PortfolioConfig] = None,
        use_wandb: bool = True,
    ):
        """
        Initialize PortfolioManager.

        Args:
            event_manager: EventManager instance
            risk_manager: RiskManager instance
            order_manager: OrderManager instance
            initial_capital: Initial portfolio capital
            config: Portfolio configuration parameters
            use_wandb: Whether to use W&B logging
        """
        self.event_manager = event_manager
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self.initial_capital = initial_capital
        self.config = config or PortfolioConfig()
        self.use_wandb = use_wandb

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Portfolio tracking
        self.positions = {}  # Current positions
        self.position_history = []  # Historical positions
        self.cash_balance = initial_capital
        self.total_value = initial_capital

        # Performance tracking
        self.performance_metrics = {
            "returns": [],
            "drawdowns": [],
            "position_count": [],
            "exposure": [],
            "realized_pnl": 0,
            "unrealized_pnl": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
        }

        # Initialize portfolio history
        self.portfolio_history = pd.DataFrame(
            columns=[
                "timestamp",
                "total_value",
                "cash_balance",
                "position_value",
                "realized_pnl",
                "unrealized_pnl",
            ]
        )

        # Setup event subscriptions
        self._setup_event_subscriptions()

    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions."""
        self.event_manager.subscribe(
            self._handle_price_update, [EventType.PRICE_UPDATE], EventPriority.NORMAL
        )
        self.event_manager.subscribe(
            self._handle_trade_executed, [EventType.ORDER_FILLED], EventPriority.HIGH
        )

    async def open_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        metadata: Optional[Dict] = None,
    ) -> Position:
        """
        Open a new position.

        Args:
            symbol: Trading pair symbol
            side: Position side (buy/sell)
            quantity: Position size
            price: Entry price
            metadata: Additional position information

        Returns:
            Created position object
        """
        try:
            # Validate position
            await self._validate_position(symbol, quantity, price)

            # Create position object
            position = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                unrealized_pnl=0,
                realized_pnl=0,
                status=PositionStatus.PENDING,
                timestamp=datetime.now(),
                metadata=metadata or {},
            )

            # Place order
            order = await self.order_manager.create_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=price,
            )

            # Update position after order execution
            position.status = PositionStatus.OPEN
            self.positions[symbol] = position

            # Update portfolio metrics
            self.cash_balance -= quantity * price
            self._update_portfolio_metrics()

            # Log position opening
            if self.use_wandb:
                wandb.log(
                    {
                        "position/opened": 1,
                        "position/size": quantity,
                        "position/entry_price": price,
                        "position/side": side.value,
                        "portfolio/cash_balance": self.cash_balance,
                        "portfolio/total_value": self.total_value,
                    }
                )

            return position

        except Exception as e:
            self.logger.error(f"Error opening position: {e}")
            raise

    async def close_position(
        self, symbol: str, price: Optional[float] = None
    ) -> Position:
        """
        Close an existing position.

        Args:
            symbol: Symbol of position to close
            price: Optional closing price (uses market price if not provided)

        Returns:
            Closed position object
        """
        try:
            position = self.positions.get(symbol)
            if not position:
                raise ValueError(f"Position not found: {symbol}")

            # Get current market price if not provided
            close_price = price or await self._get_market_price(symbol)

            # Calculate PnL
            pnl = (close_price - position.entry_price) * position.quantity
            if position.side == OrderSide.SELL:
                pnl = -pnl

            # Place closing order
            order = await self.order_manager.create_order(
                symbol=symbol,
                side=(
                    OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
                ),
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                price=close_price,
            )

            # Update position
            position.status = PositionStatus.CLOSED
            position.realized_pnl = pnl

            # Update portfolio metrics
            self.cash_balance += position.quantity * close_price
            self.performance_metrics["realized_pnl"] += pnl
            if pnl > 0:
                self.performance_metrics["winning_trades"] += 1
            else:
                self.performance_metrics["losing_trades"] += 1

            # Remove from active positions
            self.positions.pop(symbol)
            self.position_history.append(position)

            # Update portfolio metrics
            self._update_portfolio_metrics()

            # Log position closing
            if self.use_wandb:
                wandb.log(
                    {
                        "position/closed": 1,
                        "position/pnl": pnl,
                        "position/roi": pnl
                        / (position.entry_price * position.quantity),
                        "portfolio/cash_balance": self.cash_balance,
                        "portfolio/total_value": self.total_value,
                    }
                )

            return position

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            raise

    async def rebalance_portfolio(self) -> Dict[str, float]:
        """
        Rebalance portfolio according to target allocations.

        Returns:
            Dictionary of rebalancing trades
        """
        try:
            if not self.config.target_allocation:
                return {}

            # Calculate current allocations
            current_allocations = self._calculate_allocations()

            # Calculate required trades
            trades = {}
            for symbol, target in self.config.target_allocation.items():
                current = current_allocations.get(symbol, 0)
                diff = target - current

                if abs(diff) > self.config.rebalance_threshold:
                    # Calculate trade size
                    trade_value = diff * self.total_value
                    price = await self._get_market_price(symbol)
                    quantity = abs(trade_value) / price

                    trades[symbol] = {
                        "side": OrderSide.BUY if diff > 0 else OrderSide.SELL,
                        "quantity": quantity,
                        "price": price,
                    }

            # Execute rebalancing trades
            for symbol, trade in trades.items():
                if trade["side"] == OrderSide.BUY:
                    await self.open_position(
                        symbol=symbol,
                        side=trade["side"],
                        quantity=trade["quantity"],
                        price=trade["price"],
                    )
                else:
                    await self.close_position(symbol, trade["price"])

            # Log rebalancing
            if self.use_wandb and trades:
                wandb.log(
                    {
                        "rebalance/trades": len(trades),
                        "rebalance/total_value": sum(
                            t["quantity"] * t["price"] for t in trades.values()
                        ),
                    }
                )

            return trades

        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {e}")
            raise

    async def _handle_price_update(self, event: Event) -> None:
        """Handle price update events."""
        try:
            price_data = event.data
            symbol = price_data["symbol"]
            price = price_data["price"]

            if symbol in self.positions:
                position = self.positions[symbol]

                # Update position metrics
                old_unrealized_pnl = position.unrealized_pnl
                position.current_price = price
                position.unrealized_pnl = (
                    (price - position.entry_price) * position.quantity
                    if position.side == OrderSide.BUY
                    else (position.entry_price - price) * position.quantity
                )

                # Update portfolio metrics
                self.performance_metrics["unrealized_pnl"] += (
                    position.unrealized_pnl - old_unrealized_pnl
                )
                self._update_portfolio_metrics()

        except Exception as e:
            self.logger.error(f"Error handling price update: {e}")

    async def _handle_trade_executed(self, event: Event) -> None:
        """Handle trade execution events."""
        try:
            trade_data = event.data
            symbol = trade_data["symbol"]

            # Update position if it exists
            if symbol in self.positions:
                position = self.positions[symbol]
                position.status = PositionStatus.OPEN

                # Update metrics
                self.performance_metrics["total_trades"] += 1
                self._update_portfolio_metrics()

        except Exception as e:
            self.logger.error(f"Error handling trade execution: {e}")

    def _update_portfolio_metrics(self) -> None:
        """Update portfolio metrics."""
        try:
            # Calculate total position value
            position_value = sum(
                p.quantity * p.current_price for p in self.positions.values()
            )

            # Update total value
            self.total_value = self.cash_balance + position_value

            # Calculate return
            returns = self.total_value / self.initial_capital - 1
            self.performance_metrics["returns"].append(returns)

            # Calculate drawdown
            peak = max(self.performance_metrics["returns"] + [0])
            drawdown = (1 + returns) / (1 + peak) - 1
            self.performance_metrics["drawdowns"].append(drawdown)

            # Update position metrics
            self.performance_metrics["position_count"].append(len(self.positions))
            self.performance_metrics["exposure"].append(
                position_value / self.total_value
            )

            # Update portfolio history
            self.portfolio_history = self.portfolio_history.append(
                {
                    "timestamp": datetime.now(),
                    "total_value": self.total_value,
                    "cash_balance": self.cash_balance,
                    "position_value": position_value,
                    "realized_pnl": self.performance_metrics["realized_pnl"],
                    "unrealized_pnl": self.performance_metrics["unrealized_pnl"],
                },
                ignore_index=True,
            )

            # Log metrics
            if self.use_wandb:
                wandb.log(
                    {
                        "portfolio/total_value": self.total_value,
                        "portfolio/cash_balance": self.cash_balance,
                        "portfolio/position_value": position_value,
                        "portfolio/return": returns,
                        "portfolio/drawdown": drawdown,
                        "portfolio/position_count": len(self.positions),
                        "portfolio/exposure": position_value / self.total_value,
                    }
                )

        except Exception as e:
            self.logger.error(f"Error updating portfolio metrics: {e}")
            raise

    def get_portfolio_stats(self) -> Dict:
        """Get current portfolio statistics."""
        try:
            returns = np.array(self.performance_metrics["returns"])

            stats = {
                "total_value": self.total_value,
                "cash_balance": self.cash_balance,
                "position_value": self.total_value - self.cash_balance,
                "realized_pnl": self.performance_metrics["realized_pnl"],
                "unrealized_pnl": self.performance_metrics["unrealized_pnl"],
                "return": returns[-1] if len(returns) > 0 else 0,
                "volatility": np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0,
                "sharpe_ratio": (
                    np.mean(returns) / np.std(returns) * np.sqrt(252)
                    if len(returns) > 0 and np.std(returns) > 0
                    else 0
                ),
                "max_drawdown": min(self.performance_metrics["drawdowns"] + [0]),
                "win_rate": (
                    self.performance_metrics["winning_trades"]
                    / self.performance_metrics["total_trades"]
                    if self.performance_metrics["total_trades"] > 0
                    else 0
                ),
                "position_count": len(self.positions),
                "exposure": (self.total_value - self.cash_balance) / self.total_value,
                "profit_factor": (
                    abs(
                        self.performance_metrics["winning_trades"]
                        * np.mean(
                            [
                                p.realized_pnl
                                for p in self.position_history
                                if p.realized_pnl > 0
                            ]
                            or [0]
                        )
                    )
                    / abs(
                        self.performance_metrics["losing_trades"]
                        * np.mean(
                            [
                                p.realized_pnl
                                for p in self.position_history
                                if p.realized_pnl < 0
                            ]
                            or [1]
                        )
                    )
                    if self.performance_metrics["losing_trades"] > 0
                    else np.inf
                ),
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating portfolio stats: {e}")
            raise

    async def manage_position_risk(self, symbol: str) -> None:
        """
        Manage risk for a specific position.

        Args:
            symbol: Symbol of position to manage
        """
        try:
            position = self.positions.get(symbol)
            if not position:
                return

            # Get current market data
            current_price = position.current_price
            entry_price = position.entry_price

            # Calculate position metrics
            position_value = position.quantity * current_price
            portfolio_exposure = position_value / self.total_value
            unrealized_roi = position.unrealized_pnl / (position.quantity * entry_price)

            # Check risk thresholds
            risk_factors = []

            # Position size check
            if portfolio_exposure > self.config.max_position_size:
                risk_factors.append("Position size exceeded")

            # Drawdown check
            max_position_drawdown = -0.15  # 15% maximum position drawdown
            if unrealized_roi < max_position_drawdown:
                risk_factors.append("Maximum drawdown exceeded")

            # Volatility check
            position_returns = np.array(self.performance_metrics["returns"])[
                -20:
            ]  # Last 20 returns
            if len(position_returns) >= 20:
                volatility = np.std(position_returns) * np.sqrt(252)
                max_volatility = 0.5  # 50% annualized volatility threshold
                if volatility > max_volatility:
                    risk_factors.append("Excessive volatility")

            # Take action if risk thresholds are breached
            if risk_factors:
                self.logger.warning(
                    f"Risk thresholds breached for {symbol}: {', '.join(risk_factors)}"
                )

                # Determine risk reduction action
                if "Position size exceeded" in risk_factors:
                    # Reduce position size to maximum allowed
                    excess_exposure = portfolio_exposure - self.config.max_position_size
                    reduction_quantity = (
                        excess_exposure * self.total_value
                    ) / current_price
                    await self._reduce_position(symbol, reduction_quantity)
                else:
                    # Close position
                    await self.close_position(symbol)

                # Log risk event
                if self.use_wandb:
                    wandb.log(
                        {
                            "risk/threshold_breach": 1,
                            "risk/factors": len(risk_factors),
                            "risk/symbol": symbol,
                            "risk/action_taken": (
                                "reduce"
                                if "Position size exceeded" in risk_factors
                                else "close"
                            ),
                        }
                    )

        except Exception as e:
            self.logger.error(f"Error managing position risk: {e}")
            raise

    async def _reduce_position(self, symbol: str, reduction_quantity: float) -> None:
        """
        Reduce the size of an existing position.

        Args:
            symbol: Symbol of position to reduce
            reduction_quantity: Amount to reduce position by
        """
        try:
            position = self.positions.get(symbol)
            if not position:
                raise ValueError(f"Position not found: {symbol}")

            if reduction_quantity >= position.quantity:
                await self.close_position(symbol)
                return

            # Create reducing order
            order = await self.order_manager.create_order(
                symbol=symbol,
                side=(
                    OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
                ),
                order_type=OrderType.MARKET,
                quantity=reduction_quantity,
            )

            # Update position
            position.quantity -= reduction_quantity
            position.status = PositionStatus.REDUCING

            # Update portfolio metrics
            self._update_portfolio_metrics()

        except Exception as e:
            self.logger.error(f"Error reducing position: {e}")
            raise

    async def monitor_portfolio_risk(self) -> None:
        """Monitor overall portfolio risk metrics."""
        try:
            # Calculate portfolio metrics
            stats = self.get_portfolio_stats()

            # Define risk thresholds
            risk_thresholds = {
                "max_drawdown": -0.25,  # 25% maximum drawdown
                "max_exposure": 0.8,  # 80% maximum exposure
                "min_sharpe": 0.5,  # Minimum Sharpe ratio
                "max_volatility": 0.4,  # 40% maximum volatility
            }

            # Check risk thresholds
            risk_breaches = []

            if stats["max_drawdown"] < risk_thresholds["max_drawdown"]:
                risk_breaches.append("Maximum drawdown exceeded")

            if stats["exposure"] > risk_thresholds["max_exposure"]:
                risk_breaches.append("Maximum exposure exceeded")

            if (
                stats["sharpe_ratio"] < risk_thresholds["min_sharpe"]
                and self.performance_metrics["total_trades"] > 50
            ):
                risk_breaches.append("Insufficient risk-adjusted returns")

            if stats["volatility"] > risk_thresholds["max_volatility"]:
                risk_breaches.append("Excessive portfolio volatility")

            # Take action if risk thresholds are breached
            if risk_breaches:
                self.logger.warning(
                    f"Portfolio risk thresholds breached: {', '.join(risk_breaches)}"
                )

                # Reduce risk based on breaches
                if "Maximum exposure exceeded" in risk_breaches:
                    await self._reduce_portfolio_exposure()
                elif len(risk_breaches) >= 2:
                    await self._emergency_risk_reduction()

                # Log risk event
                if self.use_wandb:
                    wandb.log(
                        {
                            "portfolio_risk/threshold_breach": 1,
                            "portfolio_risk/factors": len(risk_breaches),
                            "portfolio_risk/drawdown": stats["max_drawdown"],
                            "portfolio_risk/exposure": stats["exposure"],
                            "portfolio_risk/sharpe": stats["sharpe_ratio"],
                            "portfolio_risk/volatility": stats["volatility"],
                        }
                    )

        except Exception as e:
            self.logger.error(f"Error monitoring portfolio risk: {e}")
            raise

    async def _reduce_portfolio_exposure(self) -> None:
        """Reduce overall portfolio exposure."""
        try:
            target_exposure = 0.5  # Reduce to 50% exposure
            current_exposure = (self.total_value - self.cash_balance) / self.total_value

            if current_exposure <= target_exposure:
                return

            # Calculate required reduction
            reduction_factor = (current_exposure - target_exposure) / current_exposure

            # Reduce each position proportionally
            for symbol, position in self.positions.items():
                reduction_quantity = position.quantity * reduction_factor
                await self._reduce_position(symbol, reduction_quantity)

            # Log action
            if self.use_wandb:
                wandb.log(
                    {
                        "risk_reduction/exposure_reduction": reduction_factor,
                        "risk_reduction/positions_affected": len(self.positions),
                    }
                )

        except Exception as e:
            self.logger.error(f"Error reducing portfolio exposure: {e}")
            raise

    async def _emergency_risk_reduction(self) -> None:
        """Emergency risk reduction - close worst performing positions."""
        try:
            # Sort positions by unrealized ROI
            sorted_positions = sorted(
                self.positions.items(),
                key=lambda x: x[1].unrealized_pnl / (x[1].quantity * x[1].entry_price),
            )

            # Close worst performing positions (up to 50% of positions)
            positions_to_close = sorted_positions[: len(sorted_positions) // 2]

            for symbol, _ in positions_to_close:
                await self.close_position(symbol)

            # Log action
            if self.use_wandb:
                wandb.log(
                    {
                        "risk_reduction/emergency": 1,
                        "risk_reduction/positions_closed": len(positions_to_close),
                    }
                )

        except Exception as e:
            self.logger.error(f"Error in emergency risk reduction: {e}")
            raise

    async def _validate_position(
        self, symbol: str, quantity: float, price: float
    ) -> None:
        """
        Validate a potential position against portfolio constraints.

        Args:
            symbol: Trading pair symbol
            quantity: Position size
            price: Position price
        """
        position_value = quantity * price

        # Check maximum positions
        if (
            len(self.positions) >= self.config.max_positions
            and symbol not in self.positions
        ):
            raise ValueError(f"Maximum positions ({self.config.max_positions}) reached")

        # Check position size
        portfolio_exposure = position_value / self.total_value
        if portfolio_exposure > self.config.max_position_size:
            raise ValueError(
                f"Position size ({portfolio_exposure:.2%}) exceeds maximum "
                f"({self.config.max_position_size:.2%})"
            )

        # Check cash reserve
        if (
            self.cash_balance - position_value
            < self.total_value * self.config.min_cash_reserve
        ):
            raise ValueError("Insufficient cash reserve for position")

        # Validate with risk manager
        risk_check = await self.risk_manager.check_risk_limits(
            {
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "portfolio_value": self.total_value,
            }
        )

        if not risk_check["approved"]:
            raise ValueError(f"Risk check failed: {risk_check['reason']}")

    def _calculate_allocations(self) -> Dict[str, float]:
        """Calculate current portfolio allocations."""
        allocations = {}
        for symbol, position in self.positions.items():
            position_value = position.quantity * position.current_price
            allocations[symbol] = position_value / self.total_value
        return allocations

    async def get_position_metrics(self, symbol: str) -> Dict:
        """
        Get detailed metrics for a specific position.

        Args:
            symbol: Position symbol

        Returns:
            Dictionary of position metrics
        """
        try:
            position = self.positions.get(symbol)
            if not position:
                raise ValueError(f"Position not found: {symbol}")

            position_value = position.quantity * position.current_price

            metrics = {
                "symbol": symbol,
                "side": position.side.value,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "position_value": position_value,
                "portfolio_exposure": position_value / self.total_value,
                "unrealized_pnl": position.unrealized_pnl,
                "unrealized_roi": (
                    position.unrealized_pnl / (position.quantity * position.entry_price)
                ),
                "holding_period": (datetime.now() - position.timestamp).total_seconds()
                / 3600,  # in hours
                "status": position.status.value,
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error getting position metrics: {e}")
            raise
