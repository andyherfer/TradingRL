from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from enum import Enum
import logging
import asyncio
from abc import ABC, abstractmethod
import json
import wandb
import talib
from scipy import stats

from ..analysis.event_manager import EventManager, Event, EventType, EventPriority
from ..core.risk_manager import RiskManager
from ..core.portfolio_manager import PortfolioManager
from ..analysis.market_analyzer import MarketAnalyzer, MarketRegime


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


@dataclass
class Signal:
    """Trading signal information."""

    type: SignalType
    symbol: str
    price: float
    size: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class StrategyConfig:
    """Strategy configuration."""

    warmup_period: int = 100
    update_interval: int = 1
    min_confidence: float = 0.6
    optimization_interval: int = 1440  # minutes
    max_positions: int = 5
    position_sizing_method: str = "kelly"


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    Defines interface for strategy implementation.
    """

    def __init__(
        self,
        name: str,
        symbols: List[str],
        event_manager: EventManager,
        risk_manager: RiskManager,
        portfolio_manager: PortfolioManager,
        market_analyzer: MarketAnalyzer,
        config: Optional[StrategyConfig] = None,
        use_wandb: bool = True,
    ):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            symbols: Trading symbols
            event_manager: EventManager instance
            risk_manager: RiskManager instance
            portfolio_manager: PortfolioManager instance
            market_analyzer: MarketAnalyzer instance
            config: Strategy configuration
            use_wandb: Whether to use W&B logging
        """
        self.name = name
        self.symbols = symbols
        self.event_manager = event_manager
        self.risk_manager = risk_manager
        self.portfolio_manager = portfolio_manager
        self.market_analyzer = market_analyzer
        self.config = config or StrategyConfig()
        self.use_wandb = use_wandb

        # Initialize logging
        self.logger = logging.getLogger(f"strategy.{name}")
        self.logger.setLevel(logging.INFO)

        # Strategy state
        self.state = StrategyState.INACTIVE
        self.signals = {}
        self.positions = {}
        self.performance_metrics = {}

        # Data management
        self.data_buffer = {symbol: pd.DataFrame() for symbol in symbols}
        self.indicators = {symbol: {} for symbol in symbols}

        # Strategy parameters
        self.parameters = {}
        self.optimization_history = []

        # Performance tracking
        self.signal_history = []
        self.trade_history = []

        # Setup event subscriptions
        self._setup_event_subscriptions()

    @abstractmethod
    async def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate trading signals.
        Must be implemented by concrete strategy classes.
        """
        pass

    @abstractmethod
    async def optimize_parameters(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters.
        Must be implemented by concrete strategy classes.
        """
        pass

    async def start(self) -> None:
        """Start the strategy."""
        try:
            self.logger.info(f"Starting strategy: {self.name}")

            # Enter warmup period
            self.state = StrategyState.WARMUP
            await self._warmup()

            # Activate strategy
            self.state = StrategyState.ACTIVE
            self.logger.info(f"Strategy {self.name} activated")

            # Start optimization loop
            asyncio.create_task(self._optimization_loop())

        except Exception as e:
            self.logger.error(f"Error starting strategy: {e}")
            self.state = StrategyState.ERROR
            raise

    async def stop(self) -> None:
        """Stop the strategy."""
        try:
            self.logger.info(f"Stopping strategy: {self.name}")

            # Close all positions
            for symbol in self.positions:
                await self._close_position(symbol)

            self.state = StrategyState.INACTIVE

        except Exception as e:
            self.logger.error(f"Error stopping strategy: {e}")
            self.state = StrategyState.ERROR
            raise

    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions."""
        self.event_manager.subscribe(
            self._handle_market_update,
            [EventType.PRICE_UPDATE, EventType.MARKET_REGIME_CHANGE],
            EventPriority.NORMAL,
        )

        self.event_manager.subscribe(
            self._handle_position_update,
            [EventType.POSITION_OPENED, EventType.POSITION_CLOSED],
            EventPriority.HIGH,
        )

    async def _handle_market_update(self, event: Event) -> None:
        """Handle market update events."""
        try:
            if self.state != StrategyState.ACTIVE:
                return

            # Update data buffer
            symbol = event.data["symbol"]
            if event.type == EventType.PRICE_UPDATE:
                await self._update_data_buffer(symbol, event.data)

            # Generate signals
            if len(self.data_buffer[symbol]) >= self.config.warmup_period:
                signals = await self.generate_signals(
                    {s: self.data_buffer[s] for s in self.symbols}
                )

                # Process signals
                for signal in signals:
                    await self._process_signal(signal)

        except Exception as e:
            self.logger.error(f"Error handling market update: {e}")

    async def _handle_position_update(self, event: Event) -> None:
        """Handle position update events."""
        try:
            position_data = event.data
            symbol = position_data["symbol"]

            if event.type == EventType.POSITION_OPENED:
                self.positions[symbol] = position_data
            elif event.type == EventType.POSITION_CLOSED:
                self.positions.pop(symbol, None)

            # Update performance metrics
            await self._update_performance_metrics()

        except Exception as e:
            self.logger.error(f"Error handling position update: {e}")

    async def _warmup(self) -> None:
        """Perform strategy warmup."""
        try:
            self.logger.info("Starting warmup period...")

            # Collect initial data
            for symbol in self.symbols:
                historical_data = await self._fetch_historical_data(
                    symbol, self.config.warmup_period
                )
                self.data_buffer[symbol] = historical_data

            # Calculate initial indicators
            await self._calculate_indicators()

            # Initial parameter optimization
            self.parameters = await self.optimize_parameters(self.data_buffer)

            self.logger.info("Warmup completed")

        except Exception as e:
            self.logger.error(f"Error during warmup: {e}")
            raise

    async def _optimization_loop(self) -> None:
        """Periodic strategy optimization loop."""
        try:
            while self.state == StrategyState.ACTIVE:
                await asyncio.sleep(self.config.optimization_interval * 60)

                self.logger.info("Starting strategy optimization...")
                self.state = StrategyState.OPTIMIZING

                # Optimize parameters
                new_parameters = await self.optimize_parameters(self.data_buffer)

                # Log optimization results
                optimization_record = {
                    "timestamp": datetime.now(),
                    "old_parameters": self.parameters.copy(),
                    "new_parameters": new_parameters,
                    "performance_metrics": self.performance_metrics.copy(),
                }
                self.optimization_history.append(optimization_record)

                # Update parameters
                self.parameters = new_parameters

                self.state = StrategyState.ACTIVE
                self.logger.info("Strategy optimization completed")

                # Log to W&B
                if self.use_wandb:
                    wandb.log(
                        {
                            "optimization/parameters": new_parameters,
                            **{
                                f"optimization/metric_{k}": v
                                for k, v in self.performance_metrics.items()
                            },
                        }
                    )

        except Exception as e:
            self.logger.error(f"Error in optimization loop: {e}")
            self.state = StrategyState.ERROR

    async def _process_signal(self, signal: Signal) -> None:
        """Process a trading signal."""
        try:
            if signal.type == SignalType.NO_SIGNAL:
                return

            # Validate signal
            if not self._validate_signal(signal):
                return

            # Check risk limits
            risk_check = await self.risk_manager.check_risk_limits(
                {
                    "symbol": signal.symbol,
                    "size": signal.size,
                    "side": signal.type.value,
                }
            )

            if not risk_check["approved"]:
                self.logger.warning(
                    f"Signal rejected by risk manager: {risk_check['reason']}"
                )
                return

            # Execute signal
            if signal.type in [SignalType.LONG, SignalType.SHORT]:
                await self._open_position(signal)
            else:
                await self._close_position(signal.symbol)

            # Record signal
            self.signal_history.append(signal)

            # Log to W&B
            if self.use_wandb:
                wandb.log(
                    {
                        "signal/type": signal.type.value,
                        "signal/confidence": signal.confidence,
                        "signal/size": signal.size,
                        "signal/price": signal.price,
                    }
                )

        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")

    def _validate_signal(self, signal: Signal) -> bool:
        """Validate a trading signal."""
        try:
            # Check confidence threshold
            if signal.confidence < self.config.min_confidence:
                return False

            # Check position limits
            if signal.type in [SignalType.LONG, SignalType.SHORT]:
                if len(self.positions) >= self.config.max_positions:
                    return False

            # Check existing positions
            if signal.symbol in self.positions:
                if signal.type in [SignalType.LONG, SignalType.SHORT]:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False

    async def _calculate_indicators(self) -> None:
        """Calculate technical indicators."""
        try:
            for symbol, data in self.data_buffer.items():
                self.indicators[symbol] = {
                    "sma_20": talib.SMA(data["close"].values, timeperiod=20),
                    "sma_50": talib.SMA(data["close"].values, timeperiod=50),
                    "rsi": talib.RSI(data["close"].values, timeperiod=14),
                    "macd": talib.MACD(data["close"].values)[0],
                    "bbands": talib.BBANDS(data["close"].values),
                    "atr": talib.ATR(
                        data["high"].values,
                        data["low"].values,
                        data["close"].values,
                        timeperiod=14,
                    ),
                }

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            raise

    async def _update_performance_metrics(self) -> None:
        """Update strategy performance metrics."""
        try:
            # Calculate basic metrics
            total_trades = len(self.trade_history)
            winning_trades = len([t for t in self.trade_history if t["pnl"] > 0])

            self.performance_metrics = {
                "total_trades": total_trades,
                "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
                "avg_return": (
                    np.mean([t["roi"] for t in self.trade_history])
                    if self.trade_history
                    else 0
                ),
                "sharpe_ratio": self._calculate_sharpe_ratio(),
                "max_drawdown": self._calculate_max_drawdown(),
                "current_positions": len(self.positions),
                "signal_accuracy": self._calculate_signal_accuracy(),
            }

            # Log metrics
            if self.use_wandb:
                wandb.log(
                    {
                        f"strategy/metric_{k}": v
                        for k, v in self.performance_metrics.items()
                    }
                )

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
            raise
