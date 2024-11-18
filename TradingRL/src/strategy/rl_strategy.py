from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import gymnasium as gym
from collections import deque
import wandb

from TradingRL.src.strategy.base_strategy import (
    BaseStrategy,
    Signal,
    SignalType,
    StrategyState,
)
from TradingRL.src.core.trader import Trader
from TradingRL.src.analysis.market_analyzer import MarketRegime
from TradingRL.src.analysis.event_manager import Event, EventType


class RLStrategy(BaseStrategy):
    """
    Reinforcement Learning based trading strategy.
    Uses the RL model from Trader for decision making.
    """

    def __init__(
        self,
        trader: Trader,
        feature_window: int = 100,
        confidence_threshold: float = 0.6,
        *args,
        **kwargs,
    ):
        """
        Initialize RL Strategy.

        Args:
            trader: Trader instance with trained RL model
            feature_window: Window size for feature calculation
            confidence_threshold: Minimum confidence for signal generation
            *args, **kwargs: Arguments for BaseStrategy
        """
        super().__init__(*args, **kwargs)
        self.trader = trader
        self.feature_window = feature_window
        self.confidence_threshold = confidence_threshold

        # State management
        self.state_buffer = {
            symbol: deque(maxlen=feature_window) for symbol in self.symbols
        }
        self.last_action = {symbol: SignalType.NO_SIGNAL for symbol in self.symbols}
        self.position_history = {symbol: [] for symbol in self.symbols}

        # Performance tracking
        self.prediction_accuracy = {symbol: [] for symbol in self.symbols}
        self.action_history = {symbol: [] for symbol in self.symbols}

    async def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate trading signals using RL model."""
        try:
            signals = []

            for symbol, df in data.items():
                # Prepare state
                state = self._prepare_state(symbol, df)
                if state is None:
                    continue

                # Get RL model prediction
                action, confidence = await self.trader.predict_action(state)

                # Record action
                self.action_history[symbol].append(
                    {
                        "timestamp": datetime.now(),
                        "action": action,
                        "confidence": confidence,
                        "state": state,
                    }
                )

                # Convert action to signal if confidence is high enough
                if confidence >= self.confidence_threshold:
                    signal = await self._action_to_signal(
                        symbol, action, confidence, df.iloc[-1]
                    )
                    if signal:
                        signals.append(signal)

                # Update last action
                self.last_action[symbol] = action

            return signals

        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []

    async def optimize_parameters(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Optimize RL model parameters."""
        try:
            optimization_metrics = {}

            # Train model on recent data
            for symbol, df in data.items():
                # Prepare training data
                train_states = self._prepare_training_data(df)

                # Train model
                train_metrics = await self.trader.train_model(
                    train_states, self.position_history[symbol]
                )

                optimization_metrics[symbol] = train_metrics

            # Log optimization results
            if self.use_wandb:
                self._log_optimization_metrics(optimization_metrics)

            return optimization_metrics

        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {e}")
            return {}

    def _prepare_state(self, symbol: str, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare state representation for RL model."""
        try:
            if len(data) < self.feature_window:
                return None

            # Market data features
            price_features = self._calculate_price_features(data)
            volume_features = self._calculate_volume_features(data)
            technical_features = self._calculate_technical_features(data)

            # Market regime features
            regime_features = self._calculate_regime_features(symbol)

            # Position features
            position_features = self._calculate_position_features(symbol)

            # Combine features
            state = np.concatenate(
                [
                    price_features,
                    volume_features,
                    technical_features,
                    regime_features,
                    position_features,
                ]
            )

            # Normalize state
            state = self._normalize_state(state)

            # Update state buffer
            self.state_buffer[symbol].append(state)

            return state

        except Exception as e:
            self.logger.error(f"Error preparing state: {e}")
            return None

    def _calculate_price_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate price-based features."""
        df = data.tail(self.feature_window)

        returns = np.diff(np.log(df["close"]))
        volatility = np.std(returns)

        features = np.array(
            [
                returns[-1],  # Latest return
                np.mean(returns),  # Average return
                volatility,  # Volatility
                (df["close"].iloc[-1] - df["open"].iloc[-1])
                / df["open"].iloc[-1],  # Current candle return
                (df["high"].iloc[-1] - df["low"].iloc[-1])
                / df["low"].iloc[-1],  # Current candle range
            ]
        )

        return features

    def _calculate_volume_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate volume-based features."""
        df = data.tail(self.feature_window)

        volume = df["volume"].values
        volume_ma = np.mean(volume)

        features = np.array(
            [
                volume[-1] / volume_ma,  # Relative volume
                np.std(volume) / volume_ma,  # Volume volatility
                np.sum(volume * (df["close"] - df["open"]))
                / np.sum(volume),  # Volume-weighted price change
            ]
        )

        return features

    def _calculate_technical_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate technical indicator features."""
        df = data.tail(self.feature_window)
        close = df["close"].values

        # Calculate indicators
        sma_20 = np.mean(close[-20:])
        sma_50 = np.mean(close[-50:])
        rsi = self.indicators[df.name]["rsi"][-1]
        macd = self.indicators[df.name]["macd"][-1]
        bbands = self.indicators[df.name]["bbands"]

        features = np.array(
            [
                close[-1] / sma_20 - 1,  # Price vs SMA20
                close[-1] / sma_50 - 1,  # Price vs SMA50
                rsi / 100,  # Normalized RSI
                macd,  # MACD
                (close[-1] - bbands[0][-1])
                / (bbands[2][-1] - bbands[0][-1]),  # BB position
            ]
        )

        return features

    def _calculate_regime_features(self, symbol: str) -> np.ndarray:
        """Calculate market regime features."""
        current_regime = self.market_analyzer.current_state.regime

        # One-hot encode regime
        regime_encoding = np.zeros(len(MarketRegime))
        regime_encoding[current_regime.value] = 1

        return regime_encoding

    def _calculate_position_features(self, symbol: str) -> np.ndarray:
        """Calculate position-related features."""
        position = self.positions.get(symbol)

        if position:
            position_features = np.array(
                [
                    1,  # Has position
                    position["quantity"],  # Position size
                    position["unrealized_pnl"],  # Unrealized PnL
                    position["duration"].total_seconds()
                    / 3600,  # Position duration in hours
                ]
            )
        else:
            position_features = np.array([0, 0, 0, 0])

        return position_features

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state features."""
        # Use running statistics for normalization
        return (state - self.trader.state_mean) / (self.trader.state_std + 1e-8)

    async def _action_to_signal(
        self, symbol: str, action: int, confidence: float, current_data: pd.Series
    ) -> Optional[Signal]:
        """Convert RL action to trading signal."""
        try:
            # Map action to signal type
            action_map = {
                0: SignalType.NO_SIGNAL,
                1: SignalType.LONG,
                2: SignalType.SHORT,
                3: SignalType.EXIT_LONG,
                4: SignalType.EXIT_SHORT,
            }

            signal_type = action_map[action]

            # Check if signal is valid
            if signal_type == SignalType.NO_SIGNAL:
                return None

            # Calculate position size
            if signal_type in [SignalType.LONG, SignalType.SHORT]:
                size = await self._calculate_position_size(
                    symbol, current_data["close"], confidence
                )
            else:
                size = (
                    self.positions[symbol]["quantity"]
                    if symbol in self.positions
                    else 0
                )

            # Create signal
            signal = Signal(
                type=signal_type,
                symbol=symbol,
                price=current_data["close"],
                size=size,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    "action": action,
                    "market_regime": self.market_analyzer.current_state.regime.value,
                    "volatility": self.market_analyzer.current_state.volatility,
                },
            )

            return signal

        except Exception as e:
            self.logger.error(f"Error converting action to signal: {e}")
            return None

    async def _calculate_position_size(
        self, symbol: str, price: float, confidence: float
    ) -> float:
        """Calculate position size based on Kelly criterion and confidence."""
        try:
            # Get win rate and profit/loss ratio from history
            history = self.position_history[symbol]
            if not history:
                return self.config.min_position_size

            wins = [t for t in history if t["pnl"] > 0]
            losses = [t for t in history if t["pnl"] < 0]

            win_rate = len(wins) / len(history) if history else 0.5
            avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
            avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 0

            # Calculate Kelly fraction
            if avg_loss == 0:
                kelly_fraction = 0.01
            else:
                profit_ratio = avg_win / avg_loss
                kelly_fraction = win_rate - ((1 - win_rate) / profit_ratio)

            # Adjust by confidence and limits
            size = kelly_fraction * confidence
            size = min(size, self.config.max_position_size)
            size = max(size, self.config.min_position_size)

            return size

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return self.config.min_position_size

    def _log_optimization_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log optimization metrics to W&B."""
        if not self.use_wandb:
            return

        for symbol, symbol_metrics in metrics.items():
            wandb.log(
                {
                    f"optimization/{symbol}/loss": symbol_metrics["loss"],
                    f"optimization/{symbol}/accuracy": symbol_metrics["accuracy"],
                    f"optimization/{symbol}/reward": symbol_metrics["avg_reward"],
                }
            )
