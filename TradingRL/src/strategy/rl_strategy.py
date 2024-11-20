from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import gymnasium as gym
from collections import deque
import wandb
import talib

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyState
from .system_mode import SystemMode
from TradingRL.src.core.trader import Trader
from TradingRL.src.analysis.market_analyzer import MarketAnalyzer, MarketRegime
from TradingRL.src.analysis.event_manager import EventManager, Event, EventType
from TradingRL.src.core.risk_manager import RiskManager
from TradingRL.src.core.portfolio_manager import PortfolioManager


class RLStrategy(BaseStrategy):
    """Reinforcement Learning based trading strategy."""

    def __init__(
        self,
        name: str,
        trader: Trader,
        symbols: List[str],
        event_manager: EventManager,
        risk_manager: RiskManager,
        portfolio_manager: PortfolioManager,
        market_analyzer: MarketAnalyzer,
        feature_window: int = 100,
        confidence_threshold: float = 0.6,
        use_wandb: bool = True,
        mode: SystemMode = SystemMode.PAPER,
    ):
        """Initialize RL Strategy."""
        super().__init__(
            name=name,
            symbols=symbols,
            event_manager=event_manager,
            risk_manager=risk_manager,
            portfolio_manager=portfolio_manager,
            market_analyzer=market_analyzer,
        )

        self.trader = trader
        self.feature_window = feature_window
        self.confidence_threshold = confidence_threshold
        self.use_wandb = (
            use_wandb and mode != SystemMode.TEST
        )  # Disable wandb in test mode
        self.mode = mode

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
                train_states = self._prepare_training_data(df, symbol)

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
        try:
            if "close" not in data.columns:
                self.logger.error("Missing required column: close")
                return np.zeros(5)

            df = data.tail(self.feature_window)

            # Return zeros if data is empty or too short
            if len(df) < 2:  # Need at least 2 points for returns
                self.logger.warning("Insufficient data for price features")
                return np.zeros(5)

            # Calculate returns with explicit fill_method=None
            returns = df["close"].pct_change(fill_method=None).fillna(0)
            log_returns = np.log1p(returns)

            # Calculate features with safe operations
            features = np.array(
                [
                    (
                        float(log_returns.mean()) if len(log_returns) > 0 else 0.0
                    ),  # Average return
                    (
                        float(log_returns.std()) if len(log_returns) > 1 else 0.0
                    ),  # Volatility
                    (
                        float((df["high"] / df["low"] - 1).mean())
                        if len(df) > 0
                        else 0.0
                    ),  # Average range
                    (
                        float((df["close"] / df["open"] - 1).mean())
                        if len(df) > 0
                        else 0.0
                    ),  # Average candle return
                    (
                        float((df["close"].iloc[-1] / df["close"].mean() - 1))
                        if len(df) > 0
                        else 0.0
                    ),  # Price vs MA
                ]
            )

            # Replace any NaN or inf values with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            return features

        except Exception as e:
            self.logger.error(f"Error calculating price features: {str(e)}")
            return np.zeros(5)  # Return zeros if error

    def _calculate_volume_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate volume-based features."""
        try:
            if "volume" not in data.columns:
                self.logger.error("Missing required column: volume")
                return np.zeros(5)

            df = data.tail(self.feature_window)

            # Return zeros if data is empty or too short
            if len(df) < 2:
                self.logger.warning("Insufficient data for volume features")
                return np.zeros(5)

            # Calculate volume metrics with safe operations
            volume = df["volume"].values
            volume_ma = (
                np.mean(volume) if len(volume) > 0 else 1.0
            )  # Use 1.0 to avoid division by zero
            price_volume = df["close"] * df["volume"]

            features = np.array(
                [
                    (
                        float(volume[-1] / volume_ma) if len(volume) > 0 else 0.0
                    ),  # Relative volume
                    (
                        float(np.std(volume) / volume_ma) if len(volume) > 1 else 0.0
                    ),  # Volume volatility
                    (
                        float(np.corrcoef(df["close"], volume)[0, 1])
                        if len(volume) > 1
                        else 0.0
                    ),  # Price-volume correlation
                    (
                        float(price_volume.sum() / volume.sum())
                        if volume.sum() > 0
                        else 0.0
                    ),  # VWAP
                    (
                        float((volume[-1] - volume[-2]) / volume[-2])
                        if len(volume) > 1
                        else 0.0
                    ),  # Volume momentum
                ]
            )

            # Replace any NaN or inf values with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            return features

        except Exception as e:
            self.logger.error(f"Error calculating volume features: {str(e)}")
            return np.zeros(5)  # Return zeros if error

    def _calculate_technical_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate technical indicator features."""
        try:
            df = data.tail(self.feature_window)

            # Return zeros if data is empty or too short
            if len(df) < 50:  # Need enough data for indicators
                return np.zeros(5)

            close = df["close"].values
            high = df["high"].values
            low = df["low"].values

            # Calculate indicators with safe operations
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            rsi = talib.RSI(close, timeperiod=14)
            macd = talib.MACD(close)[0]
            bbands = talib.BBANDS(close)

            # Get latest values safely
            features = np.array(
                [
                    (
                        float((close[-1] / sma_20[-1] - 1))
                        if not np.isnan(sma_20[-1])
                        else 0.0
                    ),  # Price vs SMA20
                    (
                        float((close[-1] / sma_50[-1] - 1))
                        if not np.isnan(sma_50[-1])
                        else 0.0
                    ),  # Price vs SMA50
                    (
                        float(rsi[-1] / 100.0) if not np.isnan(rsi[-1]) else 0.5
                    ),  # Normalized RSI
                    float(macd[-1]) if not np.isnan(macd[-1]) else 0.0,  # MACD
                    (
                        float(
                            (close[-1] - bbands[0][-1])
                            / (bbands[2][-1] - bbands[0][-1])
                        )
                        if not np.isnan(bbands[0][-1])
                        else 0.0
                    ),  # BB position
                ]
            )

            # Replace any NaN or inf values with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            return features

        except Exception as e:
            self.logger.error(f"Error calculating technical features: {e}")
            return np.zeros(5)  # Return zeros if error

    def _calculate_regime_features(self, symbol: str) -> np.ndarray:
        """Calculate market regime features."""
        try:
            # Get current regime from market analyzer
            current_state = self.market_analyzer.current_state.get(symbol, {})
            current_regime = current_state.get("regime", MarketRegime.UNDEFINED)

            # Create one-hot encoding array
            num_regimes = len(MarketRegime)
            features = np.zeros(num_regimes)

            # Set the corresponding regime index to 1
            if isinstance(current_regime, MarketRegime):
                regime_idx = list(MarketRegime).index(current_regime)
                features[regime_idx] = 1

            return features

        except Exception as e:
            self.logger.error(f"Error calculating regime features: {e}")
            return np.zeros(len(MarketRegime))  # Return zeros if error

    def _calculate_position_features(self, symbol: str) -> np.ndarray:
        """Calculate position-related features."""
        try:
            position = self.positions.get(symbol)

            if position:
                features = np.array(
                    [
                        1,  # Has position
                        position["quantity"],  # Position size
                        position["unrealized_pnl"],  # Unrealized PnL
                        position["duration"].total_seconds()
                        / 3600,  # Position duration in hours
                        position["roi"],  # Return on investment
                    ]
                )
            else:
                features = np.zeros(5)

            return features

        except Exception as e:
            self.logger.error(f"Error calculating position features: {e}")
            return np.zeros(5)  # Return zeros if error

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state features."""
        try:
            # Replace infinities with large finite values
            state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)

            # Use running mean and std for normalization
            if not hasattr(self, "state_mean"):
                self.state_mean = np.zeros_like(state)
                self.state_std = np.ones_like(state)
                self.state_count = 0

            # Update running statistics
            self.state_count += 1
            delta = state - self.state_mean
            self.state_mean += delta / self.state_count
            delta2 = state - self.state_mean
            self.state_std = np.sqrt(
                (self.state_std**2 * (self.state_count - 1) + delta * delta2)
                / self.state_count
            )

            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            normalized_state = (state - self.state_mean) / (self.state_std + epsilon)

            # Clip to reasonable range
            normalized_state = np.clip(normalized_state, -10, 10)

            return normalized_state

        except Exception as e:
            self.logger.error(f"Error normalizing state: {e}")
            return np.clip(state, -10, 10)  # Return clipped state if error

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

            # Get current market state
            market_state = self.market_analyzer.current_state.get(symbol, {})

            # Calculate position size
            if signal_type in [SignalType.LONG, SignalType.SHORT]:
                size = await self._calculate_position_size(
                    symbol, current_data["close"], confidence
                )
            else:
                # For exit signals, use the current position size or a default value for testing
                size = (
                    self.positions.get(symbol, {}).get("quantity", 1.0)
                    if symbol in self.positions
                    else 1.0  # Default test value
                )

            # Create signal
            signal = Signal(
                type=signal_type,
                symbol=symbol,
                price=float(current_data["close"]),  # Ensure price is float
                size=float(size),  # Ensure size is float
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    "action": action,
                    "market_regime": market_state.get(
                        "regime", MarketRegime.UNDEFINED
                    ).value,
                    "volatility": market_state.get("volatility", 0.0),
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
        try:
            if not self.use_wandb:
                return

            # Initialize wandb if not already initialized
            if not wandb.run:
                try:
                    wandb.init(
                        project="trading_rl",
                        name=f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        config={
                            "strategy": self.name,
                            "symbols": self.symbols,
                            "feature_window": self.feature_window,
                            "confidence_threshold": self.confidence_threshold,
                        },
                        mode="disabled" if self.mode == SystemMode.TEST else "online",
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to initialize wandb: {e}")
                    self.use_wandb = False
                    return

            # Log metrics
            for symbol, symbol_metrics in metrics.items():
                wandb.log(
                    {
                        f"optimization/{symbol}/loss": symbol_metrics["loss"],
                        f"optimization/{symbol}/accuracy": symbol_metrics["accuracy"],
                        f"optimization/{symbol}/reward": symbol_metrics["avg_reward"],
                    }
                )

        except Exception as e:
            self.logger.error(f"Error logging optimization metrics: {e}")
            self.use_wandb = False  # Disable wandb on error

    def _prepare_training_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Prepare data for training."""
        try:
            # Calculate features for each row
            features_list = []
            for i in range(len(data)):
                window_data = data.iloc[max(0, i - self.feature_window + 1) : i + 1]
                if len(window_data) < self.feature_window:
                    continue

                # Calculate features
                price_features = self._calculate_price_features(window_data)
                volume_features = self._calculate_volume_features(window_data)
                technical_features = self._calculate_technical_features(window_data)
                regime_features = self._calculate_regime_features(symbol)
                position_features = self._calculate_position_features(symbol)

                # Combine all features
                all_features = np.concatenate(
                    [
                        price_features,
                        volume_features,
                        technical_features,
                        regime_features,
                        position_features,
                    ]
                )

                features_list.append(all_features)

            # Convert to DataFrame
            if not features_list:
                return pd.DataFrame()

            feature_names = [f"feature_{i}" for i in range(len(features_list[0]))]
            features_df = pd.DataFrame(
                features_list,
                index=data.index[-len(features_list) :],
                columns=feature_names,
            )

            return features_df

        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame()

    async def _warmup(self) -> None:
        """Perform strategy warmup."""
        try:
            self.logger.info("Starting warmup period...")

            # Initialize state buffer with historical data
            for symbol in self.symbols:
                # Get historical data
                if hasattr(self, "test_market_data"):
                    historical_data = self.test_market_data.get(symbol, pd.DataFrame())
                else:
                    historical_data = await self.data_fetcher.get_historical_data(
                        symbol=symbol,
                        interval="1m",
                        limit=self.feature_window * 2,  # Get extra data for warmup
                    )

                # Calculate initial states
                for i in range(len(historical_data) - self.feature_window + 1):
                    window_data = historical_data.iloc[i : i + self.feature_window]
                    state = self._prepare_state(symbol, window_data)
                    if state is not None:
                        self.state_buffer[symbol].append(state)

                self.logger.info(
                    f"Initialized state buffer for {symbol} with {len(self.state_buffer[symbol])} states"
                )

            # Initial parameter optimization
            await self.optimize_parameters(
                {symbol: self.data_buffer[symbol] for symbol in self.symbols}
            )

            self.logger.info("Warmup completed")

        except Exception as e:
            self.logger.error(f"Error during warmup: {e}")
            raise
