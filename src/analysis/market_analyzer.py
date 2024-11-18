from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from enum import Enum
from dataclasses import dataclass
import talib
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
import wandb

from .event_manager import EventManager, Event, EventType, EventPriority


class MarketRegime(Enum):
    """Market regime classification."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    UNDEFINED = "undefined"


@dataclass
class MarketState:
    """Market state information."""

    regime: MarketRegime
    volatility: float
    trend_strength: float
    support_levels: List[float]
    resistance_levels: List[float]
    liquidity: float
    timestamp: datetime
    metadata: Dict[str, Any]


class MarketAnalyzer:
    """
    Analyzes market conditions and detects regimes.
    Provides market insights for strategy adaptation.
    """

    def __init__(
        self,
        event_manager: Optional[EventManager] = None,
        config: Optional[Dict] = None,
        use_wandb: bool = True,
        training_mode: bool = False,
    ):
        """
        Initialize MarketAnalyzer.

        Args:
            event_manager: EventManager instance (optional in training mode)
            config: Analyzer configuration
            use_wandb: Whether to use W&B logging
            training_mode: Whether analyzer is being used for training
        """
        self.event_manager = event_manager
        self.config = config or self._default_config()
        self.use_wandb = use_wandb and not training_mode
        self.training_mode = training_mode

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Initialize components
        self.regime_detector = GaussianMixture(
            n_components=4, random_state=42, covariance_type="full"
        )
        self.scaler = StandardScaler()

        # State tracking
        self.current_state = {}
        self.state_history = []
        self.technical_indicators = {}

        # Setup event subscriptions only if not in training mode
        if not training_mode and event_manager is not None:
            self._setup_event_subscriptions()

    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            "regime": {
                "window_size": 100,
                "min_samples": 50,
                "volatility_threshold": 0.02,
                "trend_threshold": 0.6,
            },
            "indicators": {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bb_period": 20,
                "bb_std": 2,
            },
            "levels": {
                "support_resistance_window": 20,
                "support_resistance_threshold": 0.02,
            },
        }

    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions."""
        self.event_manager.subscribe(
            self._handle_market_update,
            [EventType.PRICE_UPDATE, EventType.MARKET_DATA],
            EventPriority.HIGH,
        )

    async def _handle_market_update(self, event: Event) -> None:
        """Handle market update events."""
        try:
            symbol = event.data["symbol"]
            data = event.data["data"]

            # Update market state
            state = await self.analyze_market_state(symbol, data)

            # Publish market state update
            await self.event_manager.publish(
                Event(
                    type=EventType.MARKET_REGIME_CHANGE,
                    data={"symbol": symbol, "state": state},
                    priority=EventPriority.NORMAL,
                )
            )

        except Exception as e:
            self.logger.error(f"Error handling market update: {e}")

    async def analyze_market_state(
        self, symbol: str, data: pd.DataFrame
    ) -> MarketState:
        """
        Analyze current market state.

        Args:
            symbol: Trading pair symbol
            data: Market data DataFrame

        Returns:
            Current market state
        """
        try:
            # Calculate technical indicators
            indicators = self._calculate_indicators(data)

            # Detect market regime
            regime = self._detect_regime(data, indicators)

            # Calculate volatility
            volatility = self._calculate_volatility(data)

            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(data, indicators)

            # Find support/resistance levels
            support_levels, resistance_levels = self._find_support_resistance(data)

            # Calculate liquidity
            liquidity = self._calculate_liquidity(data)

            # Create state object
            state = MarketState(
                regime=regime,
                volatility=volatility,
                trend_strength=trend_strength,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                liquidity=liquidity,
                timestamp=datetime.now(),
                metadata={
                    "indicators": indicators,
                    "price": data["close"].iloc[-1],
                    "volume": data["volume"].iloc[-1],
                },
            )

            # Update state tracking
            self.current_state[symbol] = state
            self.state_history.append(state)

            # Log to W&B
            if self.use_wandb:
                self._log_state(symbol, state)

            return state

        except Exception as e:
            self.logger.error(f"Error analyzing market state: {e}")
            raise

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate technical indicators."""
        try:
            close = data["close"].values
            high = data["high"].values
            low = data["low"].values
            volume = data["volume"].values

            indicators = {}

            # Trend indicators
            indicators["sma_20"] = talib.SMA(close, timeperiod=20)
            indicators["sma_50"] = talib.SMA(close, timeperiod=50)
            indicators["sma_200"] = talib.SMA(close, timeperiod=200)
            indicators["ema_20"] = talib.EMA(close, timeperiod=20)

            # Momentum indicators
            indicators["rsi"] = talib.RSI(
                close, timeperiod=self.config["indicators"]["rsi_period"]
            )

            macd, signal, hist = talib.MACD(
                close,
                fastperiod=self.config["indicators"]["macd_fast"],
                slowperiod=self.config["indicators"]["macd_slow"],
                signalperiod=self.config["indicators"]["macd_signal"],
            )
            indicators["macd"] = macd
            indicators["macd_signal"] = signal
            indicators["macd_hist"] = hist

            # Volatility indicators
            upper, middle, lower = talib.BBANDS(
                close,
                timeperiod=self.config["indicators"]["bb_period"],
                nbdevup=self.config["indicators"]["bb_std"],
                nbdevdn=self.config["indicators"]["bb_std"],
            )
            indicators["bb_upper"] = upper
            indicators["bb_middle"] = middle
            indicators["bb_lower"] = lower

            indicators["atr"] = talib.ATR(high, low, close, timeperiod=14)

            # Volume indicators
            indicators["obv"] = talib.OBV(close, volume)
            indicators["adx"] = talib.ADX(high, low, close, timeperiod=14)

            return indicators

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            raise

    def _detect_regime(
        self, data: pd.DataFrame, indicators: Dict[str, np.ndarray]
    ) -> MarketRegime:
        """Detect current market regime."""
        try:
            # Prepare features for regime detection
            features = np.column_stack(
                [
                    indicators["rsi"][-self.config["regime"]["window_size"] :],
                    indicators["adx"][-self.config["regime"]["window_size"] :],
                    indicators["macd_hist"][-self.config["regime"]["window_size"] :],
                    indicators["atr"][-self.config["regime"]["window_size"] :],
                ]
            )

            if len(features) < self.config["regime"]["min_samples"]:
                return MarketRegime.UNDEFINED

            # Scale features
            features_scaled = self.scaler.fit_transform(features)

            # Fit GMM model
            self.regime_detector.fit(features_scaled)

            # Get current regime
            current_features = features_scaled[-1].reshape(1, -1)
            regime_proba = self.regime_detector.predict_proba(current_features)[0]
            regime_idx = np.argmax(regime_proba)

            # Classify regime based on features
            close = data["close"].values
            current_price = close[-1]
            sma_20 = indicators["sma_20"][-1]
            sma_50 = indicators["sma_50"][-1]
            bb_upper = indicators["bb_upper"][-1]
            bb_lower = indicators["bb_lower"][-1]
            adx = indicators["adx"][-1]

            # Trending conditions
            if adx > 25:
                if current_price > sma_20 and sma_20 > sma_50:
                    return MarketRegime.TRENDING_UP
                elif current_price < sma_20 and sma_20 < sma_50:
                    return MarketRegime.TRENDING_DOWN

            # Ranging conditions
            if indicators["rsi"][-1] > 40 and indicators["rsi"][-1] < 60 and adx < 20:
                return MarketRegime.RANGING

            # Volatile conditions
            volatility = self._calculate_volatility(data)
            if volatility > self.config["regime"]["volatility_threshold"]:
                return MarketRegime.VOLATILE

            # Breakout conditions
            if (
                current_price > bb_upper
                and data["volume"].iloc[-1] > data["volume"].mean() * 1.5
            ):
                return MarketRegime.BREAKOUT

            # Breakdown conditions
            if (
                current_price < bb_lower
                and data["volume"].iloc[-1] > data["volume"].mean() * 1.5
            ):
                return MarketRegime.BREAKDOWN

            return MarketRegime.UNDEFINED

        except Exception as e:
            self.logger.error(f"Error detecting regime: {e}")
            raise

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate current market volatility."""
        try:
            returns = np.log(data["close"] / data["close"].shift(1)).dropna()
            return np.std(returns) * np.sqrt(252)  # Annualized volatility
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            raise

    def _calculate_trend_strength(
        self, data: pd.DataFrame, indicators: Dict[str, np.ndarray]
    ) -> float:
        """Calculate current trend strength."""
        try:
            # Use ADX as base trend strength
            adx = indicators["adx"][-1] / 100.0

            # Calculate price trend
            close = data["close"].values
            price_trend = np.polyfit(range(len(close[-20:])), close[-20:], 1)[0]

            # Normalize trend
            trend_direction = np.sign(price_trend)
            trend_magnitude = min(abs(price_trend / close[-1]), 1.0)

            # Combine metrics
            trend_strength = (adx + trend_magnitude) / 2 * trend_direction

            return trend_strength

        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            raise

    def _find_support_resistance(
        self, data: pd.DataFrame
    ) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels."""
        try:
            window = self.config["levels"]["support_resistance_window"]
            threshold = self.config["levels"]["support_resistance_threshold"]

            highs = data["high"].values
            lows = data["low"].values

            support_levels = []
            resistance_levels = []

            # Find local extrema
            for i in range(window, len(data) - window):
                # Resistance levels
                if all(highs[i] > highs[i - window : i]) and all(
                    highs[i] > highs[i + 1 : i + window + 1]
                ):
                    resistance_levels.append(highs[i])

                # Support levels
                if all(lows[i] < lows[i - window : i]) and all(
                    lows[i] < lows[i + 1 : i + window + 1]
                ):
                    support_levels.append(lows[i])

            # Cluster levels
            support_levels = self._cluster_price_levels(support_levels, threshold)
            resistance_levels = self._cluster_price_levels(resistance_levels, threshold)

            return support_levels, resistance_levels

        except Exception as e:
            self.logger.error(f"Error finding support/resistance: {e}")
            raise

    def _cluster_price_levels(
        self, levels: List[float], threshold: float
    ) -> List[float]:
        """Cluster nearby price levels."""
        if not levels:
            return []

        clusters = []
        current_cluster = [levels[0]]

        for level in levels[1:]:
            if (
                abs(level - np.mean(current_cluster)) / np.mean(current_cluster)
                <= threshold
            ):
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]

        clusters.append(np.mean(current_cluster))
        return sorted(clusters)

    def _calculate_liquidity(self, data: pd.DataFrame) -> float:
        """Calculate market liquidity score."""
        try:
            # Calculate volume-based liquidity
            recent_volume = data["volume"].iloc[-20:].mean()
            volume_ratio = data["volume"].iloc[-1] / recent_volume

            # Calculate spread-based liquidity
            typical_spread = (data["high"] - data["low"]).iloc[-20:].mean()
            current_spread = data["high"].iloc[-1] - data["low"].iloc[-1]
            spread_ratio = typical_spread / current_spread if current_spread > 0 else 0

            # Combine metrics
            liquidity_score = (volume_ratio + spread_ratio) / 2
            return min(liquidity_score, 1.0)  # Normalize to [0, 1]

        except Exception as e:
            self.logger.error(f"Error calculating liquidity: {e}")
            raise

    def _log_state(self, symbol: str, state: MarketState) -> None:
        """Log market state to W&B."""
        if not self.use_wandb:
            return

        try:
            wandb.log(
                {
                    f"market/{symbol}/regime": state.regime.value,
                    f"market/{symbol}/volatility": state.volatility,
                    f"market/{symbol}/trend_strength": state.trend_strength,
                    f"market/{symbol}/liquidity": state.liquidity,
                    f"market/{symbol}/support_levels": len(state.support_levels),
                    f"market/{symbol}/resistance_levels": len(state.resistance_levels),
                    f"market/{symbol}/price": state.metadata["price"],
                    f"market/{symbol}/volume": state.metadata["volume"],
                }
            )

            # Log indicator values
            for name, value in state.metadata["indicators"].items():
                if isinstance(value, np.ndarray):
                    value = value[-1]
                wandb.log({f"indicators/{symbol}/{name}": value})

        except Exception as e:
            self.logger.error(f"Error logging state: {e}")

    async def get_market_analysis(
        self, symbol: str, lookback_period: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive market analysis.

        Args:
            symbol: Trading pair symbol
            lookback_period: Analysis lookback period

        Returns:
            Market analysis dictionary
        """
        try:
            current_state = self.current_state.get(symbol)
            if not current_state:
                return {}

            # Get relevant historical states
            history = self.state_history
            if lookback_period:
                history = history[-lookback_period:]

            # Calculate regime statistics
            regime_stats = self._calculate_regime_stats(history)

            # Calculate price levels
            price_levels = self._analyze_price_levels(symbol)

            # Calculate market dynamics
            dynamics = self._analyze_market_dynamics(symbol)

            analysis = {
                "current_state": {
                    "regime": current_state.regime.value,
                    "volatility": current_state.volatility,
                    "trend_strength": current_state.trend_strength,
                    "liquidity": current_state.liquidity,
                },
                "regime_statistics": regime_stats,
                "price_levels": price_levels,
                "market_dynamics": dynamics,
                "timestamp": datetime.now(),
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Error getting market analysis: {e}")
            raise

    def _calculate_regime_stats(self, history: List[MarketState]) -> Dict[str, Any]:
        """Calculate regime statistics from history."""
        try:
            if not history:
                return {}

            # Count regime occurrences
            regime_counts = {}
            regime_durations = defaultdict(list)
            current_regime = None
            current_duration = 0

            for state in history:
                regime = state.regime
                regime_counts[regime] = regime_counts.get(regime, 0) + 1

                if regime == current_regime:
                    current_duration += 1
                else:
                    if current_regime is not None:
                        regime_durations[current_regime].append(current_duration)
                    current_regime = regime
                    current_duration = 1

            # Calculate statistics
            total_states = len(history)
            stats = {
                "regime_probabilities": {
                    regime.value: count / total_states
                    for regime, count in regime_counts.items()
                },
                "regime_durations": {
                    regime.value: {
                        "mean": np.mean(durations),
                        "std": np.std(durations),
                        "max": max(durations),
                    }
                    for regime, durations in regime_durations.items()
                },
                "transitions": self._calculate_regime_transitions(history),
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating regime stats: {e}")
            raise

    def _calculate_regime_transitions(
        self, history: List[MarketState]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate regime transition probabilities."""
        try:
            transitions = defaultdict(lambda: defaultdict(int))

            for i in range(len(history) - 1):
                current_regime = history[i].regime
                next_regime = history[i + 1].regime
                transitions[current_regime][next_regime] += 1

            # Convert to probabilities
            probabilities = {}
            for regime, counts in transitions.items():
                total = sum(counts.values())
                probabilities[regime.value] = {
                    next_regime.value: count / total
                    for next_regime, count in counts.items()
                }

            return probabilities

        except Exception as e:
            self.logger.error(f"Error calculating regime transitions: {e}")
            raise

    def _analyze_price_levels(self, symbol: str) -> Dict[str, Any]:
        """Analyze price levels and market structure."""
        try:
            current_state = self.current_state.get(symbol)
            if not current_state:
                return {}

            current_price = current_state.metadata["price"]

            # Find nearest levels
            nearest_support = self._find_nearest_level(
                current_price, current_state.support_levels, below=True
            )
            nearest_resistance = self._find_nearest_level(
                current_price, current_state.resistance_levels, below=False
            )

            # Calculate level strengths
            support_strength = self._calculate_level_strength(
                nearest_support, current_state.support_levels
            )
            resistance_strength = self._calculate_level_strength(
                nearest_resistance, current_state.resistance_levels
            )

            return {
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "support_strength": support_strength,
                "resistance_strength": resistance_strength,
                "price_position": {
                    "distance_to_support": (
                        (current_price - nearest_support) / current_price
                        if nearest_support
                        else None
                    ),
                    "distance_to_resistance": (
                        (nearest_resistance - current_price) / current_price
                        if nearest_resistance
                        else None
                    ),
                },
            }

        except Exception as e:
            self.logger.error(f"Error analyzing price levels: {e}")
            raise

    def _find_nearest_level(
        self, price: float, levels: List[float], below: bool = True
    ) -> Optional[float]:
        """Find nearest support/resistance level."""
        if not levels:
            return None

        levels = np.array(levels)
        if below:
            mask = levels < price
            return levels[mask].max() if mask.any() else None
        else:
            mask = levels > price
            return levels[mask].min() if mask.any() else None

    def _calculate_level_strength(
        self, level: Optional[float], all_levels: List[float], threshold: float = 0.01
    ) -> float:
        """Calculate strength of a price level."""
        if not level or not all_levels:
            return 0.0

        # Count nearby levels
        nearby_levels = sum(
            1 for l in all_levels if abs(l - level) / level <= threshold
        )

        # Normalize strength score
        strength = min(nearby_levels / 5, 1.0)  # Cap at 1.0
        return strength

    def _analyze_market_dynamics(self, symbol: str) -> Dict[str, Any]:
        """Analyze market dynamics and patterns."""
        try:
            current_state = self.current_state.get(symbol)
            if not current_state:
                return {}

            indicators = current_state.metadata["indicators"]

            # Trend analysis
            trend_analysis = {
                "short_term": self._analyze_trend(indicators, "short"),
                "medium_term": self._analyze_trend(indicators, "medium"),
                "long_term": self._analyze_trend(indicators, "long"),
            }

            # Momentum analysis
            momentum_analysis = {
                "rsi_condition": self._analyze_rsi(indicators["rsi"][-1]),
                "macd_condition": self._analyze_macd(
                    indicators["macd"][-1], indicators["macd_signal"][-1]
                ),
            }

            # Volatility analysis
            volatility_analysis = {
                "current_volatility": current_state.volatility,
                "volatility_regime": self._classify_volatility(
                    current_state.volatility
                ),
                "bb_position": self._analyze_bb_position(
                    current_state.metadata["price"],
                    indicators["bb_upper"][-1],
                    indicators["bb_lower"][-1],
                ),
            }

            return {
                "trend_analysis": trend_analysis,
                "momentum_analysis": momentum_analysis,
                "volatility_analysis": volatility_analysis,
            }

        except Exception as e:
            self.logger.error(f"Error analyzing market dynamics: {e}")
            raise

    def _analyze_trend(
        self, indicators: Dict[str, np.ndarray], timeframe: str
    ) -> Dict[str, Any]:
        """Analyze price trend for different timeframes."""
        try:
            price = indicators["sma_20"][-1]  # Use SMA20 as reference

            if timeframe == "short":
                ma1, ma2 = indicators["sma_20"][-1], indicators["sma_50"][-1]
            elif timeframe == "medium":
                ma1, ma2 = indicators["sma_50"][-1], indicators["sma_200"][-1]
            else:  # long term
                ma1 = indicators["sma_200"][-1]
                ma2 = np.mean(indicators["sma_200"][-20:])

            trend_strength = abs(ma1 - ma2) / ma2
            trend_direction = np.sign(ma1 - ma2)

            return {
                "direction": trend_direction,
                "strength": trend_strength,
                "ma_alignment": ma1 > ma2,
            }

        except Exception as e:
            self.logger.error(f"Error analyzing trend: {e}")
            raise

    def _analyze_rsi(self, rsi: float) -> str:
        """Analyze RSI conditions."""
        if rsi >= 70:
            return "overbought"
        elif rsi <= 30:
            return "oversold"
        elif rsi >= 60:
            return "bullish"
        elif rsi <= 40:
            return "bearish"
        else:
            return "neutral"

    def _analyze_macd(self, macd: float, signal: float) -> Dict[str, Any]:
        """Analyze MACD conditions."""
        return {
            "position": "above" if macd > signal else "below",
            "strength": abs(macd - signal),
            "divergence": macd - signal,
        }

    def _classify_volatility(self, volatility: float) -> str:
        """Classify volatility level."""
        if volatility < 0.01:
            return "very_low"
        elif volatility < 0.02:
            return "low"
        elif volatility < 0.03:
            return "moderate"
        elif volatility < 0.04:
            return "high"
        else:
            return "very_high"

    def _analyze_bb_position(
        self, price: float, upper: float, lower: float
    ) -> Dict[str, Any]:
        """Analyze position relative to Bollinger Bands."""
        bb_width = (upper - lower) / ((upper + lower) / 2)
        position = (price - lower) / (upper - lower)

        return {
            "width": bb_width,
            "position": position,
            "condition": (
                "overbought"
                if position > 1
                else "oversold" if position < 0 else "neutral"
            ),
        }
