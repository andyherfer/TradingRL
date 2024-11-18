import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import scipy.stats as stats
from enum import Enum
import wandb


class RiskLevel(Enum):
    """Enumeration for different risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PositionConfig:
    """Configuration for position sizing."""

    max_position_size: float = 0.1  # Maximum position size as fraction of portfolio
    base_position_size: float = 0.02  # Base position size as fraction of portfolio
    min_position_size: float = 0.01  # Minimum position size as fraction of portfolio
    max_leverage: float = 1.0  # Maximum allowed leverage
    risk_per_trade: float = 0.01  # Maximum risk per trade as fraction of portfolio


@dataclass
class RiskMetrics:
    """Container for risk metrics."""

    value_at_risk: float
    expected_shortfall: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    profit_factor: float


class RiskManager:
    """
    Advanced risk management system for crypto trading.
    Handles position sizing, risk metrics, and trading constraints.
    """

    def __init__(
        self,
        position_config: Optional[PositionConfig] = None,
        confidence_level: float = 0.95,
        lookback_window: int = 100,
        max_drawdown: float = 0.2,
        use_wandb: bool = True,
    ):
        """
        Initialize the RiskManager.

        Args:
            position_config: Configuration for position sizing
            confidence_level: Confidence level for VaR calculation
            lookback_window: Window size for calculating metrics
            max_drawdown: Maximum allowed drawdown
            use_wandb: Whether to log to Weights & Biases
        """
        self.position_config = position_config or PositionConfig()
        self.confidence_level = confidence_level
        self.lookback_window = lookback_window
        self.max_drawdown = max_drawdown
        self.use_wandb = use_wandb

        # Initialize tracking variables
        self.performance_history = []
        self.position_history = []
        self.trade_history = []
        self.current_drawdown = 0.0
        self.peak_value = 0.0
        self.winning_streak = 0
        self.losing_streak = 0

        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def calculate_risk_metrics(
        self, returns: np.ndarray, current_value: float
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.

        Args:
            returns: Array of historical returns
            current_value: Current portfolio value

        Returns:
            RiskMetrics object containing calculated metrics
        """
        try:
            # Calculate Value at Risk (VaR)
            var = -np.percentile(returns, (1 - self.confidence_level) * 100)

            # Calculate Expected Shortfall (ES/CVaR)
            es = -np.mean(returns[returns <= -var])

            # Calculate volatility
            volatility = np.std(returns) * np.sqrt(252)  # Annualized

            # Calculate Sharpe Ratio (assuming risk-free rate = 0)
            sharpe = (
                np.mean(returns) / np.std(returns) * np.sqrt(252)
                if np.std(returns) != 0
                else 0
            )

            # Calculate maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            rolling_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = np.min(drawdowns)

            # Calculate current drawdown
            if len(self.performance_history) > 0:
                peak_value = max(max(self.performance_history), current_value)
                self.current_drawdown = (current_value - peak_value) / peak_value

            # Calculate win rate and profit factor
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
            profit_factor = (
                abs(np.sum(positive_returns) / np.sum(negative_returns))
                if len(negative_returns) > 0 and np.sum(negative_returns) != 0
                else np.inf
            )

            metrics = RiskMetrics(
                value_at_risk=var,
                expected_shortfall=es,
                volatility=volatility,
                sharpe_ratio=sharpe,
                max_drawdown=max_drawdown,
                current_drawdown=self.current_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
            )

            # Log metrics to W&B if enabled
            if self.use_wandb and wandb.run is not None:
                wandb.log(
                    {
                        "risk/value_at_risk": var,
                        "risk/expected_shortfall": es,
                        "risk/volatility": volatility,
                        "risk/sharpe_ratio": sharpe,
                        "risk/max_drawdown": max_drawdown,
                        "risk/current_drawdown": self.current_drawdown,
                        "risk/win_rate": win_rate,
                        "risk/profit_factor": profit_factor,
                    }
                )

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            raise

    def calculate_position_size(
        self,
        current_balance: float,
        current_price: float,
        volatility: float,
        risk_metrics: RiskMetrics,
    ) -> Tuple[float, Dict]:
        """
        Calculate optimal position size based on various factors.

        Args:
            current_balance: Current account balance
            current_price: Current asset price
            volatility: Current volatility
            risk_metrics: Current risk metrics

        Returns:
            Tuple of (position_size, metadata)
        """
        try:
            # Base position size
            position_size = current_balance * self.position_config.base_position_size

            # Adjust for volatility
            vol_adjustment = np.exp(-volatility)  # Reduce size when volatility is high
            position_size *= vol_adjustment

            # Adjust for drawdown
            if risk_metrics.current_drawdown < -0.1:  # Reduce size in drawdown
                drawdown_factor = np.exp(2 * risk_metrics.current_drawdown)
                position_size *= drawdown_factor

            # Adjust for win/loss streaks
            streak_factor = 1.0
            if self.winning_streak > 3:
                streak_factor = min(1.2, 1 + (self.winning_streak - 3) * 0.05)
            elif self.losing_streak > 3:
                streak_factor = max(0.8, 1 - (self.losing_streak - 3) * 0.05)
            position_size *= streak_factor

            # Adjust for Sharpe ratio
            if risk_metrics.sharpe_ratio > 1:
                sharpe_factor = min(1.2, 1 + (risk_metrics.sharpe_ratio - 1) * 0.1)
                position_size *= sharpe_factor

            # Apply position limits
            position_size = min(
                position_size, current_balance * self.position_config.max_position_size
            )
            position_size = max(
                position_size, current_balance * self.position_config.min_position_size
            )

            # Calculate number of units based on current price
            units = position_size / current_price

            # Record metadata
            metadata = {
                "base_size": self.position_config.base_position_size * current_balance,
                "vol_adjustment": vol_adjustment,
                "drawdown_factor": (
                    drawdown_factor if risk_metrics.current_drawdown < -0.1 else 1.0
                ),
                "streak_factor": streak_factor,
                "sharpe_factor": (
                    sharpe_factor if risk_metrics.sharpe_ratio > 1 else 1.0
                ),
                "final_position_size": position_size,
                "units": units,
            }

            # Log to W&B if enabled
            if self.use_wandb and wandb.run is not None:
                wandb.log(
                    {
                        "position/size": position_size,
                        "position/units": units,
                        "position/vol_adjustment": vol_adjustment,
                        "position/streak_factor": streak_factor,
                        **{f"position/{k}": v for k, v in metadata.items()},
                    }
                )

            return units, metadata

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            raise

    def calculate_stop_levels(
        self,
        entry_price: float,
        position_size: float,
        volatility: float,
        risk_metrics: RiskMetrics,
    ) -> Dict[str, float]:
        """
        Calculate stop-loss and take-profit levels.

        Args:
            entry_price: Entry price for the position
            position_size: Size of the position
            volatility: Current volatility
            risk_metrics: Current risk metrics

        Returns:
            Dictionary containing stop levels
        """
        try:
            # Calculate ATR-based stops
            atr_multiple = 2.0
            stop_distance = volatility * entry_price * atr_multiple

            # Adjust based on risk metrics
            if risk_metrics.sharpe_ratio < 1:
                stop_distance *= 0.8  # Tighter stops in poor performance

            # Calculate levels
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * 1.5)  # 1.5 risk-reward ratio

            # Additional dynamic levels
            trailing_stop = entry_price - (stop_distance * 0.8)
            breakeven_stop = entry_price + (stop_distance * 0.5)

            levels = {
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing_stop": trailing_stop,
                "breakeven_stop": breakeven_stop,
                "risk_reward_ratio": 1.5,
            }

            # Log to W&B if enabled
            if self.use_wandb and wandb.run is not None:
                wandb.log(
                    {
                        "stops/stop_loss": stop_loss,
                        "stops/take_profit": take_profit,
                        "stops/trailing_stop": trailing_stop,
                        "stops/breakeven_stop": breakeven_stop,
                        "stops/risk_reward_ratio": 1.5,
                    }
                )

            return levels

        except Exception as e:
            self.logger.error(f"Error calculating stop levels: {e}")
            raise

    def update_performance(
        self, trade_result: float, current_value: float, timestamp: datetime
    ) -> None:
        """
        Update performance tracking after a trade.

        Args:
            trade_result: PnL from the trade
            current_value: Current portfolio value
            timestamp: Time of the trade
        """
        try:
            # Update performance history
            self.performance_history.append(current_value)

            # Update peak value
            self.peak_value = max(self.peak_value, current_value)

            # Update current drawdown
            self.current_drawdown = (current_value - self.peak_value) / self.peak_value

            # Update win/loss streaks
            if trade_result > 0:
                self.winning_streak += 1
                self.losing_streak = 0
            elif trade_result < 0:
                self.losing_streak += 1
                self.winning_streak = 0

            # Record trade result
            self.trade_history.append(
                {
                    "timestamp": timestamp,
                    "result": trade_result,
                    "portfolio_value": current_value,
                    "drawdown": self.current_drawdown,
                }
            )

            # Log to W&B if enabled
            if self.use_wandb and wandb.run is not None:
                wandb.log(
                    {
                        "performance/trade_result": trade_result,
                        "performance/portfolio_value": current_value,
                        "performance/drawdown": self.current_drawdown,
                        "performance/winning_streak": self.winning_streak,
                        "performance/losing_streak": self.losing_streak,
                    }
                )

        except Exception as e:
            self.logger.error(f"Error updating performance: {e}")
            raise

    def check_risk_limits(self, risk_metrics: RiskMetrics) -> Tuple[bool, str]:
        """
        Check if current risk levels are within acceptable limits.

        Args:
            risk_metrics: Current risk metrics

        Returns:
            Tuple of (is_safe, reason)
        """
        try:
            # Check maximum drawdown
            if abs(risk_metrics.current_drawdown) > self.max_drawdown:
                return (
                    False,
                    f"Maximum drawdown exceeded: {risk_metrics.current_drawdown:.2%}",
                )

            # Check losing streak
            if self.losing_streak >= 5:
                return False, f"Maximum losing streak reached: {self.losing_streak}"

            # Check Sharpe ratio
            if len(self.performance_history) > 50 and risk_metrics.sharpe_ratio < -1:
                return (
                    False,
                    f"Sharpe ratio below threshold: {risk_metrics.sharpe_ratio:.2f}",
                )

            # Check profit factor
            if len(self.trade_history) > 20 and risk_metrics.profit_factor < 0.5:
                return (
                    False,
                    f"Profit factor below threshold: {risk_metrics.profit_factor:.2f}",
                )

            return True, "Risk levels acceptable"

        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            raise

    def optimize_portfolio_exposure(
        self,
        current_positions: Dict[str, float],
        correlations: pd.DataFrame,
        risk_metrics: Dict[str, RiskMetrics],
    ) -> Dict[str, float]:
        """
        Optimize portfolio exposure based on correlations and individual asset metrics.

        Args:
            current_positions: Dictionary of current positions and their sizes
            correlations: Correlation matrix of assets
            risk_metrics: Dictionary of risk metrics for each asset

        Returns:
            Dictionary of suggested position adjustments
        """
        try:
            adjustments = {}
            total_exposure = sum(abs(size) for size in current_positions.values())

            # Calculate portfolio risk contribution
            for asset, size in current_positions.items():
                # Calculate correlation-weighted exposure
                corr_exposure = sum(
                    size * other_size * correlations.loc[asset, other_asset]
                    for other_asset, other_size in current_positions.items()
                )

                # Calculate risk-adjusted position size
                risk_factor = risk_metrics[asset].volatility * (
                    1 + abs(risk_metrics[asset].current_drawdown)
                )

                # Calculate suggested adjustment
                optimal_size = (
                    size * (1 / risk_factor) * (1 - abs(corr_exposure) / total_exposure)
                )

                adjustments[asset] = optimal_size - size

                # Log to W&B if enabled
                if self.use_wandb and wandb.run is not None:
                    wandb.log(
                        {
                            f"portfolio/correlation_{asset}": corr_exposure,
                            f"portfolio/risk_factor_{asset}": risk_factor,
                            f"portfolio/adjustment_{asset}": adjustments[asset],
                        }
                    )

            return adjustments

        except Exception as e:
            self.logger.error(f"Error optimizing portfolio exposure: {e}")
            raise

    def manage_active_trade(
        self,
        entry_price: float,
        current_price: float,
        position_size: float,
        stop_levels: Dict[str, float],
        unrealized_pnl: float,
        time_in_trade: int,
    ) -> Tuple[str, Optional[float]]:
        """
        Manage an active trade with dynamic exit conditions.

        Args:
            entry_price: Entry price of the position
            current_price: Current market price
            position_size: Size of the position
            stop_levels: Dictionary of stop levels
            unrealized_pnl: Current unrealized PnL
            time_in_trade: Time in trade in minutes

        Returns:
            Tuple of (action, exit_price) where action is one of:
            'hold', 'take_profit', 'stop_loss', 'trailing_stop', 'time_exit'
        """
        try:
            # Check stop loss
            if current_price <= stop_levels["stop_loss"]:
                return "stop_loss", current_price

            # Check take profit
            if current_price >= stop_levels["take_profit"]:
                return "take_profit", current_price

            # Check trailing stop
            if current_price <= stop_levels["trailing_stop"]:
                return "trailing_stop", current_price

            # Check time-based exits
            if time_in_trade > 1440:  # 24 hours
                return "time_exit", current_price

            # Dynamic exit conditions
            profit_ratio = unrealized_pnl / (position_size * entry_price)

            # Move to breakeven if in profit
            if profit_ratio > 0.01 and current_price <= entry_price:
                return "breakeven_stop", current_price

            # Log trade management metrics
            if self.use_wandb and wandb.run is not None:
                wandb.log(
                    {
                        "trade/unrealized_pnl": unrealized_pnl,
                        "trade/profit_ratio": profit_ratio,
                        "trade/time_in_trade": time_in_trade,
                        "trade/price_vs_entry": current_price / entry_price - 1,
                    }
                )

            return "hold", None

        except Exception as e:
            self.logger.error(f"Error managing active trade: {e}")
            raise

    def calculate_risk_adjusted_returns(
        self, returns: np.ndarray, window: int = 30
    ) -> pd.DataFrame:
        """
        Calculate various risk-adjusted return metrics.

        Args:
            returns: Array of historical returns
            window: Rolling window size for calculations

        Returns:
            DataFrame with risk-adjusted metrics
        """
        try:
            metrics_df = pd.DataFrame()

            # Rolling Sharpe Ratio
            rolling_sharpe = (
                pd.Series(returns).rolling(window).mean()
                / pd.Series(returns).rolling(window).std()
            ) * np.sqrt(252)
            metrics_df["rolling_sharpe"] = rolling_sharpe

            # Rolling Sortino Ratio
            negative_returns = returns.copy()
            negative_returns[returns > 0] = 0
            rolling_sortino = (
                pd.Series(returns).rolling(window).mean()
                / pd.Series(negative_returns).rolling(window).std()
            ) * np.sqrt(252)
            metrics_df["rolling_sortino"] = rolling_sortino

            # Rolling Calmar Ratio
            rolling_returns = pd.Series(returns).rolling(window)
            rolling_max_drawdown = rolling_returns.apply(
                lambda x: np.min(
                    np.maximum.accumulate(np.cumprod(1 + x)) - np.cumprod(1 + x)
                )
            )
            metrics_df["rolling_calmar"] = (
                rolling_returns.mean() * 252 / abs(rolling_max_drawdown)
            )

            # Risk-adjusted position size multiplier
            metrics_df["position_multiplier"] = (
                (1 + metrics_df["rolling_sharpe"].clip(-2, 2))
                * (1 + metrics_df["rolling_sortino"].clip(-2, 2))
                * (1 + metrics_df["rolling_calmar"].clip(-2, 2))
            ) / 8  # Normalize to reasonable range

            # Log metrics
            if self.use_wandb and wandb.run is not None:
                latest_metrics = metrics_df.iloc[-1]
                wandb.log(
                    {
                        "risk_adjusted/sharpe": latest_metrics["rolling_sharpe"],
                        "risk_adjusted/sortino": latest_metrics["rolling_sortino"],
                        "risk_adjusted/calmar": latest_metrics["rolling_calmar"],
                        "risk_adjusted/position_multiplier": latest_metrics[
                            "position_multiplier"
                        ],
                    }
                )

            return metrics_df

        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted returns: {e}")
            raise

    def generate_risk_report(self) -> Dict:
        """
        Generate a comprehensive risk report.

        Returns:
            Dictionary containing risk report data
        """
        try:
            if len(self.trade_history) == 0:
                return {"status": "No trade history available"}

            # Convert trade history to DataFrame
            trades_df = pd.DataFrame(self.trade_history)
            returns = np.diff(self.performance_history) / self.performance_history[:-1]

            # Calculate aggregate metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df["result"] > 0])
            losing_trades = len(trades_df[trades_df["result"] < 0])

            report = {
                "performance_metrics": {
                    "total_return": (
                        self.performance_history[-1] / self.performance_history[0] - 1
                    ),
                    "win_rate": (
                        winning_trades / total_trades if total_trades > 0 else 0
                    ),
                    "average_win": trades_df[trades_df["result"] > 0]["result"].mean(),
                    "average_loss": abs(
                        trades_df[trades_df["result"] < 0]["result"].mean()
                    ),
                    "largest_win": trades_df["result"].max(),
                    "largest_loss": trades_df["result"].min(),
                    "profit_factor": (
                        abs(
                            trades_df[trades_df["result"] > 0]["result"].sum()
                            / trades_df[trades_df["result"] < 0]["result"].sum()
                        )
                        if len(trades_df[trades_df["result"] < 0]) > 0
                        else np.inf
                    ),
                },
                "risk_metrics": {
                    "max_drawdown": abs(
                        min(
                            (v - max(self.performance_history[: i + 1]))
                            / max(self.performance_history[: i + 1])
                            for i, v in enumerate(self.performance_history)
                        )
                    ),
                    "volatility": np.std(returns) * np.sqrt(252),
                    "var_95": np.percentile(returns, 5),
                    "expected_shortfall": np.mean(
                        returns[returns < np.percentile(returns, 5)]
                    ),
                    "sharpe_ratio": (
                        np.mean(returns) / np.std(returns) * np.sqrt(252)
                        if np.std(returns) != 0
                        else 0
                    ),
                },
                "position_metrics": {
                    "average_position_size": np.mean(
                        [
                            abs(t["result"] / t["portfolio_value"])
                            for t in self.trade_history
                        ]
                    ),
                    "max_position_size": max(
                        [
                            abs(t["result"] / t["portfolio_value"])
                            for t in self.trade_history
                        ]
                    ),
                    "current_exposure": (
                        self.position_history[-1] if self.position_history else 0
                    ),
                },
                "streak_metrics": {
                    "current_winning_streak": self.winning_streak,
                    "current_losing_streak": self.losing_streak,
                    "max_winning_streak": max(
                        len(list(g))
                        for k, g in itertools.groupby(trades_df["result"] > 0)
                        if k
                    ),
                    "max_losing_streak": max(
                        len(list(g))
                        for k, g in itertools.groupby(trades_df["result"] < 0)
                        if k
                    ),
                },
            }

            # Log report to W&B if enabled
            if self.use_wandb and wandb.run is not None:
                wandb.log(
                    {
                        f"report/{k1}/{k2}": v2
                        for k1, v1 in report.items()
                        for k2, v2 in v1.items()
                    }
                )

            return report

        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            raise
