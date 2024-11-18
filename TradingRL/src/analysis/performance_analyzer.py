from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging
import wandb


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    volatility: float
    trades_count: int
    timestamp: datetime


class PerformanceAnalyzer:
    """Analyzes trading performance without pyfolio dependency."""

    def __init__(self, initial_capital: float = 10000.0):
        """Initialize the analyzer."""
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[PerformanceMetrics] = []

    def calculate_metrics(
        self, returns: pd.Series, trades: pd.DataFrame
    ) -> PerformanceMetrics:
        """Calculate performance metrics from returns and trades."""
        try:
            # Basic return metrics
            total_return = (1 + returns).prod() - 1
            volatility = returns.std() * np.sqrt(252)  # Annualized
            sharpe_ratio = (
                np.sqrt(252) * returns.mean() / returns.std()
                if returns.std() != 0
                else 0
            )

            # Maximum drawdown
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = cum_returns / rolling_max - 1
            max_drawdown = drawdowns.min()

            # Trade metrics
            if not trades.empty:
                winning_trades = trades[trades["pnl"] > 0]
                win_rate = len(winning_trades) / len(trades)

                total_profits = trades[trades["pnl"] > 0]["pnl"].sum()
                total_losses = abs(trades[trades["pnl"] < 0]["pnl"].sum())
                profit_factor = (
                    total_profits / total_losses if total_losses != 0 else float("inf")
                )

                avg_trade_return = trades["pnl"].mean()
                trades_count = len(trades)
            else:
                win_rate = 0.0
                profit_factor = 0.0
                avg_trade_return = 0.0
                trades_count = 0

            metrics = PerformanceMetrics(
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_trade_return=avg_trade_return,
                volatility=volatility,
                trades_count=trades_count,
                timestamp=datetime.now(),
            )

            self.metrics_history.append(metrics)
            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            raise

    def plot_performance(
        self, returns: pd.Series, trades: pd.DataFrame
    ) -> Dict[str, plt.Figure]:
        """Create performance visualization plots."""
        try:
            plots = {}

            # Cumulative returns plot
            fig_returns, ax = plt.subplots(figsize=(12, 6))
            cum_returns = (1 + returns).cumprod()
            cum_returns.plot(ax=ax)
            ax.set_title("Cumulative Returns")
            ax.grid(True)
            plots["cumulative_returns"] = fig_returns

            # Drawdown plot
            fig_dd, ax = plt.subplots(figsize=(12, 6))
            rolling_max = cum_returns.expanding().max()
            drawdowns = cum_returns / rolling_max - 1
            drawdowns.plot(ax=ax)
            ax.set_title("Drawdowns")
            ax.grid(True)
            plots["drawdowns"] = fig_dd

            if not trades.empty:
                # Trade returns distribution
                fig_dist, ax = plt.subplots(figsize=(12, 6))
                sns.histplot(trades["pnl"], ax=ax)
                ax.set_title("Trade Returns Distribution")
                plots["trade_distribution"] = fig_dist

            return plots

        except Exception as e:
            self.logger.error(f"Error creating performance plots: {e}")
            raise

    def log_metrics(self, metrics: PerformanceMetrics) -> None:
        """Log metrics to wandb if available."""
        try:
            if wandb.run is not None:
                wandb.log(
                    {
                        "performance/total_return": metrics.total_return,
                        "performance/sharpe_ratio": metrics.sharpe_ratio,
                        "performance/max_drawdown": metrics.max_drawdown,
                        "performance/win_rate": metrics.win_rate,
                        "performance/profit_factor": metrics.profit_factor,
                        "performance/avg_trade_return": metrics.avg_trade_return,
                        "performance/volatility": metrics.volatility,
                        "performance/trades_count": metrics.trades_count,
                    }
                )
        except Exception as e:
            self.logger.error(f"Error logging metrics: {e}")
