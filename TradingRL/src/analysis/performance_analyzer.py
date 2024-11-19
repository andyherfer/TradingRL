from typing import Dict, List, Optional, Any
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
    """Analyzes trading performance."""

    def __init__(self, initial_capital: float = 10000.0):
        """Initialize the analyzer."""
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[PerformanceMetrics] = []

    async def analyze_performance(
        self, portfolio_history: pd.DataFrame, trades: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze trading performance."""
        try:
            # Calculate returns
            returns = portfolio_history["portfolio_value"].pct_change().dropna()

            # Calculate metrics
            metrics = {
                "total_return": (
                    portfolio_history["portfolio_value"].iloc[-1] / self.initial_capital
                )
                - 1,
                "sharpe_ratio": self._calculate_sharpe_ratio(returns),
                "max_drawdown": self._calculate_max_drawdown(
                    portfolio_history["portfolio_value"]
                ),
                "win_rate": self._calculate_win_rate(trades),
                "profit_factor": self._calculate_profit_factor(trades),
                "avg_trade_return": trades["pnl"].mean() if not trades.empty else 0,
                "volatility": returns.std() * np.sqrt(252),  # Annualized
                "trades_count": len(trades),
            }

            # Create performance metrics object
            perf_metrics = PerformanceMetrics(timestamp=datetime.now(), **metrics)
            self.metrics_history.append(perf_metrics)

            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({f"performance/{k}": v for k, v in metrics.items()})

            return metrics

        except Exception as e:
            self.logger.error(f"Error analyzing performance: {e}")
            raise

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if returns.empty or returns.std() == 0:
            return 0
        return np.sqrt(252) * (returns.mean() / returns.std())

    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = portfolio_values.expanding(min_periods=1).max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()

    def _calculate_win_rate(self, trades: pd.DataFrame) -> float:
        """Calculate win rate."""
        if trades.empty:
            return 0
        winning_trades = len(trades[trades["pnl"] > 0])
        return winning_trades / len(trades)

    def _calculate_profit_factor(self, trades: pd.DataFrame) -> float:
        """Calculate profit factor."""
        if trades.empty:
            return 0
        profits = trades[trades["pnl"] > 0]["pnl"].sum()
        losses = abs(trades[trades["pnl"] < 0]["pnl"].sum())
        return profits / losses if losses != 0 else float("inf")
