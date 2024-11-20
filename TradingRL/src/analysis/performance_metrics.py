from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Trading performance metrics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = float("inf")
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_trade_duration: float = 0.0
    roi: float = 0.0

    @classmethod
    def from_trade_history(cls, trades: List[Dict[str, Any]]) -> "PerformanceMetrics":
        """Calculate metrics from trade history."""
        if not trades:
            return cls()

        # Basic counts
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t["pnl"] > 0])
        losing_trades = len([t for t in trades if t["pnl"] < 0])

        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Average profits and losses
        profits = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in trades if t["pnl"] < 0]
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = abs(np.mean(losses)) if losses else 0

        # Profit factor (total profits / total losses)
        total_profits = sum(profits) if profits else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = (
            float("inf") if total_losses == 0 else total_profits / total_losses
        )

        # Returns for Sharpe ratio
        returns = pd.Series([t["roi"] for t in trades])
        sharpe_ratio = (
            np.sqrt(252) * (returns.mean() / returns.std()) if len(returns) > 1 else 0
        )

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())

        # Average trade duration
        durations = [
            (t["exit_time"] - t["entry_time"]).total_seconds() / 3600 for t in trades
        ]
        avg_duration = np.mean(durations) if durations else 0

        # Overall ROI
        roi = (1 + returns).prod() - 1

        return cls(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_trade_duration=avg_duration,
            roi=roi,
        )
