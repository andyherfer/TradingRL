import pytest
from datetime import datetime, timedelta
from TradingRL.src.analysis.performance_metrics import PerformanceMetrics


def test_performance_calculation():
    """Test performance metrics calculation."""
    # Create test trade history
    trades = [
        {
            "pnl": 100,
            "roi": 0.05,
            "entry_time": datetime.now() - timedelta(hours=2),
            "exit_time": datetime.now() - timedelta(hours=1),
        },
        {
            "pnl": -50,
            "roi": -0.02,
            "entry_time": datetime.now() - timedelta(hours=4),
            "exit_time": datetime.now() - timedelta(hours=3),
        },
        {
            "pnl": 200,
            "roi": 0.08,
            "entry_time": datetime.now() - timedelta(hours=6),
            "exit_time": datetime.now() - timedelta(hours=5),
        },
    ]

    metrics = PerformanceMetrics.from_trade_history(trades)

    # Verify metrics
    assert metrics.total_trades == 3
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 1
    assert 0.66 < metrics.win_rate < 0.67  # Approximately 2/3
    assert metrics.avg_profit == 150  # (100 + 200) / 2
    assert metrics.avg_loss == 50
    assert metrics.profit_factor == 6.0  # (100 + 200) / 50
    assert metrics.sharpe_ratio > 0
    assert 0 <= metrics.max_drawdown <= 1
    assert metrics.avg_trade_duration > 0
    assert metrics.roi > 0


def test_empty_trade_history():
    """Test metrics calculation with empty trade history."""
    metrics = PerformanceMetrics.from_trade_history([])

    assert metrics.total_trades == 0
    assert metrics.winning_trades == 0
    assert metrics.losing_trades == 0
    assert metrics.win_rate == 0
    assert metrics.avg_profit == 0
    assert metrics.avg_loss == 0
    assert metrics.profit_factor == float("inf")
    assert metrics.sharpe_ratio == 0
    assert metrics.max_drawdown == 0
    assert metrics.avg_trade_duration == 0
    assert metrics.roi == 0


def test_all_winning_trades():
    """Test metrics calculation with all winning trades."""
    trades = [
        {
            "pnl": 100,
            "roi": 0.05,
            "entry_time": datetime.now() - timedelta(hours=2),
            "exit_time": datetime.now() - timedelta(hours=1),
        },
        {
            "pnl": 150,
            "roi": 0.07,
            "entry_time": datetime.now() - timedelta(hours=4),
            "exit_time": datetime.now() - timedelta(hours=3),
        },
    ]

    metrics = PerformanceMetrics.from_trade_history(trades)

    assert metrics.win_rate == 1.0
    assert metrics.profit_factor == float("inf")
    assert metrics.max_drawdown == 0
