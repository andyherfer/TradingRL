from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
import psutil
import json
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import curses
from enum import Enum
import yaml


class MonitorView(Enum):
    """Monitor view types."""

    OVERVIEW = "overview"
    PERFORMANCE = "performance"
    POSITIONS = "positions"
    ORDERS = "orders"
    SYSTEM = "system"
    LOGS = "logs"


@dataclass
class MonitorConfig:
    """Monitor configuration."""

    update_interval: float = 1.0
    max_log_lines: int = 1000
    history_size: int = 3600
    save_interval: int = 300
    metrics_dir: str = "metrics"


class SystemMonitor:
    """
    Local system monitor for trading system.
    Provides real-time monitoring and basic control capabilities.
    """

    def __init__(self, config: Optional[MonitorConfig] = None):
        """
        Initialize SystemMonitor.

        Args:
            config: Monitor configuration
        """
        self.config = config or MonitorConfig()

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Initialize metrics storage
        self.metrics_dir = Path(self.config.metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics tracking
        self.metrics = {
            "system": deque(maxlen=self.config.history_size),
            "performance": deque(maxlen=self.config.history_size),
            "positions": deque(maxlen=self.config.history_size),
            "orders": deque(maxlen=self.config.history_size),
        }

        # Initialize log buffer
        self.logs = deque(maxlen=self.config.max_log_lines)

        # State tracking
        self.current_view = MonitorView.OVERVIEW
        self.running = False
        self.last_save = datetime.now()

    async def start(self) -> None:
        """Start system monitoring."""
        try:
            self.running = True

            # Start monitoring tasks
            monitor_task = asyncio.create_task(self._monitor_loop())
            display_task = asyncio.create_task(self._display_loop())
            save_task = asyncio.create_task(self._save_loop())

            # Wait for tasks
            await asyncio.gather(monitor_task, display_task, save_task)

        except Exception as e:
            self.logger.error(f"Error starting monitor: {e}")
            raise
        finally:
            self.running = False

    async def stop(self) -> None:
        """Stop system monitoring."""
        self.running = False

        # Save final metrics
        self._save_metrics()

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        try:
            while self.running:
                # Update system metrics
                self.metrics["system"].append(
                    {
                        "timestamp": datetime.now(),
                        "cpu_usage": psutil.cpu_percent(interval=1),
                        "memory_usage": psutil.Process().memory_info().rss
                        / 1024
                        / 1024,
                        "disk_usage": psutil.disk_usage("/").percent,
                        "network_io": psutil.net_io_counters()._asdict(),
                    }
                )

                # Update performance metrics
                # This would be connected to the PerformanceAnalyzer

                # Update position metrics
                # This would be connected to the PortfolioManager

                # Update order metrics
                # This would be connected to the OrderManager

                await asyncio.sleep(self.config.update_interval)

        except Exception as e:
            self.logger.error(f"Error in monitor loop: {e}")
            self.running = False

    async def _display_loop(self) -> None:
        """Display update loop."""
        try:
            # Initialize curses
            stdscr = curses.initscr()
            curses.start_color()
            curses.use_default_colors()
            curses.curs_set(0)

            # Initialize color pairs
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_RED, -1)
            curses.init_pair(3, curses.COLOR_YELLOW, -1)

            while self.running:
                # Clear screen
                stdscr.clear()

                # Display current view
                if self.current_view == MonitorView.OVERVIEW:
                    self._display_overview(stdscr)
                elif self.current_view == MonitorView.PERFORMANCE:
                    self._display_performance(stdscr)
                elif self.current_view == MonitorView.POSITIONS:
                    self._display_positions(stdscr)
                elif self.current_view == MonitorView.ORDERS:
                    self._display_orders(stdscr)
                elif self.current_view == MonitorView.SYSTEM:
                    self._display_system(stdscr)
                elif self.current_view == MonitorView.LOGS:
                    self._display_logs(stdscr)

                # Refresh screen
                stdscr.refresh()

                await asyncio.sleep(self.config.update_interval)
        except Exception as e:
            self.logger.error(f"Error in display loop: {e}")
        finally:
            curses.endwin()

    def _display_overview(self, stdscr) -> None:
        """Display system overview."""
        try:
            max_y, max_x = stdscr.getmaxyx()
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Display header
            stdscr.addstr(0, 0, f"Trading System Monitor - {current_time}")
            stdscr.addstr(1, 0, "=" * max_x)

            # Display system status
            if self.metrics["system"]:
                latest = self.metrics["system"][-1]
                stdscr.addstr(3, 0, "System Status:")
                stdscr.addstr(4, 2, f"CPU Usage: {latest['cpu_usage']}%")
                stdscr.addstr(5, 2, f"Memory Usage: {latest['memory_usage']:.2f} MB")
                stdscr.addstr(6, 2, f"Disk Usage: {latest['disk_usage']}%")

            # Display performance summary
            if self.metrics["performance"]:
                latest = self.metrics["performance"][-1]
                stdscr.addstr(8, 0, "Performance Summary:")
                stdscr.addstr(9, 2, f"Daily PnL: {latest.get('daily_pnl', 0):.2f} USDT")
                stdscr.addstr(
                    10, 2, f"Open Positions: {latest.get('open_positions', 0)}"
                )
                stdscr.addstr(11, 2, f"Today's Trades: {latest.get('daily_trades', 0)}")

            # Display active orders
            if self.metrics["orders"]:
                latest = self.metrics["orders"][-1]
                stdscr.addstr(13, 0, "Active Orders:")
                for i, order in enumerate(latest.get("active_orders", [])[:3]):
                    stdscr.addstr(
                        14 + i,
                        2,
                        f"{order['symbol']} - {order['side']} - {order['quantity']}",
                    )

            # Display navigation help
            stdscr.addstr(max_y - 2, 0, "=" * max_x)
            stdscr.addstr(
                max_y - 1,
                0,
                "Press: (O)verview (P)erformance (T)rades (S)ystem (L)ogs (Q)uit",
            )

            stdscr.refresh()

        except Exception as e:
            self.logger.error(f"Error displaying overview: {e}")

    def _display_performance(self, stdscr) -> None:
        """Display detailed performance metrics."""
        try:
            max_y, max_x = stdscr.getmaxyx()

            # Display header
            stdscr.addstr(0, 0, "Performance Metrics")
            stdscr.addstr(1, 0, "=" * max_x)

            if self.metrics["performance"]:
                latest = self.metrics["performance"][-1]

                # Display daily metrics
                stdscr.addstr(3, 0, "Daily Statistics:")
                stdscr.addstr(4, 2, f"Total PnL: {latest.get('daily_pnl', 0):.2f} USDT")
                stdscr.addstr(5, 2, f"Win Rate: {latest.get('win_rate', 0):.2%}")
                stdscr.addstr(6, 2, f"Sharpe Ratio: {latest.get('sharpe', 0):.2f}")
                stdscr.addstr(
                    7, 2, f"Max Drawdown: {latest.get('max_drawdown', 0):.2%}"
                )

                # Display monthly metrics
                stdscr.addstr(9, 0, "Monthly Statistics:")
                stdscr.addstr(
                    10, 2, f"Monthly PnL: {latest.get('monthly_pnl', 0):.2f} USDT"
                )
                stdscr.addstr(
                    11, 2, f"Best Day: {latest.get('best_day_pnl', 0):.2f} USDT"
                )
                stdscr.addstr(
                    12, 2, f"Worst Day: {latest.get('worst_day_pnl', 0):.2f} USDT"
                )

                # Display risk metrics
                stdscr.addstr(14, 0, "Risk Metrics:")
                stdscr.addstr(15, 2, f"Value at Risk: {latest.get('var', 0):.2f} USDT")
                stdscr.addstr(
                    16, 2, f"Current Exposure: {latest.get('exposure', 0):.2%}"
                )
                stdscr.addstr(
                    17,
                    2,
                    f"Average Trade Size: {latest.get('avg_trade_size', 0):.4f} BTC",
                )

            stdscr.refresh()

        except Exception as e:
            self.logger.error(f"Error displaying performance: {e}")

    def _display_positions(self, stdscr) -> None:
        """Display current positions and related metrics."""
        try:
            max_y, max_x = stdscr.getmaxyx()

            # Display header
            stdscr.addstr(0, 0, "Current Positions")
            stdscr.addstr(1, 0, "=" * max_x)

            if self.metrics["positions"]:
                latest = self.metrics["positions"][-1]
                positions = latest.get("positions", [])

                # Display column headers
                stdscr.addstr(
                    3, 0, "Symbol    Size      Entry     Current   PnL      ROI"
                )
                stdscr.addstr(4, 0, "-" * max_x)

                # Display positions
                for i, pos in enumerate(positions):
                    if i + 5 >= max_y - 3:  # Leave space for footer
                        break

                    # Format position data
                    symbol = f"{pos['symbol']:<9}"
                    size = f"{pos['size']:.4f}".rjust(9)
                    entry = f"{pos['entry_price']:.2f}".rjust(9)
                    current = f"{pos['current_price']:.2f}".rjust(9)
                    pnl = f"{pos['unrealized_pnl']:.2f}".rjust(8)
                    roi = f"{pos['roi']:.2%}".rjust(8)

                    # Color code PnL
                    color = (
                        curses.color_pair(1)
                        if pos["unrealized_pnl"] > 0
                        else curses.color_pair(2)
                    )

                    # Display position
                    stdscr.addstr(i + 5, 0, symbol)
                    stdscr.addstr(i + 5, 9, size)
                    stdscr.addstr(i + 5, 18, entry)
                    stdscr.addstr(i + 5, 27, current)
                    stdscr.addstr(i + 5, 36, pnl, color)
                    stdscr.addstr(i + 5, 44, roi, color)

            # Display totals
            if self.metrics["positions"]:
                total_pnl = sum(p["unrealized_pnl"] for p in positions)
                total_exposure = sum(p["size"] * p["current_price"] for p in positions)

                stdscr.addstr(max_y - 4, 0, "=" * max_x)
                stdscr.addstr(max_y - 3, 0, f"Total Positions: {len(positions)}")
                stdscr.addstr(
                    max_y - 3, 25, f"Total Exposure: {total_exposure:.2f} USDT"
                )
                stdscr.addstr(max_y - 3, 50, f"Total PnL: {total_pnl:.2f} USDT")

            stdscr.refresh()

        except Exception as e:
            self.logger.error(f"Error displaying positions: {e}")

    def _display_orders(self, stdscr) -> None:
        """Display active orders and order history."""
        try:
            max_y, max_x = stdscr.getmaxyx()

            # Display header
            stdscr.addstr(0, 0, "Order Management")
            stdscr.addstr(1, 0, "=" * max_x)

            if self.metrics["orders"]:
                latest = self.metrics["orders"][-1]

                # Display active orders
                active_orders = latest.get("active_orders", [])
                stdscr.addstr(3, 0, f"Active Orders ({len(active_orders)})")
                stdscr.addstr(
                    4, 0, "Symbol    Type     Side     Quantity  Price     Status"
                )
                stdscr.addstr(5, 0, "-" * max_x)

                for i, order in enumerate(active_orders):
                    if i + 6 >= max_y // 2:  # Use upper half for active orders
                        break

                    # Format order data
                    symbol = f"{order['symbol']:<9}"
                    order_type = f"{order['type']:<8}"
                    side = f"{order['side']:<8}"
                    quantity = f"{order['quantity']:.4f}".rjust(9)
                    price = f"{order.get('price', 'MARKET')!s}".rjust(9)
                    status = f"{order['status']:<8}"

                    # Display order
                    stdscr.addstr(i + 6, 0, symbol)
                    stdscr.addstr(i + 6, 9, order_type)
                    stdscr.addstr(i + 6, 17, side)
                    stdscr.addstr(i + 6, 25, quantity)
                    stdscr.addstr(i + 6, 34, price)
                    stdscr.addstr(i + 6, 43, status)

                # Display recent trades
                recent_trades = latest.get("recent_trades", [])
                start_row = max_y // 2
                stdscr.addstr(start_row, 0, f"Recent Trades ({len(recent_trades)})")
                stdscr.addstr(
                    start_row + 1,
                    0,
                    "Time     Symbol    Side     Quantity  Price     PnL",
                )
                stdscr.addstr(start_row + 2, 0, "-" * max_x)

                for i, trade in enumerate(recent_trades):
                    if i + start_row + 3 >= max_y - 3:
                        break

                    # Format trade data
                    time = trade["time"].strftime("%H:%M:%S")
                    symbol = f"{trade['symbol']:<9}"
                    side = f"{trade['side']:<8}"
                    quantity = f"{trade['quantity']:.4f}".rjust(9)
                    price = f"{trade['price']:.2f}".rjust(9)
                    pnl = f"{trade.get('pnl', 0):.2f}".rjust(8)

                    # Color code PnL
                    color = (
                        curses.color_pair(1)
                        if trade.get("pnl", 0) > 0
                        else curses.color_pair(2)
                    )

                    # Display trade
                    row = i + start_row + 3
                    stdscr.addstr(row, 0, time)
                    stdscr.addstr(row, 9, symbol)
                    stdscr.addstr(row, 18, side)
                    stdscr.addstr(row, 26, quantity)
                    stdscr.addstr(row, 35, price)
                    stdscr.addstr(row, 44, pnl, color)

            stdscr.refresh()

        except Exception as e:
            self.logger.error(f"Error displaying orders: {e}")

    def _save_metrics(self) -> None:
        """Save current metrics to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save each metric type
            for metric_type, data in self.metrics.items():
                if data:
                    filename = f"{metric_type}_{timestamp}.json"
                    filepath = self.metrics_dir / filename

                    with open(filepath, "w") as f:
                        json.dump(list(data), f)

            self.last_save = datetime.now()

        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")

    def update_metrics(self, metric_type: str, data: Dict[str, Any]) -> None:
        """
        Update metrics with new data.

        Args:
            metric_type: Type of metric to update
            data: New metric data
        """
        try:
            if metric_type in self.metrics:
                self.metrics[metric_type].append({"timestamp": datetime.now(), **data})

        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")

    def add_log(self, message: str, level: str = "INFO") -> None:
        """
        Add log message to buffer.

        Args:
            message: Log message
            level: Log level
        """
        try:
            self.logs.append(
                {"timestamp": datetime.now(), "level": level, "message": message}
            )

        except Exception as e:
            self.logger.error(f"Error adding log: {e}")

    async def handle_input(self, key: str) -> None:
        """
        Handle user input.

        Args:
            key: Input key
        """
        try:
            if key.lower() == "o":
                self.current_view = MonitorView.OVERVIEW
            elif key.lower() == "p":
                self.current_view = MonitorView.PERFORMANCE
            elif key.lower() == "t":
                self.current_view = MonitorView.POSITIONS
            elif key.lower() == "s":
                self.current_view = MonitorView.SYSTEM
            elif key.lower() == "l":
                self.current_view = MonitorView.LOGS
            elif key.lower() == "q":
                self.running = False
        except Exception as e:
            self.logger.error(f"Error handling input: {e}")
