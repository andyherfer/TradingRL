#!/usr/bin/env python3

import asyncio
import click
import logging
import sys
from pathlib import Path
from typing import Optional
import yaml
import signal
from datetime import datetime

# Import system components
from TradingRL.src.core.config_manager import ConfigManager, Environment, SystemMode
from TradingRL.src.analysis.event_manager import EventManager
from TradingRL.src.core.trader import Trader
from TradingRL.src.core.risk_manager import RiskManager
from TradingRL.src.core.order_manager import OrderManager
from TradingRL.src.core.order_executor import OrderExecutor
from TradingRL.src.core.portfolio_manager import PortfolioManager
from TradingRL.src.data.data_fetcher import DataFetcher
from TradingRL.src.data.database import Database
from TradingRL.src.strategy.rl_strategy import RLStrategy
from TradingRL.src.analysis.market_analyzer import MarketAnalyzer
from TradingRL.src.analysis.performance_analyzer import PerformanceAnalyzer
from TradingRL.src.monitoring.system_monitor import SystemMonitor, MonitorConfig


class TradingSystem:
    """Main trading system class that coordinates all components."""

    def __init__(self, config_path: str, env: str = "development", mode: str = "paper"):
        """
        Initialize trading system.

        Args:
            config_path: Path to configuration directory
            env: Environment type
            mode: System operation mode
        """
        self.logger = self._setup_logging()
        self.logger.info("Initializing trading system...")

        # Initialize configuration
        self.config_manager = ConfigManager(base_path=config_path, env=env, mode=mode)

        # Initialize components
        self.components = {}
        self.is_running = False
        self.shutdown_requested = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self) -> logging.Logger:
        """Setup system logging."""
        logger = logging.getLogger("trading_system")
        logger.setLevel(logging.INFO)

        # Create handlers
        console_handler = logging.StreamHandler(sys.stdout)
        file_handler = logging.FileHandler(
            f"logs/trading_system_{datetime.now().strftime('%Y%m%d')}.log"
        )

        # Create formatters and add it to handlers
        log_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(log_format)
        file_handler.setFormatter(log_format)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    async def initialize(self) -> None:
        """Initialize all system components."""
        try:
            self.logger.info("Starting system initialization...")

            # Initialize core components
            self.components["event_manager"] = EventManager()
            self.components["database"] = Database(
                config=self.config_manager.get("database")
            )

            # Initialize data components
            self.components["data_fetcher"] = DataFetcher(
                api_key=self.config_manager.get_secret("EXCHANGE_API_KEY"),
                api_secret=self.config_manager.get_secret("EXCHANGE_API_SECRET"),
                symbols=self.config_manager.get("trading.symbols"),
                timeframes=self.config_manager.get("trading.timeframes"),
            )

            # Initialize analysis components
            self.components["market_analyzer"] = MarketAnalyzer(
                event_manager=self.components["event_manager"]
            )

            self.components["performance_analyzer"] = PerformanceAnalyzer(
                initial_capital=self.config_manager.get("trading.initial_capital")
            )

            # Initialize trading components
            self.components["risk_manager"] = RiskManager(
                event_manager=self.components["event_manager"],
                config=self.config_manager.get("risk"),
            )

            self.components["order_manager"] = OrderManager(
                api_key=self.config_manager.get_secret("EXCHANGE_API_KEY"),
                api_secret=self.config_manager.get_secret("EXCHANGE_API_SECRET"),
                event_manager=self.components["event_manager"],
                risk_manager=self.components["risk_manager"],
            )

            self.components["order_executor"] = OrderExecutor(
                client=self.components["data_fetcher"].client,
                event_manager=self.components["event_manager"],
                order_manager=self.components["order_manager"],
            )

            self.components["portfolio_manager"] = PortfolioManager(
                event_manager=self.components["event_manager"],
                risk_manager=self.components["risk_manager"],
                order_manager=self.components["order_manager"],
                initial_capital=self.config_manager.get("trading.initial_capital"),
            )

            # Initialize trader and strategy
            self.components["trader"] = Trader(
                model_dir=self.config_manager.get("trader.model_dir"),
                tensorboard_log=self.config_manager.get("trader.tensorboard_log"),
            )

            self.components["strategy"] = RLStrategy(
                trader=self.components["trader"],
                symbols=self.config_manager.get("trading.symbols"),
                event_manager=self.components["event_manager"],
                risk_manager=self.components["risk_manager"],
                portfolio_manager=self.components["portfolio_manager"],
                market_analyzer=self.components["market_analyzer"],
            )

            # Initialize monitor
            self.components["monitor"] = SystemMonitor(
                config=MonitorConfig(
                    update_interval=self.config_manager.get(
                        "monitor.update_interval", 1.0
                    )
                )
            )

            self.logger.info("System initialization completed")

        except Exception as e:
            self.logger.error(f"Error during initialization: {e}")
            raise

    async def start(self) -> None:
        """Start the trading system."""
        try:
            self.logger.info("Starting trading system...")
            self.is_running = True

            # Start database
            await self.components["database"].initialize()

            # Start data fetching
            await self.components["data_fetcher"].start()

            # Start event manager
            await self.components["event_manager"].start()

            # Start strategy
            await self.components["strategy"].start()

            # Start monitor
            monitor_task = asyncio.create_task(self.components["monitor"].start())

            # Main system loop
            while not self.shutdown_requested:
                await asyncio.sleep(1)

            await self.shutdown()

        except Exception as e:
            self.logger.error(f"Error starting system: {e}")
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Shutdown the trading system."""
        try:
            self.logger.info("Initiating system shutdown...")
            self.is_running = False

            # Close all positions if in live/paper mode
            if self.config_manager.mode != SystemMode.BACKTEST:
                await self.components["portfolio_manager"].close_all_positions()

            # Stop components in reverse order
            await self.components["monitor"].stop()
            await self.components["strategy"].stop()
            await self.components["event_manager"].stop()
            await self.components["data_fetcher"].stop()
            await self.components["database"].close()

            self.logger.info("System shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise

    def _signal_handler(self, signum, frame):
        """Handle system signals."""
        self.logger.info(f"Received signal {signum}")
        self.shutdown_requested = True


@click.command()
@click.option("--config", default="config", help="Path to configuration directory")
@click.option(
    "--env", default="development", help="Environment (development/production)"
)
@click.option("--mode", default="paper", help="Operation mode (live/paper/backtest)")
def main(config: str, env: str, mode: str):
    """Main entry point for the trading system."""
    try:
        # Initialize and start the system
        system = TradingSystem(config, env, mode)

        # Run the system
        asyncio.run(system.run())

    except Exception as e:
        logging.error(f"System error: {e}")
        sys.exit(1)


async def run_backtest(
    config_path: str, start_date: datetime, end_date: datetime
) -> None:
    """Run system in backtest mode."""
    try:
        # Initialize system
        system = TradingSystem(
            config_path=config_path, env="development", mode="backtest"
        )

        # Initialize components
        await system.initialize()

        # Load historical data
        data = await system.components["data_fetcher"].get_historical_data(
            start_date=start_date, end_date=end_date
        )

        # Run backtest
        results = await system.components["strategy"].backtest(data)

        # Generate and save results
        await system.components["performance_analyzer"].analyze_backtest(results)

    except Exception as e:
        logging.error(f"Backtest error: {e}")
        raise


if __name__ == "__main__":
    main()
