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
import traceback

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
        """Initialize trading system."""
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)

        self.logger = self._setup_logging()
        self.logger.info("Initializing trading system...")

        # Initialize configuration
        self.config_manager = ConfigManager(base_path=config_path, env=env, mode=mode)

        # Get full configuration
        try:
            self.config = {
                "database": self.config_manager.get("database"),
                "exchange": {
                    "api_key": self.config_manager.get_secret("EXCHANGE_API_KEY"),
                    "api_secret": self.config_manager.get_secret("EXCHANGE_API_SECRET"),
                    **self.config_manager.get("exchange"),
                },
                "monitor": self.config_manager.get("monitor"),
                "risk": self.config_manager.get("risk"),
                "system": self.config_manager.get("system"),
                "trader": self.config_manager.get("trader"),
                "trading": {
                    **self.config_manager.get("trading"),
                    "initial_capital": 10000.0,
                    "symbols": ["BTC/USDT"],
                },
            }
        except Exception as e:
            self.logger.error(f"Error loading configuration:\n{traceback.format_exc()}")
            raise

        # Initialize components
        self.components = {}
        self.is_running = False
        self.shutdown_requested = False
        self.shutdown_event = asyncio.Event()

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

    def _signal_handler(self, signum, frame):
        """Handle system signals."""
        self.logger.info(f"Received signal {signum}")
        self.shutdown_requested = True
        if hasattr(self, "shutdown_event"):
            asyncio.create_task(self.shutdown())

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

        except Exception as e:
            self.logger.error(f"Error starting system:\n{traceback.format_exc()}")
            await self.shutdown()
            raise

    async def run(self) -> None:
        """Run the trading system."""
        try:
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # Initialize components
            await self.initialize()

            # Start the system
            await self.start()

            # Wait for shutdown signal
            await self.shutdown_event.wait()

        except Exception as e:
            self.logger.error(f"Error running system:\n{traceback.format_exc()}")
            raise
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown the trading system."""
        if not self.is_running:
            return

        try:
            self.logger.info("Initiating system shutdown...")
            self.is_running = False

            # Stop components in reverse order
            if "monitor" in self.components:
                await self.components["monitor"].stop()
            if "strategy" in self.components:
                await self.components["strategy"].stop()
            if "event_manager" in self.components:
                await self.components["event_manager"].stop()
            if "data_fetcher" in self.components:
                await self.components["data_fetcher"].stop()
            if "database" in self.components:
                await self.components["database"].close()

            self.logger.info("System shutdown completed")
            self.shutdown_event.set()

        except Exception as e:
            self.logger.error(f"Error during shutdown:\n{traceback.format_exc()}")
            raise

    async def initialize(self) -> None:
        """Initialize all system components."""
        try:
            self.logger.info("Starting system initialization...")

            # Initialize core components
            self.components["event_manager"] = EventManager()
            self.components["database"] = Database(config=self.config["database"])

            # Initialize data components
            self.components["data_fetcher"] = DataFetcher(
                api_key=self.config["exchange"]["api_key"],
                api_secret=self.config["exchange"]["api_secret"],
                symbols=self.config["trading"]["symbols"],
                timeframes=self.config["trading"]["timeframes"],
            )

            # Initialize analysis components
            self.components["market_analyzer"] = MarketAnalyzer(
                event_manager=self.components["event_manager"]
            )

            self.components["performance_analyzer"] = PerformanceAnalyzer(
                initial_capital=self.config["trading"]["initial_capital"]
            )

            # Initialize trading components
            self.components["risk_manager"] = RiskManager(config=self.config["risk"])

            self.components["order_manager"] = OrderManager(
                api_key=self.config["exchange"]["api_key"],
                api_secret=self.config["exchange"]["api_secret"],
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
                initial_capital=self.config["trading"]["initial_capital"],
            )

            # Initialize trader and strategy
            self.components["trader"] = Trader(
                model_dir=self.config["trader"]["model_dir"],
                tensorboard_log=self.config["trader"]["tensorboard_log"],
            )

            self.components["strategy"] = RLStrategy(
                name="rl_strategy",
                trader=self.components["trader"],
                symbols=self.config["trading"]["symbols"],
                event_manager=self.components["event_manager"],
                risk_manager=self.components["risk_manager"],
                portfolio_manager=self.components["portfolio_manager"],
                market_analyzer=self.components["market_analyzer"],
            )

            # Initialize monitor
            self.components["monitor"] = SystemMonitor(
                config=MonitorConfig(
                    update_interval=self.config["monitor"]["update_interval"]
                )
            )

            self.logger.info("System initialization completed")

        except Exception as e:
            self.logger.error(f"Error during initialization:\n{traceback.format_exc()}")
            raise

    # ... (rest of the implementation remains the same)


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

        # Run the system using asyncio
        asyncio.run(system.run())

    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logging.error(f"System error:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
