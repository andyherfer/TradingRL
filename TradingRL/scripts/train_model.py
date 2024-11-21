import asyncio
import click
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path
from typing import Optional

from TradingRL.src.core.config_manager import ConfigManager
from TradingRL.src.data.data_fetcher import DataFetcher
from TradingRL.src.core.trader import Trader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def train_model(config_path: str, symbol: str, device: Optional[str] = None):
    """Train the RL model using historical data."""
    data_fetcher = None
    try:
        # Initialize components
        config_manager = ConfigManager(
            base_path=config_path, env="development", mode="train"
        )

        # Get configuration
        training_config = config_manager.config["trader"]["training"]

        # Initialize components and fetch data (async operations)
        data_fetcher = DataFetcher(
            api_key=config_manager.get_secret("BINANCE_API_KEY"),
            api_secret=config_manager.get_secret("BINANCE_API_SECRET"),
            symbols=[symbol],
            timeframes=config_manager.config["trading"]["timeframes"],
        )

        await data_fetcher.initialize()  # Async operation

        # Calculate date range for training data
        end_date = datetime.now()
        days = int(training_config["data_window"].replace("d", ""))
        start_date = end_date - timedelta(days=days)

        logger.info(f"Fetching training data from {start_date} to {end_date}")

        # Fetch historical data (async operation)
        data = await data_fetcher.get_historical_data(
            symbol=symbol,
            timeframe="1h",
            start_time=start_date,
            end_time=end_date,
        )

        # From here on, everything is synchronous
        logger.info(f"Fetched {len(data)} data points")

        if len(data) < 1000:
            raise ValueError("Insufficient data points for training")

        # Prepare training data
        train_size = int(len(data) * training_config["train_test_split"])
        train_data = data[:train_size]
        test_data = data[train_size:]

        logger.info(f"Training set size: {len(train_data)} points")
        logger.info(f"Test set size: {len(test_data)} points")

        # Train model
        logger.info("Training model...")
        total_timesteps = training_config["n_epochs"] * len(train_data)
        logger.info(f"Total training steps: {total_timesteps}")

        trader = Trader(
            model_dir=training_config["model_dir"],
            tensorboard_log=training_config["tensorboard_log"],
            device=device,
        )

        # Call synchronous train_model
        trader.train_model(
            train_data=train_data,
            eval_data=test_data,
            hyperparams=training_config,
            total_timesteps=total_timesteps,
        )

        logger.info("Training completed!")

        # Evaluate model
        logger.info("Evaluating model...")
        metrics = trader.evaluate_performance(test_data)

        logger.info("\nEvaluation Metrics:")
        logger.info(f"Total Return: {metrics['total_return']:.2%}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
        logger.info(f"Total Trades: {metrics['total_trades']}")
        logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    finally:
        if data_fetcher:
            await data_fetcher.close()  # Async cleanup


@click.command()
@click.option("--config", default="config", help="Path to configuration directory")
@click.option("--symbol", default="BTC/USDT", help="Trading pair to train on")
@click.option(
    "--device",
    default=None,
    help="Device to use (cuda/mps/cpu). Auto-detected if not specified",
)
def main(config: str, symbol: str, device: Optional[str]):
    """Train the trading model."""
    asyncio.run(train_model(config, symbol, device))


if __name__ == "__main__":
    main()
