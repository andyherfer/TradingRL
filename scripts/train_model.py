import asyncio
import click
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config_manager import ConfigManager, Environment, SystemMode
from src.data.data_fetcher import DataFetcher
from src.core.trader import Trader
from src.analysis.market_analyzer import MarketAnalyzer


async def train_model(config_path: str, symbol: str, device: Optional[str] = None):
    """Train the RL model using historical data."""
    try:
        # Initialize components
        config_manager = ConfigManager(
            base_path=config_path, env="development", mode="train"
        )

        # Get configuration
        training_config = config_manager.config["trader"]["training"]

        # Initialize components
        data_fetcher = DataFetcher(
            api_key=config_manager.get_secret("EXCHANGE_API_KEY"),
            api_secret=config_manager.get_secret("EXCHANGE_API_SECRET"),
            symbols=[symbol],
            timeframes=config_manager.config["trading"]["timeframes"],
        )

        await data_fetcher.initialize()

        # Calculate date range for training data
        end_date = datetime.now()
        days = int(training_config["data_window"].replace("d", ""))
        start_date = end_date - timedelta(days=days)

        print(f"Fetching training data from {start_date} to {end_date}")

        # Fetch historical data with progress updates
        data = await data_fetcher.get_historical_data(
            symbol=symbol,
            timeframe="1h",  # Base timeframe for training
            start_time=start_date,
            end_time=end_date,
        )

        print(f"Fetched {len(data)} data points")

        if len(data) < 1000:
            raise ValueError("Insufficient data points for training")

        # Prepare training data
        train_size = int(len(data) * training_config["train_test_split"])
        train_data = data[:train_size]
        test_data = data[train_size:]

        print(f"Training set size: {len(train_data)} points")
        print(f"Test set size: {len(test_data)} points")

        # Train model
        print("Training model...")
        total_timesteps = training_config["epochs"] * len(train_data)
        print(f"Total training steps: {total_timesteps}")

        trader = Trader(
            model_dir=config_manager.config["trader"]["model_dir"],
            tensorboard_log=config_manager.config["trader"]["tensorboard_log"],
            device=device,
        )

        trader.train_model(
            train_data=train_data,
            eval_data=test_data,
            hyperparams=training_config["model_params"],
            total_timesteps=total_timesteps,
        )

        print("Training completed!")

        # Evaluate model
        print("Evaluating model...")
        metrics = trader.evaluate_performance(test_data)

        print("\nEvaluation Metrics:")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")

    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        if data_fetcher:
            await data_fetcher.close()


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
