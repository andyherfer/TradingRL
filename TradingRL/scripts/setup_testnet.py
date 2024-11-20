import asyncio
import logging
import os
from pathlib import Path
import yaml
import requests
from TradingRL.scripts.setup_credentials import setup_credentials

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default testnet credentials from Binance docs
DEFAULT_TESTNET_CREDENTIALS = {
    "api_key": "vmPUZE6mv9SD5VNHk4HlWFsOr6aKE2zvsw0MuIgwCIPy6utIco14y7Ju91duEh8A",
    "api_secret": "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j",
}


async def create_testnet_credentials():
    """Create and set up Binance testnet credentials."""
    try:
        # Use default testnet credentials
        api_key = DEFAULT_TESTNET_CREDENTIALS["api_key"]
        api_secret = DEFAULT_TESTNET_CREDENTIALS["api_secret"]

        # Set up credentials
        success = setup_credentials(
            api_key=api_key, api_secret=api_secret, config_dir="config"
        )

        if success:
            logger.info(f"Using default testnet credentials")
            logger.info(f"Testnet API Key: {api_key}")

            # Create testnet config
            config_path = Path("config/production.yaml")
            config = {
                "exchange": {
                    "name": "binance",
                    "testnet": True,  # Force testnet mode
                    "api_key": "${BINANCE_API_KEY}",
                    "api_secret": "${BINANCE_API_SECRET}",
                    "rate_limit": 1200,
                    "timeout": 30000,
                },
                "risk": {
                    "max_position_size": 0.05,
                    "max_drawdown": 0.02,
                    "stop_loss": 0.01,
                    "take_profit": 0.02,
                    "max_leverage": 1.0,
                    "daily_loss_limit": 50,
                    "max_positions": 1,
                },
                "monitoring": {
                    "health_check_interval": 60,
                    "balance_check_interval": 300,
                    "log_level": "INFO",
                    "alert_channels": ["log", "email"],
                },
                "trading": {
                    "symbols": ["BTC/USDT"],
                    "timeframes": ["5m", "15m"],
                    "initial_capital": 1000.0,
                    "min_trade_interval": 300,
                    "max_orders_per_minute": 2,
                },
            }

            # Create config directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Save config
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f)

            logger.info("Testnet configuration created successfully")
            return True
        else:
            logger.error("Failed to set up testnet credentials")
            return False

    except Exception as e:
        logger.error(f"Error setting up testnet credentials: {e}")
        return False


if __name__ == "__main__":
    try:
        result = asyncio.run(create_testnet_credentials())
        exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Setup failed with error: {e}")
        exit(1)
