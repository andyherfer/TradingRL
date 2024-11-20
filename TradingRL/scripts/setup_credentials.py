import os
import yaml
from pathlib import Path
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_credentials(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    config_dir: str = "config",
) -> bool:
    """Set up API credentials and configuration files."""
    try:
        # Create config directory if it doesn't exist
        config_path = Path(config_dir)
        config_path.mkdir(parents=True, exist_ok=True)

        # Get credentials from environment or parameters
        api_key = api_key or os.environ.get("BINANCE_API_KEY")
        api_secret = api_secret or os.environ.get("BINANCE_API_SECRET")

        if not api_key or not api_secret:
            logger.error("API credentials not provided")
            return False

        # Create .env file with credentials
        env_path = config_path / ".env"
        with open(env_path, "w") as f:
            f.write(f"BINANCE_API_KEY={api_key}\n")
            f.write(f"BINANCE_API_SECRET={api_secret}\n")

        # Create production config
        production_config = {
            "exchange": {
                "name": "binance",
                "testnet": True,  # Start with testnet for safety
                "api_key": "${BINANCE_API_KEY}",
                "api_secret": "${BINANCE_API_SECRET}",
                "rate_limit": 1200,
                "timeout": 30000,
            },
            "risk": {
                "max_position_size": 0.05,  # 5% max position size
                "max_drawdown": 0.02,  # 2% max drawdown
                "stop_loss": 0.01,  # 1% stop loss
                "take_profit": 0.02,  # 2% take profit
                "max_leverage": 1.0,  # No leverage initially
                "daily_loss_limit": 50,  # $50 max daily loss
                "max_positions": 1,  # Start with single position
            },
            "monitoring": {
                "health_check_interval": 60,  # Check system health every minute
                "balance_check_interval": 300,  # Check balance every 5 minutes
                "log_level": "INFO",
                "alert_channels": ["log", "email"],
            },
            "trading": {
                "symbols": ["BTC/USDT"],  # Start with single pair
                "timeframes": ["5m", "15m"],
                "initial_capital": 1000.0,
                "min_trade_interval": 300,  # 5 minutes minimum between trades
                "max_orders_per_minute": 2,
            },
        }

        # Save production config
        prod_config_path = config_path / "production.yaml"
        with open(prod_config_path, "w") as f:
            yaml.safe_dump(production_config, f)

        logger.info("Credentials and configuration set up successfully")
        return True

    except Exception as e:
        logger.error(f"Error setting up credentials: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Set up trading credentials")
    parser.add_argument("--api-key", help="Binance API key")
    parser.add_argument("--api-secret", help="Binance API secret")
    parser.add_argument(
        "--config-dir", default="config", help="Configuration directory path"
    )

    args = parser.parse_args()
    success = setup_credentials(args.api_key, args.api_secret, args.config_dir)
    exit(0 if success else 1)
