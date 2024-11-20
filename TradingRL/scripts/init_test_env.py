import os
import shutil
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_test_environment():
    """Initialize test environment with required directories and files."""
    try:
        # Create necessary directories
        dirs = ["config", "logs", "data"]
        for dir_name in dirs:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_name}")

        # Create temporary directory for Binance config
        temp_dir = Path("config/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created temporary directory for Binance config")

        # Set environment variables
        os.environ["BINANCE_CONFIG"] = str(temp_dir / "binance.ini")
        os.environ["CONFIG_PATH"] = "config"
        os.environ["SYSTEM_ENV"] = "production"
        os.environ["SYSTEM_MODE"] = "live"

        logger.info("Test environment initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Error initializing test environment: {e}")
        return False


if __name__ == "__main__":
    init_test_environment()
