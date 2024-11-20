import asyncio
import logging
import os
import tempfile
from pathlib import Path
from binance.client import AsyncClient
from TradingRL.src.core.config_manager import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def verify_exchange_setup():
    """Verify exchange connectivity and API functionality."""
    client = None
    temp_dir = None
    original_env = {}
    original_cwd = os.getcwd()

    try:
        # Store original environment variables and working directory
        original_env = {
            key: os.environ.get(key)
            for key in ["BINANCE_CONFIG", "BINANCE_TESTNET", "HOME", "CONFIG_PATH"]
        }

        # Create temporary directory structure
        temp_dir = tempfile.mkdtemp(prefix="trading_test_")
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir(parents=True)

        # Change to temp directory
        os.chdir(temp_dir)

        # Set environment variables
        os.environ["BINANCE_CONFIG"] = str(config_dir / "binance.ini")
        os.environ["BINANCE_TESTNET"] = "true"
        os.environ["CONFIG_PATH"] = str(config_dir)
        os.environ["HOME"] = temp_dir

        # Load configuration
        config = ConfigManager(base_path=str(config_dir), env="production", mode="live")
        api_key = config.get_secret("BINANCE_API_KEY")
        api_secret = config.get_secret("BINANCE_API_SECRET")

        if not api_key or not api_secret:
            logger.error("API credentials not found in configuration")
            return False

        # Initialize client with testnet first
        client = await AsyncClient.create(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True,  # Start with testnet for safety
            # tld="com",  # Use .com TLD
        )

        # Test basic API calls
        logger.info("Testing API connectivity...")

        # 1. Account information
        # account = await client.get_account()
        # balances = {
        #     asset["asset"]: float(asset["free"])
        #     for asset in account["balances"]
        #     if float(asset["free"]) > 0
        # }
        # logger.info(f"Account balances: {balances}")

        # 2. Test market data
        symbol = "BTCUSDT"
        ticker = await client.get_symbol_ticker(symbol=symbol)
        logger.info(f"Current {symbol} price: {ticker['price']}")

        # 3. Test order book
        depth = await client.get_order_book(symbol=symbol)
        logger.info(
            f"Order book depth: {len(depth['bids'])} bids, {len(depth['asks'])} asks"
        )

        # 4. Test small order (if allowed)
        try:
            test_order = await client.create_test_order(
                symbol=symbol,
                side="BUY",
                type="LIMIT",
                timeInForce="GTC",
                quantity=0.001,
                price=float(ticker["price"]) * 0.9,
            )
            logger.info("Test order successful")
        except Exception as e:
            logger.warning(f"Test order failed: {e}")

        # 5. Check trading permissions
        exchange_info = await client.get_exchange_info()
        symbol_info = next(
            (s for s in exchange_info["symbols"] if s["symbol"] == symbol), None
        )
        if symbol_info:
            logger.info(f"Trading permissions: {symbol_info['permissions']}")
            logger.info(f"Filters: {symbol_info['filters']}")

        logger.info("Exchange verification completed successfully")
        return True

    except Exception as e:
        logger.error(f"Exchange verification failed: {e}")
        return False
    finally:
        # Restore original environment variables
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        # Restore original working directory
        os.chdir(original_cwd)

        if client:
            await client.close_connection()

        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            import shutil

            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary directory: {e}")


if __name__ == "__main__":
    try:
        result = asyncio.run(verify_exchange_setup())
        exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Verification interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Verification failed with error: {e}")
        exit(1)
