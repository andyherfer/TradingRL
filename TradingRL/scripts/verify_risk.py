import asyncio
import logging
import os
import tempfile
from pathlib import Path
from TradingRL.src.core.risk_manager import RiskManager
from TradingRL.src.core.config_manager import ConfigManager
from TradingRL.src.core.portfolio_manager import PortfolioManager
from TradingRL.src.analysis.event_manager import EventManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def verify_risk_management():
    """Verify risk management functionality."""
    temp_dir = None
    original_env = {}
    original_cwd = os.getcwd()

    try:
        # Store original environment variables and working directory
        original_env = {
            key: os.environ.get(key)
            for key in ["CONFIG_PATH", "SYSTEM_ENV", "SYSTEM_MODE"]
        }

        # Create temporary directory structure
        temp_dir = tempfile.mkdtemp(prefix="trading_test_")
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir(parents=True)

        # Change to temp directory
        os.chdir(temp_dir)

        # Set environment variables
        os.environ["CONFIG_PATH"] = str(config_dir)
        os.environ["SYSTEM_ENV"] = "production"
        os.environ["SYSTEM_MODE"] = "live"

        # Initialize components
        config = ConfigManager(base_path=str(config_dir), env="production", mode="live")
        event_manager = EventManager()
        risk_manager = RiskManager(config=config.get("risk"))

        # Test position sizing
        logger.info("Testing position sizing...")
        test_amount = 1000.0  # $1000 test amount
        size = await risk_manager.calculate_position_size(
            amount=test_amount, risk_factor=0.5
        )
        logger.info(f"Test position size: ${size:.2f}")
        assert size <= config.get(
            "risk.max_position_size"
        ), "Position size exceeds limit"

        # Test drawdown calculation
        logger.info("Testing drawdown limits...")
        test_drawdown = 0.05  # 5% drawdown
        within_limits = await risk_manager.check_risk_limits(
            position_size=size, current_drawdown=test_drawdown
        )
        assert within_limits, "Risk limits check failed"

        # Test emergency shutdown
        logger.info("Testing emergency shutdown...")
        await risk_manager.update_risk_status(emergency=True)
        assert risk_manager.metrics["emergency_status"], "Emergency status not set"

        logger.info("Risk management verification completed successfully")
        return True

    except Exception as e:
        logger.error(f"Risk management verification failed: {e}")
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

        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            import shutil

            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary directory: {e}")


if __name__ == "__main__":
    try:
        result = asyncio.run(verify_risk_management())
        exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Verification interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Verification failed with error: {e}")
        exit(1)
