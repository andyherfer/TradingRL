import asyncio
import logging
from TradingRL.scripts.verify_exchange import verify_exchange_setup
from TradingRL.scripts.verify_risk import verify_risk_management
from TradingRL.scripts.verify_monitoring import verify_monitoring

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_all_verifications():
    """Run all system verifications."""
    try:
        logger.info("\n1. Verifying exchange setup...")
        if not await verify_exchange_setup():
            logger.error("Exchange verification failed")
            return False

        logger.info("\n2. Verifying risk management...")
        if not await verify_risk_management():
            logger.error("Risk management verification failed")
            return False

        logger.info("\n3. Verifying monitoring systems...")
        if not await verify_monitoring():
            logger.error("Monitoring verification failed")
            return False

        logger.info("\nAll verifications completed successfully")
        return True

    except Exception as e:
        logger.error(f"Verification process failed: {e}")
        return False


if __name__ == "__main__":
    try:
        result = asyncio.run(run_all_verifications())
        exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Verification process interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Verification process failed with error: {e}")
        exit(1)
