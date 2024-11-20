import asyncio
import logging
import psutil
from pathlib import Path
import tempfile
import os
from TradingRL.src.monitoring.system_monitor import SystemMonitor
from TradingRL.src.monitoring.critical_monitor import CriticalMonitor
from TradingRL.src.core.safety_monitor import SafetyMonitor, SafetyLimits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def verify_monitoring():
    """Verify monitoring systems."""
    temp_dir = None
    original_env = {}
    original_cwd = os.getcwd()
    system_monitor = None

    try:
        # Store original environment variables
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

        # Initialize monitors
        system_monitor = SystemMonitor()
        critical_monitor = CriticalMonitor()
        safety_monitor = SafetyMonitor(
            limits=SafetyLimits(
                max_daily_loss=100.0,
                max_position_size=0.1,
                max_orders_per_minute=5,
                emergency_stop_loss=0.05,
                max_leverage=1.0,
            )
        )

        # Start monitoring
        await system_monitor.start()
        await asyncio.sleep(1)  # Wait for initial metrics

        # Test system metrics
        metrics = system_monitor.metrics
        logger.info(f"System metrics: {metrics}")

        # Verify CPU monitoring
        assert "cpu" in metrics, "CPU monitoring failed"
        assert isinstance(metrics["cpu"], (int, float)), "Invalid CPU metric type"
        assert 0 <= metrics["cpu"] <= 100, "CPU usage out of range"

        # Verify memory monitoring
        assert "memory" in metrics, "Memory monitoring failed"
        assert isinstance(metrics["memory"], (int, float)), "Invalid memory metric type"
        assert 0 <= metrics["memory"] <= 100, "Memory usage out of range"

        # Verify disk monitoring
        assert "disk" in metrics, "Disk monitoring failed"
        assert isinstance(metrics["disk"], (int, float)), "Invalid disk metric type"
        assert 0 <= metrics["disk"] <= 100, "Disk usage out of range"

        # Test critical checks
        health_status = await critical_monitor.check_system_health()
        logger.info(f"System health status: {health_status}")
        assert health_status, "System health check failed"

        # Test safety checks
        test_order = {"symbol": "BTC/USDT", "size": 0.1, "type": "market"}
        is_safe = await safety_monitor.check_trade(test_order)
        logger.info(f"Trade safety check: {is_safe}")
        assert is_safe, "Trade safety check failed"

        logger.info("Monitoring verification completed successfully")
        return True

    except Exception as e:
        logger.error(f"Monitoring verification failed: {e}")
        return False

    finally:
        # Stop system monitor
        if system_monitor:
            await system_monitor.stop()

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
        result = asyncio.run(verify_monitoring())
        exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Verification interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Verification failed with error: {e}")
        exit(1)
