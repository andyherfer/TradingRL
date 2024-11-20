from typing import Dict, Any
import psutil
import logging
from datetime import datetime
import asyncio
from pathlib import Path


class SystemMonitor:
    """Monitors system health and performance."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {}
        self.alerts = []
        self.is_running = False

    async def start(self):
        """Start system monitoring."""
        self.is_running = True
        asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop system monitoring."""
        self.is_running = False

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # System metrics
                self.metrics["cpu"] = psutil.cpu_percent()
                self.metrics["memory"] = psutil.virtual_memory().percent
                self.metrics["disk"] = psutil.disk_usage("/").percent

                # Check thresholds
                await self._check_alerts()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

    async def _check_alerts(self):
        """Check for alert conditions."""
        if self.metrics["cpu"] > 80:
            await self._raise_alert("High CPU usage")
        if self.metrics["memory"] > 80:
            await self._raise_alert("High memory usage")

    async def _raise_alert(self, message: str):
        """Raise system alert."""
        alert = {
            "timestamp": datetime.now(),
            "message": message,
            "metrics": self.metrics.copy(),
        }
        self.alerts.append(alert)
        self.logger.warning(f"System alert: {message}")
