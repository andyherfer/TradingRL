from datetime import datetime
from TradingRL.src.analysis.market_analyzer import MarketRegime


class MockMarketAnalyzer:
    """Mock market analyzer for testing."""

    def __init__(self):
        self.current_state = {
            "BTC/USDT": {
                "regime": MarketRegime.TRENDING_UP,
                "volatility": 0.1,
                "trend_strength": 0.8,
                "support_levels": [45000.0, 44000.0],
                "resistance_levels": [52000.0, 53000.0],
                "liquidity": 1.0,
                "timestamp": datetime.now(),
                "metadata": {},
            }
        }

    async def start(self):
        """Mock start."""
        pass

    async def stop(self):
        """Mock stop."""
        pass

    async def analyze_market(self, data):
        """Mock market analysis."""
        return self.current_state
