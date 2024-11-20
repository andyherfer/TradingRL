from typing import Dict, Optional
import pandas as pd


class MockDataFetcher:
    """Mock data fetcher for testing."""

    def __init__(self, market_data: Dict[str, pd.DataFrame]):
        self.market_data = market_data
        self.test_market_data = market_data

    async def get_latest_data(self) -> Dict[str, pd.DataFrame]:
        """Return test market data."""
        return self.market_data

    async def get_historical_data(
        self, symbol: str, interval: str, limit: int
    ) -> pd.DataFrame:
        """Return historical test data."""
        if symbol in self.market_data:
            return self.market_data[symbol].tail(limit)
        return pd.DataFrame()

    async def start(self) -> None:
        """Mock start."""
        pass

    async def stop(self) -> None:
        """Mock stop."""
        pass
