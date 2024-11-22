from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime, timedelta
import logging
from TradingRL.src.data.data_fetcher import DataFetcher, DataConfig
from TradingRL.src.data.database import Database, DatabaseConfig
from TradingRL.src.analysis.event_manager import EventManager, Event, EventType
import asyncio
import aiohttp


class DataFetcherAdapter:
    """
    Adapter for DataFetcher that provides a unified interface for data access
    and handles caching and database storage.
    """

    def __init__(
        self,
        data_fetcher: DataFetcher,
        database: Optional[Database] = None,
        cache_data: bool = True,
    ):
        """Initialize DataFetcherAdapter."""
        self.data_fetcher = data_fetcher
        self.database = database or Database(DatabaseConfig())
        self.cache_data = cache_data
        self.event_manager = None
        self._test_market_data = None
        self._running = False
        self._session = None

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.data_cache = {}

    def _format_symbol(self, symbol: str) -> str:
        """Convert symbol to Binance format (e.g., 'BTC/USDT' -> 'BTCUSDT')."""
        return symbol.replace("/", "")

    def _unformat_symbol(self, symbol: str) -> str:
        """Convert symbol from Binance format (e.g., 'BTCUSDT' -> 'BTC/USDT')."""
        if "/" not in symbol:
            # Try to find the split point between base and quote currencies
            for i in range(len(symbol) - 4, 0, -1):  # Start from end to handle USDT
                if symbol[i:] in ["USDT", "BTC", "ETH", "BNB"]:
                    return f"{symbol[:i]}/{symbol[i:]}"
        return symbol

    async def start(self) -> None:
        """Start the adapter."""
        if self._running:
            return

        try:
            # Create new session if needed
            if not self._session:
                self._session = aiohttp.ClientSession()

            # Initialize components
            await self.data_fetcher.initialize()
            if self.database:
                await self.database.initialize()

            self._running = True
            self.logger.info("DataFetcherAdapter initialized successfully")

        except Exception as e:
            self.logger.error(f"Error starting adapter: {e}")
            await self._cleanup()
            raise

    async def stop(self) -> None:
        """Stop the adapter."""
        if not self._running:
            return

        self._running = False
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Close session
            if self._session:
                await self._session.close()
                self._session = None

            # Close database
            if self.database:
                await self.database.close()

            # Close data fetcher
            await self.data_fetcher.close()

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def _fetch_latest(self) -> None:
        """Fetch and publish latest market data."""
        if not self._running:
            return

        try:
            if self._test_market_data and self.event_manager:
                # Use test data
                for symbol, data in self._test_market_data.items():
                    await self.event_manager.publish(
                        Event(
                            type=EventType.MARKET_DATA,
                            data={
                                "symbol": symbol,
                                "data": self._dataframe_to_dict(data),
                                "timestamp": datetime.now(),
                            },
                        )
                    )
                return

            # Use live data
            for symbol in self.data_fetcher.symbols:
                if not self._running:
                    break

                data_dict = await self.get_latest_data(symbol, "1m")
                if data_dict and self.event_manager:
                    for sym, data in data_dict.items():
                        await self.event_manager.publish(
                            Event(
                                type=EventType.MARKET_DATA,
                                data={
                                    "symbol": sym,
                                    "data": data,
                                    "timestamp": datetime.now(),
                                },
                            )
                        )

        except Exception as e:
            self.logger.error(f"Error fetching latest data: {e}")

    @property
    def test_market_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        return self._test_market_data

    @test_market_data.setter
    def test_market_data(self, value: Optional[Dict[str, pd.DataFrame]]) -> None:
        self._test_market_data = value

    async def get_latest_data(
        self, symbol: Optional[str] = None, timeframe: str = "1m"
    ) -> Optional[Dict[str, Dict]]:
        """Get latest market data."""
        try:
            # Handle test mode
            if self._test_market_data:
                if not symbol:
                    return {
                        k: self._dataframe_to_dict(v)
                        for k, v in self._test_market_data.items()
                    }
                if symbol in self._test_market_data:
                    return {
                        symbol: self._dataframe_to_dict(self._test_market_data[symbol])
                    }
                return None

            # Handle live mode
            if not symbol and self.data_fetcher.symbols:
                symbol = self.data_fetcher.symbols[0]
            if not symbol:
                return None

            data = await self.get_historical_data(
                symbol,
                timeframe,
                datetime.now() - timedelta(minutes=100),
                datetime.now(),
                include_indicators=True,
                use_cache=False,
            )

            if data is not None and not data.empty:
                return {self._unformat_symbol(symbol): self._dataframe_to_dict(data)}
            return None

        except Exception as e:
            self.logger.error(f"Error getting latest data: {e}")
            return None

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        include_indicators: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Get historical market data."""
        try:
            if self.test_market_data and symbol in self.test_market_data:
                return self.test_market_data[symbol]

            cache_key = f"{symbol}_{timeframe}_{start_time}_{end_time}"

            if use_cache and self.cache_data and cache_key in self.data_cache:
                return self.data_cache[cache_key]

            if self.database:
                db_data = await self.database.get_market_data(
                    symbol, timeframe, start_time, end_time
                )
                if not db_data.empty:
                    if self.cache_data:
                        self.data_cache[cache_key] = db_data
                    return db_data

            data = await self.data_fetcher.get_historical_data(
                symbol, timeframe, start_time, end_time, include_indicators
            )

            if self.database and not data.empty:
                await self.database.store_market_data(symbol, timeframe, data)

            if self.cache_data:
                self.data_cache[cache_key] = data

            return data

        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            raise

    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, List]:
        """
        Get current orderbook.

        Args:
            symbol: Trading pair
            limit: Depth limit

        Returns:
            Orderbook data
        """
        try:
            return await self.data_fetcher.get_orderbook(symbol, limit)
        except Exception as e:
            self.logger.error(f"Error fetching orderbook: {e}")
            raise

    async def get_recent_trades(
        self, symbol: str, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get recent trades.

        Args:
            symbol: Trading pair
            limit: Number of trades

        Returns:
            List of recent trades
        """
        try:
            return await self.data_fetcher.get_recent_trades(symbol, limit)
        except Exception as e:
            self.logger.error(f"Error fetching recent trades: {e}")
            raise

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear the data cache.

        Args:
            symbol: Optional symbol to clear specific cache
        """
        if symbol:
            keys_to_remove = [k for k in self.data_cache.keys() if k.startswith(symbol)]
            for key in keys_to_remove:
                del self.data_cache[key]
        else:
            self.data_cache.clear()

    async def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get market information.

        Args:
            symbol: Trading pair

        Returns:
            Market information
        """
        try:
            return await self.data_fetcher.get_market_info(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching market info: {e}")
            raise

    @property
    def client(self):
        """Get the underlying Binance client."""
        return self.data_fetcher.client

    async def __aenter__(self):
        """Async context manager enter."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    def _dataframe_to_dict(self, df: Union[pd.DataFrame, dict]) -> dict:
        """Convert DataFrame to dictionary format."""
        try:
            if isinstance(df, dict):
                return df
            if df is None or df.empty:
                return {}
            return df.to_dict(orient="index")
        except Exception as e:
            self.logger.error(f"Error converting data to dict: {e}")
            return {}

    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        if self._session and not self._session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._session.close())
                else:
                    loop.run_until_complete(self._session.close())
            except Exception:
                pass  # Ignore errors in destructor
