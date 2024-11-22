from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException
import aiohttp
import logging
import json
import time
from dataclasses import dataclass
from enum import Enum
import sqlite3
import pickle
from functools import lru_cache
import talib
import websockets
import ujson
from collections import defaultdict, deque
import os


class DataType(Enum):
    """Types of market data."""

    TRADE = "trade"
    KLINE = "kline"
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    AGG_TRADE = "agg_trade"


@dataclass
class DataConfig:
    """Configuration for data fetching."""

    cache_dir: str = "data_cache"
    max_cache_size: int = 1000
    update_interval: float = 1.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    use_cache: bool = True
    batch_size: int = 1000
    max_workers: int = 4


class DataFetcher:
    """
    Handles market data fetching, processing, and caching.
    Supports both real-time and historical data from Binance.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: List[str],
        timeframes: List[str] = ["1m", "5m", "15m", "1h"],
        config: Optional[DataConfig] = None,
    ):
        """Initialize DataFetcher."""
        self.symbols = [self._format_symbol(symbol) for symbol in symbols]
        self.timeframes = timeframes
        self.config = config or DataConfig()

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Initialize clients
        self.client = None
        self.socket_manager = None
        self.ws_client = None

        # Store credentials
        self.api_key = api_key
        self.api_secret = api_secret

        # Data management
        self.data_buffers = {
            symbol: {tf: pd.DataFrame() for tf in timeframes} for symbol in symbols
        }

        # Cache management
        self.cache = {}
        self.cache_timestamps = {}

        # Subscription management
        self.active_streams = {}
        self.callbacks = defaultdict(list)

        # Initialize database connection
        self.db_conn = None

        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(10)
        self.last_requests = deque(maxlen=100)

    async def initialize(self) -> None:
        """Initialize connections and start data fetching."""
        try:
            # Initialize Binance client
            self.client = await AsyncClient.create(self.api_key, self.api_secret)
            self.socket_manager = BinanceSocketManager(self.client)

            # Initialize database connection
            self.db_conn = sqlite3.connect("market_data.db")
            self._create_tables()

            # Create cache directory if it doesn't exist
            os.makedirs(self.config.cache_dir, exist_ok=True)

            # Load cache if exists
            self._load_cache()

            # Get exchange info
            await self._get_exchange_info()

            self.logger.info("DataFetcher initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing DataFetcher: {e}")
            raise

    async def close(self) -> None:
        """Clean up resources."""
        try:
            # Close data streams
            for stream in self.active_streams.values():
                await stream.close()

            # Close Binance client
            if self.client:
                await self.client.close_connection()

            # Close database connection
            if self.db_conn:
                self.db_conn.close()

            # Save cache
            self._save_cache()

            self.logger.info("DataFetcher closed successfully")

        except Exception as e:
            self.logger.error(f"Error closing DataFetcher: {e}")
            raise

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        include_indicators: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical market data.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Timeframe (e.g., "1m", "5m")
            start_time: Start time
            end_time: End time (optional)
            include_indicators: Whether to include technical indicators

        Returns:
            DataFrame with market data
        """
        try:
            # Format symbol for Binance API
            formatted_symbol = self._format_symbol(symbol)
            cache_key = f"{formatted_symbol}_{timeframe}_{start_time}_{end_time}"

            # Check cache
            if self.config.use_cache and cache_key in self.cache:
                data = self.cache[cache_key]
                if self._is_cache_valid(cache_key):
                    return data

            # Fetch data
            end_time = end_time or datetime.now()
            data = await self._fetch_historical_klines(
                formatted_symbol, timeframe, start_time, end_time
            )

            # Process data
            df = self._process_klines(data)

            # Calculate indicators if requested
            if include_indicators:
                df = self._add_technical_indicators(df)

            # Cache data
            if self.config.use_cache:
                self.cache[cache_key] = df
                self.cache_timestamps[cache_key] = time.time()

            return df

        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            raise

    async def subscribe_data(
        self, symbol: str, data_type: DataType, callback: callable
    ) -> None:
        """
        Subscribe to real-time data updates.

        Args:
            symbol: Trading pair
            data_type: Type of data to subscribe to
            callback: Callback function for updates
        """
        try:
            # Add callback
            stream_key = f"{symbol}_{data_type.value}"
            self.callbacks[stream_key].append(callback)

            # Start stream if not active
            if stream_key not in self.active_streams:
                if data_type == DataType.KLINE:
                    for timeframe in self.timeframes:
                        stream = await self.socket_manager.kline_socket(
                            symbol, timeframe
                        )
                        self.active_streams[f"{stream_key}_{timeframe}"] = stream
                else:
                    stream = await self._get_data_stream(symbol, data_type)
                    self.active_streams[stream_key] = stream

                # Start processing stream
                asyncio.create_task(self._process_stream(stream_key))

        except Exception as e:
            self.logger.error(f"Error subscribing to data: {e}")
            raise

    async def _process_stream(self, stream_key: str) -> None:
        """Process incoming data stream."""
        try:
            stream = self.active_streams[stream_key]
            async with stream as tscm:
                while True:
                    msg = await tscm.recv()
                    await self._handle_stream_message(stream_key, msg)

        except Exception as e:
            self.logger.error(f"Error processing stream {stream_key}: {e}")
            # Attempt to reconnect
            await self._reconnect_stream(stream_key)

    async def _handle_stream_message(self, stream_key: str, msg: Dict) -> None:
        """Handle incoming stream message."""
        try:
            # Process message based on type
            if "k" in msg:  # Kline data
                data = self._process_kline_message(msg)
                symbol, timeframe = stream_key.split("_")[0:2]
                self._update_data_buffer(symbol, timeframe, data)
            elif "b" in msg:  # Orderbook data
                data = self._process_orderbook_message(msg)
            else:  # Trade or ticker data
                data = msg

            # Notify callbacks
            for callback in self.callbacks[stream_key]:
                await callback(data)

            # Update database
            self._store_data(stream_key, data)

        except Exception as e:
            self.logger.error(f"Error handling stream message: {e}")

    def _update_data_buffer(
        self, symbol: str, timeframe: str, data: pd.DataFrame
    ) -> None:
        """Update in-memory data buffer."""
        try:
            buffer = self.data_buffers[symbol][timeframe]

            # Append new data
            buffer = pd.concat([buffer, data]).drop_duplicates()

            # Maintain buffer size
            if len(buffer) > self.config.max_cache_size:
                buffer = buffer.iloc[-self.config.max_cache_size :]

            # Update indicators
            buffer = self._add_technical_indicators(buffer)

            self.data_buffers[symbol][timeframe] = buffer

        except Exception as e:
            self.logger.error(f"Error updating data buffer: {e}")

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame."""
        try:
            # Price indicators
            df["sma_20"] = talib.SMA(df["close"], timeperiod=20)
            df["sma_50"] = talib.SMA(df["close"], timeperiod=50)
            df["sma_200"] = talib.SMA(df["close"], timeperiod=200)
            df["ema_20"] = talib.EMA(df["close"], timeperiod=20)

            # Momentum indicators
            df["rsi"] = talib.RSI(df["close"], timeperiod=14)
            df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
                df["close"], fastperiod=12, slowperiod=26, signalperiod=9
            )

            # Volatility indicators
            df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
            df["bbands_upper"], df["bbands_middle"], df["bbands_lower"] = talib.BBANDS(
                df["close"], timeperiod=20, nbdevup=2, nbdevdn=2
            )

            # Volume indicators
            df["obv"] = talib.OBV(df["close"], df["volume"])
            df["adx"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)

            return df

        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return df

    async def _fetch_historical_klines(
        self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime
    ) -> List[List]:
        """
        Fetch historical klines from Binance in batches.
        """
        try:
            klines = []
            current_time = start_time
            batch_count = 0
            last_timestamp = None

            # Calculate time delta based on timeframe
            time_deltas = {
                "1m": timedelta(minutes=self.config.batch_size),
                "5m": timedelta(minutes=5 * self.config.batch_size),
                "15m": timedelta(minutes=15 * self.config.batch_size),
                "1h": timedelta(hours=self.config.batch_size),
                "4h": timedelta(hours=4 * self.config.batch_size),
                "1d": timedelta(days=self.config.batch_size),
            }

            batch_delta = time_deltas.get(
                timeframe, timedelta(minutes=self.config.batch_size)
            )

            self.logger.info(
                f"Fetching historical data for {symbol} from {start_time} to {end_time}"
            )

            while current_time < end_time:
                async with self.rate_limiter:
                    # Calculate batch end time
                    batch_end = min(current_time + batch_delta, end_time)

                    # Convert timestamps to milliseconds
                    start_ts = int(current_time.timestamp() * 1000)
                    end_ts = int(batch_end.timestamp() * 1000)

                    # Fetch batch with retries
                    for attempt in range(self.config.retry_attempts):
                        try:
                            batch = await self.client.get_klines(
                                symbol=symbol,
                                interval=timeframe,
                                startTime=start_ts,
                                endTime=end_ts,
                                limit=self.config.batch_size,
                            )

                            if batch:
                                # Check if we're getting new data
                                new_last_timestamp = batch[-1][0]
                                if (
                                    last_timestamp
                                    and new_last_timestamp <= last_timestamp
                                ):
                                    # We've reached the end of available data
                                    self.logger.info("Reached end of available data")
                                    return klines

                                klines.extend(batch)
                                batch_count += 1
                                last_timestamp = new_last_timestamp

                                if (
                                    batch_count % 10 == 0
                                ):  # Log progress every 10 batches
                                    self.logger.info(
                                        f"Fetched {len(klines)} klines up to "
                                        f"{datetime.fromtimestamp(batch[-1][0]/1000)}"
                                    )

                                break  # Success, exit retry loop
                            else:
                                # No data returned for this time period
                                current_time = batch_end
                                break

                        except BinanceAPIException as e:
                            if (
                                attempt == self.config.retry_attempts - 1
                            ):  # Last attempt
                                raise
                            await asyncio.sleep(self.config.retry_delay * (2**attempt))

                    # Update current time for next batch
                    if batch:
                        # Move forward by the actual data received
                        current_time = datetime.fromtimestamp(batch[-1][0] / 1000)
                        # Add a small offset to avoid duplicate data
                        current_time += timedelta(milliseconds=1)
                    else:
                        # If no data, check if we're too close to current time
                        if batch_end >= datetime.now() - timedelta(minutes=1):
                            self.logger.info("Reached current time, ending data fetch")
                            break
                        current_time = batch_end

                    # Rate limiting
                    self.last_requests.append(time.time())
                    if len(self.last_requests) >= 10:
                        await asyncio.sleep(1)

                    # Additional delay between batches
                    await asyncio.sleep(0.1)

            self.logger.info(
                f"Successfully fetched {len(klines)} klines for {symbol} "
                f"in {batch_count} batches"
            )

            return klines

        except BinanceAPIException as e:
            self.logger.error(f"Binance API error fetching klines for {symbol}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error fetching historical klines for {symbol}: {e}")
            raise

    def _process_klines(self, klines: List[List]) -> pd.DataFrame:
        """Process raw klines data into DataFrame."""
        try:
            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_volume",
                    "taker_buy_quote_volume",
                    "ignore",
                ],
            )

            # Convert types
            numeric_columns = ["open", "high", "low", "close", "volume"]
            df[numeric_columns] = df[numeric_columns].astype(float)

            # Convert timestamps
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"Error processing klines: {e}")
            raise

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        try:
            cursor = self.db_conn.cursor()

            # Create market data table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timeframe TEXT,
                    timestamp DATETIME,
                    data BLOB,
                    type TEXT
                )
            """
            )

            # Create index
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_market_data 
                ON market_data (symbol, timeframe, timestamp)
            """
            )

            self.db_conn.commit()

        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise

    def _store_data(self, stream_key: str, data: Dict) -> None:
        """Store data in database."""
        try:
            cursor = self.db_conn.cursor()

            # Prepare data
            symbol, data_type = stream_key.split("_")[0:2]
            timestamp = datetime.now()
            serialized_data = pickle.dumps(data)

            # Insert data
            cursor.execute(
                """
                INSERT INTO market_data 
                (symbol, timeframe, timestamp, data, type)
                VALUES (?, ?, ?, ?, ?)
                """,
                (symbol, data_type, timestamp, serialized_data, data_type),
            )

            self.db_conn.commit()

        except Exception as e:
            self.logger.error(f"Error storing data: {e}")

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(f"{self.config.cache_dir}/cache.pickle", "wb") as f:
                pickle.dump(
                    {"data": self.cache, "timestamps": self.cache_timestamps}, f
                )

            self.logger.info("Cache saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving cache: {e}")

    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            cache_path = f"{self.config.cache_dir}/cache.pickle"
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                    self.cache = cache_data["data"]
                    self.cache_timestamps = cache_data["timestamps"]

                self.logger.info("Cache loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        try:
            if cache_key not in self.cache_timestamps:
                return False

            cache_time = self.cache_timestamps[cache_key]
            current_time = time.time()

            # Cache validity depends on timeframe
            timeframe = cache_key.split("_")[1]
            max_age = self._get_cache_max_age(timeframe)

            return (current_time - cache_time) < max_age

        except Exception as e:
            self.logger.error(f"Error checking cache validity: {e}")
            return False

    def _get_cache_max_age(self, timeframe: str) -> int:
        """Get maximum cache age for timeframe."""
        timeframe_minutes = {
            "1m": 60,
            "5m": 240,
            "15m": 720,
            "1h": 1440,
            "4h": 4320,
            "1d": 10080,
        }

        base_minutes = timeframe_minutes.get(timeframe, 60)
        return base_minutes * 60  # Convert to seconds

    async def _reconnect_stream(self, stream_key: str) -> None:
        """Reconnect to a broken stream."""
        try:
            self.logger.info(f"Attempting to reconnect stream: {stream_key}")

            # Close existing stream
            if stream_key in self.active_streams:
                await self.active_streams[stream_key].close()

            # Wait before reconnecting
            await asyncio.sleep(self.config.retry_delay)

            # Create new stream
            symbol, data_type = stream_key.split("_")[0:2]
            new_stream = await self._get_data_stream(symbol, DataType(data_type))

            self.active_streams[stream_key] = new_stream

            # Restart stream processing
            asyncio.create_task(self._process_stream(stream_key))

            self.logger.info(f"Successfully reconnected stream: {stream_key}")

        except Exception as e:
            self.logger.error(f"Error reconnecting stream: {e}")
            # Schedule another reconnection attempt
            asyncio.create_task(self._schedule_reconnection(stream_key))

    async def _schedule_reconnection(self, stream_key: str) -> None:
        """Schedule a reconnection attempt with exponential backoff."""
        try:
            for attempt in range(self.config.retry_attempts):
                delay = self.config.retry_delay * (2**attempt)
                await asyncio.sleep(delay)

                try:
                    await self._reconnect_stream(stream_key)
                    return
                except Exception:
                    continue

            self.logger.error(
                f"Failed to reconnect stream after {self.config.retry_attempts} attempts"
            )

        except Exception as e:
            self.logger.error(f"Error scheduling reconnection: {e}")

    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, List]:
        """
        Get current orderbook snapshot.

        Args:
            symbol: Trading pair
            limit: Depth limit

        Returns:
            Dictionary with bids and asks
        """
        try:
            async with self.rate_limiter:
                orderbook = await self.client.get_order_book(symbol=symbol, limit=limit)

                return {
                    "bids": [[float(p), float(q)] for p, q in orderbook["bids"]],
                    "asks": [[float(p), float(q)] for p, q in orderbook["asks"]],
                }

        except Exception as e:
            self.logger.error(f"Error fetching orderbook: {e}")
            raise

    async def get_recent_trades(self, symbol: str, limit: int = 1000) -> List[Dict]:
        """
        Get recent trades.

        Args:
            symbol: Trading pair
            limit: Number of trades

        Returns:
            List of recent trades
        """
        try:
            async with self.rate_limiter:
                trades = await self.client.get_recent_trades(symbol=symbol, limit=limit)

                return [
                    {
                        "id": t["id"],
                        "price": float(t["price"]),
                        "quantity": float(t["qty"]),
                        "time": datetime.fromtimestamp(t["time"] / 1000),
                        "is_buyer_maker": t["isBuyerMaker"],
                    }
                    for t in trades
                ]

        except Exception as e:
            self.logger.error(f"Error fetching recent trades: {e}")
            raise

    def get_current_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get current data from buffer.

        Args:
            symbol: Trading pair
            timeframe: Timeframe

        Returns:
            Current market data
        """
        try:
            if symbol not in self.data_buffers:
                return None

            if timeframe not in self.data_buffers[symbol]:
                return None

            return self.data_buffers[symbol][timeframe].copy()

        except Exception as e:
            self.logger.error(f"Error getting current data: {e}")
            return None

    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Get current funding rate.

        Args:
            symbol: Trading pair

        Returns:
            Current funding rate
        """
        try:
            async with self.rate_limiter:
                funding_rate = await self.client.futures_funding_rate(
                    symbol=symbol, limit=1
                )

                if funding_rate:
                    return float(funding_rate[0]["fundingRate"])
                return None

        except Exception as e:
            self.logger.error(f"Error fetching funding rate: {e}")
            return None

    async def get_historical_funding_rates(
        self, symbol: str, start_time: datetime, end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical funding rates.

        Args:
            symbol: Trading pair
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with funding rates
        """
        try:
            async with self.rate_limiter:
                rates = []
                current_time = start_time
                end_time = end_time or datetime.now()

                while current_time < end_time:
                    batch = await self.client.futures_funding_rate(
                        symbol=symbol,
                        startTime=int(current_time.timestamp() * 1000),
                        limit=1000,
                    )

                    if not batch:
                        break

                    rates.extend(batch)

                    # Update current time
                    last_timestamp = batch[-1]["fundingTime"]
                    current_time = datetime.fromtimestamp(last_timestamp / 1000)

                    await asyncio.sleep(0.1)  # Rate limiting

                df = pd.DataFrame(rates)
                df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
                df["fundingRate"] = df["fundingRate"].astype(float)
                df.set_index("timestamp", inplace=True)

                return df

        except Exception as e:
            self.logger.error(f"Error fetching historical funding rates: {e}")
            raise

    async def get_market_info(self, symbol: str) -> Dict:
        """
        Get market information.

        Args:
            symbol: Trading pair

        Returns:
            Market information
        """
        try:
            async with self.rate_limiter:
                info = await self.client.get_symbol_info(symbol)

                # Extract relevant information
                filters = {f["filterType"]: f for f in info["filters"]}

                return {
                    "base_asset": info["baseAsset"],
                    "quote_asset": info["quoteAsset"],
                    "status": info["status"],
                    "min_price": float(filters["PRICE_FILTER"]["minPrice"]),
                    "max_price": float(filters["PRICE_FILTER"]["maxPrice"]),
                    "tick_size": float(filters["PRICE_FILTER"]["tickSize"]),
                    "min_qty": float(filters["LOT_SIZE"]["minQty"]),
                    "max_qty": float(filters["LOT_SIZE"]["maxQty"]),
                    "step_size": float(filters["LOT_SIZE"]["stepSize"]),
                    "min_notional": float(
                        filters.get("MIN_NOTIONAL", {}).get("minNotional", 0)
                    ),
                }

        except Exception as e:
            self.logger.error(f"Error fetching market info: {e}")
            raise

    async def _get_exchange_info(self) -> None:
        """Get and cache exchange information."""
        try:
            async with self.rate_limiter:
                info = await self.client.get_exchange_info()

                # Cache exchange info
                self.exchange_info = {s["symbol"]: s for s in info["symbols"]}

                # Cache trading rules
                self.trading_rules = {}
                for symbol in self.symbols:
                    if symbol in self.exchange_info:
                        self.trading_rules[symbol] = self._extract_trading_rules(
                            self.exchange_info[symbol]
                        )

        except Exception as e:
            self.logger.error(f"Error fetching exchange info: {e}")
            raise

    def _extract_trading_rules(self, symbol_info: Dict) -> Dict:
        """Extract trading rules from symbol information."""
        filters = {f["filterType"]: f for f in symbol_info["filters"]}

        return {
            "price_precision": symbol_info["quotePrecision"],
            "quantity_precision": symbol_info["baseAsset"],
            "price_filter": filters.get("PRICE_FILTER", {}),
            "lot_size": filters.get("LOT_SIZE", {}),
            "min_notional": filters.get("MIN_NOTIONAL", {}).get("minNotional", 0),
            "market_lot_size": filters.get("MARKET_LOT_SIZE", {}),
            "max_num_orders": filters.get("MAX_NUM_ORDERS", {}).get("maxNumOrders", 0),
            "max_num_algo_orders": filters.get("MAX_NUM_ALGO_ORDERS", {}).get(
                "maxNumAlgoOrders", 0
            ),
        }

    def _format_symbol(self, symbol: str) -> str:
        """Convert symbol to Binance format (e.g., 'BTC/USDT' -> 'BTCUSDT')."""
        return symbol.replace("/", "")
