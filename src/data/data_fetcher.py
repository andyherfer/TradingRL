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
from typing import List, Dict, Optional
import os
from pathlib import Path
import ccxt.async_support as ccxt
from logging import getLogger

logger = getLogger(__name__)


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
        """
        Initialize DataFetcher.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            symbols: List of trading pairs
            timeframes: List of timeframes to fetch
            config: DataFetcher configuration
        """
        self.symbols = symbols
        self.timeframes = timeframes
        self.config = config or DataConfig()

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Initialize clients
        self.client = None
        self.socket_manager = None
        self.ws_client = None

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
        self.db_conn = sqlite3.connect("market_data.db")
        self._create_tables()

        # Technical indicators
        self.indicators = {}

        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(10)
        self.last_requests = deque(maxlen=100)

        # Initialize API credentials
        self.api_key = api_key
        self.api_secret = api_secret

        # Create cache directory if it doesn't exist
        self.cache_dir = Path("data/market_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.testnet = False

    async def start(self):
        """Initialize connections and start data fetching."""
        try:
            # Initialize Binance client
            self.client = await AsyncClient.create(self.api_key, self.api_secret)
            self.socket_manager = BinanceSocketManager(self.client)

            # Start data streams
            await self._start_data_streams()

            self.logger.info("DataFetcher started successfully")

        except Exception as e:
            self.logger.error(f"Error starting DataFetcher: {e}")
            raise

    async def stop(self):
        """Stop data fetching and clean up."""
        try:
            # Close data streams
            for stream in self.active_streams.values():
                await stream.close()

            # Close clients
            await self.client.close_connection()

            # Save cache to disk
            self._save_cache()

            # Close database connection
            self.db_conn.close()

            self.logger.info("DataFetcher stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping DataFetcher: {e}")
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
            symbol: Trading pair
            timeframe: Timeframe (e.g., "1m", "5m")
            start_time: Start time
            end_time: End time (optional)
            include_indicators: Whether to include technical indicators

        Returns:
            DataFrame with market data
        """
        try:
            cache_key = f"{symbol}_{timeframe}_{start_time}_{end_time}"

            # Check cache
            if self.config.use_cache and cache_key in self.cache:
                data = self.cache[cache_key]
                if self._is_cache_valid(cache_key):
                    return data

            # Fetch data
            end_time = end_time or datetime.now()
            data = await self._fetch_historical_klines(
                symbol, timeframe, start_time, end_time
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
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        retry_count: int = 3,
    ) -> pd.DataFrame:
        """Fetch historical klines (candlestick data) from the exchange."""
        try:
            # Convert symbol format if needed (e.g., BTC/USDT -> BTCUSDT)
            formatted_symbol = symbol.replace("/", "")

            # Convert timestamps to milliseconds
            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)

            all_klines = []
            current_start = start_ts

            while current_start < end_ts:
                try:
                    # Fetch a batch of klines with correct parameters
                    params = {
                        "symbol": formatted_symbol,
                        "interval": interval,
                        "startTime": current_start,
                        "endTime": end_ts,
                        "limit": 1000,
                    }

                    klines = await self.client.get_historical_klines(
                        symbol=formatted_symbol,
                        interval=interval,
                        start_str=str(current_start),
                        end_str=str(end_ts),
                        limit=1000,
                    )

                    if not klines:
                        break

                    all_klines.extend(klines)

                    # Update the start time for the next batch
                    if klines:
                        current_start = klines[-1][0] + 1
                    else:
                        break

                    # Rate limiting
                    await asyncio.sleep(0.5)  # Add delay to avoid rate limits

                except Exception as e:
                    logger.error(f"Error fetching batch of klines: {e}")
                    if retry_count > 0:
                        await asyncio.sleep(1)
                        return await self._fetch_historical_klines(
                            symbol, interval, start_time, end_time, retry_count - 1
                        )
                    raise

            # Convert to DataFrame
            if not all_klines:
                return pd.DataFrame()

            df = pd.DataFrame(
                all_klines,
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
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignored",
                ],
            )

            # Convert types
            numeric_columns = ["open", "high", "low", "close", "volume"]
            df[numeric_columns] = df[numeric_columns].astype(float)

            # Convert timestamp to datetime index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            return df[["open", "high", "low", "close", "volume"]]

        except Exception as e:
            logger.error(f"Error fetching historical klines: {e}")
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

    async def initialize(self) -> None:
        """Initialize exchange clients."""
        try:
            # Initialize Binance client with testnet
            self.client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=True,  # Enable testnet
            )

            # Initialize CCXT client as backup
            self.exchange = ccxt.binance(
                {
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "enableRateLimit": True,
                    "options": {"defaultType": "future"},
                    "urls": {
                        "api": {
                            "public": "https://testnet.binance.vision/api/v3",
                            "private": "https://testnet.binance.vision/api/v3",
                        }
                    },
                }
            )

            self.exchange.set_sandbox_mode(True)
            logger.info("Data fetcher initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing data fetcher: {e}")
            raise

    async def close(self) -> None:
        """Close all client connections."""
        if self.client:
            await self.client.close_connection()
        if self.exchange:
            await self.exchange.close()

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical market data for a given symbol and timeframe.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Candlestick timeframe (e.g., '1h', '4h', '1d')
            start_time: Start time
            end_time: End time
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with OHLCV data and technical indicators
        """
        try:
            # Convert timeframe to Binance format
            timeframe_map = {
                "1m": "1m",
                "3m": "3m",
                "5m": "5m",
                "15m": "15m",
                "30m": "30m",
                "1h": "1h",
                "2h": "2h",
                "4h": "4h",
                "6h": "6h",
                "8h": "8h",
                "12h": "12h",
                "1d": "1d",
                "3d": "3d",
                "1w": "1w",
                "1M": "1M",
            }

            binance_timeframe = timeframe_map.get(timeframe)
            if not binance_timeframe:
                raise ValueError(f"Unsupported timeframe: {timeframe}")

            # Check cache first if enabled
            if use_cache:
                cached_data = await self._load_from_cache(
                    symbol, timeframe, start_time, end_time
                )
                if cached_data is not None:
                    return cached_data

            # Fetch historical klines
            data = await self._fetch_historical_klines(
                symbol=symbol,
                interval=binance_timeframe,
                start_time=start_time,
                end_time=end_time,
            )

            if data.empty:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return pd.DataFrame()

            # Add technical indicators
            data = self._add_technical_indicators(data)

            # Cache the data if enabled
            if use_cache:
                await self._save_to_cache(data, symbol, timeframe, start_time, end_time)

            return data

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise

    async def _fetch_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        retry_count: int = 3,
    ) -> pd.DataFrame:
        """Fetch historical klines (candlestick data) from the exchange."""
        try:
            # Convert symbol format if needed (e.g., BTC/USDT -> BTCUSDT)
            formatted_symbol = symbol.replace("/", "")

            # Convert timestamps to milliseconds
            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)

            all_klines = []
            current_start = start_ts

            while current_start < end_ts:
                try:
                    # Fetch a batch of klines with correct parameters
                    params = {
                        "symbol": formatted_symbol,
                        "interval": interval,
                        "startTime": current_start,
                        "endTime": end_ts,
                        "limit": 1000,
                    }

                    klines = await self.client.get_historical_klines(
                        symbol=formatted_symbol,
                        interval=interval,
                        start_str=str(current_start),
                        end_str=str(end_ts),
                        limit=1000,
                    )

                    if not klines:
                        break

                    all_klines.extend(klines)

                    # Update the start time for the next batch
                    if klines:
                        current_start = klines[-1][0] + 1
                    else:
                        break

                    # Rate limiting
                    await asyncio.sleep(0.5)  # Add delay to avoid rate limits

                except Exception as e:
                    logger.error(f"Error fetching batch of klines: {e}")
                    if retry_count > 0:
                        await asyncio.sleep(1)
                        return await self._fetch_historical_klines(
                            symbol, interval, start_time, end_time, retry_count - 1
                        )
                    raise

            # Convert to DataFrame
            if not all_klines:
                return pd.DataFrame()

            df = pd.DataFrame(
                all_klines,
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
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignored",
                ],
            )

            # Convert types
            numeric_columns = ["open", "high", "low", "close", "volume"]
            df[numeric_columns] = df[numeric_columns].astype(float)

            # Convert timestamp to datetime index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            return df[["open", "high", "low", "close", "volume"]]

        except Exception as e:
            logger.error(f"Error fetching historical klines: {e}")
            raise

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame."""
        try:
            # Calculate basic indicators
            df["sma_20"] = df["close"].rolling(window=20).mean()
            df["sma_50"] = df["close"].rolling(window=50).mean()
            df["rsi"] = self._calculate_rsi(df["close"], periods=14)
            df["atr"] = self._calculate_atr(df, periods=14)

            # Add momentum indicators
            df["macd"], df["macd_signal"], df["macd_hist"] = self._calculate_macd(
                df["close"]
            )

            # Add volatility indicators
            df["bollinger_upper"], df["bollinger_middle"], df["bollinger_lower"] = (
                self._calculate_bollinger_bands(df["close"])
            )

            return df.dropna()  # Remove any rows with NaN values

        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            raise

    # Add helper methods for technical indicators...
    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> tuple:
        """Calculate MACD, Signal line, and MACD histogram."""
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def _calculate_bollinger_bands(
        self, prices: pd.Series, window: int = 20, num_std: float = 2
    ) -> tuple:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower

    def _calculate_atr(self, df: pd.DataFrame, periods: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=periods).mean()

    async def _load_from_cache(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        cache_file = (
            self.cache_dir
            / f"{symbol.replace('/', '_')}_{timeframe}_{start_time.date()}_{end_time.date()}.csv"
        )

        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, index_col="timestamp", parse_dates=True)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"Error loading from cache: {e}")

        return None

    async def _save_to_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Save data to cache."""
        if df.empty:
            return

        cache_file = (
            self.cache_dir
            / f"{symbol.replace('/', '_')}_{timeframe}_{start_time.date()}_{end_time.date()}.csv"
        )
        try:
            df.to_csv(cache_file)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
