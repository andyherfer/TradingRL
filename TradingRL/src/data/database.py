from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import aiosqlite
import logging
import json
from enum import Enum
from pathlib import Path
import pickle
import zlib
from dataclasses import dataclass
import asyncpg
import os


class DataCategory(Enum):
    """Categories of data to be stored."""

    MARKET_DATA = "market_data"
    TRADES = "trades"
    ORDERS = "orders"
    PORTFOLIO = "portfolio"
    PERFORMANCE = "performance"
    SYSTEM_STATE = "system_state"
    STRATEGY = "strategy"
    MODEL = "model"


@dataclass
class DatabaseConfig:
    """Database configuration."""

    db_path: str = "data/trading.db"
    postgres_url: Optional[str] = None
    use_postgres: bool = False
    backup_dir: str = "data/backups"
    compression_level: int = 6
    max_connections: int = 10
    enable_journal: bool = True
    cache_size: int = 2000
    backup_interval: int = 3600  # seconds


class AsyncConnectionManager:
    """Async context manager for database connections."""

    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.connection.rollback()
        else:
            await self.connection.commit()


class Database:
    """
    Handles data persistence and retrieval for the trading system.
    Supports both SQLite and PostgreSQL backends.
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize Database."""
        self.config = config or DatabaseConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.connection = None
        self.cache = {}

        # Create directories
        os.makedirs(os.path.dirname(self.config.db_path), exist_ok=True)

    async def initialize(self):
        """Initialize database connection and tables."""
        try:
            # Create connection
            self.connection = await aiosqlite.connect(self.config.db_path)

            # Create tables
            await self._create_tables()

            self.logger.info(f"Database initialized at {self.config.db_path}")

        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise

    async def close(self) -> None:
        """Close database connection."""
        try:
            if self.connection:
                await self.connection.commit()  # Commit any pending changes
                await self.connection.close()
                self.connection = None
            self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing database: {e}")
            # Don't raise - just log the error

    async def _create_tables(self):
        """Create database tables."""
        try:
            # Drop existing indices first to avoid conflicts
            await self.connection.execute("DROP INDEX IF EXISTS idx_market_data_lookup")
            await self.connection.execute("DROP INDEX IF EXISTS idx_trades_lookup")

            # Drop existing tables to ensure clean state
            await self.connection.execute("DROP TABLE IF EXISTS market_data")
            await self.connection.execute("DROP TABLE IF EXISTS trades")

            # Create market data table
            await self.connection.execute(
                """
                CREATE TABLE market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    data BLOB NOT NULL,
                    type TEXT,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """
            )

            # Create trades table
            await self.connection.execute(
                """
                CREATE TABLE trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    fees REAL NOT NULL,
                    pnl REAL,
                    metadata TEXT
                )
            """
            )

            # Create indices after tables are created
            await self.connection.execute(
                """
                CREATE INDEX idx_market_data_lookup 
                ON market_data(symbol, timeframe, timestamp)
            """
            )

            await self.connection.execute(
                """
                CREATE INDEX idx_trades_lookup
                ON trades(symbol, timestamp)
            """
            )

            await self.connection.commit()

        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise

    async def store_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Store market data."""
        try:
            compressed_data = self._compress_data(data)
            timestamp = data.index[-1].timestamp()

            await self.connection.execute(
                """
                INSERT OR REPLACE INTO market_data 
                (symbol, timeframe, timestamp, data)
                VALUES (?, ?, ?, ?)
                """,
                (symbol, timeframe, timestamp, compressed_data),
            )
            await self.connection.commit()

        except Exception as e:
            self.logger.error(f"Error storing market data: {e}")
            raise

    async def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Retrieve market data."""
        try:
            end_time = end_time or datetime.now()

            cursor = await self.connection.execute(
                """
                SELECT data FROM market_data
                WHERE symbol = ? 
                AND timeframe = ?
                AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
                """,
                (symbol, timeframe, start_time.timestamp(), end_time.timestamp()),
            )

            rows = await cursor.fetchall()

            if not rows:
                return pd.DataFrame()

            dfs = [self._decompress_data(row[0]) for row in rows]
            return pd.concat(dfs) if dfs else pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error retrieving market data: {e}")
            raise

    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage."""
        try:
            return zlib.compress(pickle.dumps(data))
        except Exception as e:
            self.logger.error(f"Error compressing data: {e}")
            raise

    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress stored data."""
        try:
            return pickle.loads(zlib.decompress(compressed_data))
        except Exception as e:
            self.logger.error(f"Error decompressing data: {e}")
            raise
