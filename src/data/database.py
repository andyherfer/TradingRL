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

class Database:
    """
    Handles data persistence and retrieval for the trading system.
    Supports both SQLite and PostgreSQL backends.
    """
    
    def __init__(
        self,
        config: Optional[DatabaseConfig] = None
    ):
        """
        Initialize Database.
        
        Args:
            config: Database configuration
        """
        self.config = config or DatabaseConfig()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Connection management
        self.pool = None
        self.connections = {}
        
        # Initialize directories
        os.makedirs(os.path.dirname(self.config.db_path), exist_ok=True)
        os.makedirs(self.config.backup_dir, exist_ok=True)
        
        # Cache management
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Backup management
        self.last_backup = None
        self.backup_task = None
    
    async def initialize(self):
        """Initialize database connection and tables."""
        try:
            if self.config.use_postgres:
                # Initialize PostgreSQL connection pool
                self.pool = await asyncpg.create_pool(
                    self.config.postgres_url,
                    max_size=self.config.max_connections,
                    min_size=2
                )
            else:
                # Initialize SQLite database
                await self._init_sqlite()
            
            # Create tables
            await self._create_tables()
            
            # Start backup task
            self.backup_task = asyncio.create_task(self._backup_loop())
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    async def close(self):
        """Close database connections."""
        try:
            # Stop backup task
            if self.backup_task:
                self.backup_task.cancel()
            
            # Close connections
            if self.config.use_postgres:
                await self.pool.close()
            else:
                for conn in self.connections.values():
                    await conn.close()
            
            self.logger.info("Database connections closed")
            
        except Exception as e:
            self.logger.error(f"Error closing database: {e}")
    
    async def store_market_data(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame
    ) -> None:
        """
        Store market data.
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe
            data: Market data DataFrame
        """
        try:
            # Prepare data for storage
            compressed_data = self._compress_data(data)
            
            if self.config.use_postgres:
                async with self.pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO market_data (symbol, timeframe, timestamp, data)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (symbol, timeframe, timestamp)
                        DO UPDATE SET data = EXCLUDED.data
                        """,
                        symbol,
                        timeframe,
                        data.index[-1],
                        compressed_data
                    )
            else:
                async with self._get_connection() as conn:
                    await conn.execute(
                        """
                        INSERT OR REPLACE INTO market_data 
                        (symbol, timeframe, timestamp, data)
                        VALUES (?, ?, ?, ?)
                        """,
                        (symbol, timeframe, data.index[-1].timestamp(), compressed_data)
                    )
                    await conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing market data: {e}")
            raise
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Retrieve market data.
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe
            start_time: Start time
            end_time: End time
            
        Returns:
            Market data DataFrame
        """
        try:
            cache_key = f"market_{symbol}_{timeframe}_{start_time}_{end_time}"
            
            # Check cache
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]
            
            if self.config.use_postgres:
                async with self.pool.acquire() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT data FROM market_data
                        WHERE symbol = $1 
                        AND timeframe = $2
                        AND timestamp BETWEEN $3 AND $4
                        ORDER BY timestamp
                        """,
                        symbol,
                        timeframe,
                        start_time,
                        end_time or datetime.now()
                    )
            else:
                async with self._get_connection() as conn:
                    cursor = await conn.execute(
                        """
                        SELECT data FROM market_data
                        WHERE symbol = ? 
                        AND timeframe = ?
                        AND timestamp BETWEEN ? AND ?
                        ORDER BY timestamp
                        """,
                        (symbol, timeframe, start_time.timestamp(), 
                         (end_time or datetime.now()).timestamp())
                    )
                    rows = await cursor.fetchall()
            
            # Decompress and combine data
            dfs = [
                self._decompress_data(row[0])
                for row in rows
            ]
            
            if not dfs:
                return pd.DataFrame()
            
            result = pd.concat(dfs)
            
            # Update cache
            self.cache[cache_key] = result
            self.cache_misses += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving market data: {e}")
            raise
    
    async def store_trade(
        self,
        trade: Dict[str, Any]
    ) -> None:
        """
        Store trade information.
        
        Args:
            trade: Trade information
        """
        try:
            if self.config.use_postgres:
                async with self.pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO trades (
                            trade_id, symbol, side, quantity, price,
                            timestamp, fees, pnl, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """,
                        trade['id'],
                        trade['symbol'],
                        trade['side'],
                        trade['quantity'],
                        trade['price'],
                        trade['timestamp'],
                        trade['fees'],
                        trade.get('pnl', 0),
                        json.dumps(trade.get('metadata', {}))
                    )
            else:
                async with self._get_connection() as conn:
                    await conn.execute(
                        """
                        INSERT INTO trades (
                            trade_id, symbol, side, quantity, price,
                            timestamp, fees, pnl, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (trade['id'], trade['symbol'], trade['side'],
                         trade['quantity'], trade['price'], trade['timestamp'].timestamp(),
                         trade['fees'], trade.get('pnl', 0),
                         json.dumps(trade.get('metadata', {})))
                    )
                    await conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing trade: {e}")
            raise
    
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Retrieve trades.
        
        Args:
            symbol: Optional symbol filter
            start_time: Start time
            end_time: End time
            
        Returns:
            Trades DataFrame
        """
        try:
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.timestamp())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.timestamp())
            
            query += " ORDER BY timestamp"
            
            async with self._get_connection() as conn:
                cursor = await conn.execute(query, params)
                rows = await cursor.fetchall()
            
            df = pd.DataFrame(rows, columns=[
                'trade_id', 'symbol', 'side', 'quantity', 'price',
                'timestamp', 'fees', 'pnl', 'metadata'
            ])
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df['metadata'] = df['metadata'].apply(json.loads)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving trades: {e}")
            raise
    
    async def store_portfolio_state(
        self,
        state: Dict[str, Any]
    ) -> None:
        """
        Store portfolio state.
        
        Args:
            state: Portfolio state
        """
        try:
            compressed_state = self._compress_data(state)
            
            if self.config.use_postgres:
                async with self.pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO portfolio_states (timestamp, state)
                        VALUES ($1, $2)
                        """,
                        datetime.now(),
                        compressed_state
                    )
            else:
                async with self._get_connection() as conn:
                    await conn.execute(
                        """
                        INSERT INTO portfolio_states (timestamp, state)
                        VALUES (?, ?)
                        """,
                        (datetime.now().timestamp(), compressed_state)
                    )
                    await conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing portfolio state: {e}")
            raise
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage."""
        try:
            pickled_data = pickle.dumps(data)
            return zlib.compress(pickled_data, self.config.compression_level)
        except Exception as e:
            self.logger.error(f"Error compressing data: {e}")
            raise
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress stored data."""
        try:
            pickled_data = zlib.decompress(compressed_data)
            return pickle.loads(pickled_data)
        except Exception as e:
            self.logger.error(f"Error decompressing data: {e}")
            raise
    
    async def _create_tables(self) -> None:
        """Create database tables."""
        try:
            async with self._get_connection() as conn:
                # Market data table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        data BLOB NOT NULL,
                        UNIQUE(symbol, timeframe, timestamp)
                    )
                ''')
                
                # Trades table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
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
                ''')
                
                # Portfolio states table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS portfolio_states (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        state BLOB NOT NULL
                    )
                ''')
                
                # Create indices
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_market_data_lookup 
                    ON market_data(symbol, timeframe, timestamp)
                ''')
                
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_trades_lookup
                    ON trades(symbol, timestamp)
                ''')
                
                await conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise
    
    async def _backup_loop(self) -> None:
        """Periodic backup task."""
        try:
            while True:
                await asyncio.sleep(self.config.backup_interval)
                await self.create_backup()
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in backup loop: {e}")
    
    async def create_backup(self) -> None:
        """Create database backup."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(
                self.config.backup_dir,
                f"trading_db_backup_{timestamp}.db"
            )
            
            if self.config.use_postgres:
                # Use pg_dump for PostgreSQL
                os.system(f"pg_dump {self.config.postgres_url} > {backup_path}")
            else:
                # Copy SQLite database
                async with self._get_connection() as conn:
                    await conn.execute("VACUUM INTO ?", (backup_path,))
            
            self.last_backup = datetime.now()
            self.logger.info(f"Database backup created: {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            
        async def store_system_state(
        self,
        state: Dict[str, Any],
        category: str = "general"
    ) -> None:
        """
        Store system state information.
        
        Args:
            state: System state
            category: State category
        """
        try:
            compressed_state = self._compress_data(state)
            timestamp = datetime.now()
            
            if self.config.use_postgres:
                async with self.pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO system_states 
                        (timestamp, category, state)
                        VALUES ($1, $2, $3)
                        """,
                        timestamp,
                        category,
                        compressed_state
                    )
            else:
                async with self._get_connection() as conn:
                    await conn.execute(
                        """
                        INSERT INTO system_states 
                        (timestamp, category, state)
                        VALUES (?, ?, ?)
                        """,
                        (timestamp.timestamp(), category, compressed_state)
                    )
                    await conn.commit()
            
            # Update cache
            cache_key = f"system_state_{category}"
            self.cache[cache_key] = state
            
        except Exception as e:
            self.logger.error(f"Error storing system state: {e}")
            raise
    
    async def store_performance_metrics(
        self,
        metrics: Dict[str, Any],
        strategy_name: Optional[str] = None
    ) -> None:
        """
        Store performance metrics.
        
        Args:
            metrics: Performance metrics
            strategy_name: Optional strategy name
        """
        try:
            timestamp = datetime.now()
            
            if self.config.use_postgres:
                async with self.pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO performance_metrics
                        (timestamp, strategy_name, metrics)
                        VALUES ($1, $2, $3)
                        """,
                        timestamp,
                        strategy_name,
                        json.dumps(metrics)
                    )
            else:
                async with self._get_connection() as conn:
                    await conn.execute(
                        """
                        INSERT INTO performance_metrics
                        (timestamp, strategy_name, metrics)
                        VALUES (?, ?, ?)
                        """,
                        (timestamp.timestamp(), strategy_name, json.dumps(metrics))
                    )
                    await conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing performance metrics: {e}")
            raise
    
    async def get_performance_metrics(
        self,
        strategy_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        aggregation: str = "none"
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Retrieve performance metrics.
        
        Args:
            strategy_name: Optional strategy filter
            start_time: Start time
            end_time: End time
            aggregation: Aggregation method ('none', 'daily', 'weekly', 'monthly')
            
        Returns:
            Performance metrics
        """
        try:
            query = "SELECT * FROM performance_metrics WHERE 1=1"
            params = []
            
            if strategy_name:
                query += " AND strategy_name = ?"
                params.append(strategy_name)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.timestamp())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.timestamp())
            
            async with self._get_connection() as conn:
                cursor = await conn.execute(query, params)
                rows = await cursor.fetchall()
            
            df = pd.DataFrame(rows, columns=[
                'id', 'timestamp', 'strategy_name', 'metrics'
            ])
            
            if df.empty:
                return pd.DataFrame()
            
            # Process data
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['metrics'] = df['metrics'].apply(json.loads)
            
            # Apply aggregation if requested
            if aggregation != "none":
                return self._aggregate_metrics(df, aggregation)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving performance metrics: {e}")
            raise
    
    def _aggregate_metrics(
        self,
        df: pd.DataFrame,
        aggregation: str
    ) -> pd.DataFrame:
        """Aggregate performance metrics."""
        try:
            # Expand metrics dictionary into columns
            metrics_df = pd.json_normalize(df['metrics'])
            df = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Define aggregation frequency
            freq_map = {
                'daily': 'D',
                'weekly': 'W',
                'monthly': 'M'
            }
            freq = freq_map.get(aggregation)
            
            if not freq:
                raise ValueError(f"Invalid aggregation method: {aggregation}")
            
            # Define aggregation functions for different metric types
            agg_funcs = {
                'pnl': 'sum',
                'win_rate': 'mean',
                'sharpe_ratio': 'last',
                'max_drawdown': 'min',
                'trade_count': 'sum',
                'position_count': 'mean'
            }
            
            # Perform aggregation
            return df.resample(freq).agg(agg_funcs)
            
        except Exception as e:
            self.logger.error(f"Error aggregating metrics: {e}")
            raise
    
    async def store_model_state(
        self,
        model_name: str,
        state: bytes,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Store ML model state.
        
        Args:
            model_name: Name of the model
            state: Serialized model state
            metadata: Optional model metadata
        """
        try:
            timestamp = datetime.now()
            
            if self.config.use_postgres:
                async with self.pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO model_states
                        (timestamp, model_name, state, metadata)
                        VALUES ($1, $2, $3, $4)
                        """,
                        timestamp,
                        model_name,
                        state,
                        json.dumps(metadata or {})
                    )
            else:
                async with self._get_connection() as conn:
                    await conn.execute(
                        """
                        INSERT INTO model_states
                        (timestamp, model_name, state, metadata)
                        VALUES (?, ?, ?, ?)
                        """,
                        (timestamp.timestamp(), model_name, state, 
                         json.dumps(metadata or {}))
                    )
                    await conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing model state: {e}")
            raise
    
    async def get_model_state(
        self,
        model_name: str,
        timestamp: Optional[datetime] = None
    ) -> Tuple[bytes, Dict]:
        """
        Retrieve ML model state.
        
        Args:
            model_name: Name of the model
            timestamp: Optional specific timestamp
            
        Returns:
            Tuple of (model_state, metadata)
        """
        try:
            query = """
                SELECT state, metadata 
                FROM model_states 
                WHERE model_name = ?
            """
            params = [model_name]
            
            if timestamp:
                query += " AND timestamp <= ?"
                params.append(timestamp.timestamp())
            
            query += " ORDER BY timestamp DESC LIMIT 1"
            
            async with self._get_connection() as conn:
                cursor = await conn.execute(query, params)
                row = await cursor.fetchone()
            
            if not row:
                raise ValueError(f"No model state found for {model_name}")
            
            return row[0], json.loads(row[1])
            
        except Exception as e:
            self.logger.error(f"Error retrieving model state: {e}")
            raise
    
    async def aggregate_trade_stats(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        grouping: str = "daily"
    ) -> pd.DataFrame:
        """
        Aggregate trade statistics.
        
        Args:
            symbol: Optional symbol filter
            start_time: Start time
            end_time: End time
            grouping: Grouping period ('daily', 'weekly', 'monthly')
            
        Returns:
            Aggregated trade statistics
        """
        try:
            # Get raw trade data
            trades_df = await self.get_trades(symbol, start_time, end_time)
            
            if trades_df.empty:
                return pd.DataFrame()
            
            # Set timestamp as index
            trades_df.set_index('timestamp', inplace=True)
            
            # Define grouping frequency
            freq_map = {
                'daily': 'D',
                'weekly': 'W',
                'monthly': 'M'
            }
            freq = freq_map.get(grouping, 'D')
            
            # Calculate aggregations
            agg_df = trades_df.resample(freq).agg({
                'quantity': 'sum',
                'pnl': 'sum',
                'fees': 'sum'
            })
            
            # Calculate additional metrics
            trades_grouped = trades_df.groupby(pd.Grouper(freq=freq))
            
            agg_df['trade_count'] = trades_grouped.size()
            agg_df['win_rate'] = trades_grouped.apply(
                lambda x: (x['pnl'] > 0).mean()
            )
            agg_df['avg_trade_size'] = trades_grouped['quantity'].mean()
            agg_df['avg_pnl_per_trade'] = agg_df['pnl'] / agg_df['trade_count']
            
            return agg_df
            
        except Exception as e:
            self.logger.error(f"Error aggregating trade stats: {e}")
            raise
    
    async def get_system_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get system performance metrics.
        
        Args:
            start_time: Start time
            end_time: End time
            metrics: List of specific metrics to retrieve
            
        Returns:
            System metrics DataFrame
        """
        try:
            # Get performance metrics
            perf_metrics = await self.get_performance_metrics(
                start_time=start_time,
                end_time=end_time
            )
            
            # Get trade statistics
            trade_stats = await self.aggregate_trade_stats(
                start_time=start_time,
                end_time=end_time,
                grouping='daily'
            )
            
            # Combine metrics
            combined_df = pd.merge(
                perf_metrics,
                trade_stats,
                left_index=True,
                right_index=True,
                how='outer'
            )
            
            # Filter specific metrics if requested
            if metrics:
                combined_df = combined_df[metrics]
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error retrieving system metrics: {e}")
            raise
    
    async def cleanup_old_data(
        self,
        older_than: datetime,
        categories: Optional[List[DataCategory]] = None
    ) -> None:
        """
        Clean up old data.
        
        Args:
            older_than: Timestamp threshold
            categories: Optional list of data categories to clean
        """
        try:
            if not categories:
                categories = list(DataCategory)
            
            for category in categories:
                table_name = f"{category.value}"
                
                async with self._get_connection() as conn:
                    await conn.execute(
                        f"DELETE FROM {table_name} WHERE timestamp < ?",
                        (older_than.timestamp(),)
                    )
                    await conn.commit()
            
            # Optimize database
            await self._optimize_database()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            raise
    
    async def _optimize_database(self) -> None:
        """Optimize database structure and indices."""
        try:
            async with self._get_connection() as conn:
                await conn.execute("VACUUM")
                await conn.execute("ANALYZE")
            
        except Exception as e:
            self.logger.error(f"Error optimizing database: {e}")
            raise