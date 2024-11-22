import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml

from TradingRL.src.core.config_manager import ConfigManager
from TradingRL.src.analysis.event_manager import EventManager, EventType
from TradingRL.src.core.trader import Trader
from TradingRL.src.core.risk_manager import RiskManager
from TradingRL.src.core.portfolio_manager import PortfolioManager
from TradingRL.src.analysis.market_analyzer import MarketAnalyzer
from TradingRL.src.data.data_fetcher import DataFetcher
from TradingRL.src.data.data_fetcher_adapter import DataFetcherAdapter
from TradingRL.src.core.order_manager import OrderManager


@pytest.fixture
def test_config():
    """Create test configuration."""
    config_dir = Path("config")
    config_dir.mkdir(parents=True, exist_ok=True)

    # Use a temporary config file name for tests
    temp_config_name = f"test_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    temp_config_path = config_dir / temp_config_name

    # Create test config
    test_config = {
        "exchange": {
            "name": "binance_test",
            "api_key": "test_key",
            "api_secret": "test_secret",
            "testnet": True,
            "rate_limit": 1200,
            "timeout": 30000,
        },
        "risk": {
            "max_position_size": 1.0,
            "max_drawdown": 0.1,
            "position_sizing_method": "kelly",
            "risk_per_trade": 0.02,
            "max_leverage": 1.0,
        },
        "trading": {
            "symbols": ["BTC/USDT"],
            "timeframes": ["1m", "5m", "15m"],
            "initial_capital": 10000.0,
            "base_currency": "USDT",
        },
    }

    with open(temp_config_path, "w") as f:
        yaml.dump(test_config, f)

    # Create ConfigManager with the temporary config file
    config_manager = ConfigManager(
        base_path=str(config_dir),
        env="test",
        mode="test",
        config_file=temp_config_name,  # Use the temporary config file
    )

    yield config_manager

    # Cleanup: remove temporary config file
    if temp_config_path.exists():
        temp_config_path.unlink()


@pytest.fixture
async def event_manager():
    manager = EventManager()
    try:
        await manager.start()
        await asyncio.sleep(0.1)
        yield manager
    finally:
        await manager.stop()


@pytest.fixture
def market_data():
    """Generate test market data."""
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="1min")
    data = pd.DataFrame(
        {
            "open": np.random.random(1000) * 1000 + 50000,
            "high": np.random.random(1000) * 1000 + 50000,
            "low": np.random.random(1000) * 1000 + 50000,
            "close": np.random.random(1000) * 1000 + 50000,
            "volume": np.random.random(1000) * 100,
        },
        index=dates,
    )
    return {"BTC/USDT": data}


@pytest.fixture
async def data_fetcher(market_data, event_manager):
    """Create test data fetcher."""
    fetcher = DataFetcher(api_key="test", api_secret="test", symbols=["BTC/USDT"])
    adapter = DataFetcherAdapter(fetcher)
    adapter.event_manager = event_manager
    try:
        async with asyncio.timeout(5):
            await adapter.start()
        yield adapter
    finally:
        async with asyncio.timeout(5):
            await adapter.stop()


@pytest.fixture
async def market_analyzer(event_manager):
    analyzer = MarketAnalyzer(event_manager=event_manager)
    return analyzer


@pytest.fixture
async def risk_manager(test_config):
    return RiskManager(config=test_config.get("risk"))


@pytest.fixture
async def order_manager(event_manager, risk_manager):
    """Create test order manager."""
    return OrderManager(
        api_key="test_key",
        api_secret="test_secret",
        event_manager=event_manager,
        risk_manager=risk_manager,
    )


@pytest.fixture
async def portfolio_manager(event_manager, risk_manager, order_manager):
    """Create test portfolio manager."""
    return PortfolioManager(
        event_manager=event_manager,
        risk_manager=risk_manager,
        order_manager=order_manager,
        initial_capital=10000.0,
    )
