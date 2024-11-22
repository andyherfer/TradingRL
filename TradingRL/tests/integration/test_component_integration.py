import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from TradingRL.src.core.trader import Trader
from TradingRL.src.core.risk_manager import RiskManager
from TradingRL.src.analysis.market_analyzer import MarketAnalyzer, MarketRegime
from TradingRL.src.analysis.event_manager import EventManager, EventType
from TradingRL.tests.mocks.mock_data_fetcher import MockDataFetcher
from TradingRL.tests.mocks.mock_trader import MockTrader
import os
import shutil


@pytest.fixture(autouse=True)
def setup_test_dirs():
    """Create and cleanup test directories."""
    os.makedirs("test_models", exist_ok=True)
    os.makedirs("test_logs", exist_ok=True)
    yield
    shutil.rmtree("test_models", ignore_errors=True)
    shutil.rmtree("test_logs", ignore_errors=True)


@pytest.fixture
def market_data():
    """Create test market data."""
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="1min")
    data = pd.DataFrame(
        {
            "open": np.linspace(50000, 55000, 1000) + np.random.normal(0, 100, 1000),
            "high": np.linspace(50200, 55200, 1000) + np.random.normal(0, 100, 1000),
            "low": np.linspace(49800, 54800, 1000) + np.random.normal(0, 100, 1000),
            "close": np.linspace(50000, 55000, 1000) + np.random.normal(0, 100, 1000),
            "volume": np.random.random(1000) * 100,
        },
        index=dates,
    )
    return {"BTC/USDT": data}


@pytest.mark.asyncio
async def test_trader_risk_manager_integration(market_data):
    """Test integration between Trader and RiskManager components."""
    trader = MockTrader(model_dir="test_models")
    risk_manager = RiskManager(
        config={
            "max_position_size": 0.1,
            "max_drawdown": 0.02,
            "max_leverage": 1.0,
            "daily_loss_limit": 100,
        }
    )

    try:
        for i in range(5):
            state = np.random.random(50)
            action, confidence = await trader.predict_action(state)

            position_size = await risk_manager.calculate_position_size(
                amount=10000.0, risk_factor=confidence
            )

            assert position_size <= risk_manager.config["max_position_size"]
            assert position_size > 0

            risk_check = await risk_manager.check_risk_limits(
                position_size=position_size, current_drawdown=0.01
            )
            assert risk_check is True

            risk_check = await risk_manager.check_risk_limits(
                position_size=0.5,
                current_drawdown=0.03,
            )
            assert risk_check is False

    finally:
        await trader.close()


@pytest.mark.asyncio
async def test_full_component_interaction(market_data):
    """Test interaction between Trader, RiskManager, and MarketAnalyzer."""
    event_manager = EventManager()
    trader = MockTrader(model_dir="test_models")
    risk_manager = RiskManager(
        config={
            "max_position_size": 0.1,
            "max_drawdown": 0.02,
            "max_leverage": 1.0,
            "daily_loss_limit": 100,
        }
    )
    market_analyzer = MarketAnalyzer(event_manager=event_manager)

    decisions = []

    async def record_decision(decision):
        decisions.append(decision)

    try:
        await market_analyzer.start()

        symbol = "BTC/USDT"
        data = market_data[symbol]

        for i in range(10):
            await market_analyzer.update_market_state(symbol, data.iloc[i : i + 100])
            market_state = market_analyzer.current_state.get(symbol, {})

            base_state = np.random.random(50)
            market_features = np.array(
                [
                    list(MarketRegime).index(
                        market_state.get("regime", MarketRegime.UNDEFINED)
                    ),
                    market_state.get("volatility", 0),
                    market_state.get("trend_strength", 0),
                ]
            )
            state = np.concatenate([base_state, market_features])

            action, confidence = await trader.predict_action(state)

            position_size = await risk_manager.calculate_position_size(
                amount=10000.0, risk_factor=confidence
            )
            risk_approved = await risk_manager.check_risk_limits(
                position_size=position_size, current_drawdown=0.01
            )

            await record_decision(
                {
                    "timestamp": datetime.now(),
                    "action": action,
                    "confidence": confidence,
                    "position_size": position_size,
                    "risk_approved": risk_approved,
                    "market_regime": market_state.get("regime", MarketRegime.UNDEFINED),
                    "volatility": market_state.get("volatility", 0),
                }
            )

            await asyncio.sleep(0.1)

        assert len(decisions) == 10
        for decision in decisions:
            assert "action" in decision
            assert "confidence" in decision
            assert "position_size" in decision
            assert "risk_approved" in decision
            assert "market_regime" in decision
            assert decision["position_size"] <= risk_manager.config["max_position_size"]

    finally:
        await trader.close()
        await market_analyzer.stop()
