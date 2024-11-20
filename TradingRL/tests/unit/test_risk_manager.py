import pytest
import asyncio
from TradingRL.src.core.risk_manager import RiskManager


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_risk_manager_position_sizing(test_config):
    """Test position sizing calculation."""
    risk_manager = RiskManager(config=test_config.get("risk"))

    async with asyncio.timeout(2):
        position_size = await risk_manager.calculate_position_size(
            amount=50000.0,
            risk_factor=0.8,
        )

    assert 0 < position_size <= 1.0
    assert position_size <= risk_manager.config["max_position_size"]


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_risk_limits(test_config):
    """Test risk limit checks."""
    risk_manager = RiskManager(config=test_config.get("risk"))

    async with asyncio.timeout(2):
        within_limits = await risk_manager.check_risk_limits(
            position_size=0.5, current_drawdown=0.05
        )
        assert within_limits is True

        # Test exceeding limits
        exceeded_limits = await risk_manager.check_risk_limits(
            position_size=1.5, current_drawdown=0.15
        )
        assert exceeded_limits is False
