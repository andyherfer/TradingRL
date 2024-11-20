import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from TradingRL.src.analysis.market_analyzer import MarketAnalyzer, MarketRegime
from TradingRL.src.analysis.event_manager import EventType
import asyncio


@pytest.mark.asyncio
async def test_regime_detection(event_manager):
    """Test market regime detection."""
    analyzer = MarketAnalyzer(event_manager=event_manager)

    # Generate test data with clear trends
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1min")

    # Trending up data with low volatility
    trend_up_data = pd.DataFrame(
        {
            "open": np.linspace(50000, 54800, 100),
            "high": np.linspace(50200, 55000, 100),
            "low": np.linspace(49800, 54600, 100),
            "close": np.linspace(50000, 55000, 100),
            "volume": np.random.random(100) * 100,
        },
        index=dates,
    )

    # Ranging data with low volatility
    base_price = 50000
    amplitude = 500  # Reduced amplitude for lower volatility
    x = np.linspace(0, 4 * np.pi, 100)
    ranging_data = pd.DataFrame(
        {
            "open": base_price + amplitude * np.sin(x),
            "high": base_price + amplitude * (1.1 + np.sin(x)),
            "low": base_price + amplitude * (-0.1 + np.sin(x)),
            "close": base_price + amplitude * np.sin(x),
            "volume": np.random.random(100) * 100,
        },
        index=dates,
    )

    # Test regime detection
    up_regime = await analyzer.detect_regime("BTC/USDT", trend_up_data)
    assert up_regime == MarketRegime.TRENDING_UP

    range_regime = await analyzer.detect_regime("BTC/USDT", ranging_data)
    assert range_regime == MarketRegime.RANGING


@pytest.mark.asyncio
async def test_support_resistance_levels(event_manager):
    """Test support and resistance level detection."""
    analyzer = MarketAnalyzer(event_manager=event_manager)

    # Generate data with clear support/resistance
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1min")
    prices = []
    for i in range(100):
        if i < 33:
            prices.append(50000 + np.random.normal(0, 100))
        elif i < 66:
            prices.append(52000 + np.random.normal(0, 100))
        else:
            prices.append(51000 + np.random.normal(0, 100))

    data = pd.DataFrame(
        {
            "high": prices + np.random.normal(0, 50, 100),
            "low": prices - np.random.normal(0, 50, 100),
            "close": prices,
            "volume": np.random.random(100) * 100,
        },
        index=dates,
    )

    levels = await analyzer.find_support_resistance("BTC/USDT", data)

    assert len(levels["support"]) > 0
    assert len(levels["resistance"]) > 0
    assert any(49800 < level < 50200 for level in levels["support"])
    assert any(51800 < level < 52200 for level in levels["resistance"])


@pytest.mark.asyncio
async def test_volatility_calculation(event_manager):
    """Test volatility calculation."""
    analyzer = MarketAnalyzer(event_manager=event_manager)

    # Generate low and high volatility data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1min")

    low_vol_data = pd.DataFrame(
        {
            "close": 50000 + np.random.normal(0, 100, 100),
            "volume": np.random.random(100) * 100,
        },
        index=dates,
    )

    high_vol_data = pd.DataFrame(
        {
            "close": 50000 + np.random.normal(0, 1000, 100),
            "volume": np.random.random(100) * 100,
        },
        index=dates,
    )

    low_vol = await analyzer.calculate_volatility("BTC/USDT", low_vol_data)
    high_vol = await analyzer.calculate_volatility("BTC/USDT", high_vol_data)

    assert low_vol < high_vol


@pytest.mark.asyncio
async def test_market_state_updates(event_manager):
    """Test market state updates and event emission."""
    analyzer = MarketAnalyzer(event_manager=event_manager)
    events_received = []

    async def test_handler(event):
        if event.type == EventType.MARKET_REGIME_CHANGE:
            events_received.append(event)
            print(
                f"Received regime change: {event.data['old_regime']} -> {event.data['new_regime']}"
            )

    event_manager.subscribe(test_handler, [EventType.MARKET_REGIME_CHANGE])

    # Start analyzer
    await analyzer.start()

    try:
        # Generate state change
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1min")

        # First update with ranging data (low volatility)
        ranging_data = pd.DataFrame(
            {
                "open": 50000 + np.random.normal(0, 50, 100),  # Reduced volatility
                "high": 50050 + np.random.normal(0, 50, 100),
                "low": 49950 + np.random.normal(0, 50, 100),
                "close": 50000 + np.random.normal(0, 50, 100),
                "volume": np.random.random(100) * 100,
            },
            index=dates,
        )
        await analyzer.update_market_state("BTC/USDT", ranging_data)

        # Wait for event processing
        await asyncio.sleep(0.2)

        # Then update with strong trend data
        trend_data = pd.DataFrame(
            {
                "open": np.linspace(50000, 55000, 100),  # Clear uptrend
                "high": np.linspace(50200, 55200, 100),
                "low": np.linspace(49800, 54800, 100),
                "close": np.linspace(50000, 55000, 100),
                "volume": np.random.random(100) * 200,  # Higher volume
            },
            index=dates,
        )
        await analyzer.update_market_state("BTC/USDT", trend_data)

        # Wait for event processing
        await asyncio.sleep(0.2)

        # Verify events were emitted
        assert len(events_received) >= 2, (
            f"Expected at least 2 regime change events, got {len(events_received)}. "
            f"Events: {[(e.data['old_regime'], e.data['new_regime']) for e in events_received]}"
        )

    finally:
        await analyzer.stop()
