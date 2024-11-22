import pytest
import asyncio
from TradingRL.src.analysis.event_manager import EventType
from typing import Dict, List, Any


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_event_propagation(event_manager, data_fetcher, market_data):
    """Test event propagation through the system."""
    events_received = []

    async def test_handler(event):
        events_received.append(event)
        print(f"Received event: {event.type}, data: {event.data}")  # Debug print

    # Subscribe to events
    event_manager.subscribe(
        test_handler, [EventType.MARKET_DATA, EventType.PRICE_UPDATE]
    )

    # Set test market data and event manager
    data_fetcher.test_market_data = market_data
    data_fetcher.event_manager = event_manager

    try:
        # Simulate market data update with timeout
        async with asyncio.timeout(5):
            await data_fetcher._fetch_latest()

        # Wait for event processing with timeout
        async with asyncio.timeout(2):
            # Wait a bit longer for event processing
            for _ in range(5):  # Reduced retries
                if events_received:
                    break
                await asyncio.sleep(0.1)  # Reduced sleep time

        assert len(events_received) > 0, "No events were received"

        # Verify event data
        event = events_received[0]
        assert event.type == EventType.MARKET_DATA
        assert "symbol" in event.data
        assert event.data["symbol"] == "BTC/USDT"
        assert "data" in event.data
        assert isinstance(event.data["data"], dict)

    except asyncio.TimeoutError:
        pytest.fail(
            f"Event propagation test timed out. Events received: {len(events_received)}"
        )
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")
