import pytest
import asyncio
from datetime import datetime, timedelta
from TradingRL.src.analysis.event_manager import EventType
from TradingRL.src.strategy.system_mode import SystemMode
import uuid
from TradingRL.src.core.order_manager import (
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
)


@pytest.mark.asyncio
async def test_full_trading_cycle(
    event_manager,
    data_fetcher,
    market_analyzer,
    risk_manager,
    portfolio_manager,
    order_manager,
    market_data,
):
    """Test complete trading cycle from signal generation to order execution."""
    events_received = []

    async def event_handler(event):
        events_received.append(event)
        print(f"Received event: {event.type}")  # Debug print

    # Subscribe to all relevant events
    event_types = [
        EventType.MARKET_DATA,
        EventType.MARKET_REGIME_CHANGE,
        EventType.ORDER_UPDATE,
        EventType.TRADE_UPDATE,
        EventType.POSITION_UPDATE,
        EventType.RISK_UPDATE,
        EventType.ORDER_FILLED,
    ]
    event_manager.subscribe(event_handler, event_types)

    components = [data_fetcher, market_analyzer, portfolio_manager, order_manager]

    try:
        # Start all components
        for component in components:
            await component.start()

        # Set test market data and ensure event manager is set
        data_fetcher.test_market_data = market_data
        data_fetcher.event_manager = event_manager  # Ensure event manager is set

        # Trigger market data update and wait for processing
        await data_fetcher._fetch_latest()
        await asyncio.sleep(1)  # Increased wait time for event processing

        # Create and submit a test order
        test_order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
            params={},
            fills=[],
        )

        # Add order to active orders
        order_manager.active_orders[test_order.id] = test_order

        # Process the order fill
        await order_manager._process_fill(
            test_order,
            {
                "quantity": 0.1,
                "price": 50000.0,
                "commission": 0.0,
            },
        )

        # Wait for event processing
        await asyncio.sleep(1)

        # Print received events for debugging
        print("\nReceived events:")
        for event in events_received:
            print(f"- {event.type}")

        # Verify event sequence with more detailed error message
        expected_events = {
            EventType.MARKET_DATA: False,
            EventType.ORDER_FILLED: False,
            EventType.POSITION_UPDATE: False,
        }

        for event in events_received:
            expected_events[event.type] = True

        # Verify complete cycle with detailed error message
        missing_events = [
            event_type
            for event_type, received in expected_events.items()
            if not received
        ]
        assert not missing_events, (
            f"Missing expected events: {missing_events}. "
            f"Received events: {[e.type for e in events_received]}"
        )

    finally:
        # Stop all components in reverse order
        for component in reversed(components):
            try:
                await component.stop()
            except Exception as e:
                print(f"Error stopping component: {e}")

        # Cancel any remaining tasks
        for task in asyncio.all_tasks():
            if not task.done() and task != asyncio.current_task():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


@pytest.mark.asyncio
async def test_risk_management_integration(
    event_manager, risk_manager, portfolio_manager, market_data
):
    """Test risk management integration with portfolio management."""
    # Setup test position
    position = {
        "symbol": "BTC/USDT",
        "size": 0.1,
        "entry_price": 50000.0,
        "current_price": 49000.0,
        "unrealized_pnl": -100.0,
        "realized_pnl": 0.0,
    }

    # Test position sizing limits
    size = await risk_manager.calculate_position_size(amount=100000.0, risk_factor=0.8)
    assert size <= risk_manager.config["max_position_size"]

    # Test risk limits
    within_limits = await risk_manager.check_risk_limits(
        position_size=size, current_drawdown=0.05
    )
    assert within_limits is True

    # Test emergency risk handling
    emergency_event = {
        "type": EventType.RISK_UPDATE,
        "data": {"drawdown": 0.25, "positions": [position]},  # Exceeds max_drawdown
    }
    await portfolio_manager.handle_risk_event(emergency_event)


@pytest.mark.asyncio
async def test_system_recovery(
    event_manager,
    data_fetcher,
    market_analyzer,
    portfolio_manager,
    order_manager,
    market_data,
):
    """Test system recovery after simulated failures."""

    async def simulate_component_failure(data_fetcher):
        """Simulate component failure and recovery."""
        await data_fetcher.stop()
        await asyncio.sleep(1)
        await data_fetcher.start()

    try:
        # Start system
        await data_fetcher.start()
        await market_analyzer.start()
        await portfolio_manager.start()

        # Set test data
        data_fetcher.test_market_data = market_data

        # Initial data fetch
        await data_fetcher._fetch_latest()
        await asyncio.sleep(0.1)

        # Simulate component failure and recovery
        await simulate_component_failure(data_fetcher)

        # Wait for recovery
        await asyncio.sleep(1)

        # Verify system state after recovery
        assert data_fetcher._running
        latest_data = await data_fetcher.get_latest_data()
        assert isinstance(latest_data, dict)
        assert len(latest_data) > 0
        assert "BTC/USDT" in latest_data

    finally:
        await data_fetcher.stop()
        await market_analyzer.stop()
        await portfolio_manager.stop()
