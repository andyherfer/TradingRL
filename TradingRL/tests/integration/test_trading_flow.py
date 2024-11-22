import pytest
import asyncio
from TradingRL.tests.mocks.mock_trader import MockTrader  # Import mock trader
from TradingRL.src.strategy.rl_strategy import RLStrategy
from TradingRL.src.analysis.event_manager import EventType
from TradingRL.src.strategy.system_mode import SystemMode


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_trading_flow(
    event_manager,
    data_fetcher,
    market_analyzer,
    risk_manager,
    portfolio_manager,
    market_data,
):
    """Test complete trading flow."""
    # Initialize mock trader instead of real trader
    trader = MockTrader(
        model_dir="test_models",
        tensorboard_log="test_logs",
        project_name="test_trading",
    )

    # Initialize strategy in test mode
    strategy = RLStrategy(
        name="test_strategy",
        trader=trader,
        symbols=["BTC/USDT"],
        event_manager=event_manager,
        risk_manager=risk_manager,
        portfolio_manager=portfolio_manager,
        market_analyzer=market_analyzer,
        use_wandb=False,  # Disable wandb for tests
        mode=SystemMode.TEST,  # Set test mode
    )

    # Set test data
    strategy.data_fetcher = data_fetcher
    strategy.test_market_data = market_data

    # Initialize data buffer with test data
    for symbol, data in market_data.items():
        strategy.data_buffer[symbol] = data.copy()

    try:
        # Start components with timeout
        async with asyncio.timeout(5):
            await strategy.start()

        # Generate signals with timeout
        async with asyncio.timeout(5):
            signals = await strategy.generate_signals(market_data)

        # Verify signal generation
        assert isinstance(signals, list)

        # Verify trader was trained
        assert trader.trained, "Trader should have been trained during warmup"
        assert len(trader.training_history) > 0, "Training history should not be empty"

        # Verify predictions were made
        assert len(trader.predictions) > 0, "No predictions were made"

        # Test signal processing with timeout
        if signals:
            async with asyncio.timeout(5):
                signal = signals[0]
                position = await portfolio_manager.process_signal(signal)
                assert position is not None

    finally:
        # Ensure cleanup with timeout
        async with asyncio.timeout(5):
            await strategy.stop()


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
    data_fetcher.event_manager = event_manager  # Set event manager

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
