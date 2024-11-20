import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from TradingRL.tests.mocks.mock_trader import MockTrader
from TradingRL.src.strategy.rl_strategy import RLStrategy
from TradingRL.src.strategy.system_mode import SystemMode
from TradingRL.src.analysis.market_analyzer import MarketRegime
from TradingRL.src.strategy.base_strategy import SignalType, StrategyState
from TradingRL.tests.mocks.mock_data_fetcher import MockDataFetcher
import asyncio
from TradingRL.tests.mocks.mock_market_analyzer import MockMarketAnalyzer


@pytest.fixture
async def strategy(event_manager, risk_manager, portfolio_manager):
    """Create test strategy instance."""
    trader = MockTrader()
    market_analyzer = MockMarketAnalyzer()
    strategy = RLStrategy(
        name="test_strategy",
        trader=trader,
        symbols=["BTC/USDT"],
        event_manager=event_manager,
        risk_manager=risk_manager,
        portfolio_manager=portfolio_manager,
        market_analyzer=market_analyzer,
        mode=SystemMode.TEST,
    )
    return strategy


@pytest.mark.asyncio
async def test_feature_calculation_with_empty_data(strategy):
    """Test feature calculation with empty data."""
    # Create empty DataFrame with correct columns
    empty_data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # Test each feature calculation method with empty data
    price_features = strategy._calculate_price_features(empty_data)
    assert isinstance(price_features, np.ndarray)
    assert len(price_features) == 5
    assert np.all(price_features == 0)  # Should return zeros for empty data

    volume_features = strategy._calculate_volume_features(empty_data)
    assert isinstance(volume_features, np.ndarray)
    assert len(volume_features) == 5
    assert np.all(volume_features == 0)


@pytest.mark.asyncio
async def test_feature_calculation_with_invalid_data(strategy):
    """Test feature calculation with invalid data."""
    # Create data with missing values
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1min")
    invalid_data = pd.DataFrame(
        {
            "open": np.random.random(100),
            "high": np.random.random(100),
            "low": np.random.random(100),
            "close": [
                np.nan if i % 10 == 0 else x
                for i, x in enumerate(np.random.random(100))
            ],
            "volume": np.random.random(100),
        },
        index=dates,
    )

    # Should handle NaN values gracefully
    features = strategy._calculate_price_features(invalid_data)
    assert not np.isnan(features).any()


@pytest.mark.asyncio
async def test_state_normalization_edge_cases(strategy):
    """Test state normalization with edge cases."""
    # Test with zeros
    zero_state = np.zeros(10)
    norm_zeros = strategy._normalize_state(zero_state)
    assert np.allclose(norm_zeros, 0)

    # Test with infinities
    inf_state = np.array([np.inf, -np.inf] + [1] * 8)
    norm_inf = strategy._normalize_state(inf_state)
    assert not np.isinf(norm_inf).any()
    assert np.all(norm_inf >= -10) and np.all(norm_inf <= 10)

    # Test with NaN values
    nan_state = np.array([np.nan] * 5 + [1] * 5)
    norm_nan = strategy._normalize_state(nan_state)
    assert not np.isnan(norm_nan).any()


@pytest.mark.asyncio
async def test_action_to_signal_conversion(strategy, market_data):
    """Test conversion of actions to trading signals."""
    current_data = market_data["BTC/USDT"].iloc[-1]

    # Mock market analyzer state
    strategy.market_analyzer.current_state = {
        "BTC/USDT": {
            "regime": MarketRegime.TRENDING_UP,
            "volatility": 0.1,
            "trend_strength": 0.8,
            "support_levels": [45000.0, 44000.0],
            "resistance_levels": [52000.0, 53000.0],
            "liquidity": 1.0,
            "timestamp": datetime.now(),
            "metadata": {},
        }
    }

    # Add a mock position for testing exit signals
    strategy.positions = {
        "BTC/USDT": {
            "quantity": 0.5,
            "entry_price": 50000.0,
            "side": "long",
        }
    }

    # Test each action type
    actions = {
        0: SignalType.NO_SIGNAL,
        1: SignalType.LONG,
        2: SignalType.SHORT,
        3: SignalType.EXIT_LONG,
        4: SignalType.EXIT_SHORT,
    }

    for action, expected_type in actions.items():
        signal = await strategy._action_to_signal(
            symbol="BTC/USDT", action=action, confidence=0.8, current_data=current_data
        )

        if action == 0:
            assert signal is None
        else:
            assert signal is not None
            assert signal.type == expected_type
            assert signal.symbol == "BTC/USDT"
            assert signal.confidence == 0.8
            assert isinstance(signal.price, float)
            assert isinstance(signal.size, float)
            assert signal.size > 0.0  # Size should always be positive

            # Verify size based on signal type
            if signal.type in [SignalType.LONG, SignalType.SHORT]:
                assert signal.size <= strategy.config.max_position_size
            else:  # Exit signals
                assert signal.size == strategy.positions["BTC/USDT"]["quantity"]


@pytest.mark.asyncio
async def test_position_size_calculation(strategy):
    """Test position size calculation logic."""
    # Test with no history
    size = await strategy._calculate_position_size("BTC/USDT", 50000.0, 0.8)
    assert isinstance(size, float)
    assert 0 < size <= 1.0

    # Add some mock trade history
    strategy.position_history["BTC/USDT"] = [
        {"pnl": 100, "roi": 0.05},
        {"pnl": -50, "roi": -0.02},
        {"pnl": 200, "roi": 0.08},
    ]

    # Test with history
    size = await strategy._calculate_position_size("BTC/USDT", 50000.0, 0.9)
    assert isinstance(size, float)
    assert 0 < size <= 1.0

    # Test with different confidence levels
    sizes = []
    for conf in [0.1, 0.5, 0.9]:
        size = await strategy._calculate_position_size("BTC/USDT", 50000.0, conf)
        sizes.append(size)

    # Higher confidence should generally mean larger position sizes
    assert sizes[0] <= sizes[1] <= sizes[2]


@pytest.mark.asyncio
async def test_strategy_warmup(strategy, market_data):
    """Test strategy warmup process."""
    # Set test data
    strategy.data_fetcher = MockDataFetcher(market_data)
    strategy.test_market_data = market_data

    # Initialize data buffer
    for symbol, data in market_data.items():
        strategy.data_buffer[symbol] = data.copy()

    try:
        # Start strategy
        await strategy.start()

        # Verify initialization
        assert strategy.state == StrategyState.ACTIVE
        assert len(strategy.state_buffer["BTC/USDT"]) > 0
        assert hasattr(strategy, "state_mean")
        assert hasattr(strategy, "state_std")

    finally:
        # Ensure cleanup
        await strategy.stop()
        # Cancel any pending tasks
        for task in asyncio.all_tasks():
            if not task.done() and task != asyncio.current_task():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


@pytest.mark.asyncio
async def test_strategy_error_handling(strategy, market_data):
    """Test strategy error handling."""
    # Test with invalid market data
    invalid_data = {
        "BTC/USDT": pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    }
    signals = await strategy.generate_signals(invalid_data)
    assert len(signals) == 0  # Should handle empty data gracefully

    # Test with missing symbol (expected error log)
    unknown_symbol_data = {"ETH/USDT": market_data["BTC/USDT"]}
    signals = await strategy.generate_signals(unknown_symbol_data)
    assert len(signals) == 0  # Should handle unknown symbol gracefully

    # Test with simulated trader error (expected error log)
    strategy.trader.force_error = True
    signals = await strategy.generate_signals(market_data)
    assert len(signals) == 0  # Should handle trader errors gracefully
    strategy.trader.force_error = False  # Reset for other tests


@pytest.mark.asyncio
async def test_feature_calculation_error_handling(strategy):
    """Test error handling in feature calculations."""
    # Create DataFrame with missing required columns
    invalid_data = pd.DataFrame(
        {
            "price": np.random.random(100),  # Wrong column name
            "vol": np.random.random(100),  # Wrong column name
        }
    )

    # Should handle missing columns gracefully
    price_features = strategy._calculate_price_features(invalid_data)
    assert isinstance(price_features, np.ndarray)
    assert len(price_features) == 5
    assert np.all(price_features == 0)  # Should return zeros on error

    volume_features = strategy._calculate_volume_features(invalid_data)
    assert isinstance(volume_features, np.ndarray)
    assert len(volume_features) == 5
    assert np.all(volume_features == 0)  # Should return zeros on error

    technical_features = strategy._calculate_technical_features(invalid_data)
    assert isinstance(technical_features, np.ndarray)
    assert len(technical_features) == 5
    assert np.all(technical_features == 0)  # Should return zeros on error
