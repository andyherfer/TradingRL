# TradingRL

A cryptocurrency trading system using deep reinforcement learning, designed for automated trading strategies with comprehensive risk management and real-time market analysis.
(Built Using Claude sonnet-3.6 and Cursor, so replicate at your own risk)
## Overview

TradingRL is an advanced cryptocurrency trading platform that leverages deep reinforcement learning to make intelligent trading decisions. The system is designed to operate autonomously while maintaining strict risk management protocols.

Key Features:
- Deep Reinforcement Learning based decision making
- Real-time market regime detection and analysis
- Smart order execution with slippage prevention
- Multi-level risk management system
- Comprehensive performance analytics and monitoring
- Support for multiple exchanges and trading pairs
- Backtesting capabilities

## System Architecture

```
Trading System
├── Core Components
│   ├── DataFetcher: Real-time market data collection
│   ├── MarketAnalyzer: Technical analysis and feature engineering
│   ├── RLStrategy: Deep RL-based trading decisions
│   ├── RiskManager: Position and portfolio risk management
│   └── OrderExecutor: Smart order execution and monitoring
│
├── Event System
│   └── EventManager: Asynchronous component communication
│
└── Monitoring
    ├── Performance Tracker: Real-time P&L and metrics
    └── System Monitor: Health checks and diagnostics
```

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for training)
- Redis (for real-time data processing)
- PostgreSQL (for data storage)
- TA-Lib

### Installing Dependencies

1. **TA-Lib Installation**

For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y ta-lib
```

For macOS:
```bash
brew install ta-lib
```

For Windows:
- Download ta-lib from [here](http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip)
- Extract and follow the installation instructions

2. **Python Dependencies**
```bash
pip install -r requirements.txt
```

## Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/TradingRL.git
cd TradingRL
```

2. **Set up configuration**
```bash
cp config/development_config.yaml.example config/development_config.yaml
cp config/.env.example .env
```

3. **Configure your environment**

Edit `.env`:
```env
EXCHANGE_API_KEY=your_exchange_api_key
EXCHANGE_API_SECRET=your_exchange_api_secret

```

4. **Train the model**
```bash
make train
```

5. **Run the trading system**
```bash
make run
```

## Configuration

### System Configuration (development_config.yaml)
```yaml
system:
  environment: development  # development/production
  mode: paper              # paper/live
  log_level: info
  max_positions: 3
  base_currency: USDT

trading:
  symbols:
    - BTC/USDT
    - ETH/USDT
  timeframes:
    - 1m
    - 5m
    - 1h
  
risk_management:
  max_position_size: 1000  # in USDT
  stop_loss_pct: 2.0
  take_profit_pct: 4.0
  max_drawdown_pct: 15.0
```

## Features in Detail

### Reinforcement Learning Model
- Custom DRL agent based on PPO (Proximal Policy Optimization)
- State space: Technical indicators, market microstructure features
- Action space: Discrete actions (BUY, SELL, HOLD)
- Reward function: Risk-adjusted returns (Sharpe ratio)

### Risk Management
- Position sizing based on Kelly Criterion
- Dynamic stop-loss and take-profit levels
- Portfolio-level risk constraints
- Drawdown protection

### Market Analysis
- Real-time market regime detection
- Volume profile analysis
- Order book imbalance monitoring
- Multi-timeframe analysis

## Development

### Running Tests
```bash
# Run all tests
make test

# Run specific test suite
python -m pytest tests/unit/test_risk_manager.py
```

### Code Style
We use Black for formatting and Flake8 for linting:
```bash
make lint
make format
```

## Monitoring

Access the monitoring dashboard at `http://localhost:8501` when running the system.

Metrics available:
- Real-time P&L
- Position statistics
- Risk metrics
- System health
- Model performance

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

