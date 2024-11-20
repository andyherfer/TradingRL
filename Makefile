.PHONY: setup setup-full train clean test test-system lint run help verify-all verify-exchange verify-risk verify-monitoring setup-credentials setup-testnet

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && pip install -e .
	@echo "Setup complete. Remember to activate the virtual environment with: source .venv/bin/activate"

setup-full: setup
	mkdir -p data/market_data
	mkdir -p data/models
	mkdir -p logs
	mkdir -p config

train:
	python -m TradingRL.scripts.train_model --config config --symbol BTC/USDT

clean:
	rm -rf data/models/*
	rm -rf logs/*
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf *.egg-info
	rm -rf build/ dist/
	rm -rf .pytest_cache
	rm -rf .coverage

test:
	pytest tests/

test-system: ## Run system tests with detailed output
	@echo "Running system tests..."
	@echo "1. Running unit tests..."
	pytest -v -s --log-cli-level=DEBUG TradingRL/tests/unit/ || exit 1
	@echo "\n2. Running integration tests..."
	pytest -v -s --log-cli-level=DEBUG TradingRL/tests/integration/ || exit 1
	@echo "\n3. Generating coverage report..."
	pytest --cov=TradingRL TradingRL/tests/ --cov-report=term-missing
	@echo "\nTest execution complete."

test-quick: ## Run tests without coverage and minimal output
	pytest -v TradingRL/tests/

lint:
	black .
	flake8 .

run: ## Run the trading system
	@echo "Starting trading system..."
	@if [ -z "$(env)" ]; then \
		echo "Environment not specified. Using development as default."; \
		ENV="development"; \
	else \
		ENV="$(env)"; \
	fi
	@if [ -z "$(mode)" ]; then \
		echo "Mode not specified. Using paper as default."; \
		MODE="paper"; \
	else \
		MODE="$(mode)"; \
	fi
	@echo "Ensuring virtual environment is activated..."
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "Virtual environment not activated. Please run: source .venv/bin/activate"; \
		exit 1; \
	fi
	PYTHONPATH=. python -m TradingRL.main --config config --env="$$ENV" --mode="$$MODE"

help: ## Show help message
	@echo "Usage: make [target] [env=<environment>] [mode=<mode>]"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*##"; printf "\033[36m\033[0m"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""
	@echo "Live Trading Commands:"
	@echo "  verify-exchange  - Verify exchange connectivity"
	@echo "  verify-risk     - Verify risk management"
	@echo "  verify-monitoring - Verify monitoring systems"
	@echo "  verify-all      - Run all verifications"
	@echo "  run-live        - Run live trading with verifications"
	@echo ""
	@echo "Test targets:"
	@echo "  test         - Run all tests"
	@echo "  test-system  - Run system tests with detailed output and coverage"
	@echo "  test-quick   - Run tests with minimal output"
	@echo ""
	@echo "Environment options:"
	@echo "  development (default)"
	@echo "  production"
	@echo ""
	@echo "Mode options:"
	@echo "  paper (default)"
	@echo "  live"
	@echo "  backtest"
	@echo ""
	@echo "Example:"
	@echo "  make verify-all"
	@echo "  make run-live"

# Verification targets
init-test-env: ## Initialize test environment
	@echo "Initializing test environment..."
	PYTHONPATH=. python -m TradingRL.scripts.init_test_env

verify-exchange: init-test-env ## Verify exchange connectivity and API functionality
	@echo "Verifying exchange setup..."
	PYTHONPATH=. python -m TradingRL.scripts.verify_exchange

verify-risk: init-test-env ## Verify risk management functionality
	@echo "Verifying risk management..."
	PYTHONPATH=. python -m TradingRL.scripts.verify_risk

verify-monitoring: init-test-env ## Verify monitoring systems
	@echo "Verifying monitoring systems..."
	PYTHONPATH=. python -m TradingRL.scripts.verify_monitoring

setup-credentials: ## Set up trading credentials
	@if [ -z "$$BINANCE_API_KEY" ] || [ -z "$$BINANCE_API_SECRET" ]; then \
		echo "Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables"; \
		exit 1; \
	fi
	@echo "Setting up credentials..."
	PYTHONPATH=. python -m TradingRL.scripts.setup_credentials

setup-testnet: ## Set up Binance testnet credentials
	@echo "Setting up testnet credentials..."
	PYTHONPATH=. python -m TradingRL.scripts.setup_testnet

verify-all: setup-testnet ## Run all verifications for live trading
	@echo "Running all verifications..."
	PYTHONPATH=. python -m TradingRL.scripts.run_verifications

run-live: verify-all ## Run the trading system in live mode with verifications
	@echo "Starting live trading system..."
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "Virtual environment not activated. Please run: source .venv/bin/activate"; \
		exit 1; \
	fi
	PYTHONPATH=. python -m TradingRL.main --config config --env=production --mode=live
