.PHONY: setup setup-full train clean test lint run help

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

test:
	pytest tests/

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
	python -m TradingRL.main --config config --env $$ENV --mode $$MODE

help: ## Show help message
	@echo "Usage: make [target] [env=<environment>] [mode=<mode>]"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*##"; printf "\033[36m\033[0m"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
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
	@echo "  make run env=production mode=live"
