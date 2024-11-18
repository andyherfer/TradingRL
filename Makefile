.PHONY: setup setup-full train clean test lint

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

setup-full: setup
	mkdir -p data/market_data
	mkdir -p data/models
	mkdir -p logs
	mkdir -p config
#	test -f config/development_config.yaml || cp config/development_config.yaml.example config/development_config.yaml
#	test -f config/.env || cp config/.env.example config/.env

train:
	python scripts/train_model.py --config config --symbol BTC/USDT

clean:
	rm -rf data/models/*
	rm -rf logs/*
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

test:
	pytest tests/

lint:
	black .
	flake8 .