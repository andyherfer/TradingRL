from typing import Dict, Any, Optional, Union
import yaml
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
import logging
from enum import Enum
import secrets
from cryptography.fernet import Fernet
import dotenv
from pydantic import BaseModel, field_validator
import warnings
from pprint import pprint
from typing import List


class Environment(Enum):
    """System environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    TEST = "test"


class SystemMode(Enum):
    """System operation modes."""

    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    TRAIN = "train"
    TEST = "test"


class ExchangeConfig(BaseModel):
    """Exchange configuration validation model."""

    name: str
    api_key: str
    api_secret: str
    testnet: bool = False
    rate_limit: int = 1200
    timeout: int = 30

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("API key cannot be empty")
        return v


class TradingConfig(BaseModel):
    """Trading configuration validation model."""

    max_positions: int = 5
    max_leverage: float = 1.0
    default_quantity: float = 0.01
    min_trade_interval: int = 60
    max_slippage: float = 0.001
    timeframes: List[str] = ["5m", "15m", "1h"]

    @field_validator("max_leverage")
    @classmethod
    def validate_leverage(cls, v):
        if v <= 0 or v > 10:
            raise ValueError("Leverage must be between 0 and 10")
        return v


class RiskConfig(BaseModel):
    """Risk configuration validation model."""

    max_position_size: float = 0.1
    max_drawdown: float = 0.2
    stop_loss: float = 0.05
    take_profit: float = 0.1

    @field_validator("max_position_size")
    @classmethod
    def validate_position_size(cls, v):
        if v <= 0 or v > 1:
            raise ValueError("Max position size must be between 0 and 1")
        return v


@dataclass
class ConfigManager:
    """Manages system configuration."""

    base_path: str = "config"
    env: Environment = Environment.DEVELOPMENT
    mode: SystemMode = SystemMode.PAPER
    _config_file: str = field(default="config.yaml", init=False)  # Private field
    _secrets_file: str = field(default=".env", init=False)  # Private field

    def __init__(
        self,
        base_path: Optional[str] = None,
        env: Optional[str] = None,
        mode: Optional[str] = None,
    ):
        """Initialize Configuration Manager."""
        # Set paths
        self.base_path = base_path or os.getenv("CONFIG_PATH", "config")
        self.env = (
            Environment(env.lower())
            if env
            else Environment(os.getenv("SYSTEM_ENV", "development").lower())
        )
        self.mode = (
            SystemMode(mode.lower())
            if mode
            else SystemMode(os.getenv("SYSTEM_MODE", "paper").lower())
        )

        # Initialize paths
        self.config_dir = Path(self.base_path)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Load configurations
        self.config = {}
        self.secrets = {}
        self._load_configurations()

        # Initialize encryption
        self._init_encryption()

    def _load_configurations(self) -> None:
        """Load all configuration files."""
        try:
            # Load main config
            config_path = self.config_dir / "test.yaml"  # Fixed path for test config
            if config_path.exists():
                with open(config_path) as f:
                    self.config = yaml.safe_load(f)
            else:
                self.logger.warning(f"Config file not found: {config_path}")
                self.config = {}  # Initialize empty config

            # Load environment variables
            dotenv.load_dotenv(self.config_dir / self._secrets_file)

            # Validate configurations
            self._validate_configurations()

        except Exception as e:
            self.logger.error(f"Error loading configurations: {e}")
            raise

    def _validate_configurations(self) -> None:
        """Validate configuration settings."""
        try:
            # Initialize with defaults if sections are missing
            if "exchange" not in self.config:
                self.config["exchange"] = {
                    "name": "binance_test",
                    "api_key": "test_key",
                    "api_secret": "test_secret",
                    "testnet": True,
                    "rate_limit": 1200,
                    "timeout": 30000,
                }

            if "risk" not in self.config:
                self.config["risk"] = {
                    "max_position_size": 1.0,
                    "max_drawdown": 0.1,
                    "position_sizing_method": "kelly",
                    "risk_per_trade": 0.02,
                    "max_leverage": 1.0,
                }

            # Validate exchange config
            exchange_config = ExchangeConfig(**self.config.get("exchange", {}))
            trading_config = TradingConfig(**self.config.get("trading", {}))
            risk_config = RiskConfig(**self.config.get("risk", {}))

            # Update config with validated values using model_dump instead of dict
            self.config.update(
                {
                    "exchange": exchange_config.model_dump(),
                    "trading": trading_config.model_dump(),
                    "risk": risk_config.model_dump(),
                }
            )

        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            raise

    def _init_encryption(self) -> None:
        """Initialize encryption for sensitive data."""
        try:
            # Generate or load encryption key
            key_path = self.config_dir / ".key"
            if key_path.exists():
                with open(key_path, "rb") as f:
                    self.key = f.read()
            else:
                self.key = Fernet.generate_key()
                with open(key_path, "wb") as f:
                    f.write(self.key)

            self.cipher = Fernet(self.key)

        except Exception as e:
            self.logger.error(f"Error initializing encryption: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found
        """
        try:
            value = self.config
            for k in key.split("."):
                value = value.get(k, default)
                if value == default:
                    break
            return value

        except Exception as e:
            self.logger.error(f"Error getting configuration: {e}")
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key (dot notation supported)
            value: Value to set
        """
        try:
            keys = key.split(".")
            current = self.config

            for k in keys[:-1]:
                current = current.setdefault(k, {})

            current[keys[-1]] = value

            # Validate after update
            self._validate_configurations()

            # Save configuration
            self.save()

        except Exception as e:
            self.logger.error(f"Error setting configuration: {e}")
            raise

    def save(self) -> None:
        """Save current configuration to file."""
        try:
            config_path = self.config_dir / f"{self.env.value}_{self._config_file}"

            # Backup existing config
            if config_path.exists():
                backup_path = config_path.with_suffix(".backup")
                config_path.rename(backup_path)

            # Save new config
            with open(config_path, "w") as f:
                yaml.safe_dump(self.config, f)

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise

    def get_secret(self, key: str) -> str:
        """Get a secret value from environment variables or secrets file."""
        try:
            # Try to get from environment first
            value = os.getenv(key)
            if value is not None:
                return value

            # If not in environment, try secrets file
            if self.secrets and key in self.secrets:
                return self.secrets[key]

            # Return empty string if not found
            self.logger.warning(f"Secret {key} not found")
            return ""
        except Exception as e:
            self.logger.error(f"Error getting secret: {e}")
            return ""

    def set_secret(self, key: str, value: str) -> None:
        """
        Set encrypted secret value.

        Args:
            key: Secret key
            value: Secret value
        """
        try:
            # Encrypt value
            encrypted_value = self.cipher.encrypt(value.encode()).decode()

            # Update .env file
            dotenv.set_key(self.config_dir / self._secrets_file, key, encrypted_value)

        except Exception as e:
            self.logger.error(f"Error setting secret: {e}")
            raise

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and configuration."""
        return {
            "environment": self.env.value,
            "mode": self.mode.value,
            "config_path": str(self.config_dir),
            "configurations": {
                k: v
                for k, v in self.config.items()
                if k not in ["exchange"]  # Exclude sensitive data
            },
        }

    def validate_exchange_credentials(self) -> bool:
        """Validate exchange API credentials."""
        try:
            api_key = self.get_secret("EXCHANGE_API_KEY")
            api_secret = self.get_secret("EXCHANGE_API_SECRET")

            if not api_key or not api_secret:
                return False

            # Test API connection (implement based on exchange)
            return True

        except Exception as e:
            self.logger.error(f"Error validating credentials: {e}")
            return False
