import pytest
import yaml
import os
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration without modifying original test.yaml."""
    config_dir = Path("config")
    test_yaml = config_dir / "test.yaml"
    temp_yaml = config_dir / "temp_test.yaml"

    # Create a copy of test.yaml if it exists
    if test_yaml.exists():
        shutil.copy(test_yaml, temp_yaml)

    yield str(config_dir)

    # Clean up temp file
    if temp_yaml.exists():
        temp_yaml.unlink()


@pytest.fixture(autouse=True)
def prevent_config_save(monkeypatch):
    """Prevent saving to test.yaml during tests."""

    def mock_save_config(*args, **kwargs):
        pass

    from TradingRL.src.core.config_manager import ConfigManager

    monkeypatch.setattr(ConfigManager, "save_config", mock_save_config)
