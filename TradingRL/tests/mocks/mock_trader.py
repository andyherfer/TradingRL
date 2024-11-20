import numpy as np
from typing import Dict, Any, Tuple, Optional
import pandas as pd


class MockTrader:
    """Mock trader for testing."""

    def __init__(self, model_dir: str = "test_models", **kwargs):
        self.model_dir = model_dir
        self.trained = False
        self.predictions = []
        self.training_history = []
        self.force_error = False

    async def predict_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Mock prediction with error simulation."""
        if self.force_error:
            raise ValueError("Simulated prediction error")

        # Cycle through different actions for testing
        action = len(self.predictions) % 5  # 0-4 for different signal types
        confidence = 0.5 + (len(self.predictions) % 5) * 0.1  # 0.5-0.9

        prediction = (action, confidence)
        self.predictions.append(
            {"state": state, "action": prediction[0], "confidence": prediction[1]}
        )
        return prediction

    async def train_model(
        self, train_states: pd.DataFrame, position_history: list
    ) -> Dict[str, Any]:
        """Mock training."""
        self.trained = True
        metrics = {"loss": 0.1, "accuracy": 0.8, "avg_reward": 1.0}
        self.training_history.append(
            {"states": train_states, "history": position_history, "metrics": metrics}
        )
        return metrics

    def save(self, path: str) -> None:
        """Mock save."""
        pass

    def load(self, path: str) -> None:
        """Mock load."""
        self.trained = True
