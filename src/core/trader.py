import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
import torch
import logging
from datetime import datetime
import os
import json
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from collections import deque
import platform


@dataclass
class Trade:
    """Class to store trade information."""

    timestamp: pd.Timestamp
    action: str
    price: float
    position: float
    portfolio_value: float
    profit_loss: float


class WandBCallback(BaseCallback):
    """Custom callback for logging to Weights & Biases during training."""

    def __init__(self, check_freq: int, verbose: int = 1):
        super(WandBCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.portfolio_values = []
        self.trades = []

    def _on_step(self) -> bool:
        """Called after each step in the environment."""
        if self.n_calls % self.check_freq == 0:
            # Log basic metrics
            wandb.log(
                {
                    "train/timesteps": self.n_calls,
                    "train/episode_reward": (
                        np.mean(self.episode_rewards) if self.episode_rewards else 0
                    ),
                    "train/episode_length": (
                        np.mean(self.episode_lengths) if self.episode_lengths else 0
                    ),
                },
                step=self.n_calls,
            )

            # Clear episode buffers
            self.episode_rewards = []
            self.episode_lengths = []

        return True

    def _on_rollout_end(self) -> None:
        """Called after each rollout."""
        # Get current environment info
        env = self.training_env.envs[0]
        self.portfolio_values.append(env.balance + env.position_value)

        # Log portfolio value
        wandb.log(
            {"train/portfolio_value": self.portfolio_values[-1]}, step=self.n_calls
        )


class TradingEnvironment(gym.Env):
    """Custom Trading Environment that follows gym interface."""

    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0):
        super(TradingEnvironment, self).__init__()

        self.data = data
        self.initial_balance = initial_balance
        self.commission_rate = 0.001  # 0.1% commission per trade

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell

        # Calculate the number of features
        self.num_features = len(data.columns)

        # Observation space: price data + technical indicators + position info
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features + 3,),  # +3 for balance, position, position_value
            dtype=np.float32,
        )

        # Initialize trade history
        self.trade_history = []
        self.np_random = None
        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Additional options for reset

        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)  # Initialize self.np_random

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_value = 0.0
        self.trade_history = []

        initial_observation = self._get_observation()
        info = {
            "balance": self.balance,
            "position": self.position,
            "position_value": self.position_value,
        }

        return initial_observation, info

    def _get_observation(self):
        """Get the current observation."""
        # Get current market data
        market_data = self.data.iloc[self.current_step].values

        # Add account information
        account_info = np.array([self.balance, self.position, self.position_value])

        return np.concatenate([market_data, account_info])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step within the environment.

        Args:
            action: Action to take (0: hold, 1: buy, 2: sell)

        Returns:
            observation: New observation
            reward: Reward for the action
            terminated: Whether the episode is terminated
            truncated: Whether the episode was truncated
            info: Additional information
        """
        current_price = self.data.iloc[self.current_step]["close"]
        old_portfolio_value = self.balance + self.position_value

        # Execute trading action
        if action == 1:  # Buy
            if self.balance > 0:
                purchase_amount = self.balance * 0.95  # Keep some balance for fees
                new_position = (
                    purchase_amount * (1 - self.commission_rate)
                ) / current_price
                self.position += new_position
                self.balance -= purchase_amount

                # Record trade
                self.trade_history.append(
                    Trade(
                        timestamp=self.data.index[self.current_step],
                        action="buy",
                        price=current_price,
                        position=new_position,
                        portfolio_value=self.balance + self.position_value,
                        profit_loss=0,
                    )
                )

        elif action == 2:  # Sell
            if self.position > 0:
                sale_amount = self.position * current_price * (1 - self.commission_rate)
                profit_loss = sale_amount - (
                    self.position * self.trade_history[-1].price
                    if self.trade_history
                    else 0
                )
                self.balance += sale_amount

                # Record trade
                self.trade_history.append(
                    Trade(
                        timestamp=self.data.index[self.current_step],
                        action="sell",
                        price=current_price,
                        position=self.position,
                        portfolio_value=self.balance,
                        profit_loss=profit_loss,
                    )
                )

                self.position = 0

        # Update position value
        self.position_value = self.position * current_price

        # Calculate reward
        new_portfolio_value = self.balance + self.position_value
        reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value

        # Move to next step
        self.current_step += 1

        # Check if episode is finished
        terminated = self.current_step >= len(self.data) - 1
        truncated = False  # We don't use truncation in this environment

        # Get new observation
        obs = self._get_observation()

        # Additional info
        info = {
            "portfolio_value": new_portfolio_value,
            "position": self.position,
            "balance": self.balance,
            "trade_history": self.trade_history,
        }

        return obs, reward, terminated, truncated, info


def get_device() -> str:
    """
    Automatically detect and return the best available device for PyTorch.
    Returns 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif platform.processor() == "arm" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Trader:
    """Main trader class for managing the RL model and trading decisions."""

    def __init__(
        self,
        model_dir: str = "models",
        tensorboard_log: str = "logs/tensorboard",
        project_name: str = "crypto_trader",
        device: Optional[str] = None,
    ):
        """Initialize the trader with paths for model and log storage."""
        self.model_dir = model_dir
        self.tensorboard_log = tensorboard_log
        self.project_name = project_name
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(tensorboard_log, exist_ok=True)

        # Set device - use provided device or auto-detect
        self.device = device if device else get_device()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")

        logging.basicConfig(level=logging.INFO)
        self.model = None
        self.env = None

    def _plot_trading_actions(
        self,
        data: pd.DataFrame,
        trade_history: List[Trade],
        title: str = "Trading Actions",
    ) -> plt.Figure:
        """Create a plot of price with buy/sell markers."""
        fig, ax = plt.subplots(figsize=(15, 7))

        # Plot price
        ax.plot(data.index, data["close"], label="Price", alpha=0.7)

        # Plot buy points
        buy_trades = [t for t in trade_history if t.action == "buy"]
        if buy_trades:
            buy_times = [t.timestamp for t in buy_trades]
            buy_prices = [t.price for t in buy_trades]
            ax.scatter(
                buy_times,
                buy_prices,
                color="green",
                marker="^",
                s=100,
                label="Buy",
                alpha=0.7,
            )

        # Plot sell points
        sell_trades = [t for t in trade_history if t.action == "sell"]
        if sell_trades:
            sell_times = [t.timestamp for t in sell_trades]
            sell_prices = [t.price for t in sell_trades]
            ax.scatter(
                sell_times,
                sell_prices,
                color="red",
                marker="v",
                s=100,
                label="Sell",
                alpha=0.7,
            )

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def train_model(
        self,
        train_data: pd.DataFrame,
        eval_data: pd.DataFrame,
        hyperparams: Dict = None,
        total_timesteps: int = 100000,
    ) -> None:
        """Train the RL model with given data and hyperparameters."""
        # Initialize wandb
        wandb.init(
            project=self.project_name,
            config={
                "model_type": "PPO",
                "total_timesteps": total_timesteps,
                "device": self.device,
                **(hyperparams or {}),
            },
        )

        # Default hyperparameters if none provided
        if hyperparams is None:
            hyperparams = {
                "learning_rate": 0.0003,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
            }

        # Create environments
        self.env = Monitor(self.create_environment(train_data))
        eval_env = Monitor(self.create_environment(eval_data))

        # Initialize model with device
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=self.tensorboard_log,
            device=self.device,
            **hyperparams,
        )

        # Setup callbacks
        wandb_callback = WandBCallback(check_freq=1000)

        checkpoint_callback = CheckpointCallback(
            save_freq=10000, save_path=self.model_dir, name_prefix="ppo_trader"
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.model_dir}/best_model",
            log_path=f"{self.model_dir}/eval_results",
            eval_freq=10000,
            deterministic=True,
            render=False,
        )

        # Train the model
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=[wandb_callback, checkpoint_callback, eval_callback],
                tb_log_name=f"ppo_trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

            # Evaluate final performance
            metrics = self.evaluate_performance(eval_data)
            wandb.log({"final_evaluation": metrics})

            # Plot final training results
            fig = self._plot_trading_actions(
                eval_data, self.env.trade_history, "Final Training Performance"
            )
            wandb.log({"training_plot": wandb.Image(fig)})
            plt.close(fig)

            # Save final model
            final_model_path = os.path.join(self.model_dir, "final_model")
            self.model.save(final_model_path)
            self.logger.info(f"Final model saved to {final_model_path}")

            # Close wandb run
            wandb.finish()

        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            wandb.finish()
            raise

    def evaluate_performance(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate model performance on test data."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        try:
            env = self.create_environment(test_data)
            obs = env.reset()
            done = False
            portfolio_values = []
            actions_taken = []

            # Add logging to debug observation shape
            self.logger.info(
                f"First observation shape: {obs.shape if hasattr(obs, 'shape') else None}"
            )

            while not done:
                # Convert observation to numpy array with consistent shape
                if isinstance(obs, (list, tuple)):
                    obs = np.array(obs, dtype=np.float32)
                if len(obs.shape) == 1:
                    obs = obs.reshape(1, -1)  # Ensure 2D shape

                self.logger.info(f"Processed observation shape: {obs.shape}")

                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                portfolio_values.append(info["portfolio_value"])
                actions_taken.append(action)

            # Calculate performance metrics
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            metrics = {
                "total_return": (portfolio_values[-1] - portfolio_values[0])
                / portfolio_values[0],
                "sharpe_ratio": np.mean(returns)
                / np.std(returns)
                * np.sqrt(252),  # Annualized
                "max_drawdown": np.min(
                    np.minimum.accumulate(portfolio_values)
                    / np.maximum.accumulate(portfolio_values)
                    - 1
                ),
                "win_rate": np.sum(returns > 0) / len(returns),
                "total_trades": len([a for a in actions_taken if a != 0]),
                "final_portfolio_value": portfolio_values[-1],
                "avg_return_per_trade": (
                    np.mean(returns[returns != 0])
                    if len(returns[returns != 0]) > 0
                    else 0
                ),
                "volatility": np.std(returns) * np.sqrt(252),  # Annualized
                "profit_factor": (
                    abs(np.sum(returns[returns > 0]) / np.sum(returns[returns < 0]))
                    if np.sum(returns[returns < 0]) != 0
                    else np.inf
                ),
            }

            # Create and log performance plot
            if wandb.run is not None:
                fig = self._plot_trading_actions(
                    test_data, env.trade_history, "Backtest Performance"
                )
                wandb.log(
                    {"backtest_metrics": metrics, "backtest_plot": wandb.Image(fig)}
                )
                plt.close(fig)

            return metrics

        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise

    def create_environment(
        self, data: pd.DataFrame, initial_balance: float = 10000.0
    ) -> TradingEnvironment:
        """Create and return a trading environment."""
        return TradingEnvironment(data, initial_balance)

    def predict_action(self, data: pd.DataFrame) -> Tuple[str, float, Dict]:
        """
        Predict trading action for given market data.

        Returns:
            Tuple containing:
            - action: str ("buy", "sell", or "hold")
            - confidence: float (prediction confidence)
            - metadata: Dict (additional prediction information)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        try:
            # Create a temporary environment for prediction
            temp_env = self.create_environment(data)

            # Get the observation
            obs = temp_env._get_observation()

            # Predict action
            action, _ = self.model.predict(obs, deterministic=True)

            # Convert action to trading signal
            action_map = {0: "hold", 1: "buy", 2: "sell"}

            # Calculate confidence and action probabilities
            with torch.no_grad():
                action_probs = self.model.policy.get_distribution(
                    torch.FloatTensor(obs.reshape(1, -1))
                ).distribution.probs
                confidence = float(action_probs[0][action])

                # Get all action probabilities
                probs_dict = {
                    action_map[i]: float(action_probs[0][i])
                    for i in range(len(action_map))
                }

            # Create metadata dictionary
            metadata = {
                "action_probabilities": probs_dict,
                "current_price": float(data.iloc[-1]["close"]),
                "timestamp": data.index[-1],
                "technical_indicators": {
                    col: float(data.iloc[-1][col])
                    for col in data.columns
                    if col not in ["open", "high", "low", "close", "volume"]
                },
            }

            # Log prediction to W&B if in active run
            if wandb.run is not None:
                wandb.log(
                    {
                        "prediction/action": action_map[action],
                        "prediction/confidence": confidence,
                        "prediction/price": metadata["current_price"],
                        **{f"prediction/prob_{k}": v for k, v in probs_dict.items()},
                        **{
                            f"prediction/indicator_{k}": v
                            for k, v in metadata["technical_indicators"].items()
                        },
                    }
                )

            return action_map[action], confidence, metadata

        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise

    def backtest(
        self,
        test_data: pd.DataFrame,
        initial_balance: float = 10000.0,
        log_to_wandb: bool = True,
    ) -> Dict:
        """
        Perform detailed backtesting of the model.

        Args:
            test_data: Historical data for backtesting
            initial_balance: Starting balance for backtesting
            log_to_wandb: Whether to log results to W&B

        Returns:
            Dictionary containing backtest results and metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        try:
            # Initialize backtest environment
            env = self.create_environment(test_data, initial_balance)
            obs = env.reset()
            done = False

            # Initialize tracking variables
            portfolio_history = []
            trade_history = []
            action_history = []
            position_history = []

            # Run backtest
            while not done:
                action, confidence, metadata = self.predict_action(
                    test_data.iloc[: env.current_step + 1]
                )

                # Convert action to numeric
                action_num = {"hold": 0, "buy": 1, "sell": 2}[action]

                # Take step in environment
                obs, reward, done, info = env.step(action_num)

                # Record step information
                portfolio_history.append(
                    {
                        "timestamp": test_data.index[env.current_step],
                        "portfolio_value": info["portfolio_value"],
                        "balance": info["balance"],
                        "position": info["position"],
                        "price": test_data.iloc[env.current_step]["close"],
                        "action": action,
                        "confidence": confidence,
                    }
                )

                # Record trade if one occurred
                if action != "hold":
                    trade_history.append(
                        {
                            "timestamp": test_data.index[env.current_step],
                            "action": action,
                            "price": test_data.iloc[env.current_step]["close"],
                            "position_size": abs(info["position"]),
                            "portfolio_value": info["portfolio_value"],
                        }
                    )

            # Convert histories to DataFrames
            portfolio_df = pd.DataFrame(portfolio_history)
            portfolio_df.set_index("timestamp", inplace=True)

            trades_df = pd.DataFrame(trade_history)
            if not trades_df.empty:
                trades_df.set_index("timestamp", inplace=True)

            # Calculate metrics
            returns = portfolio_df["portfolio_value"].pct_change().dropna()

            metrics = {
                "total_return": (
                    portfolio_df["portfolio_value"].iloc[-1] - initial_balance
                )
                / initial_balance,
                "sharpe_ratio": (
                    np.mean(returns) / np.std(returns) * np.sqrt(252)
                    if len(returns) > 0
                    else 0
                ),
                "max_drawdown": np.min(
                    np.minimum.accumulate(portfolio_df["portfolio_value"])
                    / np.maximum.accumulate(portfolio_df["portfolio_value"])
                    - 1
                ),
                "total_trades": len(trades_df),
                "win_rate": (
                    np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
                ),
                "profit_factor": (
                    abs(np.sum(returns[returns > 0]) / np.sum(returns[returns < 0]))
                    if np.sum(returns[returns < 0]) != 0
                    else np.inf
                ),
                "volatility": np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0,
                "final_portfolio_value": portfolio_df["portfolio_value"].iloc[-1],
            }

            if log_to_wandb and wandb.run is not None:
                # Create and log performance plot
                fig = self._plot_trading_actions(
                    test_data, env.trade_history, "Detailed Backtest Results"
                )

                # Create returns distribution plot
                fig_returns = plt.figure(figsize=(10, 6))
                sns.histplot(returns, kde=True)
                plt.title("Distribution of Returns")
                plt.xlabel("Return")
                plt.ylabel("Frequency")

                # Log to W&B
                wandb.log(
                    {
                        "backtest/metrics": metrics,
                        "backtest/trading_plot": wandb.Image(fig),
                        "backtest/returns_distribution": wandb.Image(fig_returns),
                        "backtest/portfolio_history": wandb.Table(
                            dataframe=portfolio_df
                        ),
                        "backtest/trades": wandb.Table(
                            dataframe=(
                                trades_df if not trades_df.empty else pd.DataFrame()
                            )
                        ),
                    }
                )

                plt.close(fig)
                plt.close(fig_returns)

            return {
                "metrics": metrics,
                "portfolio_history": portfolio_df,
                "trades": trades_df,
            }

        except Exception as e:
            self.logger.error(f"Error during backtesting: {e}")
            raise
