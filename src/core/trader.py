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

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Store original data for price information
        self.raw_data = data.copy()

        # Preprocess data for features
        self.data = self._preprocess_data(data)

        self.initial_balance = initial_balance
        self.commission_rate = 0.001  # 0.1% commission per trade

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell

        # Calculate the number of features
        self.num_features = len(self.data.columns)

        # Observation space: price data + technical indicators + position info
        self.observation_space = spaces.Box(
            low=-10,  # Reasonable lower bound after normalization
            high=10,  # Reasonable upper bound after normalization
            shape=(self.num_features + 3,),  # +3 for balance, position, position_value
            dtype=np.float32,
        )

        # Initialize trade history
        self.trade_history = []
        self.np_random = None
        self.reset()

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and normalize the data."""
        try:
            # Make a copy to avoid modifying the original data
            df = data.copy()

            # Forward fill any missing values
            df = df.ffill()

            # Backward fill any remaining missing values at the start
            df = df.bfill()

            # Calculate returns and volatility
            df["returns"] = df["close"].pct_change()
            df["volatility"] = df["returns"].rolling(window=20).std()

            # Replace any infinite values with large numbers
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)

            # Calculate normalized price features
            price_cols = ["open", "high", "low", "close"]
            for col in price_cols:
                # Use log returns instead of percentage change
                with np.errstate(divide="ignore"):  # Handle divide by zero in log
                    df[f"{col}_ret"] = np.log(df[col]).diff()

                # Calculate rolling z-score
                rolling_mean = df[col].rolling(window=20, min_periods=1).mean()
                rolling_std = df[col].rolling(window=20, min_periods=1).std()
                df[f"{col}_zscore"] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

            # Normalize volume using log transform and z-score
            with np.errstate(divide="ignore"):  # Handle divide by zero in log
                df["volume_ret"] = np.log(df["volume"]).diff()
            volume_mean = df["volume"].rolling(window=20, min_periods=1).mean()
            volume_std = df["volume"].rolling(window=20, min_periods=1).std()
            df["volume_zscore"] = (df["volume"] - volume_mean) / (volume_std + 1e-8)

            # Calculate price momentum features
            for period in [5, 10, 20]:
                df[f"momentum_{period}"] = df["close"].pct_change(period)
                df[f"volume_momentum_{period}"] = df["volume"].pct_change(period)

            # Calculate moving averages and relative strength
            for period in [10, 20, 50]:
                df[f"ma_{period}"] = df["close"].rolling(window=period).mean()
                df[f"ma_{period}_slope"] = df[f"ma_{period}"].pct_change(5)
                df[f"price_to_ma_{period}"] = df["close"] / df[f"ma_{period}"] - 1

            # Volatility features
            df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
            with np.errstate(divide="ignore"):  # Handle divide by zero in log
                df["daily_range"] = np.log(df["high"]) - np.log(df["low"])

            # Final feature selection
            feature_cols = (
                [f"{col}_ret" for col in price_cols]
                + [f"{col}_zscore" for col in price_cols]
                + [
                    "volume_ret",
                    "volume_zscore",
                    "volatility",
                    "high_low_range",
                    "daily_range",
                ]
                + [f"momentum_{period}" for period in [5, 10, 20]]
                + [f"volume_momentum_{period}" for period in [5, 10, 20]]
                + [f"ma_{period}_slope" for period in [10, 20, 50]]
                + [f"price_to_ma_{period}" for period in [10, 20, 50]]
            )

            # Drop rows with NaN values at the beginning
            df = df.dropna()

            # Ensure all values are finite and within reasonable bounds
            for col in feature_cols:
                df[col] = df[col].clip(-10, 10)

                # Additional check for NaN values
                if df[col].isna().any():
                    self.logger.warning(f"NaN values found in {col}, filling with 0")
                    df[col] = df[col].fillna(0)

            # Final verification
            if df[feature_cols].isna().any().any():
                problematic_cols = (
                    df[feature_cols].columns[df[feature_cols].isna().any()].tolist()
                )
                raise ValueError(
                    f"NaN values still present in columns: {problematic_cols}"
                )

            # Keep only the feature columns
            result_df = df[feature_cols]

            # Log the shape of the processed data
            self.logger.info(f"Preprocessed data shape: {result_df.shape}")
            self.logger.info(f"Features used: {feature_cols}")

            return result_df

        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise ValueError(f"Error preprocessing data: {str(e)}")

    def _get_observation(self):
        """Get the current observation."""
        try:
            # Get current market data
            market_data = self.data.iloc[self.current_step].values

            # Calculate normalized account info
            # Use close_ret instead of close_pct for position normalization
            current_price = float(self.data.iloc[self.current_step]["close_ret"])

            account_info = np.array(
                [
                    self.balance / self.initial_balance - 1,  # Normalized balance
                    self.position
                    / (
                        self.initial_balance / (current_price + 1e-8)
                    ),  # Normalized position
                    self.position_value / self.initial_balance
                    - 1,  # Normalized position value
                ]
            )

            # Combine and ensure no NaN values
            obs = np.concatenate([market_data, account_info])

            # Check for and handle NaN/inf values
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                self.logger.warning(
                    "Found NaN or inf values in observation, replacing with 0"
                )
                obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

            # Clip values to observation space bounds
            obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

            return obs.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Error creating observation: {str(e)}")
            self.logger.error(f"Current data columns: {self.data.columns.tolist()}")
            self.logger.error(f"Current step: {self.current_step}")
            raise ValueError(f"Error creating observation: {str(e)}")

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

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step within the environment."""
        # Use raw_data for price information
        current_price = float(self.raw_data.iloc[self.current_step]["close"])
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
                        timestamp=self.raw_data.index[self.current_step],
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
                        timestamp=self.raw_data.index[self.current_step],
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
            "current_price": current_price,
        }

        return obs, reward, terminated, truncated, info

    def _plot_trading_actions(self, title: str = "Trading Actions") -> plt.Figure:
        """Create a plot of price with buy/sell markers."""
        fig, ax = plt.subplots(figsize=(15, 7))

        # Plot price using raw_data
        ax.plot(self.raw_data.index, self.raw_data["close"], label="Price", alpha=0.7)

        # Plot buy points
        buy_trades = [t for t in self.trade_history if t.action == "buy"]
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
        sell_trades = [t for t in self.trade_history if t.action == "sell"]
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


def get_device() -> str:
    """
    Automatically detect and return the best available device for PyTorch.
    Returns 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
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

    def train_model(
        self,
        train_data: pd.DataFrame,
        eval_data: pd.DataFrame,
        hyperparams: Dict = None,
        total_timesteps: int = 100000,
    ) -> None:
        """Train the RL model with given data and hyperparameters."""
        try:
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
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=[wandb_callback, checkpoint_callback, eval_callback],
                tb_log_name=f"ppo_trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

            # Evaluate final performance
            metrics = self.evaluate_performance(eval_data)
            wandb.log({"final_evaluation": metrics})

            # Save final model
            final_model_path = os.path.join(self.model_dir, "final_model")
            self.model.save(final_model_path)
            self.logger.info(f"Final model saved to {final_model_path}")

        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise

        finally:
            # Close wandb run
            wandb.finish()

    def evaluate_performance(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate model performance on test data."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        try:
            env = self.create_environment(test_data)
            obs, _ = env.reset()  # Unpack observation and info
            done = False
            truncated = False
            portfolio_values = []
            actions_taken = []
            trades = []
            returns = []

            initial_value = env.balance + env.position_value
            last_portfolio_value = initial_value

            while not done and not truncated:
                # Process observation
                if isinstance(obs, tuple):
                    obs = obs[0]  # Take first element if tuple

                # Ensure observation is a numpy array with correct shape
                obs_array = np.array(obs, dtype=np.float32)
                if len(obs_array.shape) == 1:
                    obs_array = obs_array.reshape(1, -1)

                # Get action from model
                action, _ = self.model.predict(obs_array, deterministic=True)

                # Take step in environment
                obs, reward, done, truncated, info = env.step(action)

                # Record portfolio value and calculate return
                current_value = info["portfolio_value"]
                portfolio_values.append(current_value)

                # Calculate period return
                period_return = (
                    (current_value - last_portfolio_value) / last_portfolio_value
                    if last_portfolio_value != 0
                    else 0
                )
                if period_return != 0:  # Only record non-zero returns
                    returns.append(period_return)

                last_portfolio_value = current_value

                # Record action and trade if any
                actions_taken.append(action)
                if action != 0:  # If not hold
                    trades.append(
                        {
                            "action": "buy" if action == 1 else "sell",
                            "price": test_data.iloc[env.current_step]["close"],
                            "portfolio_value": current_value,
                        }
                    )

            # Calculate metrics safely
            total_trades = len(trades)
            if total_trades == 0:
                self.logger.warning("No trades were executed during evaluation")
                metrics = {
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                    "final_portfolio_value": (
                        portfolio_values[-1] if portfolio_values else initial_value
                    ),
                    "avg_return_per_trade": 0.0,
                    "volatility": 0.0,
                    "profit_factor": 0.0,
                }
            else:
                # Calculate returns and metrics
                returns = np.array(returns)
                positive_returns = returns[returns > 0]
                negative_returns = returns[returns < 0]

                total_positive = (
                    np.sum(positive_returns) if len(positive_returns) > 0 else 0
                )
                total_negative = (
                    abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 0
                )

                metrics = {
                    "total_return": (portfolio_values[-1] - initial_value)
                    / initial_value,
                    "sharpe_ratio": (
                        np.mean(returns) / np.std(returns) * np.sqrt(252)
                        if len(returns) > 1
                        else 0
                    ),
                    "max_drawdown": self._calculate_max_drawdown(portfolio_values),
                    "win_rate": (
                        len(positive_returns) / len(returns) if len(returns) > 0 else 0
                    ),
                    "total_trades": total_trades,
                    "final_portfolio_value": portfolio_values[-1],
                    "avg_return_per_trade": np.mean(returns) if len(returns) > 0 else 0,
                    "volatility": (
                        np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
                    ),
                    "profit_factor": (
                        total_positive / total_negative if total_negative > 0 else 0
                    ),
                }

            # Log detailed metrics
            self.logger.info(f"Evaluation completed with {total_trades} trades")
            self.logger.info(
                f"Final portfolio value: {metrics['final_portfolio_value']:.2f}"
            )
            self.logger.info(f"Total return: {metrics['total_return']:.2%}")

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

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate the maximum drawdown from a list of portfolio values."""
        if not portfolio_values:
            return 0.0

        peak = portfolio_values[0]
        max_drawdown = 0.0

        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

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
