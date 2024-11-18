from setuptools import setup, find_packages

setup(
    name="TradingRL",
    version="0.1",
    packages=find_packages(include=["TradingRL", "TradingRL.*"]),
    install_requires=[
        "numpy",
        "pandas",
        "gymnasium",
        "stable-baselines3",
        "torch",
        "wandb",
        "click",
        "python-binance",
        "pyyaml",
        "scipy",
        "TA-Lib",
        "matplotlib",
        "seaborn",
    ],
    python_requires=">=3.7",
)
