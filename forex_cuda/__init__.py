from .config import PortfolioEnvConfig
from .preprocessing import LogReturnDataset, PreprocessConfig, build_or_load_preprocessed_dataset

__all__ = [
    "PortfolioEnvConfig",
    "CuPortfolioEnv",
    "LogReturnDataset",
    "PreprocessConfig",
    "build_or_load_preprocessed_dataset",
]


def __getattr__(name: str):
    if name == "CuPortfolioEnv":
        from .env import CuPortfolioEnv

        return CuPortfolioEnv
    raise AttributeError(f"module 'forex_cuda' has no attribute {name!r}")
