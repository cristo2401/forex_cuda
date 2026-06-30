from .cache import (
    CachePaths,
    build_or_load_preprocessed_dataset,
    cache_paths_for_config,
    load_preprocessed_cache,
    save_preprocessed_cache,
)
from .parquet import discover_parquet_pairs, load_close_prices, load_log_returns_from_parquet
from .returns import compute_log_returns
from .types import LogReturnDataset, PreprocessConfig
from .windows import make_numpy_rolling_market, make_start_indices

__all__ = [
    "CachePaths",
    "LogReturnDataset",
    "PreprocessConfig",
    "build_or_load_preprocessed_dataset",
    "cache_paths_for_config",
    "compute_log_returns",
    "discover_parquet_pairs",
    "load_close_prices",
    "load_log_returns_from_parquet",
    "load_preprocessed_cache",
    "make_numpy_rolling_market",
    "make_start_indices",
    "save_preprocessed_cache",
]

