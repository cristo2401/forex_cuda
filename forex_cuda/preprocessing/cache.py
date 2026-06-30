from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .parquet import discover_parquet_pairs, load_close_prices
from .returns import compute_log_returns
from .types import LogReturnDataset, PreprocessConfig

CACHE_VERSION = 1


@dataclass(frozen=True)
class CachePaths:
    root: Path
    metadata: Path
    log_returns: Path
    timestamps: Path
    close_prices: Path

    def exists(self) -> bool:
        return (
            self.metadata.exists()
            and self.log_returns.exists()
            and self.timestamps.exists()
            and self.close_prices.exists()
        )


def _selected_pairs(config: PreprocessConfig) -> list[str]:
    return list(config.pairs) if config.pairs is not None else discover_parquet_pairs(config.data_dir)


def _source_file_stats(config: PreprocessConfig, pairs: list[str]) -> list[dict[str, Any]]:
    stats = []
    for pair in pairs:
        path = config.data_dir / f"{pair}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        stat = path.stat()
        stats.append(
            {
                "pair": pair,
                "file": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return stats


def _cache_payload(config: PreprocessConfig, pairs: list[str]) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "data_dir": str(config.data_dir.resolve()),
        "pairs": pairs,
        "price_col": config.price_col,
        "timestamp_col": config.timestamp_col,
        "drop_data_gaps": config.drop_data_gaps,
        "max_rows": config.max_rows,
        "source_files": _source_file_stats(config, pairs),
    }


def _cache_key(config: PreprocessConfig, pairs: list[str]) -> str:
    payload = _cache_payload(config, pairs)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    if config.cache_name is None:
        return digest
    safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in config.cache_name)
    return f"{safe_name}-{digest}"


def cache_paths_for_config(config: PreprocessConfig) -> CachePaths:
    config.validate()
    pairs = _selected_pairs(config)
    root = config.cache_dir / _cache_key(config, pairs)
    return CachePaths(
        root=root,
        metadata=root / "metadata.json",
        log_returns=root / "log_returns.float32.npy",
        timestamps=root / "timestamps.parquet",
        close_prices=root / "close_prices.parquet",
    )


def save_preprocessed_cache(
    dataset: LogReturnDataset,
    paths: CachePaths,
    *,
    config: PreprocessConfig,
    preprocessing_seconds: float | None = None,
) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    np.save(paths.log_returns, np.ascontiguousarray(dataset.log_returns, dtype=np.float32))
    pd.DataFrame({"timestamp": dataset.timestamps}).to_parquet(paths.timestamps, index=False)
    dataset.close_prices.to_parquet(paths.close_prices, index=True)

    pairs = _selected_pairs(config)
    metadata = {
        "cache_version": CACHE_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": _cache_payload(config, pairs),
        "pairs": dataset.pairs,
        "log_returns_shape": list(dataset.log_returns.shape),
        "close_prices_shape": list(dataset.close_prices.shape),
        "timestamp_start": str(dataset.timestamps[0]),
        "timestamp_end": str(dataset.timestamps[-1]),
        "preprocessing_seconds": preprocessing_seconds,
    }
    paths.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_preprocessed_cache(paths: CachePaths) -> LogReturnDataset:
    if not paths.exists():
        raise FileNotFoundError(paths.root)

    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    log_returns = np.load(paths.log_returns).astype(np.float32, copy=False)
    timestamp_frame = pd.read_parquet(paths.timestamps)
    close_prices = pd.read_parquet(paths.close_prices)

    timestamps = pd.DatetimeIndex(timestamp_frame["timestamp"])
    pairs = list(metadata["pairs"])
    runtime_metadata = dict(metadata)
    runtime_metadata["source"] = "cache"
    runtime_metadata["loaded_from_cache"] = True
    runtime_metadata["cache_root"] = str(paths.root)

    return LogReturnDataset(
        log_returns=np.ascontiguousarray(log_returns),
        timestamps=timestamps,
        pairs=pairs,
        close_prices=close_prices,
        metadata=runtime_metadata,
    )


def build_or_load_preprocessed_dataset(
    data_dir: str | Path,
    *,
    pairs: Iterable[str] | None = None,
    price_col: str = "close",
    timestamp_col: str = "timestamp",
    drop_data_gaps: bool = True,
    max_rows: int | None = None,
    cache_dir: str | Path = Path("data_cache"),
    cache_name: str | None = None,
    refresh: bool = False,
) -> LogReturnDataset:
    config = PreprocessConfig.from_values(
        data_dir,
        pairs=pairs,
        price_col=price_col,
        timestamp_col=timestamp_col,
        drop_data_gaps=drop_data_gaps,
        max_rows=max_rows,
        cache_dir=cache_dir,
        cache_name=cache_name,
    )
    paths = cache_paths_for_config(config)
    if paths.exists() and not refresh:
        return load_preprocessed_cache(paths)

    started = datetime.now(UTC)
    perf_started = time.perf_counter()
    close_prices = load_close_prices(
        config.data_dir,
        pairs=config.pairs,
        price_col=config.price_col,
        timestamp_col=config.timestamp_col,
        drop_data_gaps=config.drop_data_gaps,
        max_rows=config.max_rows,
    )
    log_returns, timestamps, loaded_pairs = compute_log_returns(close_prices)
    preprocessing_seconds = time.perf_counter() - perf_started
    dataset = LogReturnDataset(
        log_returns=log_returns,
        timestamps=timestamps,
        pairs=loaded_pairs,
        close_prices=close_prices,
        metadata={
            "source": "parquet",
            "loaded_from_cache": False,
            "started_at_utc": started.isoformat(),
            "cache_root": str(paths.root),
        },
    )
    save_preprocessed_cache(
        dataset,
        paths,
        config=config,
        preprocessing_seconds=preprocessing_seconds,
    )
    return dataset
