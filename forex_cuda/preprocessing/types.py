from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LogReturnDataset:
    log_returns: np.ndarray
    timestamps: pd.DatetimeIndex
    pairs: list[str]
    close_prices: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessConfig:
    data_dir: Path
    pairs: tuple[str, ...] | None = None
    price_col: str = "close"
    timestamp_col: str = "timestamp"
    drop_data_gaps: bool = True
    max_rows: int | None = None
    cache_dir: Path = Path("data_cache")
    cache_name: str | None = None

    @classmethod
    def from_values(
        cls,
        data_dir: str | Path,
        *,
        pairs: Iterable[str] | None = None,
        price_col: str = "close",
        timestamp_col: str = "timestamp",
        drop_data_gaps: bool = True,
        max_rows: int | None = None,
        cache_dir: str | Path = Path("data_cache"),
        cache_name: str | None = None,
    ) -> "PreprocessConfig":
        pair_tuple = tuple(pairs) if pairs is not None else None
        config = cls(
            data_dir=Path(data_dir),
            pairs=pair_tuple,
            price_col=price_col,
            timestamp_col=timestamp_col,
            drop_data_gaps=drop_data_gaps,
            max_rows=max_rows,
            cache_dir=Path(cache_dir),
            cache_name=cache_name,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(self.data_dir)
        if not self.data_dir.is_dir():
            raise NotADirectoryError(self.data_dir)
        if self.pairs is not None and len(self.pairs) == 0:
            raise ValueError("pairs must be non-empty when provided")
        if self.max_rows is not None and self.max_rows <= 1:
            raise ValueError("max_rows must be greater than 1")
        if not self.price_col:
            raise ValueError("price_col must be non-empty")
        if not self.timestamp_col:
            raise ValueError("timestamp_col must be non-empty")

