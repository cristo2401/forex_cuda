from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .returns import compute_log_returns
from .types import LogReturnDataset


def discover_parquet_pairs(data_dir: str | Path) -> list[str]:
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise NotADirectoryError(path)
    return sorted(file.stem for file in path.glob("*.parquet"))


def _read_pair_series(
    pair_path: Path,
    *,
    pair: str,
    price_col: str,
    timestamp_col: str,
    drop_data_gaps: bool,
) -> pd.Series:
    columns = [timestamp_col, price_col]
    if drop_data_gaps:
        columns.append("is_data_gap")

    try:
        frame = pd.read_parquet(pair_path, columns=columns)
    except (KeyError, ValueError):
        frame = pd.read_parquet(pair_path, columns=[timestamp_col, price_col])

    if timestamp_col not in frame.columns:
        raise ValueError(f"{pair_path.name} is missing {timestamp_col!r}")
    if price_col not in frame.columns:
        raise ValueError(f"{pair_path.name} is missing {price_col!r}")

    if drop_data_gaps and "is_data_gap" in frame.columns:
        frame = frame[~frame["is_data_gap"]]

    frame = frame[[timestamp_col, price_col]]
    frame = frame[frame[price_col].notna() & (frame[price_col] > 0)]
    frame = frame.drop_duplicates(subset=timestamp_col, keep="last")
    frame = frame.sort_values(timestamp_col)
    series = frame.set_index(timestamp_col)[price_col]
    series.index = pd.DatetimeIndex(series.index)
    series.name = pair
    return series


def load_close_prices(
    data_dir: str | Path,
    *,
    pairs: Iterable[str] | None = None,
    price_col: str = "close",
    timestamp_col: str = "timestamp",
    drop_data_gaps: bool = True,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load and timestamp-align close prices from one parquet file per FX pair."""
    path = Path(data_dir)
    selected_pairs = list(pairs) if pairs is not None else discover_parquet_pairs(path)
    if not selected_pairs:
        raise ValueError("at least one pair is required")

    series = []
    for pair in selected_pairs:
        pair_path = path / f"{pair}.parquet"
        if not pair_path.exists():
            raise FileNotFoundError(pair_path)
        series.append(
            _read_pair_series(
                pair_path,
                pair=pair,
                price_col=price_col,
                timestamp_col=timestamp_col,
                drop_data_gaps=drop_data_gaps,
            )
        )

    aligned = pd.concat(series, axis=1, join="inner")
    aligned = aligned.dropna().sort_index()
    if aligned.empty:
        raise ValueError("no overlapping timestamps found for selected pairs")

    if max_rows is not None:
        if max_rows <= 1:
            raise ValueError("max_rows must be greater than 1")
        aligned = aligned.tail(max_rows)

    return aligned.astype(np.float64)


def load_log_returns_from_parquet(
    data_dir: str | Path,
    *,
    pairs: Iterable[str] | None = None,
    price_col: str = "close",
    timestamp_col: str = "timestamp",
    drop_data_gaps: bool = True,
    max_rows: int | None = None,
) -> LogReturnDataset:
    close_prices = load_close_prices(
        data_dir,
        pairs=pairs,
        price_col=price_col,
        timestamp_col=timestamp_col,
        drop_data_gaps=drop_data_gaps,
        max_rows=max_rows,
    )
    log_returns, timestamps, loaded_pairs = compute_log_returns(close_prices)
    return LogReturnDataset(
        log_returns=log_returns,
        timestamps=timestamps,
        pairs=loaded_pairs,
        close_prices=close_prices,
        metadata={
            "source": "parquet",
            "loaded_from_cache": False,
        },
    )

