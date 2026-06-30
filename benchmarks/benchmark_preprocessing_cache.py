from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from forex_cuda.preprocessing import build_or_load_preprocessed_dataset

DEFAULT_DATA_DIR = Path(os.environ.get("FOREX_CUDA_DATA_DIR", "data/parquet_h1"))
DEFAULT_PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "USDCAD",
    "AUDUSD",
    "NZDUSD",
    "EURJPY",
]


def plot_results(results: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["Parquet build", "Cache load"]
    values = [
        results["parquet_build_seconds"],
        results["cache_load_seconds"],
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=["#64748b", "#0f766e"])
    ax.set_ylabel("Seconds")
    ax.set_title("Preprocessing: parquet alignment vs cache")
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.4f}s", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--max-rows", type=int, default=80_000)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--cache-name", type=str, default="h1_major8")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()

    parquet_started = time.perf_counter()
    parquet_dataset = build_or_load_preprocessed_dataset(
        args.data_dir,
        pairs=args.pairs,
        max_rows=args.max_rows,
        drop_data_gaps=True,
        cache_dir=args.cache_dir,
        cache_name=args.cache_name,
        refresh=True,
    )
    parquet_build_seconds = time.perf_counter() - parquet_started

    cache_started = time.perf_counter()
    cache_dataset = build_or_load_preprocessed_dataset(
        args.data_dir,
        pairs=args.pairs,
        max_rows=args.max_rows,
        drop_data_gaps=True,
        cache_dir=args.cache_dir,
        cache_name=args.cache_name,
        refresh=False,
    )
    cache_load_seconds = time.perf_counter() - cache_started

    results = {
        "pairs": cache_dataset.pairs,
        "log_returns_shape": list(cache_dataset.log_returns.shape),
        "close_prices_shape": list(cache_dataset.close_prices.shape),
        "date_start": str(cache_dataset.timestamps[0]),
        "date_end": str(cache_dataset.timestamps[-1]),
        "cache_root": cache_dataset.metadata.get("cache_root"),
        "parquet_build_seconds": parquet_build_seconds,
        "cache_load_seconds": cache_load_seconds,
        "speedup": parquet_build_seconds / cache_load_seconds,
        "cache_loaded_from_cache": bool(cache_dataset.metadata.get("loaded_from_cache", False)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "benchmark_preprocessing_cache.json"
    plot_path = args.output_dir / "benchmark_preprocessing_cache.png"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_results(results, plot_path)

    print(json.dumps(results, indent=2))
    print(f"saved {json_path}")
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
