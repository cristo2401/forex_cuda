from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from forex_cuda import CuPortfolioEnv, PortfolioEnvConfig
from forex_cuda.preprocessing import (
    build_or_load_preprocessed_dataset,
    make_numpy_rolling_market,
    make_start_indices,
)


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

def cpu_python_loop(
    market: np.ndarray,
    actions: np.ndarray,
    commission_bps: np.ndarray,
    spread_bps: np.ndarray,
    slippage_bps: np.ndarray,
    long_margin_requirement: np.ndarray,
    short_margin_requirement: np.ndarray,
    *,
    initial_balance: float,
    max_abs_weight: float,
    max_leverage: float,
    margin_call_ratio: float,
) -> dict[str, float]:
    num_envs, num_assets, num_steps = market.shape
    balances = np.full(num_envs, initial_balance, dtype=np.float32)
    positions = np.zeros((num_envs, num_assets), dtype=np.float32)
    terminated = np.zeros(num_envs, dtype=np.bool_)
    margin_called = np.zeros(num_envs, dtype=np.bool_)
    total_cost_bps = commission_bps + spread_bps + slippage_bps
    margin_used = np.zeros(num_envs, dtype=np.float32)

    started = time.perf_counter()
    for step in range(num_steps):
        for env in range(num_envs):
            if terminated[env]:
                continue

            raw_action = actions[step, env]
            clipped_action = np.clip(raw_action, -max_abs_weight, max_abs_weight)
            abs_sum = float(np.abs(clipped_action).sum())
            scale = max_leverage / abs_sum if abs_sum > max_leverage and abs_sum > 0.0 else 1.0
            target = clipped_action * scale

            equity_before = float(balances[env])
            margin_requirement = np.where(
                target >= 0.0,
                long_margin_requirement,
                short_margin_requirement,
            )
            margin_used[env] = float((np.abs(target) * equity_before * margin_requirement).sum())
            if margin_used[env] / max(equity_before, 1.0e-8) > margin_call_ratio:
                terminated[env] = True
                margin_called[env] = True
                continue

            weighted_log_return = float((target * market[env, :, step]).sum())
            turnover = float(np.abs(target - positions[env]).sum())
            gross_equity = equity_before * float(np.exp(weighted_log_return))
            transaction_cost = equity_before * turnover * float(total_cost_bps[env]) * 1.0e-4
            balances[env] = max(gross_equity - transaction_cost, 0.0)
            positions[env] = target

    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "mean_balance": float(balances.mean()),
        "final_terminated_count": num_envs,
        "margin_call_count": int(margin_called.sum()),
        "mean_margin_used": float(margin_used.mean()),
    }


def gpu_cuda_loop(
    market: np.ndarray,
    actions: np.ndarray,
    commission_bps: np.ndarray,
    spread_bps: np.ndarray,
    slippage_bps: np.ndarray,
    long_margin_requirement: np.ndarray,
    short_margin_requirement: np.ndarray,
    *,
    initial_balance: float,
    max_abs_weight: float,
    max_leverage: float,
    margin_call_ratio: float,
) -> dict[str, float]:
    import torch

    cuda_market = torch.as_tensor(market, dtype=torch.float32, device="cuda").contiguous()
    cuda_actions = torch.as_tensor(actions, dtype=torch.float32, device="cuda").contiguous()

    env = CuPortfolioEnv(
        cuda_market,
        config=PortfolioEnvConfig(
            initial_balance=initial_balance,
            max_abs_weight=max_abs_weight,
            max_leverage=max_leverage,
            margin_call_ratio=margin_call_ratio,
        ),
    )
    env.reset()
    env.set_margin_requirements(
        long_margin_requirement=long_margin_requirement,
        short_margin_requirement=short_margin_requirement,
    )
    env.set_costs(
        commission_bps=commission_bps,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
    )
    torch.cuda.synchronize()

    started = time.perf_counter()
    for step in range(cuda_market.shape[2]):
        env.step(cuda_actions[step])
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "mean_balance": float(env.state.balances.mean().item()),
        "final_terminated_count": int(env.terminated.sum().item()),
        "margin_call_count": int((env.state.margin_ratio > margin_call_ratio).sum().item()),
        "mean_margin_used": float(env.state.margin_used.mean().item()),
    }


def plot_results(results: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["CPU Python/Numpy", "GPU CUDA"]
    values = [
        results["cpu_python_numpy"]["seconds"],
        results["gpu_cuda_vectorized"]["seconds"],
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=["#6b7280", "#0f766e"])
    ax.set_ylabel("Seconds")
    ax.set_title("Real H1 parquet simulation")
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.4f}s", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--max-rows", type=int, default=80_000)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--episode-steps", type=int, default=256)
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--asset-leverage", type=float, default=30.0)
    parser.add_argument("--margin-call-ratio", type=float, default=1.0)
    parser.add_argument("--max-abs-weight", type=float, default=1.0)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--cache-name", type=str, default="h1_major8")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()

    load_started = time.perf_counter()
    dataset = build_or_load_preprocessed_dataset(
        args.data_dir,
        pairs=args.pairs,
        max_rows=args.max_rows,
        drop_data_gaps=True,
        cache_dir=args.cache_dir,
        cache_name=args.cache_name,
        refresh=args.refresh_cache,
    )
    dataset_load_seconds = time.perf_counter() - load_started

    num_assets, total_steps = dataset.log_returns.shape
    if args.episode_steps > total_steps:
        raise ValueError("episode_steps exceeds the available log-return history")

    rng = np.random.default_rng(args.seed)
    starts = make_start_indices(
        total_steps=total_steps,
        num_envs=args.num_envs,
        episode_steps=args.episode_steps,
        seed=args.seed,
    )
    market = make_numpy_rolling_market(
        dataset.log_returns,
        starts=starts,
        episode_steps=args.episode_steps,
    )
    actions = rng.uniform(
        low=-0.25,
        high=0.25,
        size=(args.episode_steps, args.num_envs, num_assets),
    ).astype(np.float32)
    commission_bps = rng.uniform(0.5, 2.0, size=args.num_envs).astype(np.float32)
    spread_bps = rng.uniform(0.0, 0.8, size=args.num_envs).astype(np.float32)
    slippage_bps = rng.uniform(0.0, 0.4, size=args.num_envs).astype(np.float32)
    margin_requirement = 1.0 / args.asset_leverage
    long_margin_requirement = np.full(num_assets, margin_requirement, dtype=np.float32)
    short_margin_requirement = np.full(num_assets, margin_requirement, dtype=np.float32)

    total_env_steps = args.num_envs * args.episode_steps
    cpu = cpu_python_loop(
        market,
        actions,
        commission_bps,
        spread_bps,
        slippage_bps,
        long_margin_requirement,
        short_margin_requirement,
        initial_balance=args.initial_balance,
        max_abs_weight=args.max_abs_weight,
        max_leverage=args.max_leverage,
        margin_call_ratio=args.margin_call_ratio,
    )
    gpu = gpu_cuda_loop(
        market,
        actions,
        commission_bps,
        spread_bps,
        slippage_bps,
        long_margin_requirement,
        short_margin_requirement,
        initial_balance=args.initial_balance,
        max_abs_weight=args.max_abs_weight,
        max_leverage=args.max_leverage,
        margin_call_ratio=args.margin_call_ratio,
    )

    results = {
        "metadata": {
            "pairs": dataset.pairs,
            "log_returns_shape": list(dataset.log_returns.shape),
            "market_shape": list(market.shape),
            "date_start": str(dataset.timestamps[0]),
            "date_end": str(dataset.timestamps[-1]),
            "total_env_steps": total_env_steps,
            "dataset_load_seconds": dataset_load_seconds,
            "loaded_from_cache": bool(dataset.metadata.get("loaded_from_cache", False)),
            "cache_root": dataset.metadata.get("cache_root"),
            "asset_leverage": args.asset_leverage,
            "long_margin_requirement": float(margin_requirement),
            "short_margin_requirement": float(margin_requirement),
            "margin_call_ratio": args.margin_call_ratio,
        },
        "cpu_python_numpy": {
            **cpu,
            "env_steps_per_second": total_env_steps / cpu["seconds"],
        },
        "gpu_cuda_vectorized": {
            **gpu,
            "env_steps_per_second": total_env_steps / gpu["seconds"],
        },
    }
    results["speedup"] = (
        results["cpu_python_numpy"]["seconds"]
        / results["gpu_cuda_vectorized"]["seconds"]
    )
    results["mean_balance_abs_diff"] = abs(
        results["cpu_python_numpy"]["mean_balance"]
        - results["gpu_cuda_vectorized"]["mean_balance"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "benchmark_real_parquet.json"
    plot_path = args.output_dir / "benchmark_real_parquet.png"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_results(results, plot_path)

    print(json.dumps(results, indent=2))
    print(f"saved {json_path}")
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
