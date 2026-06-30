from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from forex_cuda import CuPortfolioEnv, PortfolioEnvConfig


def numpy_loop(log_returns: np.ndarray, actions: np.ndarray, transaction_cost_bps: float) -> float:
    num_envs, num_assets, num_steps = log_returns.shape
    balances = np.full(num_envs, 10_000.0, dtype=np.float32)
    positions = np.zeros((num_envs, num_assets), dtype=np.float32)
    cost_rate = transaction_cost_bps * 1.0e-4

    started = time.perf_counter()
    for step in range(num_steps):
        for env in range(num_envs):
            action = actions[step, env]
            abs_sum = np.abs(action).sum()
            if abs_sum > 1.0:
                action = action / abs_sum
            weighted_log_return = float((action * log_returns[env, :, step]).sum())
            turnover = float(np.abs(action - positions[env]).sum())
            equity_before = balances[env]
            gross_equity = equity_before * np.exp(weighted_log_return)
            transaction_cost = equity_before * turnover * cost_rate
            balances[env] = max(gross_equity - transaction_cost, 0.0)
            positions[env] = action
    return time.perf_counter() - started


def cuda_loop(log_returns: np.ndarray, actions: np.ndarray, transaction_cost_bps: float) -> float:
    import torch

    market = torch.as_tensor(log_returns, dtype=torch.float32, device="cuda").contiguous()
    cuda_actions = torch.as_tensor(actions, dtype=torch.float32, device="cuda").contiguous()
    env = CuPortfolioEnv(
        market,
        config=PortfolioEnvConfig(transaction_cost_bps=transaction_cost_bps),
    )
    env.reset()
    torch.cuda.synchronize()

    started = time.perf_counter()
    for step in range(market.shape[2]):
        env.step(cuda_actions[step])
    torch.cuda.synchronize()
    return time.perf_counter() - started


def plot_results(results: dict[str, float], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(results)
    values = [results[label] for label in labels]
    ax.bar(labels, values, color=["#6b7280", "#0f766e"])
    ax.set_ylabel("Seconds")
    ax.set_title("One million portfolio steps")
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.3f}s", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=10_000)
    parser.add_argument("--num-assets", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()

    total_steps = args.num_envs * args.num_steps
    if total_steps != 1_000_000:
        print(f"configured total steps: {total_steps:,}")

    rng = np.random.default_rng(123)
    log_returns = rng.normal(
        loc=0.0,
        scale=0.001,
        size=(args.num_envs, args.num_assets, args.num_steps),
    ).astype(np.float32)
    actions = rng.uniform(
        low=-0.25,
        high=0.25,
        size=(args.num_steps, args.num_envs, args.num_assets),
    ).astype(np.float32)

    results = {
        "python_numpy": numpy_loop(log_returns, actions, args.transaction_cost_bps),
        "cuda_vectorized": cuda_loop(log_returns, actions, args.transaction_cost_bps),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "benchmark_steps.json"
    plot_path = args.output_dir / "benchmark_steps.png"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_results(results, plot_path)
    print(json.dumps(results, indent=2))
    print(f"saved {json_path}")
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
