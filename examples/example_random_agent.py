from __future__ import annotations

import torch

from forex_cuda import CuPortfolioEnv, PortfolioEnvConfig


def main() -> None:
    num_envs = 4096
    num_assets = 8
    num_steps = 512

    generator = torch.Generator(device="cuda").manual_seed(7)
    log_returns = torch.randn(
        num_envs,
        num_assets,
        num_steps,
        device="cuda",
        generator=generator,
    ) * 0.001

    env = CuPortfolioEnv(
        log_returns,
        config=PortfolioEnvConfig(
            initial_balance=10_000.0,
            commission_bps_range=(0.5, 2.0),
            spread_bps_range=(0.0, 0.8),
            slippage_bps_range=(0.0, 0.4),
            max_abs_weight=1.0,
            max_leverage=1.0,
        ),
    )

    obs, info = env.reset(seed=123)
    print("reset", obs.shape, info)

    for _ in range(10):
        actions = torch.empty(num_envs, num_assets, device="cuda").uniform_(-0.25, 0.25)
        obs, rewards, terminated, truncated, info = env.step(actions)
        if terminated.all():
            break

    print(
        "done",
        {
            "obs_shape": tuple(obs.shape),
            "mean_reward": float(rewards.mean().item()),
            "mean_balance": float(env.state.balances.mean().item()),
            "mean_cost_bps": float(env.costs.total_bps().mean().item()),
            "time": int(info["time_indices"].max().item()),
            "terminated": bool(terminated.all().item()),
            "truncated": bool(truncated.any().item()),
        },
    )


if __name__ == "__main__":
    main()
