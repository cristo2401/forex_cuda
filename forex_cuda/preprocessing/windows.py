from __future__ import annotations

import numpy as np


def make_start_indices(
    *,
    total_steps: int,
    num_envs: int,
    episode_steps: int,
    seed: int | None = None,
    random_starts: bool = True,
) -> np.ndarray:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if episode_steps <= 0:
        raise ValueError("episode_steps must be positive")
    if episode_steps > total_steps:
        raise ValueError("episode_steps exceeds total_steps")

    max_start = total_steps - episode_steps
    if max_start == 0:
        return np.zeros(num_envs, dtype=np.int64)
    if random_starts:
        rng = np.random.default_rng(seed)
        return rng.integers(0, max_start + 1, size=num_envs, dtype=np.int64)

    starts = np.linspace(0, max_start, num=num_envs)
    return np.rint(starts).astype(np.int64)


def make_numpy_rolling_market(
    log_returns: np.ndarray,
    *,
    starts: np.ndarray,
    episode_steps: int,
) -> np.ndarray:
    """Build [num_envs, num_assets, episode_steps] CPU windows from [assets, time]."""
    if log_returns.ndim != 2:
        raise ValueError("log_returns must have shape [num_assets, total_steps]")
    if starts.ndim != 1:
        raise ValueError("starts must have shape [num_envs]")
    if episode_steps <= 0:
        raise ValueError("episode_steps must be positive")

    num_assets, total_steps = log_returns.shape
    max_start = total_steps - episode_steps
    if max_start < 0:
        raise ValueError("episode_steps exceeds total_steps")
    if np.any(starts < 0) or np.any(starts > max_start):
        raise ValueError("starts contain values outside the valid episode range")

    offsets = np.arange(episode_steps, dtype=np.int64)
    indices = starts.astype(np.int64, copy=False)[:, None] + offsets[None, :]
    market = log_returns[:, indices]
    market = np.transpose(market, (1, 0, 2))
    if market.shape != (len(starts), num_assets, episode_steps):
        raise RuntimeError(f"unexpected market shape: {market.shape}")
    return np.ascontiguousarray(market, dtype=np.float32)

