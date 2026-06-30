from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StateTensors:
    balances: Any
    free_cash: Any
    margin_used: Any
    margin_ratio: Any
    positions: Any
    time_indices: Any
    reward_ema: Any
    reward_ema_sq: Any

    def observation(self):
        import torch

        return torch.cat(
            [
                self.balances.unsqueeze(1),
                self.free_cash.unsqueeze(1),
                self.margin_used.unsqueeze(1),
                self.margin_ratio.unsqueeze(1),
                self.positions,
            ],
            dim=1,
        )


@dataclass
class CostTensors:
    commission_bps: Any
    spread_bps: Any
    slippage_bps: Any

    def total_bps(self):
        return self.commission_bps + self.spread_bps + self.slippage_bps


@dataclass
class BrokerTensors:
    long_margin_requirement: Any
    short_margin_requirement: Any


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("forex_cuda requires PyTorch for tensor operations") from exc
    return torch


def ensure_cuda_contiguous_float32(name: str, tensor, ndim: int):
    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tensor.ndim}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must live on CUDA")
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def make_market_tensor(log_returns, *, num_envs: int | None = None, device: str = "cuda"):
    """Return a contiguous CUDA float32 market tensor with shape [N, A, T]."""
    torch = _torch()
    market = torch.as_tensor(log_returns, dtype=torch.float32, device=device)

    if market.ndim == 2:
        if num_envs is None:
            raise ValueError("num_envs is required when log_returns has shape [A, T]")
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        market = market.unsqueeze(0).repeat(num_envs, 1, 1)
    elif market.ndim == 3:
        if num_envs is not None and market.shape[0] != num_envs:
            raise ValueError(
                f"log_returns has {market.shape[0]} envs but num_envs={num_envs}"
            )
    else:
        raise ValueError("log_returns must have shape [A, T] or [N, A, T]")

    if min(market.shape) <= 0:
        raise ValueError("market tensor dimensions must be non-empty")
    return market.contiguous()


def make_rolling_market_tensor(
    log_returns,
    *,
    num_envs: int,
    episode_steps: int,
    start_indices=None,
    random_starts: bool = True,
    device: str = "cuda",
    generator=None,
):
    """Build [num_envs, num_assets, episode_steps] windows from [num_assets, total_steps].

    This is the practical bridge from one aligned historical FX matrix into many
    parallel CUDA environments. Each env receives a different time window.
    """
    torch = _torch()
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if episode_steps <= 0:
        raise ValueError("episode_steps must be positive")

    source = torch.as_tensor(log_returns, dtype=torch.float32, device=device)
    if source.ndim != 2:
        raise ValueError("log_returns must have shape [num_assets, total_steps]")

    num_assets, total_steps = map(int, source.shape)
    if num_assets <= 0 or total_steps <= 0:
        raise ValueError("log_returns dimensions must be non-empty")
    if episode_steps > total_steps:
        raise ValueError(
            f"episode_steps={episode_steps} exceeds total_steps={total_steps}"
        )

    max_start = total_steps - episode_steps
    if start_indices is None:
        if max_start == 0:
            starts = torch.zeros(num_envs, dtype=torch.int64, device=device)
        elif random_starts:
            starts = torch.randint(
                low=0,
                high=max_start + 1,
                size=(num_envs,),
                dtype=torch.int64,
                device=device,
                generator=generator,
            )
        else:
            starts = torch.linspace(0, max_start, steps=num_envs, device=device)
            starts = starts.round().to(torch.int64)
    else:
        starts = torch.as_tensor(start_indices, dtype=torch.int64, device=device)
        if starts.shape != (num_envs,):
            raise ValueError(f"start_indices must have shape [{num_envs}]")

    if bool(torch.any(starts < 0).item()) or bool(torch.any(starts > max_start).item()):
        raise ValueError("start_indices are outside the valid episode range")

    offsets = torch.arange(episode_steps, dtype=torch.int64, device=device)
    gather_indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
    gather_indices = gather_indices.unsqueeze(1).expand(-1, num_assets, -1)
    source_by_env = source.unsqueeze(0).expand(num_envs, -1, -1)
    return source_by_env.gather(2, gather_indices).contiguous()


def make_state_tensors(
    *,
    num_envs: int,
    num_assets: int,
    initial_balance: float,
    device,
) -> StateTensors:
    torch = _torch()
    balances = torch.full((num_envs,), initial_balance, dtype=torch.float32, device=device)
    free_cash = torch.full((num_envs,), initial_balance, dtype=torch.float32, device=device)
    margin_used = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    margin_ratio = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    positions = torch.zeros((num_envs, num_assets), dtype=torch.float32, device=device)
    time_indices = torch.zeros((num_envs,), dtype=torch.int64, device=device)
    reward_ema = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    reward_ema_sq = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    return StateTensors(
        balances=balances.contiguous(),
        free_cash=free_cash.contiguous(),
        margin_used=margin_used.contiguous(),
        margin_ratio=margin_ratio.contiguous(),
        positions=positions.contiguous(),
        time_indices=time_indices.contiguous(),
        reward_ema=reward_ema.contiguous(),
        reward_ema_sq=reward_ema_sq.contiguous(),
    )


def _fill_cost_tensor(tensor, fixed_value: float, value_range: tuple[float, float] | None):
    if value_range is None:
        tensor.fill_(fixed_value)
        return tensor

    low, high = value_range
    if low == high:
        tensor.fill_(low)
        return tensor
    tensor.uniform_(low, high)
    return tensor


def reset_cost_tensors(costs: CostTensors, config) -> CostTensors:
    _fill_cost_tensor(
        costs.commission_bps,
        float(config.transaction_cost_bps),
        config.commission_bps_range,
    )
    _fill_cost_tensor(
        costs.spread_bps,
        float(config.spread_bps),
        config.spread_bps_range,
    )
    _fill_cost_tensor(
        costs.slippage_bps,
        float(config.slippage_bps),
        config.slippage_bps_range,
    )
    return costs


def make_cost_tensors(*, num_envs: int, config, device) -> CostTensors:
    torch = _torch()
    costs = CostTensors(
        commission_bps=torch.empty(num_envs, dtype=torch.float32, device=device).contiguous(),
        spread_bps=torch.empty(num_envs, dtype=torch.float32, device=device).contiguous(),
        slippage_bps=torch.empty(num_envs, dtype=torch.float32, device=device).contiguous(),
    )
    return reset_cost_tensors(costs, config)


def assign_margin_requirement_tensor(
    name: str,
    destination,
    value,
    *,
    num_envs: int,
    num_assets: int,
    device,
):
    torch = _torch()
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        destination.fill_(float(value))
        return destination

    source = torch.as_tensor(value, dtype=torch.float32, device=device)
    if not bool(torch.all(source > 0).item()):
        raise ValueError(f"{name} values must be positive")

    if source.shape == (num_assets,):
        destination.copy_(source.unsqueeze(0).expand(num_envs, -1).contiguous())
        return destination
    if source.shape == (num_envs, num_assets):
        destination.copy_(source.contiguous())
        return destination

    raise ValueError(
        f"{name} must be a scalar, have shape [{num_assets}], "
        f"or have shape [{num_envs}, {num_assets}]"
    )
    return destination


def assign_margin_requirement_from_leverage(
    name: str,
    long_destination,
    short_destination,
    value,
    *,
    num_envs: int,
    num_assets: int,
    device,
) -> BrokerTensors:
    torch = _torch()
    leverage = torch.as_tensor(value, dtype=torch.float32, device=device)
    if leverage.ndim == 0:
        if float(leverage.item()) <= 0:
            raise ValueError(f"{name} must be positive")
        margin_requirement = torch.full(
            (num_envs, num_assets),
            1.0 / float(leverage.item()),
            dtype=torch.float32,
            device=device,
        )
    else:
        if not bool(torch.all(leverage > 0).item()):
            raise ValueError(f"{name} values must be positive")
        if leverage.shape == (num_assets,):
            margin_requirement = (1.0 / leverage).unsqueeze(0).expand(num_envs, -1).contiguous()
        elif leverage.shape == (num_envs, num_assets):
            margin_requirement = (1.0 / leverage).contiguous()
        else:
            raise ValueError(
                f"{name} must be a scalar, have shape [{num_assets}], "
                f"or have shape [{num_envs}, {num_assets}]"
            )

    long_destination.copy_(margin_requirement)
    short_destination.copy_(margin_requirement)
    return BrokerTensors(
        long_margin_requirement=long_destination,
        short_margin_requirement=short_destination,
    )


def reset_broker_tensors(
    broker: BrokerTensors,
    config,
    *,
    num_envs: int,
    num_assets: int,
    device,
) -> BrokerTensors:
    if config.margin_requirement is None:
        assign_margin_requirement_from_leverage(
            "asset_leverage",
            broker.long_margin_requirement,
            broker.short_margin_requirement,
            config.asset_leverage,
            num_envs=num_envs,
            num_assets=num_assets,
            device=device,
        )
    else:
        assign_margin_requirement_tensor(
            "margin_requirement",
            broker.long_margin_requirement,
            config.margin_requirement,
            num_envs=num_envs,
            num_assets=num_assets,
            device=device,
        )
        broker.short_margin_requirement.copy_(broker.long_margin_requirement)

    if config.long_margin_requirement is not None:
        assign_margin_requirement_tensor(
            "long_margin_requirement",
            broker.long_margin_requirement,
            config.long_margin_requirement,
            num_envs=num_envs,
            num_assets=num_assets,
            device=device,
        )
    if config.short_margin_requirement is not None:
        assign_margin_requirement_tensor(
            "short_margin_requirement",
            broker.short_margin_requirement,
            config.short_margin_requirement,
            num_envs=num_envs,
            num_assets=num_assets,
            device=device,
        )
    return broker


def make_broker_tensors(*, num_envs: int, num_assets: int, config, device) -> BrokerTensors:
    torch = _torch()
    broker = BrokerTensors(
        long_margin_requirement=torch.empty(
            num_envs,
            num_assets,
            dtype=torch.float32,
            device=device,
        ).contiguous(),
        short_margin_requirement=torch.empty(
            num_envs,
            num_assets,
            dtype=torch.float32,
            device=device,
        ).contiguous(),
    )
    return reset_broker_tensors(
        broker,
        config,
        num_envs=num_envs,
        num_assets=num_assets,
        device=device,
    )


def assign_cost_tensor(name: str, destination, value, *, num_envs: int, device):
    torch = _torch()
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        destination.fill_(float(value))
        return destination

    source = torch.as_tensor(value, dtype=torch.float32, device=device)
    if source.shape != (num_envs,):
        raise ValueError(f"{name} must be a scalar or have shape [{num_envs}]")
    if not bool(torch.all(source >= 0).item()):
        raise ValueError(f"{name} values must be non-negative")
    destination.copy_(source.contiguous())
    return destination


def randomize_cost_tensor(name: str, destination, value_range: tuple[float, float]):
    low, high = value_range
    if low < 0 or high < 0:
        raise ValueError(f"{name} range values must be non-negative")
    if low > high:
        raise ValueError(f"{name} lower bound must be <= upper bound")
    return _fill_cost_tensor(destination, low, (low, high))


def make_action_tensor(actions, *, num_envs: int, num_assets: int, device):
    torch = _torch()
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
    if action_tensor.shape != (num_envs, num_assets):
        raise ValueError(
            "actions must have shape "
            f"[num_envs, num_assets] = [{num_envs}, {num_assets}], "
            f"got {tuple(action_tensor.shape)}"
        )
    return action_tensor.contiguous()
