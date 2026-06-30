from __future__ import annotations

from typing import Any

import numpy as np

from .config import PortfolioEnvConfig
from .extension import load_extension
from .tensors import (
    assign_cost_tensor,
    assign_margin_requirement_from_leverage,
    assign_margin_requirement_tensor,
    make_action_tensor,
    make_broker_tensors,
    make_cost_tensors,
    make_market_tensor,
    make_state_tensors,
    randomize_cost_tensor,
    reset_cost_tensors,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - gymnasium is an install dependency.
    gym = None
    spaces = None


class CuPortfolioEnv(gym.Env if gym is not None else object):
    """Vectorized CUDA portfolio environment.

    Tensor contract:
    - market: [num_envs, num_assets, num_steps] float32 CUDA contiguous log returns.
    - balances/free_cash/time_indices: [num_envs] contiguous CUDA tensors.
    - positions/actions: [num_envs, num_assets] float32 CUDA contiguous target weights.
    - commission/spread/slippage: [num_envs] float32 CUDA contiguous cost tensors.
    - long/short_margin_requirement: [num_envs, num_assets] float32 CUDA contiguous margin rules.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        log_returns,
        *,
        num_envs: int | None = None,
        config: PortfolioEnvConfig | None = None,
        device: str = "cuda",
    ) -> None:
        self.config = config or PortfolioEnvConfig()
        self.config.validate()

        self.market = make_market_tensor(log_returns, num_envs=num_envs, device=device)
        self.num_envs, self.num_assets, self.num_steps = map(int, self.market.shape)
        self.device = self.market.device

        self.state = make_state_tensors(
            num_envs=self.num_envs,
            num_assets=self.num_assets,
            initial_balance=self.config.initial_balance,
            device=self.device,
        )
        self.costs = make_cost_tensors(
            num_envs=self.num_envs,
            config=self.config,
            device=self.device,
        )
        self.broker = make_broker_tensors(
            num_envs=self.num_envs,
            num_assets=self.num_assets,
            config=self.config,
            device=self.device,
        )

        import torch

        self.rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if spaces is not None:
            self.action_space = spaces.Box(
                low=-self.config.max_abs_weight,
                high=self.config.max_abs_weight,
                shape=(self.num_envs, self.num_assets),
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_envs, self.num_assets + 4),
                dtype=np.float32,
            )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if gym is not None:
            super().reset(seed=seed)

        import torch

        if seed is not None:
            torch.manual_seed(seed)

        self.state.balances.fill_(self.config.initial_balance)
        self.state.free_cash.fill_(self.config.initial_balance)
        self.state.margin_used.zero_()
        self.state.margin_ratio.zero_()
        self.state.positions.zero_()
        self.state.time_indices.zero_()
        self.state.reward_ema.zero_()
        self.state.reward_ema_sq.zero_()
        reset_cost_tensors(self.costs, self.config)
        self.rewards.zero_()
        self.terminated.zero_()
        self.truncated.zero_()

        info = {
            "market_shape": tuple(self.market.shape),
            "device": str(self.device),
            "cost_shape": (self.num_envs,),
            "margin_requirement_shape": (self.num_envs, self.num_assets),
            "options": options or {},
        }
        return self.state.observation(), info

    def set_asset_leverage(self, asset_leverage: float | Any) -> None:
        assign_margin_requirement_from_leverage(
            "asset_leverage",
            self.broker.long_margin_requirement,
            self.broker.short_margin_requirement,
            asset_leverage,
            num_envs=self.num_envs,
            num_assets=self.num_assets,
            device=self.device,
        )

    def set_margin_requirements(
        self,
        *,
        margin_requirement: float | Any | None = None,
        long_margin_requirement: float | Any | None = None,
        short_margin_requirement: float | Any | None = None,
    ) -> None:
        if margin_requirement is not None:
            assign_margin_requirement_tensor(
                "margin_requirement",
                self.broker.long_margin_requirement,
                margin_requirement,
                num_envs=self.num_envs,
                num_assets=self.num_assets,
                device=self.device,
            )
            self.broker.short_margin_requirement.copy_(
                self.broker.long_margin_requirement
            )
        if long_margin_requirement is not None:
            assign_margin_requirement_tensor(
                "long_margin_requirement",
                self.broker.long_margin_requirement,
                long_margin_requirement,
                num_envs=self.num_envs,
                num_assets=self.num_assets,
                device=self.device,
            )
        if short_margin_requirement is not None:
            assign_margin_requirement_tensor(
                "short_margin_requirement",
                self.broker.short_margin_requirement,
                short_margin_requirement,
                num_envs=self.num_envs,
                num_assets=self.num_assets,
                device=self.device,
            )

    def set_costs(
        self,
        *,
        commission_bps: float | Any | None = None,
        spread_bps: float | Any | None = None,
        slippage_bps: float | Any | None = None,
    ) -> None:
        if commission_bps is not None:
            assign_cost_tensor(
                "commission_bps",
                self.costs.commission_bps,
                commission_bps,
                num_envs=self.num_envs,
                device=self.device,
            )
        if spread_bps is not None:
            assign_cost_tensor(
                "spread_bps",
                self.costs.spread_bps,
                spread_bps,
                num_envs=self.num_envs,
                device=self.device,
            )
        if slippage_bps is not None:
            assign_cost_tensor(
                "slippage_bps",
                self.costs.slippage_bps,
                slippage_bps,
                num_envs=self.num_envs,
                device=self.device,
            )

    def randomize_costs(
        self,
        *,
        commission_bps_range: tuple[float, float] | None = None,
        spread_bps_range: tuple[float, float] | None = None,
        slippage_bps_range: tuple[float, float] | None = None,
    ) -> None:
        if commission_bps_range is not None:
            randomize_cost_tensor(
                "commission_bps",
                self.costs.commission_bps,
                commission_bps_range,
            )
        if spread_bps_range is not None:
            randomize_cost_tensor("spread_bps", self.costs.spread_bps, spread_bps_range)
        if slippage_bps_range is not None:
            randomize_cost_tensor(
                "slippage_bps",
                self.costs.slippage_bps,
                slippage_bps_range,
            )

    def step(self, actions):
        action_tensor = make_action_tensor(
            actions,
            num_envs=self.num_envs,
            num_assets=self.num_assets,
            device=self.device,
        )

        ext = load_extension()
        ext.step_inplace(
            self.market,
            self.state.balances,
            self.state.free_cash,
            self.state.margin_used,
            self.state.margin_ratio,
            self.state.positions,
            self.state.time_indices,
            self.state.reward_ema,
            self.state.reward_ema_sq,
            action_tensor,
            self.costs.commission_bps,
            self.costs.spread_bps,
            self.costs.slippage_bps,
            self.broker.long_margin_requirement,
            self.broker.short_margin_requirement,
            self.rewards,
            self.terminated,
            float(self.config.max_abs_weight),
            float(self.config.max_leverage),
            float(self.config.margin_call_ratio),
            float(self.config.margin_call_penalty),
            float(self.config.risk_ema_alpha),
            float(self.config.risk_penalty_coef),
        )

        observation = self.state.observation()
        info = {"time_indices": self.state.time_indices}

        if self.config.gymnasium_api:
            return observation, self.rewards, self.terminated, self.truncated, info
        return observation, self.rewards, self.terminated, info

    def step_legacy(self, actions):
        result = self.step(actions)
        if len(result) == 5:
            observation, rewards, terminated, _truncated, info = result
            return observation, rewards, terminated, info
        return result
