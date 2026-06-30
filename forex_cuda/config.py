from __future__ import annotations

from dataclasses import dataclass

CostRange = tuple[float, float]
AssetLeverage = float | tuple[float, ...]
MarginRequirement = float | tuple[float, ...]


@dataclass(frozen=True)
class PortfolioEnvConfig:
    initial_balance: float = 10_000.0
    transaction_cost_bps: float = 1.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    commission_bps_range: CostRange | None = None
    spread_bps_range: CostRange | None = None
    slippage_bps_range: CostRange | None = None
    asset_leverage: AssetLeverage = 30.0
    margin_requirement: MarginRequirement | None = None
    long_margin_requirement: MarginRequirement | None = None
    short_margin_requirement: MarginRequirement | None = None
    margin_call_ratio: float = 1.0
    margin_call_penalty: float = 1.0
    risk_ema_alpha: float = 0.01
    risk_penalty_coef: float = 0.0
    max_abs_weight: float = 1.0
    max_leverage: float = 1.0
    gymnasium_api: bool = True

    def validate(self) -> None:
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be non-negative")
        if self.spread_bps < 0:
            raise ValueError("spread_bps must be non-negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        self._validate_range("commission_bps_range", self.commission_bps_range)
        self._validate_range("spread_bps_range", self.spread_bps_range)
        self._validate_range("slippage_bps_range", self.slippage_bps_range)
        self._validate_asset_leverage()
        self._validate_margin_requirement("margin_requirement", self.margin_requirement)
        self._validate_margin_requirement(
            "long_margin_requirement",
            self.long_margin_requirement,
        )
        self._validate_margin_requirement(
            "short_margin_requirement",
            self.short_margin_requirement,
        )
        if self.margin_call_ratio <= 0:
            raise ValueError("margin_call_ratio must be positive")
        if self.margin_call_penalty < 0:
            raise ValueError("margin_call_penalty must be non-negative")
        if not 0 <= self.risk_ema_alpha <= 1:
            raise ValueError("risk_ema_alpha must be in [0, 1]")
        if self.risk_penalty_coef < 0:
            raise ValueError("risk_penalty_coef must be non-negative")
        if self.max_abs_weight <= 0:
            raise ValueError("max_abs_weight must be positive")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")

    @staticmethod
    def _validate_range(name: str, value: CostRange | None) -> None:
        if value is None:
            return
        low, high = value
        if low < 0 or high < 0:
            raise ValueError(f"{name} values must be non-negative")
        if low > high:
            raise ValueError(f"{name} lower bound must be <= upper bound")

    def _validate_asset_leverage(self) -> None:
        if isinstance(self.asset_leverage, (int, float)):
            if self.asset_leverage <= 0:
                raise ValueError("asset_leverage must be positive")
            return
        if len(self.asset_leverage) == 0:
            raise ValueError("asset_leverage sequence must be non-empty")
        if any(value <= 0 for value in self.asset_leverage):
            raise ValueError("asset_leverage values must be positive")

    @staticmethod
    def _validate_margin_requirement(name: str, value: MarginRequirement | None) -> None:
        if value is None:
            return
        if isinstance(value, (int, float)):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            return
        if len(value) == 0:
            raise ValueError(f"{name} sequence must be non-empty")
        if any(item <= 0 for item in value):
            raise ValueError(f"{name} values must be positive")
