#pragma once

#include <torch/extension.h>

namespace forex_cuda {

void step_inplace(
    torch::Tensor market,
    torch::Tensor balances,
    torch::Tensor free_cash,
    torch::Tensor margin_used,
    torch::Tensor margin_ratio,
    torch::Tensor positions,
    torch::Tensor time_indices,
    torch::Tensor reward_ema,
    torch::Tensor reward_ema_sq,
    torch::Tensor actions,
    torch::Tensor commission_bps,
    torch::Tensor spread_bps,
    torch::Tensor slippage_bps,
    torch::Tensor long_margin_requirement,
    torch::Tensor short_margin_requirement,
    torch::Tensor rewards,
    torch::Tensor terminated,
    double max_abs_weight,
    double max_leverage,
    double margin_call_ratio,
    double margin_call_penalty,
    double risk_ema_alpha,
    double risk_penalty_coef
);

}  // namespace forex_cuda
