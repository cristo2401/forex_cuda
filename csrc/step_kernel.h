#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace forex_cuda {

void launch_step_kernel(
    const float* market,
    float* balances,
    float* free_cash,
    float* margin_used,
    float* margin_ratio,
    float* positions,
    int64_t* time_indices,
    float* reward_ema,
    float* reward_ema_sq,
    const float* actions,
    const float* commission_bps,
    const float* spread_bps,
    const float* slippage_bps,
    const float* long_margin_requirement,
    const float* short_margin_requirement,
    float* rewards,
    bool* terminated,
    int64_t num_envs,
    int64_t num_assets,
    int64_t num_steps,
    float max_abs_weight,
    float max_leverage,
    float margin_call_ratio,
    float margin_call_penalty,
    float risk_ema_alpha,
    float risk_penalty_coef,
    cudaStream_t stream
);

}  // namespace forex_cuda
