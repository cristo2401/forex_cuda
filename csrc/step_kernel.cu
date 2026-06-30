#include "step_kernel.h"

#include "portfolio_math.h"

namespace forex_cuda {

namespace {

__global__ void step_kernel(
    const float* __restrict__ market,
    float* __restrict__ balances,
    float* __restrict__ free_cash,
    float* __restrict__ margin_used,
    float* __restrict__ margin_ratio,
    float* __restrict__ positions,
    int64_t* __restrict__ time_indices,
    float* __restrict__ reward_ema,
    float* __restrict__ reward_ema_sq,
    const float* __restrict__ actions,
    const float* __restrict__ commission_bps,
    const float* __restrict__ spread_bps,
    const float* __restrict__ slippage_bps,
    const float* __restrict__ long_margin_requirement,
    const float* __restrict__ short_margin_requirement,
    float* __restrict__ rewards,
    bool* __restrict__ terminated,
    int64_t num_envs,
    int64_t num_assets,
    int64_t num_steps,
    float max_abs_weight,
    float max_leverage,
    float margin_call_ratio,
    float margin_call_penalty,
    float risk_ema_alpha,
    float risk_penalty_coef
) {
    const int64_t env = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (env >= num_envs) {
        return;
    }

    const int64_t time_index = time_indices[env];
    if (terminated[env] || time_index < 0 || time_index >= num_steps) {
        rewards[env] = 0.0f;
        terminated[env] = true;
        return;
    }

    const int64_t market_env_offset = env * num_assets * num_steps + time_index;
    const int64_t state_env_offset = env * num_assets;
    const float equity_before = balances[env];

    if (!(equity_before > kMinEquity)) {
        balances[env] = 0.0f;
        free_cash[env] = 0.0f;
        margin_used[env] = 0.0f;
        margin_ratio[env] = 0.0f;
        rewards[env] = 0.0f;
        terminated[env] = true;
        return;
    }

    const float* asset_log_returns = market + market_env_offset;
    float* env_positions = positions + state_env_offset;
    const float* env_actions = actions + state_env_offset;
    const float transaction_cost_bps =
        commission_bps[env] + spread_bps[env] + slippage_bps[env];

    const StepStats stats = compute_portfolio_step(
        asset_log_returns,
        num_steps,
        env_positions,
        env_actions,
        long_margin_requirement,
        short_margin_requirement,
        equity_before,
        num_assets,
        transaction_cost_bps,
        max_abs_weight,
        max_leverage
    );

    margin_used[env] = stats.margin_used;
    margin_ratio[env] = stats.margin_ratio;

    if (stats.margin_ratio > margin_call_ratio) {
        free_cash[env] = 0.0f;
        rewards[env] = -margin_call_penalty;
        terminated[env] = true;
        return;
    }

    const float abs_sum = action_abs_sum(env_actions, num_assets, max_abs_weight);
    const float scale = leverage_scale(abs_sum, max_leverage);
    for (int64_t asset = 0; asset < num_assets; ++asset) {
        env_positions[asset] = effective_target_weight(
            env_actions[asset],
            max_abs_weight,
            scale
        );
    }

    balances[env] = stats.equity_after;
    free_cash[env] = stats.free_cash;

    float adjusted_reward = stats.reward;
    if (risk_ema_alpha > 0.0f) {
        const float previous_mean = reward_ema[env];
        const float previous_sq = reward_ema_sq[env];
        const float alpha = risk_ema_alpha > 1.0f ? 1.0f : risk_ema_alpha;
        const float new_mean = previous_mean + alpha * (stats.reward - previous_mean);
        const float new_sq = previous_sq + alpha * (stats.reward * stats.reward - previous_sq);
        const float variance = fx_max(new_sq - new_mean * new_mean, 0.0f);
        reward_ema[env] = new_mean;
        reward_ema_sq[env] = new_sq;
        adjusted_reward -= risk_penalty_coef * sqrtf(variance);
    }
    rewards[env] = adjusted_reward;

    const int64_t next_time_index = time_index + 1;
    time_indices[env] = next_time_index;
    terminated[env] = next_time_index >= num_steps || stats.equity_after <= kMinEquity;
}

}  // namespace

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
) {
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((num_envs + threads_per_block - 1) / threads_per_block);
    step_kernel<<<blocks, threads_per_block, 0, stream>>>(
        market,
        balances,
        free_cash,
        margin_used,
        margin_ratio,
        positions,
        time_indices,
        reward_ema,
        reward_ema_sq,
        actions,
        commission_bps,
        spread_bps,
        slippage_bps,
        long_margin_requirement,
        short_margin_requirement,
        rewards,
        terminated,
        num_envs,
        num_assets,
        num_steps,
        max_abs_weight,
        max_leverage,
        margin_call_ratio,
        margin_call_penalty,
        risk_ema_alpha,
        risk_penalty_coef
    );
}

}  // namespace forex_cuda
