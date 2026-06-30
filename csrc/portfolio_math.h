#pragma once

#include <cstdint>
#include <math.h>

#ifdef __CUDACC__
#define FX_CUDA_HOST_DEVICE __host__ __device__
#else
#define FX_CUDA_HOST_DEVICE
#endif

namespace forex_cuda {

constexpr float kMinEquity = 1.0e-8f;
constexpr float kBpsToFraction = 1.0e-4f;

struct StepStats {
    float weighted_log_return;
    float turnover;
    float transaction_cost;
    float margin_used;
    float margin_ratio;
    float equity_after;
    float reward;
    float free_cash;
};

FX_CUDA_HOST_DEVICE inline float fx_abs(float value) {
    return value < 0.0f ? -value : value;
}

FX_CUDA_HOST_DEVICE inline float fx_max(float lhs, float rhs) {
    return lhs > rhs ? lhs : rhs;
}

FX_CUDA_HOST_DEVICE inline float sanitize_weight(float weight, float max_abs_weight) {
    if (!(weight == weight)) {
        return 0.0f;
    }
    if (weight > max_abs_weight) {
        return max_abs_weight;
    }
    if (weight < -max_abs_weight) {
        return -max_abs_weight;
    }
    return weight;
}

FX_CUDA_HOST_DEVICE inline float action_abs_sum(
    const float* actions,
    int64_t num_assets,
    float max_abs_weight
) {
    float total = 0.0f;
    for (int64_t asset = 0; asset < num_assets; ++asset) {
        total += fx_abs(sanitize_weight(actions[asset], max_abs_weight));
    }
    return total;
}

FX_CUDA_HOST_DEVICE inline float leverage_scale(float abs_sum, float max_leverage) {
    if (abs_sum > max_leverage && abs_sum > 0.0f) {
        return max_leverage / abs_sum;
    }
    return 1.0f;
}

FX_CUDA_HOST_DEVICE inline float effective_target_weight(
    float action,
    float max_abs_weight,
    float scale
) {
    return sanitize_weight(action, max_abs_weight) * scale;
}

FX_CUDA_HOST_DEVICE inline StepStats compute_portfolio_step(
    const float* asset_log_returns,
    int64_t return_stride,
    const float* current_positions,
    const float* actions,
    const float* long_margin_requirement,
    const float* short_margin_requirement,
    float equity_before,
    int64_t num_assets,
    float transaction_cost_bps,
    float max_abs_weight,
    float max_leverage
) {
    const float abs_sum = action_abs_sum(actions, num_assets, max_abs_weight);
    const float scale = leverage_scale(abs_sum, max_leverage);
    const float cost_rate = transaction_cost_bps * kBpsToFraction;

    StepStats stats{};
    float effective_abs_sum = 0.0f;

    for (int64_t asset = 0; asset < num_assets; ++asset) {
        const float target_weight = effective_target_weight(
            actions[asset],
            max_abs_weight,
            scale
        );
        const float log_return = asset_log_returns[asset * return_stride];

        stats.weighted_log_return += target_weight * log_return;
        stats.turnover += fx_abs(target_weight - current_positions[asset]);
        effective_abs_sum += fx_abs(target_weight);
        const float margin_requirement =
            target_weight >= 0.0f ? long_margin_requirement[asset] : short_margin_requirement[asset];
        stats.margin_used += fx_abs(target_weight) * equity_before * margin_requirement;
    }

    stats.margin_ratio = stats.margin_used / fx_max(equity_before, kMinEquity);
    const float gross_equity = equity_before * expf(stats.weighted_log_return);
    stats.transaction_cost = equity_before * stats.turnover * cost_rate;
    stats.equity_after = gross_equity - stats.transaction_cost;
    if (!(stats.equity_after > 0.0f)) {
        stats.equity_after = 0.0f;
    }

    stats.reward = logf(
        fx_max(stats.equity_after, kMinEquity) / fx_max(equity_before, kMinEquity)
    );
    stats.free_cash = fx_max(stats.equity_after - stats.margin_used, 0.0f);
    return stats;
}

}  // namespace forex_cuda
