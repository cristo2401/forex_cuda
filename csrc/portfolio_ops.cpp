#include "portfolio_ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "step_kernel.h"

namespace forex_cuda {

namespace {

void check_cuda_contiguous_float(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
}

void check_cuda_contiguous_bool(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kBool, name, " must be bool");
}

void check_cuda_contiguous_int64(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt64, name, " must be int64");
}

}  // namespace

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
) {
    check_cuda_contiguous_float(market, "market");
    check_cuda_contiguous_float(balances, "balances");
    check_cuda_contiguous_float(free_cash, "free_cash");
    check_cuda_contiguous_float(margin_used, "margin_used");
    check_cuda_contiguous_float(margin_ratio, "margin_ratio");
    check_cuda_contiguous_float(positions, "positions");
    check_cuda_contiguous_int64(time_indices, "time_indices");
    check_cuda_contiguous_float(reward_ema, "reward_ema");
    check_cuda_contiguous_float(reward_ema_sq, "reward_ema_sq");
    check_cuda_contiguous_float(actions, "actions");
    check_cuda_contiguous_float(commission_bps, "commission_bps");
    check_cuda_contiguous_float(spread_bps, "spread_bps");
    check_cuda_contiguous_float(slippage_bps, "slippage_bps");
    check_cuda_contiguous_float(long_margin_requirement, "long_margin_requirement");
    check_cuda_contiguous_float(short_margin_requirement, "short_margin_requirement");
    check_cuda_contiguous_float(rewards, "rewards");
    check_cuda_contiguous_bool(terminated, "terminated");

    TORCH_CHECK(market.dim() == 3, "market must have shape [num_envs, num_assets, num_steps]");
    const int64_t num_envs = market.size(0);
    const int64_t num_assets = market.size(1);
    const int64_t num_steps = market.size(2);
    TORCH_CHECK(num_envs > 0 && num_assets > 0 && num_steps > 0, "market dimensions must be non-empty");

    TORCH_CHECK(balances.dim() == 1 && balances.size(0) == num_envs, "balances must have shape [num_envs]");
    TORCH_CHECK(free_cash.dim() == 1 && free_cash.size(0) == num_envs, "free_cash must have shape [num_envs]");
    TORCH_CHECK(margin_used.dim() == 1 && margin_used.size(0) == num_envs, "margin_used must have shape [num_envs]");
    TORCH_CHECK(margin_ratio.dim() == 1 && margin_ratio.size(0) == num_envs, "margin_ratio must have shape [num_envs]");
    TORCH_CHECK(time_indices.dim() == 1 && time_indices.size(0) == num_envs, "time_indices must have shape [num_envs]");
    TORCH_CHECK(reward_ema.dim() == 1 && reward_ema.size(0) == num_envs, "reward_ema must have shape [num_envs]");
    TORCH_CHECK(reward_ema_sq.dim() == 1 && reward_ema_sq.size(0) == num_envs, "reward_ema_sq must have shape [num_envs]");
    TORCH_CHECK(commission_bps.dim() == 1 && commission_bps.size(0) == num_envs, "commission_bps must have shape [num_envs]");
    TORCH_CHECK(spread_bps.dim() == 1 && spread_bps.size(0) == num_envs, "spread_bps must have shape [num_envs]");
    TORCH_CHECK(slippage_bps.dim() == 1 && slippage_bps.size(0) == num_envs, "slippage_bps must have shape [num_envs]");
    TORCH_CHECK(
        long_margin_requirement.dim() == 1 && long_margin_requirement.size(0) == num_assets,
        "long_margin_requirement must have shape [num_assets]"
    );
    TORCH_CHECK(
        short_margin_requirement.dim() == 1 && short_margin_requirement.size(0) == num_assets,
        "short_margin_requirement must have shape [num_assets]"
    );
    TORCH_CHECK(rewards.dim() == 1 && rewards.size(0) == num_envs, "rewards must have shape [num_envs]");
    TORCH_CHECK(terminated.dim() == 1 && terminated.size(0) == num_envs, "terminated must have shape [num_envs]");
    TORCH_CHECK(
        positions.dim() == 2 && positions.size(0) == num_envs && positions.size(1) == num_assets,
        "positions must have shape [num_envs, num_assets]"
    );
    TORCH_CHECK(
        actions.dim() == 2 && actions.size(0) == num_envs && actions.size(1) == num_assets,
        "actions must have shape [num_envs, num_assets]"
    );

    TORCH_CHECK(market.device() == balances.device(), "all tensors must be on the same CUDA device");
    TORCH_CHECK(market.device() == free_cash.device(), "all tensors must be on the same CUDA device");
    TORCH_CHECK(market.device() == margin_used.device(), "all tensors must be on the same CUDA device");
    TORCH_CHECK(market.device() == margin_ratio.device(), "all tensors must be on the same CUDA device");
    TORCH_CHECK(market.device() == positions.device(), "all tensors must be on the same CUDA device");
    TORCH_CHECK(market.device() == time_indices.device(), "all tensors must be on the same CUDA device");
    TORCH_CHECK(market.device() == reward_ema.device(), "all tensors must be on the same CUDA device");
    TORCH_CHECK(market.device() == reward_ema_sq.device(), "all tensors must be on the same CUDA device");
    TORCH_CHECK(market.device() == actions.device(), "all tensors must be on the same CUDA device");
    TORCH_CHECK(market.device() == commission_bps.device(), "all tensors must be on the same CUDA device");
    TORCH_CHECK(market.device() == spread_bps.device(), "all tensors must be on the same CUDA device");
    TORCH_CHECK(market.device() == slippage_bps.device(), "all tensors must be on the same CUDA device");
    TORCH_CHECK(market.device() == long_margin_requirement.device(), "all tensors must be on the same CUDA device");
    TORCH_CHECK(market.device() == short_margin_requirement.device(), "all tensors must be on the same CUDA device");
    TORCH_CHECK(market.device() == rewards.device(), "all tensors must be on the same CUDA device");
    TORCH_CHECK(market.device() == terminated.device(), "all tensors must be on the same CUDA device");

    TORCH_CHECK(max_abs_weight > 0.0, "max_abs_weight must be positive");
    TORCH_CHECK(max_leverage > 0.0, "max_leverage must be positive");
    TORCH_CHECK(margin_call_ratio > 0.0, "margin_call_ratio must be positive");
    TORCH_CHECK(margin_call_penalty >= 0.0, "margin_call_penalty must be non-negative");
    TORCH_CHECK(risk_ema_alpha >= 0.0 && risk_ema_alpha <= 1.0, "risk_ema_alpha must be in [0, 1]");
    TORCH_CHECK(risk_penalty_coef >= 0.0, "risk_penalty_coef must be non-negative");

    const c10::cuda::CUDAGuard device_guard(market.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(market.device().index());

    launch_step_kernel(
        market.data_ptr<float>(),
        balances.data_ptr<float>(),
        free_cash.data_ptr<float>(),
        margin_used.data_ptr<float>(),
        margin_ratio.data_ptr<float>(),
        positions.data_ptr<float>(),
        time_indices.data_ptr<int64_t>(),
        reward_ema.data_ptr<float>(),
        reward_ema_sq.data_ptr<float>(),
        actions.data_ptr<float>(),
        commission_bps.data_ptr<float>(),
        spread_bps.data_ptr<float>(),
        slippage_bps.data_ptr<float>(),
        long_margin_requirement.data_ptr<float>(),
        short_margin_requirement.data_ptr<float>(),
        rewards.data_ptr<float>(),
        terminated.data_ptr<bool>(),
        num_envs,
        num_assets,
        num_steps,
        static_cast<float>(max_abs_weight),
        static_cast<float>(max_leverage),
        static_cast<float>(margin_call_ratio),
        static_cast<float>(margin_call_penalty),
        static_cast<float>(risk_ema_alpha),
        static_cast<float>(risk_penalty_coef),
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace forex_cuda
