#include <torch/extension.h>

#include "portfolio_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "CUDA kernels for vectorized FX portfolio simulation";
    module.def(
        "step_inplace",
        &forex_cuda::step_inplace,
        "Run one in-place portfolio environment step on CUDA tensors",
        pybind11::arg("market"),
        pybind11::arg("balances"),
        pybind11::arg("free_cash"),
        pybind11::arg("margin_used"),
        pybind11::arg("margin_ratio"),
        pybind11::arg("positions"),
        pybind11::arg("time_indices"),
        pybind11::arg("reward_ema"),
        pybind11::arg("reward_ema_sq"),
        pybind11::arg("actions"),
        pybind11::arg("commission_bps"),
        pybind11::arg("spread_bps"),
        pybind11::arg("slippage_bps"),
        pybind11::arg("long_margin_requirement"),
        pybind11::arg("short_margin_requirement"),
        pybind11::arg("rewards"),
        pybind11::arg("terminated"),
        pybind11::arg("max_abs_weight"),
        pybind11::arg("max_leverage"),
        pybind11::arg("margin_call_ratio"),
        pybind11::arg("margin_call_penalty"),
        pybind11::arg("risk_ema_alpha"),
        pybind11::arg("risk_penalty_coef")
    );
}
