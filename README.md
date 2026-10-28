# forex-cuda

Vectorized CUDA environment for FX portfolio simulations with PyTorch tensors and a Gymnasium-style Python API.

[![CPU math checks](https://github.com/cristo2401/forex_cuda/actions/workflows/cpu-math.yml/badge.svg)](https://github.com/cristo2401/forex_cuda/actions/workflows/cpu-math.yml)

**Start here:** [run a CPU check](#check-the-math-without-a-gpu) · [install CUDA/PyTorch](docs/setup_cuda.md) · [benchmark scope](docs/BENCHMARKS.md) · [dataset format](docs/dataset_format.md).

The project explores how to keep many independent portfolio simulations on the
GPU and expose them through Python. It is a simulation component for research;
it does not provide a trained strategy or evidence of trading profitability.

> Research software only. This is not financial advice or a live-trading system.

## Highlights

- One CUDA thread per parallel market environment.
- SoA tensor layout for contiguous PyTorch/CUDA memory access.
- Multi-asset portfolio actions with per-environment transaction costs.
- Long/short margin requirements per asset and CUDA-side margin calls.
- Cacheable parquet preprocessing for real H1 FX datasets.
- A stored H1-data run reports 117.97M env-steps/sec for the timed simulation loop;
  see the [baseline, excluded work and reproducibility limits](docs/BENCHMARKS.md).

## Check the math without a GPU

Requires a C++17 compiler and Bash; Python, PyTorch and market data are not needed.

```bash
git clone https://github.com/cristo2401/forex_cuda.git
cd forex_cuda
bash scripts/run_cpp_math_test.sh
```

Expected output: `portfolio_math tests passed`. These checks cover two deterministic
fixtures for exposure, turnover, costs, margin, equity and reward. CI runs this
CPU check. It does not compile CUDA or establish full CPU/GPU numerical equivalence.

## Tensor Contract

All tensors that cross the Python/C++ boundary must be `.contiguous().cuda()` and use `torch.float32`, except `time_indices` (`torch.int64`) and `terminated` (`torch.bool`).

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `market` | `[num_envs, num_assets, num_steps]` | Historical log returns. Layout is SoA: environment, asset, then time as the contiguous inner axis. |
| `balances` | `[num_envs]` | Current equity per parallel environment. |
| `free_cash` | `[num_envs]` | Free margin after margin requirements and transaction costs. |
| `margin_used` | `[num_envs]` | Broker margin currently required by target exposure. |
| `margin_ratio` | `[num_envs]` | `margin_used / equity`; margin call triggers above `margin_call_ratio`. |
| `positions` | `[num_envs, num_assets]` | Current open portfolio weights. |
| `actions` | `[num_envs, num_assets]` | Target portfolio weights produced by the agent. |
| `commission_bps` | `[num_envs]` | Per-environment commission in basis points. |
| `spread_bps` | `[num_envs]` | Per-environment spread cost in basis points. |
| `slippage_bps` | `[num_envs]` | Per-environment slippage cost in basis points. |
| `long_margin_requirement` | `[num_assets]` | Margin requirement for long exposure per asset. |
| `short_margin_requirement` | `[num_assets]` | Margin requirement for short exposure per asset. |
| `reward_ema` | `[num_envs]` | EWMA state for risk-adjusted reward. |
| `reward_ema_sq` | `[num_envs]` | EWMA second moment for volatility penalty. |
| `rewards` | `[num_envs]` | Log reward written by the CUDA kernel. |
| `terminated` | `[num_envs]` | Done flag written by the CUDA kernel. |
| `time_indices` | `[num_envs]` | Per-environment cursor into the time axis. |

The CUDA kernel assigns one thread per environment. For each step it reads the log returns at `market[env, asset, time]`, applies the target action weights, reads `commission_bps[env] + spread_bps[env] + slippage_bps[env]`, computes per-asset margin from the long/short margin requirement tensors, subtracts transaction costs from turnover, and writes the updated balance/reward directly into output tensors.

Actions are notional portfolio exposure weights. `asset_leverage` is accepted as a convenience and converted internally to `margin_requirement = 1 / leverage`. Explicit `long_margin_requirement` and `short_margin_requirement` can override it when broker rules differ by side.

This design avoids a common PCIe bottleneck: the market matrix and state live in VRAM for the whole episode. Python only launches kernels and supplies action tensors already on CUDA, instead of copying market windows or per-environment state between CPU and GPU every step.

## Install

Use the project CUDA/PyTorch conda environment:

```bash
conda env create -f environment.yml
conda activate forex-cuda
# Install CUDA-enabled PyTorch for your CUDA version first. See docs/setup_cuda.md.
pip install -e . --no-build-isolation --no-deps
```

## Quick Start

```python
import torch
from forex_cuda import CuPortfolioEnv, PortfolioEnvConfig

log_returns = torch.randn(4096, 8, 512, device="cuda") * 0.001
env = CuPortfolioEnv(
    log_returns,
    config=PortfolioEnvConfig(
        commission_bps_range=(0.5, 2.0),
        spread_bps_range=(0.0, 0.8),
        slippage_bps_range=(0.0, 0.4),
        long_margin_requirement=(0.0333, 0.0333, 0.05, 0.05, 0.0333, 0.0333, 0.0333, 0.05),
        short_margin_requirement=(0.0333, 0.0333, 0.05, 0.05, 0.0333, 0.0333, 0.0333, 0.05),
        margin_call_ratio=1.0,
        risk_penalty_coef=0.0,
    ),
)

obs, info = env.reset()
actions = torch.empty(4096, 8, device="cuda").uniform_(-0.25, 0.25)
obs, rewards, terminated, truncated, info = env.step(actions)
```

For controlled simulation experiments, set explicit cost and margin assumptions:

```python
env.set_costs(
    commission_bps=0.8,
    spread_bps=0.35,
    slippage_bps=0.15,
)
env.set_margin_requirements(
    long_margin_requirement=[0.0333, 0.0333, 0.05, 0.05, 0.0333, 0.0333, 0.0333, 0.05],
    short_margin_requirement=[0.04, 0.04, 0.06, 0.06, 0.04, 0.04, 0.04, 0.06],
)
```

## Real H1 Parquet Data

The environment is multi-asset/multi-currency by construction: `num_assets` is the number of FX pairs, and each action row contains one target portfolio weight per pair.

Use the parquet loader and rolling window builder to turn one aligned historical timeline into many parallel CUDA episodes:

```python
from pathlib import Path

from forex_cuda import CuPortfolioEnv, PortfolioEnvConfig
from forex_cuda.preprocessing import build_or_load_preprocessed_dataset
from forex_cuda.tensors import make_rolling_market_tensor

data = build_or_load_preprocessed_dataset(
    Path("/path/to/parquet_h1"),
    pairs=["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "EURJPY"],
    max_rows=80_000,
    cache_dir="data_cache",
    cache_name="h1_major8",
)

market = make_rolling_market_tensor(
    data.log_returns,
    num_envs=4096,
    episode_steps=512,
    random_starts=True,
    device="cuda",
)

env = CuPortfolioEnv(market, config=PortfolioEnvConfig())
```

The preprocessing layer is intentionally separate from the CUDA env:

- `forex_cuda/preprocessing/parquet.py`: parquet loading and timestamp alignment.
- `forex_cuda/preprocessing/returns.py`: close-price matrix to log-return matrix.
- `forex_cuda/preprocessing/cache.py`: `.npy`/parquet/json cache keyed by source files and config.
- `forex_cuda/preprocessing/windows.py`: CPU rolling-window helpers for fair CPU/GPU comparisons.

The cache stores:

- `log_returns.float32.npy`
- `timestamps.parquet`
- `close_prices.parquet`
- `metadata.json`

Benchmark preprocessing and cache load:

```bash
conda activate forex-cuda
export FOREX_CUDA_DATA_DIR=/path/to/parquet_h1
python benchmarks/benchmark_preprocessing_cache.py
```

Current preprocessing result for 8 pairs and 80k aligned rows:

| Path | Time |
| --- | ---: |
| Build from parquet | 0.2074s |
| Load from cache | 0.0081s |

Cache speedup: `25.75x`.

The real-data notebook is available at:

- `pruebas_reales/forex_cuda_parquet_h1_real_test.ipynb`

The CPU/GPU real-data benchmark is available at:

```bash
conda activate forex-cuda
export FOREX_CUDA_DATA_DIR=/path/to/parquet_h1
python benchmarks/benchmark_real_parquet.py
```

Validated with 8 FX pairs, 4096 parallel environments, 512 H1 bars per episode, and 256 random-agent steps:

| Metric | Value |
| --- | ---: |
| Log-return matrix | `[8, 79999]` |
| CUDA market tensor | `[4096, 8, 512]` |
| Runtime for 256 steps | 0.0125s |
| Environment steps/sec | 83.95M |
| Margin calls | 0 |

The dedicated CPU/GPU comparison uses the same real H1 windows, actions, and per-environment costs for both backends:

| Backend | Time | Env-steps/sec |
| --- | ---: | ---: |
| CPU Python/Numpy | 15.9604s | 65.70k |
| GPU CUDA vectorized | 0.0089s | 117.97M |

Speedup: `1795.60x` over the Python/Numpy CPU loop.

It writes:

- `benchmark_results/benchmark_real_parquet.json`
- `benchmark_results/benchmark_real_parquet.png`
- `benchmark_results/benchmark_preprocessing_cache.json`
- `benchmark_results/benchmark_preprocessing_cache.png`

Run the random agent example:

```bash
conda activate forex-cuda
python examples/example_random_agent.py
```

## Isolated C++ Math Test

The math core is isolated from PyTorch so it can be checked before importing Python:

```bash
bash scripts/run_cpp_math_test.sh
```

## Benchmark

The benchmark compares a traditional Python/Numpy loop against the vectorized CUDA environment over one million environment steps:

```bash
conda activate forex-cuda
python benchmarks/benchmark_steps.py
```

It writes:

- `benchmark_results/benchmark_steps.json`
- `benchmark_results/benchmark_steps.png`

On the RTX 3070 validation environment, the default one-million-step run produced:

| Backend | Time |
| --- | ---: |
| Python/Numpy | 6.8437s |
| CUDA vectorized | 0.0041s |

## Build Layout

- `csrc/portfolio_math.h`: shared host/device portfolio math.
- `csrc/step_kernel.cu`: one-thread-per-environment CUDA kernel.
- `csrc/portfolio_ops.cpp`: ATen validation and CUDA launch.
- `csrc/bindings.cpp`: Pybind11 module definition.
- `forex_cuda/preprocessing/`: parquet loading, alignment, cache, returns, and CPU window helpers.
- `forex_cuda/data.py`: compatibility re-exports for preprocessing helpers.
- `forex_cuda/tensors.py`: tensor creation and layout helpers.
- `forex_cuda/env.py`: Gymnasium-style environment wrapper.
