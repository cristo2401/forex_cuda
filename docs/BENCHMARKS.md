# How to read the benchmarks

The committed JSON files are records of earlier runs, not measurements refreshed
on every commit. Throughput measures simulation work, not strategy quality.

## Stored H1-data run

Source: [`benchmark_real_parquet.json`](../benchmark_results/benchmark_real_parquet.json).
Implementation: [`benchmark_real_parquet.py`](../benchmarks/benchmark_real_parquet.py).

| Item | Recorded value |
|---|---:|
| Parallel environments | 4,096 |
| FX pairs | 8 |
| Steps per environment | 256 |
| Total environment-steps | 1,048,576 |
| Python/NumPy loop | 15.9604 s |
| CUDA loop | 0.008889 s |
| CUDA throughput | 117.97 million env-steps/s |
| Ratio against this baseline | 1,795.60× |
| Absolute difference in mean final balance | 0.00390625 |

**The CPU baseline is a nested Python/NumPy loop, not optimized native CPU code.**
The ratio therefore cannot be read as a general GPU-versus-CPU speedup. Comparing
against vectorized NumPy or native parallel C++ is useful future work.

The GPU timer starts after tensor transfers, environment construction, reset and
cost/margin setup. CUDA synchronization brackets the timed loop. Dataset loading,
preprocessing, extension compilation, host-to-device transfer and model training
are outside that GPU timing. The recorded run uses cached preprocessing and matching
market windows/actions for the CPU and GPU paths.

The saved balance difference is an aggregate check, not a per-environment parity
test. The existing math tests cover a small number of cases. Further testing should
cover individual trajectories, termination, margin calls and numerical edge cases.

The original README identifies an RTX 3070. The JSON does not record a full hardware
and software fingerprint, repeated-run distribution or confidence interval. Treat
the numbers as one observed configuration rather than a portable performance promise.

## Reproduce with your own data

Follow [CUDA setup](setup_cuda.md) and the [H1 schema](dataset_format.md), then inspect:

```bash
python benchmarks/benchmark_real_parquet.py --help
python benchmarks/benchmark_steps.py --help
python benchmarks/benchmark_preprocessing_cache.py --help
```

The market-data benchmark needs separately supplied H1 Parquet files. The synthetic
benchmark provides a path without those files but still requires the CUDA extension.
Record the GPU/CPU, driver, CUDA/PyTorch versions, commit, data coverage and all flags
when comparing a new result. Report setup/preprocessing separately from the timed loop.
