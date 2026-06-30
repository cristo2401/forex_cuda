# Contributing

Contributions are welcome when they keep the project focused on high-throughput CUDA/Gymnasium simulation for reinforcement-learning research.

## Local Setup

Use a CUDA-enabled PyTorch environment, then install the package in editable mode:

```bash
pip install -e . --no-build-isolation --no-deps
```

Run the core checks before opening a pull request:

```bash
bash scripts/run_cpp_math_test.sh
nvcc -std=c++17 -O2 -I csrc -c csrc/step_kernel.cu -o /tmp/forex_cuda_step_kernel.o
python -m py_compile forex_cuda/*.py forex_cuda/preprocessing/*.py benchmarks/*.py examples/*.py
```

## Guidelines

- Keep tensors contiguous CUDA `float32` unless a different dtype is part of the contract.
- Preserve the Structure-of-Arrays layout.
- Keep CPU/GPU benchmark comparisons reproducible.
- Do not commit datasets, caches, compiled extensions, or credentials.
- Include focused tests for math or kernel contract changes.

