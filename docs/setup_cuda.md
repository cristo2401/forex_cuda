# CUDA Setup

`forex-cuda` builds a PyTorch C++/CUDA extension with `torch.utils.cpp_extension`.

## Requirements

- NVIDIA GPU
- CUDA toolkit with `nvcc`
- Python 3.10+
- CUDA-enabled PyTorch
- `ninja` is recommended by PyTorch build tooling

## Example Conda Environment

```bash
conda create -n forex-cuda python=3.11 -y
conda activate forex-cuda
python -m pip install --upgrade pip
```

Install PyTorch following the official selector for your CUDA version:

```bash
# Example only. Pick the command matching your system from pytorch.org.
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Install this package:

```bash
pip install -e . --no-build-isolation --no-deps
```

## Validate

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY

bash scripts/run_cpp_math_test.sh
python examples/example_random_agent.py
```

