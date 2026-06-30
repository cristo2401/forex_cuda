from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent.resolve()


setup(
    name="forex-cuda",
    version="0.1.0",
    description="Vectorized CUDA portfolio environment for FX log-return simulations.",
    packages=find_packages(exclude=("tests", "tests.*")),
    ext_modules=[
        CUDAExtension(
            name="forex_cuda_ext",
            sources=[
                str(ROOT / "csrc" / "bindings.cpp"),
                str(ROOT / "csrc" / "portfolio_ops.cpp"),
                str(ROOT / "csrc" / "step_kernel.cu"),
            ],
            include_dirs=[str(ROOT / "csrc")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "-std=c++17",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "pyarrow>=14.0",
        "torch",
        "gymnasium>=0.29",
        "matplotlib>=3.7",
    ],
)
