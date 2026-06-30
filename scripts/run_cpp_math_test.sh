#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${TMPDIR:-/tmp}/forex_cuda_math_test"

c++ -std=c++17 -O2 -I"${ROOT_DIR}/csrc" \
    "${ROOT_DIR}/tests/cpp/test_portfolio_math.cpp" \
    -o "${OUT}"

"${OUT}"

