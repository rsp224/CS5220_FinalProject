#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

module load cudatoolkit
module load nccl
module load craype-accel-nvidia80

cmake_args=(
  -S .
  -B build-dist
  -DUSE_MPI=ON
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CXX_COMPILER="$(which CC)"
)

if [[ -n "${NCCL_ROOT:-}" ]]; then
  cmake_args+=(-DNCCL_ROOT="${NCCL_ROOT}")
elif [[ -n "${NCCL_DIR:-}" ]]; then
  cmake_args+=(-DNCCL_ROOT="${NCCL_DIR}")
elif [[ -n "${NCCL_HOME:-}" ]]; then
  cmake_args+=(-DNCCL_ROOT="${NCCL_HOME}")
fi

cmake "${cmake_args[@]}"
cmake --build build-dist -j
