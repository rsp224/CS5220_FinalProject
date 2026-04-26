#!/bin/bash
module load cudatoolkit nccl craype-accel-nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
ntasks="${NTASKS:-4}"
srun --ntasks="${ntasks}" --gpus-per-task=1 --gpu-bind=none ./build-dist/train_dist "$@"
