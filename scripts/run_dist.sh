#!/bin/bash
module load cudatoolkit nccl craype-accel-nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
srun --gpus-per-task=1 ./build-dist/train_dist