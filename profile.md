

```bash
srun --ntasks=4 --gpus-per-task=1 --gpu-bind=none \
	nsys profile \
    -t cuda,nvtx,mpi,osrt \
    --mpi-impl=mpich \
    --force-overwrite=true \
    -o $SCRATCH/prof/nsys_report_%q{SLURM_PROCID} \
    ./build-dist/train_dist mnist 1024 -1 4

nsys stats --report nvtx_sum $SCRATCH/prof/nsys_report_0.nsys-rep
```