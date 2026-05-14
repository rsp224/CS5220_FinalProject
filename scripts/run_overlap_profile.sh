#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# ── Pre-flight ──────────────────────────────────────────────────────────────
if [[ ! -x build-dist/train_dist ]]; then
    echo "ERROR: build-dist/train_dist not found or not executable" >&2
    exit 1
fi

if [[ -z "${SCRATCH:-}" ]]; then
    echo "ERROR: \$SCRATCH is not set" >&2
    exit 1
fi

module load cudatoolkit nccl craype-accel-nvidia80 python
export MPICH_GPU_SUPPORT_ENABLED=1

echo "Active SLURM jobs:"
squeue --me

# Find the running interactive job in this allocation
JOBID=$(squeue --me -h -o "%i %t" | awk '$2=="R"{print $1; exit}')
if [[ -z "$JOBID" ]]; then
    echo "ERROR: No running SLURM job found. Start an interactive allocation first." >&2
    exit 1
fi
echo "Using SLURM job: $JOBID"

TRAIN_ARGS="mnist 16384 4 tree"

# ── Profile each GPU count ──────────────────────────────────────────────────
for N in 2 4 8 16; do
    OUTDIR="$SCRATCH/prof/overlap_${N}gpu"
    mkdir -p "$OUTDIR"

    # Whole-node allocation: urgent_gpu gives 4 GPUs/node
    NODES=$(( (N + 3) / 4 ))

    echo ""
    echo "================================================================"
    echo " Running with $N GPUs ($NODES nodes)  →  $OUTDIR"
    echo "================================================================"

    if srun --jobid="$JOBID" \
            --nodes="$NODES" --ntasks="$N" \
            --gpus-per-task=1 --gpu-bind=none \
        nsys profile \
            -t cuda,nvtx,mpi \
            --mpi-impl=mpich \
            --force-overwrite=true \
            --sample=none \
            -o "$OUTDIR/nsys_report_%q{SLURM_PROCID}" \
        ./build-dist/train_dist $TRAIN_ARGS; then
        echo "✓ $N-GPU run complete: $OUTDIR"
    else
        echo "⚠ $N-GPU srun failed (not enough GPUs in allocation?), skipping." >&2
    fi
done

echo ""
echo "All runs finished. Now running analysis..."
python3 scripts/analyze_overlap.py
