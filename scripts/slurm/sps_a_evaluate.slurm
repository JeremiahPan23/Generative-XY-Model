#!/usr/bin/env bash
#SBATCH --job-name=xy-sps-A-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Evaluate an existing checkpoint only. No training or overwriting old results.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
PYTHON_BIN="${SPS_PYTHON:?Set SPS_PYTHON to the compute-node environment Python}"
RUN_DIR="${RUN_DIR:-results/sps_a/L4_seed11_full_25906}"
OUT_DIR="$RUN_DIR/eval_long_${SLURM_JOB_ID:?}"

for file in "$RUN_DIR/train/checkpoint.pt" "$RUN_DIR/reference/reference.json" "$RUN_DIR/reference/samples.npz"; do
    if [[ ! -f "$file" ]]; then
        echo "Missing on compute node $(hostname): $file" >&2
        exit 2
    fi
done
if [[ -e "$OUT_DIR" ]]; then
    echo "Refusing to reuse output directory: $OUT_DIR" >&2
    exit 2
fi

"$PYTHON_BIN" scripts/sps_experiment.py evaluate \
    --checkpoint "$RUN_DIR/train/checkpoint.pt" \
    --reference "$RUN_DIR/reference" --out "$OUT_DIR" --device cuda \
    --steps 32 64 128 --chains 8 --draws 16384 --burn 4096 \
    --batch-size 128 --seed 22000

echo "Evaluation finished: $OUT_DIR"
echo 'Inspect physical screening and chain diagnostics before drawing conclusions.'
