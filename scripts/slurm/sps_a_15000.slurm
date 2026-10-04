#!/usr/bin/env bash
#SBATCH --job-name=xy-sps-A-15000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
PYTHON_BIN="${SPS_PYTHON:?Set the compute-node environment Python path}"
SEED="${SEED:?Submit with SEED=11, 22 or 33}"
case "$SEED" in 11|22|33) ;; *) echo 'Seeds are 11, 22 and 33.' >&2; exit 2;; esac
RUN_DIR="results/sps_a/L4_seed${SEED}_continue15000_${SLURM_JOB_ID:?}"

"$PYTHON_BIN" -u scripts/sps_budget_15000.py --seed "$SEED" --out "$RUN_DIR"
echo "Finished final training-budget point: $RUN_DIR"
