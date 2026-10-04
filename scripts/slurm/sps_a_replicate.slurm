#!/usr/bin/env bash
#SBATCH --job-name=xy-sps-A-replicate
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
SEED="${SEED:?Submit once with SEED=22 and once with SEED=33}"
case "$SEED" in 22|33) ;; *) echo 'Preselected seeds are 22 and 33.' >&2; exit 2;; esac
BASELINE_DIR="${BASELINE_DIR:-results/sps_a/L4_seed11_full_25906}"
RUN_DIR="results/sps_a/L4_seed${SEED}_replica_${SLURM_JOB_ID:?}"

"$PYTHON_BIN" -u scripts/sps_seed_replication.py \
    --seed "$SEED" --baseline "$BASELINE_DIR" --out "$RUN_DIR"
echo "Finished seed replication: $RUN_DIR"
