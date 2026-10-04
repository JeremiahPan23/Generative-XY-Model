#!/usr/bin/env bash
#SBATCH --job-name=xy-sps-A
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}"

# Activate the group's PyTorch/CUDA environment before sbatch. Override with:
# SPS_PYTHON=/absolute/path/to/env/bin/python sbatch scripts/slurm/sps_a.slurm
PYTHON_BIN="${SPS_PYTHON:-python}"
MODE="${MODE:-pilot}"
L="${L:-4}"
SEED="${SEED:-11}"
RESUME="${RESUME:-0}"
RUN_DIR="${RUN_DIR:-results/sps_a/L${L}_seed${SEED}_${MODE}_${SLURM_JOB_ID}}"

case "$L" in 4|8) ;; *) echo 'This template supports L=4 or L=8.' >&2; exit 2;; esac
case "$MODE" in pilot|full) ;; *) echo 'MODE must be pilot or full.' >&2; exit 2;; esac

"$PYTHON_BIN" -c 'import torch; print("torch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(), "No visible CUDA GPU"; print("GPU:", torch.cuda.get_device_name(0))'
"$PYTHON_BIN" -m unittest discover -s tests -p test_sps_xy.py -v

resume_args=()
if [[ "$RESUME" == 1 ]]; then resume_args+=(--resume); fi

if [[ "$MODE" == pilot ]]; then
    "$PYTHON_BIN" scripts/sps_experiment.py train --config "configs/sps_a_l${L}.json" \
        --out "$RUN_DIR/train" --device cuda --seed "$SEED" --updates 50 "${resume_args[@]}"
    REF_DIR="$RUN_DIR/reference"
    if [[ ! -f "$REF_DIR/reference.json" ]]; then
        "$PYTHON_BIN" scripts/sps_experiment.py reference --L "$L" --out "$REF_DIR" \
            --chains 2 --draws 128 --burn 200 --interval 5
    fi
    "$PYTHON_BIN" scripts/sps_experiment.py evaluate --checkpoint "$RUN_DIR/train/checkpoint.pt" \
        --reference "$REF_DIR" --out "$RUN_DIR/eval_${SLURM_JOB_ID}" --device cuda \
        --steps 32 64 128 --chains 2 --draws 64 --burn 32 --batch-size 64
else
    "$PYTHON_BIN" scripts/sps_experiment.py train --config "configs/sps_a_l${L}.json" \
        --out "$RUN_DIR/train" --device cuda --seed "$SEED" "${resume_args[@]}"
    REF_DIR="$RUN_DIR/reference"
    if [[ ! -f "$REF_DIR/reference.json" ]]; then
        "$PYTHON_BIN" scripts/sps_experiment.py reference --L "$L" --out "$REF_DIR"
    fi
    "$PYTHON_BIN" scripts/sps_experiment.py evaluate --checkpoint "$RUN_DIR/train/checkpoint.pt" \
        --reference "$REF_DIR" --out "$RUN_DIR/eval_${SLURM_JOB_ID}" --device cuda --steps 32 64 128
fi

echo "Finished: $RUN_DIR"
echo 'Pilot completion is a software check, not a scientific result. Inspect screening diagnostics.'
