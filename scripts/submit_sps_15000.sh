#!/usr/bin/env bash
# Run on ws0. Queue three sequential jobs; this script never trains on ws0.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${SPS_PYTHON:?Export the ws5 Python path before submitting}"
command -v sbatch >/dev/null
JOBFILE="budget15000_jobids.txt"
if [[ -e "$JOBFILE" ]]; then
    echo "$JOBFILE already exists. Inspect the previous submissions; do not submit duplicates." >&2
    exit 2
fi
[[ -f scripts/slurm/sps_a_15000.slurm && -f configs/sps_a_15000.json ]]
# Keep partial submissions on disk if a later sbatch fails.
: > "$JOBFILE"
previous=""
for seed in 11 22 33; do
    dependency=()
    if [[ -n "$previous" ]]; then
        dependency=("--dependency=afterany:$previous")
    fi
    job_id=$(SEED="$seed" sbatch --parsable --nodelist=ws5 "${dependency[@]}" scripts/slurm/sps_a_15000.slurm)
    job_id=${job_id%%;*}
    if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
        echo "Unexpected sbatch response: $job_id. Inspect squeue before any retry." >&2
        exit 2
    fi
    printf '%s\n' "$job_id" >> "$JOBFILE"
    printf 'seed%s=%s\n' "$seed" "$job_id"
    previous="$job_id"
done
echo "Saved three job IDs in $JOBFILE. Pending Dependency is expected."
