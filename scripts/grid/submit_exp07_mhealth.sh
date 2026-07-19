#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

tuning_job=$(sbatch --parsable scripts/grid/run_exp07_mhealth_tuning.sh)
aggregate_job=$(sbatch \
    --parsable \
    --dependency="afterok:${tuning_job}" \
    scripts/grid/run_exp07_mhealth_tuning_aggregate.sh)

echo "tuning=${tuning_job}"
echo "aggregate=${aggregate_job} dependency=afterok:${tuning_job}"

portions=(
    "generalization:0-79"
    "scale:80-169"
    "robustness_a:170-259"
    "robustness_b:260-349"
    "robustness_vfl:350-379"
    "tail:380-419"
    "redundancy:420-509"
)

for portion in "${portions[@]}"; do
    name="${portion%%:*}"
    indices="${portion#*:}"
    job_id=$(sbatch \
        --parsable \
        --dependency="afterok:${aggregate_job}" \
        --array="${indices}%4" \
        --job-name="exp07_${name}" \
        scripts/grid/run_exp07_mhealth.sh)
    echo "${name}=${job_id} array=${indices}%4 dependency=afterok:${aggregate_job}"
done
