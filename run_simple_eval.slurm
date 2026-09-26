#!/bin/bash

#SBATCH --account=<your_account>
#
# Cluster-specific: set --account and --partition, and adjust --nodes/--gpus-per-node/--time/--mem for your cluster.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --partition=<partition>
#SBATCH --time=4:00:00
#SBATCH --output=slurm/simple_eval_%j.out
#SBATCH --error=slurm/simple_eval_%j.err

set -eo pipefail

ALL_MODELS=(
  "parason-rl"
  "parason-rl-trial025"
  "parason-rl-trial050"
  "parason-rl-trial100"
  "parason-rl-subtask025"
  "parason-rl-subtask050"
  "parason-rl-subtask100"
  "parason-rl-latency100-trial025"
  "parason-rl-latency100-trial050"
  "parason-rl-latency100-trial100"
  "parason-rl-latency100-subtask025"
  "parason-rl-latency100-subtask050"
  "parason-rl-latency100-subtask100"
)
MODEL="${ALL_MODELS[${SLURM_ARRAY_TASK_ID}]}"

for conda_sh in "${HOME}/miniconda3/etc/profile.d/conda.sh" "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
  if [[ -f "${conda_sh}" ]]; then
    source "${conda_sh}"
    break
  fi
done
conda activate parason

cd "${SLURM_SUBMIT_DIR}/parason_sft"

python src/simple_eval.py \
  --model_name="${SLURM_SUBMIT_DIR}/ckpts/${MODEL}" \
  --data-type="${SLURM_SUBMIT_DIR}/data/${BENCH:-<BENCH>}/test.parquet" \
  --launch_server \
  --verbose 2 \
  --template-type model \
  --bfloat16 \
  --branching-generate \
  -n 32 \
  --max-context-length 40960 \
  ${EXTRA_ARGS}
