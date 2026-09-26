#!/bin/bash

#SBATCH --account=<your_account>
#
# Cluster-specific: set --account and --partition, and adjust --nodes/--gpus-per-node/--time/--mem for your cluster.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=<partition>
#SBATCH --time=1:00:00
#SBATCH --output=slurm/merge_%j.out
#SBATCH --error=slurm/merge_%j.err

set -eo pipefail

# Merge a verl FSDP checkpoint to HF format at ckpts/${MODEL}/ (top-level,
# matching SFT layout). Override MODEL / STEP via env vars when invoking sbatch:
#   MODEL=parason-rl STEP=200 sbatch run_merge.slurm
MODEL="${MODEL:-parason-rl}"
STEP="${STEP:-200}"

CKPT_DIR="${SLURM_SUBMIT_DIR}/ckpts/${MODEL}"
ACTOR_DIR="${CKPT_DIR}/global_step_${STEP}/actor"

[[ -d "${ACTOR_DIR}" ]] || { echo "FATAL: actor shard dir missing: ${ACTOR_DIR}"; exit 2; }
[[ -f "${CKPT_DIR}/config.json" ]] && {
  echo "[merge] ${CKPT_DIR}/config.json already exists; refusing to overwrite. Remove it or move it aside first."
  exit 3
}

for conda_sh in "${HOME}/miniconda3/etc/profile.d/conda.sh" "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
  if [[ -f "${conda_sh}" ]]; then
    source "${conda_sh}"
    break
  fi
done
conda activate parason

export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}/parason_rl"

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_DIR}" \
  --target_dir "${CKPT_DIR}"

echo "[ok] merged ${MODEL} step ${STEP} -> ${CKPT_DIR}"
ls "${CKPT_DIR}" | grep -E "^(config\.json|model-.*\.safetensors|tokenizer)" | head
