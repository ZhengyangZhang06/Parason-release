#!/bin/bash

#SBATCH --account=<your_account>
#
# Cluster-specific: set --account and --partition, and adjust --nodes/--gpus-per-node/--time/--mem for your cluster.
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --partition=<partition>
#SBATCH --time=4:00:00
#SBATCH --output=slurm/eval_%A_%a.out
#SBATCH --error=slurm/eval_%A_%a.err

set -eo pipefail

# Each row = "MODEL|BENCH|N_SAMPLES". Add rows and bump --array=0-N to scale.
CONFIGS=(
  "parason-sft|aime24|32"
  "parason-rl|aime24|32"
  "parason-sft|aime24|32"
  "parason-rl|aime24|32"
)

ROW="${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"
IFS='|' read -r MODEL BENCH N_SAMPLES <<<"${ROW}"

BENCH_PARQUET="${SLURM_SUBMIT_DIR}/data/${BENCH}/test.parquet"
# Per-(bench, model) output dir so the completion JSON is the only "_${N_SAMPLES}.json" file there.
RESULTS_DIR="${SLURM_SUBMIT_DIR}/eval_results/${BENCH}/${MODEL}"
mkdir -p "${RESULTS_DIR}"

echo "[eval] ${MODEL} x ${BENCH} -> ${RESULTS_DIR}/*_${N_SAMPLES}.json"

# Completion gate: main_generation only writes the aggregate JSON after every
# sample finishes. Its presence breaks the requeue chain (exit 0 cancels
# afternotok dependent). The filename's data_source prefix is whatever
# infer_data_type() returns for the parquet, so glob instead of hard-coding.
if compgen -G "${RESULTS_DIR}/*_${N_SAMPLES}.json" > /dev/null; then
  echo "[done] already complete; skipping"
  exit 0
fi

sbatch -d afternotok:${SLURM_JOB_ID} --array=${SLURM_ARRAY_TASK_ID} "${SLURM_SUBMIT_DIR}/run_eval.slurm"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate parason

export PYTHONUNBUFFERED=1
export CUDA_HOME="${CONDA_PREFIX}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_V1=1
export RAY_memory_monitor_refresh_ms=0
# Slurm sets ROCR_VISIBLE_DEVICES (for ROCm) alongside CUDA_VISIBLE_DEVICES on
# this cluster; verl's worker init refuses when both are set. Same workaround
# run_rl.slurm uses around `srun ray start` / `srun python`.
unset ROCR_VISIBLE_DEVICES

cd "${SLURM_SUBMIT_DIR}/parason_rl"

python3 -m verl.trainer.main_generation \
  data.path="${BENCH_PARQUET}" \
  data.n_samples=${N_SAMPLES} \
  data.batch_size=64 \
  data.output_path="${SLURM_SUBMIT_DIR}/eval_results/${BENCH}" \
  actor_rollout_ref.model.path="${SLURM_SUBMIT_DIR}/ckpts/${MODEL}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.prompt_length=40960 \
  actor_rollout_ref.rollout.response_length=40960 \
  actor_rollout_ref.rollout.max_model_len=40960 \
  actor_rollout_ref.rollout.max_num_batched_tokens=40960 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
  actor_rollout_ref.rollout.agent.enable_parallel_branching=True \
  actor_rollout_ref.rollout.agent.no_conclusion=true \
  reward_model.config.strip_comma_from_answer=True \
  trainer.n_gpus_per_node="${SLURM_GPUS_ON_NODE}"

# Missing JSON -> partial run; nonzero exit lets the queued afternotok job resume.
compgen -G "${RESULTS_DIR}/*_${N_SAMPLES}.json" > /dev/null \
  || { echo "[incomplete] no completion JSON under ${RESULTS_DIR}"; exit 1; }
echo "[ok] eval complete under ${RESULTS_DIR}"
