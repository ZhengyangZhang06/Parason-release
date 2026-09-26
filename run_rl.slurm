#!/bin/bash

#SBATCH --account=<your_account>
#
# Cluster-specific: set --account and --partition, and adjust --nodes/--gpus-per-node/--time/--mem for your cluster.
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --output=slurm/rl_%A_%a.out
#SBATCH --error=slurm/rl_%A_%a.err

set -eo pipefail

ALL_SUBTASK_BETA=(
  0.000
  0.000
  0.000
  0.000
  0.025
  0.050
  0.100
  0.000
  0.000
  0.000
  0.025
  0.050
  0.100
)
ALL_TRIAL_BETA=(
  0.000
  0.025
  0.050
  0.100
  0.000
  0.000
  0.000
  0.025
  0.050
  0.100
  0.000
  0.000
  0.000
)
ALL_LATENCY_ALPHA=(
  0.000
  0.000
  0.000
  0.000
  0.000
  0.000
  0.000
  0.100
  0.100
  0.100
  0.100
  0.100
  0.100
)
ALL_OUTPUT_CKPTS=(
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

SUBTASK_BETA="${ALL_SUBTASK_BETA[${SLURM_ARRAY_TASK_ID}]}"
TRIAL_BETA="${ALL_TRIAL_BETA[${SLURM_ARRAY_TASK_ID}]}"
LATENCY_ALPHA="${ALL_LATENCY_ALPHA[${SLURM_ARRAY_TASK_ID}]}"
OUTPUT_CKPT="${ALL_OUTPUT_CKPTS[${SLURM_ARRAY_TASK_ID}]}"

# Training / validation data and the SFT checkpoint to initialize from, both relative to the
# repo root. Set these to your own locations (see README.md, "Preparing checkpoints and data").
TRAIN_DATA="${TRAIN_DATA:-data/<TRAIN_DATA>.parquet}"
VAL_DATA="${VAL_DATA:-${TRAIN_DATA}}"
INIT_CKPT="${INIT_CKPT:-ckpts/<SFT_CKPT>}"

OUTPUT_DIR="${SLURM_SUBMIT_DIR}/ckpts/${OUTPUT_CKPT}"
mkdir -p "${OUTPUT_DIR}"

EXPECTED_WORLD_SIZE=$((SLURM_JOB_NUM_NODES * SLURM_GPUS_ON_NODE))

checkpoint_is_normal() {
  local step="$1"
  local actor_dir="${OUTPUT_DIR}/global_step_${step}/actor"
  local hf_dir="${actor_dir}/huggingface"
  local count

  [[ "$step" =~ ^[0-9]+$ ]] || return 1
  [[ -d "${actor_dir}" ]] || return 1
  [[ -s "${hf_dir}/config.json" ]] || return 1
  [[ -s "${hf_dir}/tokenizer.json" ]] || return 1

  count=$(find "${actor_dir}" -maxdepth 1 -type f -name "model_world_size_${EXPECTED_WORLD_SIZE}_rank_*.pt" | wc -l)
  [[ "$count" -eq "$EXPECTED_WORLD_SIZE" ]] || return 1
  count=$(find "${actor_dir}" -maxdepth 1 -type f -name "optim_world_size_${EXPECTED_WORLD_SIZE}_rank_*.pt" | wc -l)
  [[ "$count" -eq "$EXPECTED_WORLD_SIZE" ]] || return 1
  count=$(find "${actor_dir}" -maxdepth 1 -type f -name "extra_state_world_size_${EXPECTED_WORLD_SIZE}_rank_*.pt" | wc -l)
  [[ "$count" -eq "$EXPECTED_WORLD_SIZE" ]] || return 1
}

# We only train for 200 steps.
if [[ -f "${OUTPUT_DIR}/latest_checkpointed_iteration.txt" ]]; then
  cur_step=$(<"${OUTPUT_DIR}/latest_checkpointed_iteration.txt")
  if [[ "$cur_step" =~ ^[0-9]+$ ]] && (( cur_step >= 200 )); then
    if checkpoint_is_normal "$cur_step"; then
      exit 0
    fi
    echo "[ckpt-prune] latest_checkpointed_iteration.txt points to incomplete step ${cur_step}; recovering"
  fi
fi

sbatch -d afternotok:${SLURM_JOB_ID} --array=${SLURM_ARRAY_TASK_ID} "${SLURM_SUBMIT_DIR}/run_rl.slurm"

for conda_sh in "${HOME}/miniconda3/etc/profile.d/conda.sh" "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
  if [[ -f "${conda_sh}" ]]; then
    source "${conda_sh}"
    break
  fi
done
conda activate parason

export PYTHONUNBUFFERED=1
export CUDA_HOME="${CONDA_PREFIX}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_V1=1
export VLLM_DISABLE_COMPILE_CACHE=1
export RAY_memory_monitor_refresh_ms=0

export WANDB_PROJECT=parason
export WANDB_JOB_TYPE=rl
export WANDB_RUN_GROUP="${OUTPUT_CKPT}"

cd "${SLURM_SUBMIT_DIR}/parason_rl"

# One-shot ckpt prune before launch:
# - pick the max global_step_N/ with a complete actor checkpoint
# - force trainer to resume from it by writing that step to latest_checkpointed_iteration.txt
#   (resume_mode=auto follows that file even if a higher-step ckpt exists on disk)
# - keep that step + normal multiples of 25, prune the rest and any incomplete ckpts
if [[ -d "${OUTPUT_DIR}" ]]; then
  latest=""
  for d in "${OUTPUT_DIR}"/global_step_*; do
    [[ -d "$d" ]] || continue
    step="${d##*global_step_}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    checkpoint_is_normal "$step" || continue
    if [[ -z "$latest" || "$step" -gt "$latest" ]]; then
      latest="$step"
    fi
  done
  if [[ -n "$latest" ]]; then
    echo "[ckpt-prune] resume target = step $latest (max normal actor ckpt)"
    for d in "${OUTPUT_DIR}"/global_step_*; do
      [[ -d "$d" ]] || continue
      step="${d##*global_step_}"
      [[ "$step" =~ ^[0-9]+$ ]] || continue
      if ! checkpoint_is_normal "$step"; then
        echo "[ckpt-prune] removing incomplete $d (step=$step, latest=$latest)"
        rm -rf "$d"
      elif [[ "$step" -ne "$latest" && $((step % 25)) -ne 0 ]]; then
        echo "[ckpt-prune] removing $d (step=$step, latest=$latest)"
        rm -rf "$d"
      fi
    done
    echo "$latest" > "${OUTPUT_DIR}/latest_checkpointed_iteration.txt"
  fi
fi

# Discover nodes and pick a head node/IP for Ray
mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
ray_temp_dir="/tmp/${USER}/ray-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

for node_i in "${nodes_array[@]}"; do
  srun --nodes=1 --ntasks=1 -w "$node_i" mkdir -p "$ray_temp_dir" &
done
wait

if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
fi

port=6379
ip_head="${head_node_ip}:${port}"
export ip_head
export RAY_ADDRESS="${ip_head}"
echo "Nodes: ${nodes_array[*]}"
echo "Head: ${head_node} | IP Head: ${ip_head}"

echo "Starting Ray HEAD at ${head_node}"
# ray 2.52.1 picks several agent/metrics ports randomly and refuses to start if any of
# them land inside the default worker port range (10002-19999). Pin them all well above.
srun --nodes=1 --ntasks=1 -w "$head_node" \
  /usr/bin/env -u ROCR_VISIBLE_DEVICES \
  ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --temp-dir="$ray_temp_dir" \
    --runtime-env-agent-port=30007 \
    --dashboard-agent-grpc-port=30008 \
    --metrics-export-port=30009 \
    --num-cpus "${SLURM_CPUS_ON_NODE}" --num-gpus "${SLURM_GPUS_ON_NODE}" --block &
sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "Starting Ray WORKER ${i} at ${node_i}"
  srun --nodes=1 --ntasks=1 -w "$node_i" \
    /usr/bin/env -u ROCR_VISIBLE_DEVICES \
    ray start --address "$ip_head" \
      --runtime-env-agent-port=30007 \
      --dashboard-agent-grpc-port=30008 \
      --metrics-export-port=30009 \
      --num-cpus "${SLURM_CPUS_ON_NODE}" --num-gpus "${SLURM_GPUS_ON_NODE}" --block &
  sleep 5
done

# Wait until all workers register; trainer requires nnodes*gpus_per_node GPUs and aborts otherwise.
expected_gpus=$((SLURM_JOB_NUM_NODES * SLURM_GPUS_ON_NODE))
available=0
for attempt in $(seq 1 60); do
  available=$(timeout 25 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    /usr/bin/env -u ROCR_VISIBLE_DEVICES \
    python3 -c "import ray; ray.init(address='auto', logging_level='ERROR'); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo 0)
  [[ "$available" =~ ^[0-9]+$ ]] || available=0
  if (( available >= expected_gpus )); then
    echo "Ray cluster ready: ${available}/${expected_gpus} GPUs (attempt ${attempt})"
    break
  fi
  echo "Waiting for Ray cluster: ${available}/${expected_gpus} GPUs (attempt ${attempt})"
  sleep 5
done

srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
  /usr/bin/env -u ROCR_VISIBLE_DEVICES \
  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${SLURM_SUBMIT_DIR}/${TRAIN_DATA}" \
    data.val_files="${SLURM_SUBMIT_DIR}/${VAL_DATA}" \
    data.filter_overlong_prompts=True \
    data.train_batch_size=128 \
    data.max_prompt_length=40960 \
    data.max_response_length=40960 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${SLURM_SUBMIT_DIR}/${INIT_CKPT}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    actor_rollout_ref.actor.ppo_mini_batch_size=null \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    actor_rollout_ref.rollout.max_num_batched_tokens=40960 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=40960 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.agent.enable_parallel_branching=True \
    actor_rollout_ref.rollout.agent.no_conclusion=true \
    actor_rollout_ref.rollout.agent_return_expanded_sequences=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.broadcast_from_last=True \
    reward_model.config.acceleration_ratio_reward=1.0 \
    reward_model.config.acceleration_ratio_reward_factor=0.5 \
    reward_model.config.acceleration_ratio_clip_max=0.2 \
    reward_model.config.version=v2 \
    reward_model.config.require_think_end=False \
    reward_model.config.strip_comma_from_answer=True \
    reward_model.config.subtask_beta="${SUBTASK_BETA}" \
    reward_model.config.trial_beta="${TRIAL_BETA}" \
    reward_model.config.latency_alpha="${LATENCY_ALPHA}" \
    reward_model.reward_manager_type=reward_manager_with_server \
    trainer.critic_warmup=0 \
    trainer.logger="['console','tensorboard','wandb']" \
    trainer.project_name=parason \
    trainer.experiment_name="${OUTPUT_CKPT}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="${SLURM_GPUS_ON_NODE}" \
    trainer.nnodes="${SLURM_JOB_NUM_NODES}" \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.total_epochs=30
