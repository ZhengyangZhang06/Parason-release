#!/bin/bash

#SBATCH --account=<your_account>
#
# Cluster-specific: set --account and --partition, and adjust --nodes/--gpus-per-node/--time/--mem for your cluster.
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --partition=<partition>
#SBATCH --time=4:00:00
#SBATCH --output=slurm/sft_%A_%a.out
#SBATCH --error=slurm/sft_%A_%a.err

set -eo pipefail

sbatch -d afternotok:${SLURM_JOB_ID} --array=${SLURM_ARRAY_TASK_ID} "${SLURM_SUBMIT_DIR}/run_sft.slurm"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate parason

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Training sets, relative to data/ at the repo root. Set these to your own dataset
# directories (see README.md, "Preparing checkpoints and data").
ALL_TRAIN_FILES=(
  "<TRAIN_DATA_8x>"
  "<TRAIN_DATA_1x>"
)
ALL_NUM_EPOCHS=(
  1
  8
)
ALL_OUTPUT_CKPTS=(
  "parason-sft"
  "parason-sft-1x-8ep"
)

TRAIN_FILE="${ALL_TRAIN_FILES[${SLURM_ARRAY_TASK_ID}]}"
NUM_EPOCHS="${ALL_NUM_EPOCHS[${SLURM_ARRAY_TASK_ID}]}"
OUTPUT_CKPT="${ALL_OUTPUT_CKPTS[${SLURM_ARRAY_TASK_ID}]}"

export WANDB_PROJECT=parason
export WANDB_JOB_TYPE=sft
export WANDB_RUN_GROUP="${OUTPUT_CKPT}"

cd "${SLURM_SUBMIT_DIR}/parason_sft"

torchrun \
  --standalone \
  --nproc-per-node=gpu \
  src/sft_parallel.py \
  --block_size=40960 \
  --per_device_train_batch_size=1 \
  --per_device_eval_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --num_train_epochs=${NUM_EPOCHS} \
  --train_file_path="${SLURM_SUBMIT_DIR}/data/${TRAIN_FILE}" \
  --model_name="${SLURM_SUBMIT_DIR}/ckpts/Qwen3-8B" \
  --warmup_ratio=0.05 \
  --deepspeed configs/deepspeed_zero3.json \
  --bf16=True \
  --eval_strategy="no" \
  --logging_steps=1 \
  --save_strategy="steps" \
  --save_steps=32 \
  --save_total_limit=2 \
  --lr_scheduler_type="cosine" \
  --learning_rate=1e-5 \
  --weight_decay=1e-4 \
  --adam_beta1=0.9 \
  --adam_beta2=0.95 \
  --output_dir="${SLURM_SUBMIT_DIR}/ckpts/${OUTPUT_CKPT}" \
  --push_to_hub=false \
  --save_only_model=False \
  --gradient_checkpointing=True \
  --use-liger-kernel=True \
  --dataset_text_field="qwen_text" \
  --attn_implementation="flex_attention" \
  --template_name="qwen" \
  --report_to="wandb" \
  --wandb_project="parason"
