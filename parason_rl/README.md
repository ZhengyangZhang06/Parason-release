# Parason RL

The reinforcement-learning stage trains a model to generate **parallel reasoning paths** when
solving math problems. It is built on a vendored fork of
[verl](https://github.com/volcengine/verl) (pinned at commit
[`0d4541f`](https://github.com/volcengine/verl/tree/0d4541f397828843525b3f3a7eadff03d56ff24c));
[`MODIFICATIONS.md`](MODIFICATIONS.md) lists exactly what we changed.

## Overview

Parason reasons with structured tokens that enable parallel exploration:

- `<think>…</think>` — the chain of thought
- `<Parallel>…</Parallel>` — a block whose child threads run independently
- `<Outlines>…</Outlines>` — the header of a parallel block, one outline item per thread, framed as either
  `<Subtask>…</Subtask>` (an independent sub-task the problem is decomposed into) or
  `<Trial>…</Trial>` (one of several independent attempts/hypotheses explored in parallel)
- `<Thread>…</Thread>` — one reasoning thread, executing its matching outline item
- `<Conclusion>…</Conclusion>` — the final answer, in `\boxed{}`

Decoding the threads concurrently shortens the *critical path* (the longest single thread) to an
answer, so the model reaches equal-accuracy solutions in fewer sequential tokens.

### Reward function

The reward combines two terms:

1. **Correctness** `r_correct` — `+1.0` for a correct answer, `0.0` otherwise, via symbolic math
   verification (see [`deepscaler/`](deepscaler/)).
2. **Acceleration ratio** `r_accel` — rewards efficient parallelism:

   ```
   r_accel = factor * min(acceleration_ratio - 1.0, clip_max)
   acceleration_ratio = sequential_cost / parallel_cost
   ```

   with defaults `factor = 0.5`, `clip_max = 0.2`.

   ```
   reward = r_correct + r_accel
   ```

3. **Optional parallel-structure bonuses** (off by default; set the weights in
   `reward_model.config` to enable). These are additive, z-scored within each GRPO group:
   - `subtask_beta` — rewards **subtask-style** parallelism (decomposing into independent sub-tasks)
   - `trial_beta` — rewards **trial-style** parallelism (multiple independent attempts/hypotheses)
   - `parallel_ratio_beta` — rewards the overall fraction of reasoning done in parallel
   - `latency_alpha` — rewards shorter critical-path (token) latency

   The released model variants (`…-subtask{025,050,100}`, `…-trial{025,050,100}`,
   `…-latency{050,100,200}`) sweep these weights.

The reward manager also logs parallel-structure statistics — thread counts, token usage,
critical-path length, and the per-type `subtask_parallel_ratio` / `trial_parallel_ratio`.

## Installation

Install the shared dependencies and the verl fork (editable):

```bash
pip install -r ../requirements.txt
pip install flash-attn==2.7.4.post1 opentelemetry-sdk==1.39.1 \
    opentelemetry-exporter-prometheus==0.60b1 --no-build-isolation
pip install -e .          # installs the vendored verl fork
```

### Prepare an SFT model

RL requires an SFT checkpoint that already emits the parallel-reasoning structure and answers in
`\boxed{}`. See [../parason_sft/README.md](../parason_sft/README.md).

Both SFT and RL run on a single node with 8× 80 GB A100/H100 GPUs; multi-node is supported for RL.

## Single-node training

```bash
export VLLM_USE_V1=1
MODEL_PATH=../parason_sft/ckpts/Q3-8B-131072-SFT

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="../parason_sft/data/mult-10k-par_pq/train.parquet" \
  data.val_files="../parason_sft/data/mult-10k-par_pq/val.parquet" \
  data.train_batch_size=128 data.val_batch_size=512 \
  data.max_prompt_length=9216 data.max_response_length=8192 \
  data.filter_overlong_prompts=True data.return_raw_chat=True \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.agent.num_workers=8 \
  actor_rollout_ref.rollout.agent.enable_parallel_branching=True \
  actor_rollout_ref.rollout.agent.no_conclusion=true \
  actor_rollout_ref.rollout.agent_return_expanded_sequences=True \
  algorithm.use_kl_in_reward=False algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.broadcast_from_last=True \
  reward_model.reward_manager_type=reward_manager_with_server \
  reward_model.config.version=v2 \
  reward_model.config.acceleration_ratio_reward=1.0 \
  reward_model.config.acceleration_ratio_reward_factor=0.5 \
  reward_model.config.acceleration_ratio_clip_max=0.2 \
  reward_model.config.strip_comma_from_answer=True \
  trainer.logger=['console','tensorboard'] \
  trainer.project_name=parason trainer.experiment_name=parason-parallel \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 \
  trainer.save_freq=10 trainer.test_freq=10 trainer.total_epochs=30 \
  trainer.val_before_train=False
```

To train the **sequential baseline**, point `MODEL_PATH` at the AR-SFT checkpoint and set
`actor_rollout_ref.rollout.agent.enable_parallel_branching=False`.

## Cluster training, evaluation, and merging

The turn-key SLURM scripts live at the **repository root** and are submitted from there (they
`cd` into `parason_rl/` / `parason_sft/` for you). Set `--account` (and, if your cluster differs,
`--partition`) in each script's SBATCH header first.

```bash
# from the repo root:
sbatch run_rl.slurm            # GRPO training — sweeps the subtask/trial/latency reward weights
MODEL=parason-rl STEP=200 sbatch run_merge.slurm   # merge FSDP shards -> HF
sbatch run_eval.slurm          # evaluate the merged models with the parallel rollout
```

`run_rl.slurm` initializes from the SFT checkpoint under `ckpts/` and trains on the data under
`data/` — both are set via the arrays at the top of the script. The released checkpoints and
datasets are withheld during the anonymous review period and will be linked on publication; see
*Preparing checkpoints and data* in the root [README](../README.md) for the expected layout.

## Metrics

Training logs to TensorBoard:

```bash
tensorboard --logdir=./tensorboard_log
```

Key metrics: `reward/mean`, `metrics/acceleration_ratio`, `metrics/parallel_ratio`, and the
policy loss.

### Acceleration ratio vs. the sequential baseline

```
acceleration_ratio = total tokens in the sequential baseline
                     / tokens on the longest thread (critical path) of the parallel model
```

The former is logged as `total_num_tokens`; the latter as `num_tokens_in_the_longest_thread`.

## Reference results

Both models are trained for 400 steps on the synthetic multiplication task. The parallel model
matches the sequential baseline's accuracy with a shorter critical path:

| Setting | Tokens in longest thread | Accuracy |
| --- | :---: | :---: |
| Sequential baseline | 3322 | 99.0% |
| Parason (parallel) | 2632 | 99.0% |

That is a **1.26× reduction in token latency** at equal accuracy.
