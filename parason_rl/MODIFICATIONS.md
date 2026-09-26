# Parason modifications to verl

`parason_rl/verl/` is a **vendored fork of [verl](https://github.com/volcengine/verl)**, pinned at
upstream commit [`0d4541f397828843525b3f3a7eadff03d56ff24c`](https://github.com/volcengine/verl/tree/0d4541f397828843525b3f3a7eadff03d56ff24c).

Of the ~330 vendored files, **294 are byte-identical to upstream**. Parason changes **29 files**
and **adds 5**. This document lists exactly what we changed so the contribution is auditable and
the fork can be re-based onto a newer verl if needed.

To reproduce the diff yourself:

```bash
git clone --filter=blob:none --no-checkout https://github.com/volcengine/verl /tmp/verl
git -C /tmp/verl checkout 0d4541f397828843525b3f3a7eadff03d56ff24c
diff -rq /tmp/verl/verl parason_rl/verl
```

## Added files (5)

Parallel-reasoning reward stack (correctness + acceleration ratio) and generation I/O:

| File | Purpose |
| --- | --- |
| `verl/trainer/reward_manager.py` | Reward manager computing correctness + acceleration-ratio reward over parallel rollouts. |
| `verl/trainer/reward_manager_with_server.py` | Same, backed by an out-of-process HTTP reward server (throughput). |
| `verl/trainer/reward_server.py` | The gunicorn reward server (`create_app()`), token-id in / reward out. |
| `verl/trainer/reward_stats.py` | Parallel-structure statistics (thread counts, tokens, critical-path length). |
| `verl/trainer/generation_io.py` | Serialization helpers for parallel generations. |

## Modified files (29)

**Parallel rollout** — expand a single prompt into multiple concurrently-decoded threads and
return the expanded sequences to the trainer:

- `verl/experimental/agent_loop/agent_loop.py`
- `verl/experimental/agent_loop/single_turn_agent_loop.py`
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- `verl/workers/rollout/sglang_rollout/async_sglang_server.py`
- `verl/workers/rollout/replica.py`
- `verl/protocol.py`

**Training loop & advantage** — parallel-aware GRPO, latency/acceleration reward wiring,
`broadcast_from_last`, and parallel metrics:

- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/ppo/reward.py`
- `verl/trainer/ppo/metric_utils.py`
- `verl/trainer/main_ppo.py`
- `verl/workers/actor/dp_actor.py`
- `verl/workers/fsdp_workers.py`
- `verl/utils/seqlen_balancing.py`
- `verl/utils/model.py`
- `verl/trainer/fsdp_sft_trainer.py`

**Generation / evaluation**:

- `verl/trainer/main_generation.py`

**Config plumbing** (new fields for parallel branching, reward, and generation):

- `verl/trainer/config/reward_model/reward_model.yaml`
- `verl/trainer/config/generation.yaml`
- `verl/trainer/config/rollout/rollout.yaml`
- `verl/trainer/config/algorithm.py`
- `verl/trainer/config/ppo_trainer.yaml`
- `verl/trainer/config/ppo_megatron_trainer.yaml`
- `verl/trainer/config/model/hf_model.yaml`
- `verl/trainer/config/_generated_ppo_trainer.yaml`
- `verl/trainer/config/_generated_ppo_megatron_trainer.yaml`
- `verl/workers/config/rollout.py`
- `verl/workers/config/actor.py`
- `verl/workers/config/model.py`

The reward logic itself lives in `parason_rl/deepscaler/` (adapted from
[DeepScaleR](https://github.com/agentica-project/rllm)); verl imports `deepscaler_reward_fn` from
`deepscaler/rewards/math_rewardv2.py`.
