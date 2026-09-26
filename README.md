# Parason

**Parason** trains large language models to reason in **parallel** — exploring multiple
solution threads at once instead of a single long chain of thought. On problems with
independent sub-computations this improves solution quality and, because the threads share a
critical path, cuts the *token latency* to an answer.

A model is taught to structure its reasoning with special tags:

- `<think>…</think>` — the chain of thought
- `<Parallel>…</Parallel>` — a block whose child threads run independently
- `<Outlines>…</Outlines>` — the header of a parallel block: one outline item per thread, framed as either
  - `<Subtask>…</Subtask>` — an independent sub-task the problem is **decomposed** into (divide-and-conquer), or
  - `<Trial>…</Trial>` — one of several independent **attempts / hypotheses** explored in parallel
- `<Thread>…</Thread>` — one reasoning thread, executing its matching outline item
- `<Conclusion>…</Conclusion>` — the final answer, in `\boxed{}`

(The two framings are structurally identical — both spawn sibling `<Thread>`s — but let the model express two kinds of parallelism, which the RL reward can incentivize separately.)

Training has two stages:

1. **SFT** ([`parason_sft/`](parason_sft/)) — supervised fine-tuning that teaches the model the
   parallel-reasoning format, with a prefix-tree attention mask so sibling threads do not attend
   to one another.
2. **RL** ([`parason_rl/`](parason_rl/)) — GRPO reinforcement learning (built on a
   [verl](https://github.com/volcengine/verl) fork) whose reward combines answer **correctness**
   with an **acceleration-ratio** term that rewards shorter critical paths.

## Repository layout

| Path | What it is |
| --- | --- |
| `run_*.slurm` | **The reproduction entry points** — `run_sft.slurm`, `run_rl.slurm`, `run_merge.slurm`, `run_eval.slurm`, `run_simple_eval.slurm`. Submit from the repo root. |
| [`parason_sft/`](parason_sft/) | SFT trainer, prefix-tree collator, synthetic-data generator, quick eval. |
| [`parason_rl/`](parason_rl/) | GRPO RL on a vendored [verl](https://github.com/volcengine/verl) fork. See [`parason_rl/MODIFICATIONS.md`](parason_rl/MODIFICATIONS.md) for exactly what we changed vs. upstream. |
| [`parason_rl/deepscaler/`](parason_rl/deepscaler/) | Math reward (correctness grader), adapted from [DeepScaleR](https://github.com/agentica-project/rllm). |
| [`requirements.txt`](requirements.txt) | Pinned Python dependencies for both stages. |

## Installation

Requires Python 3.10 and CUDA 12.4. Both stages run on a single node with 8× 80 GB A100/H100
GPUs; multi-node is supported for RL.

```bash
python -m venv .venv && source .venv/bin/activate    # or conda create -n parason python=3.10
pip install -r requirements.txt
# flash-attn and the OpenTelemetry extras must be built without isolation:
pip install flash-attn==2.7.4.post1 opentelemetry-sdk==1.39.1 \
    opentelemetry-exporter-prometheus==0.60b1 --no-build-isolation
# Install the verl fork (editable):
pip install -e parason_rl
```

## Preparing checkpoints and data

The released checkpoints and datasets are withheld during the anonymous review period and will be
linked here on publication. The `run_*.slurm` scripts expect them under `ckpts/` and `data/` at the
repo root (both are gitignored, so nothing large is committed):

```
ckpts/
  Qwen3-8B/                                 # base model (SFT initialization), from Qwen/Qwen3-8B
  <SFT_CKPT>/                               # SFT checkpoint — RL initialization
  <RL_CKPT>/                                # RL checkpoint(s) to evaluate
data/
  <TRAIN_DATA>/                             # SFT + RL training data (parquet / json)
  <BENCH>/test.parquet                      # evaluation benchmark, e.g. aime24, math500
```

Only the base model is public today (`hf download Qwen/Qwen3-8B --local-dir ckpts/Qwen3-8B`,
`pip install -U huggingface_hub`). The checkpoint and data names above are the placeholders used by
the SLURM scripts — point them at your own paths, or at the released artifacts once available.
Everything in the Quickstart below runs from self-generated synthetic data and needs none of this.

## Quickstart

The fastest end-to-end smoke test uses a self-contained synthetic multiplication task (no
external data required):

```bash
cd parason_sft
# 1. Generate the synthetic dataset — JSON for SFT, Parquet for RL
python data/generate_math.py -n 10000 --create_val --parallel -p 0.5 \
    --save_format json    --dataset_dir data/mult-10k-par
python data/generate_math.py -n 10000 --create_val --parallel -p 0.5 \
    --save_format parquet --dataset_dir data/mult-10k-par_pq
# 2. Supervised fine-tune (train_parallel.sh defaults to data/mult-10k-par)
OUTPUT_DIR=ckpts/Q3-8B-131072-SFT ./train_parallel.sh
# 3. RL fine-tune the SFT checkpoint — see parason_rl/README.md for the single-node command
```

See [`parason_sft/README.md`](parason_sft/README.md) and
[`parason_rl/README.md`](parason_rl/README.md) for full instructions, hyperparameters, and how to
reproduce the math results.

## Reproducing the released models

The end-to-end cluster pipeline is driven by the SLURM scripts at the repository root — submit
them from the repo root after setting `--account`/`--partition` in each SBATCH header:

```bash
sbatch run_sft.slurm       # supervised fine-tuning
sbatch run_rl.slurm        # GRPO RL (sweeps the subtask/trial/latency reward weights)
sbatch run_merge.slurm     # merge FSDP checkpoints -> HuggingFace format
sbatch run_eval.slurm      # evaluate with the parallel rollout
sbatch run_simple_eval.slurm   # SGLang branching-generation eval
```

Each script reads its checkpoint and dataset paths from the arrays at the top of the file — set
them to your local `ckpts/` and `data/` locations (see *Preparing checkpoints and data* above)
before submitting.

## Results

On a synthetic multi-step multiplication task, after 400 RL steps the parallel model matches the
sequential baseline's accuracy while producing a shorter critical path:

| Setting | Tokens in longest thread | Accuracy |
| --- | :---: | :---: |
| Sequential baseline | 3322 | 99.0% |
| Parason (parallel) | 2632 | 99.0% |

That is a **1.26× reduction in token latency** at equal accuracy. See
[`parason_rl/README.md`](parason_rl/README.md) for how the acceleration ratio is computed.

## License and attribution

Parason is released under the [Apache License 2.0](LICENSE). It builds on two Apache-2.0 projects
whose attributions are recorded in [`NOTICE`](NOTICE):

- [verl](https://github.com/volcengine/verl) — the RL training framework (vendored fork).
- [DeepScaleR / rllm](https://github.com/agentica-project/rllm) — the math reward.
