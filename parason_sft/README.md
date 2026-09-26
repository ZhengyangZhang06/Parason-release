# Parason SFT

Supervised fine-tuning that teaches a model the Parason parallel-reasoning format. The trainer
uses a **prefix-tree data collator** that builds a custom attention mask and position ids so that
sibling `<Thread>…</Thread>` branches inside a `<Parallel>…</Parallel>` block do not attend to one
another, while the shared prefix and the post-parallel continuation attend normally.

The format's special tags are `<Parallel>`, `<Thread>`, `<Outlines>`, `<Conclusion>`, plus the
outline-item tags `<Outline>` / `<Trial>` / `<Subtask>` (a `<Trial>` frames one of several parallel
attempts; a `<Subtask>` frames one piece of a decomposed problem). The collator treats all three
outline-item tags equivalently when building the attention mask.

## Install dependencies

Dependencies are pinned in the top-level [`requirements.txt`](../requirements.txt):

```bash
pip install -r ../requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

(SFT does not need the RL-only packages such as `ray`, `vllm`, or `sglang`.)

## Generate data

A self-contained synthetic multiplication task lets you smoke-test the whole pipeline without any
external data. Each example asks for the product of a chain of integers; with `--parallel`,
independent sub-products are placed in separate `<Thread>` branches. See
[data/README.md](data/README.md) for all options.

Generate both formats: **JSON** for SFT (trl's `SFTTrainer` mis-tokenizes the extra `prompt`
column that the parquet writer adds for verl) and **Parquet** for the visualizer, `simple_eval`,
and RL.

```bash
cd data
python generate_math.py -n 10000 --task mult --parallel \
    --create_val --val_num_examples 1000 --min_len 5 --max_len 8 \
    --save_format json    --dataset_dir mult-10k-par     --overwrite
python generate_math.py -n 10000 --task mult --parallel \
    --create_val --val_num_examples 1000 --min_len 5 --max_len 8 \
    --save_format parquet --dataset_dir mult-10k-par_pq  --overwrite
cd ..
```

## Obtain a longer-context base model

Qwen3-8B ships with a 40960-token context; extend it to 131072 for long parallel traces:

```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir ./Qwen/Qwen3-8B-131072
sed -i 's/"max_position_embeddings": 40960,/"max_position_embeddings": 131072,/' \
    Qwen/Qwen3-8B-131072/config.json
```

## Training

The training data must expose a `qwen_text` field (the generator above does).

```bash
# Parallel-reasoning model (train_parallel.sh defaults to the JSON dataset ./data/mult-10k-par)
OUTPUT_DIR=ckpts/Q3-8B-131072-SFT ./train_parallel.sh

# Sequential baseline (same data; special tokens like <Parallel> are treated as plain text)
OUTPUT_DIR=ckpts/Q3-8B-131072-AR-SFT ./train_sequential.sh
```

Key knobs inside `train_parallel.sh`: `lr`, `epochs`, `micro_batch_size`,
`gradient_accumulation_steps`, `block_size`, `attn_implementation`, and `base_model`.
The scripts use `configs/deepspeed_zero3.json` by default; for memory-constrained setups switch to
`configs/deepspeed_zero3_offload.json` (CPU optimizer offload, which requires DeepSpeed's compiled
`cpu_adam` op).

The two baselines use the **same** dataset for a fair comparison — the sequential model simply
does not receive the prefix-tree attention mask.

To reproduce the released model on the real data instead of the synthetic task, use
[`run_sft.slurm`](../run_sft.slurm) (submitted from the repo root), pointing its `ALL_TRAIN_FILES`
entry at the training data under `data/`. That data is withheld during the anonymous review period
and will be linked on publication.

## Attention / position visualizer

Inspect the tokens, attention mask, and position ids the collator produces for any sample:

```bash
python src/prefix_tree_visualizer.py \
    --dataset-path data/mult-10k-par_pq/train.parquet \
    --model-name Qwen/Qwen3-8B-131072 \
    --port 8008
```

Open `http://localhost:8008`, pick a sample index, and click a token to see its attention row.
`--text-field` defaults to `qwen_text`; `--template-name` defaults to `qwen`. Uses CPU if CUDA is
unavailable.

## Quick evaluation

A lightweight evaluator that runs branching generation with SGLang and grades the answers. For
serious evaluation we recommend the verl-based parallel rollout in
[../parason_rl/README.md](../parason_rl/README.md).

```bash
TRAINED_MODEL="ckpts/Q3-8B-131072-SFT"
python src/simple_eval.py \
    --data-type data/mult-10k-par_pq/train.parquet \
    --model_name "$TRAINED_MODEL" \
    --launch_server --template-type model --bfloat16 \
    --branching-generate -n 1 --max-context-length 8192 --verbose 2
```

## Source layout

| File | Purpose |
| --- | --- |
| `src/sft_parallel.py` | Parallel-reasoning SFT trainer (uses the prefix-tree collator). |
| `src/sft_sequential.py` | Sequential baseline trainer. |
| `src/prefix_tree.py` | Prefix-tree collator: builds the per-thread attention mask and position ids. |
| `src/rewards.py` | Math grading and parallel-structure statistics used by the evaluator. |
| `src/simple_eval.py` | SGLang branching-generation evaluator. |
| `src/prefix_tree_visualizer.py` | Flask app to visualize the collator output. |
| `src/utils.py` | Shared helpers (special-token setup, server launch). |
| `data/generate_math.py`, `data/mult_utils.py` | Synthetic multiplication dataset generator. |

## Tests

```bash
python -m unittest discover -s src/tests
```
