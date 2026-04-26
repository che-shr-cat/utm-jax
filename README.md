# UTM-Jax: Universal Transformer with Memory Tokens

A JAX/Flax implementation of a **Universal Transformer with Memory Tokens** and Adaptive Computation Time (ACT), built to study depth–state trade-offs in recursive reasoning on algorithmic tasks (Sudoku-Extreme).

This repo accompanies the paper *"Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive Reasoning"* and is intended as both a reference implementation and a pedagogic asset — every load-bearing design choice is documented in [`docs/adr/`](docs/adr/) so the path from architecture to results can be retraced.

## What's interesting here

- **Memory tokens** added to the recurrent loop give the model a positional-invariant scratchpad. Without them, the model fails to solve Sudoku-Extreme in this configuration regardless of depth or seed.
- **Deep-start router init** (negative bias on the ACT halting head) eliminates a pervasive initialization trap where the router collapses to a 2-step shallow halt and never recovers. See ADR 013 / 014.
- **Graves ACT gradient correctness**. We initially had a subtle bug where the ponder penalty gradient evaluated to zero; the fix is documented in ADR 011.
- **Bounded ACT loop** via `jax.lax.scan` with a fixed `max_ponder_steps` and `jax.where` masking — gives predictable XLA compile graphs at the cost of always running the upper bound.

## Architecture summary

- Flax NNX, pure JAX
- RoPE with independent positional indices for memory and sequence tokens (ADR 005)
- Norm-free block with SwiGLU MLP (ADR 002)
- ACT halting via cumulative probability + remainder, deep-start init by default (ADR 014)
- Optional Muon optimizer for matrix params (out of scope for the v1 paper but supported)

## Repository layout

```
models/        # UT block, RoPE attention, ACT router
optimizers/    # Muon
dataset/       # Sudoku, ARC, Maze dataset builders
tests/         # Component, numerical, sharding tests
scripts/       # Eval, analysis, paper figure generation, TPU deploy
docs/adr/      # Architecture Decision Records
train.py       # Main training loop
puzzle_dataset.py
```

## Installation

Requires Python 3.10+. Designed for Google Cloud TPU v5p / v6e.

```bash
git clone https://github.com/che-shr-cat/utm-jax.git
cd utm-jax
pip install -r requirements.txt
```

For TPU runtime: the TPU VM image already includes a compatible JAX. On a CPU/GPU machine, swap `jax[tpu]` in `requirements.txt` for the variant matching your hardware.

## Local quickstart

Build the Sudoku-Extreme dataset (downloads from HuggingFace, ~3.83M train / 423K test):

```bash
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-full
```

Train with the deep-start default (ADR 014):

```bash
python train.py \
    --data_paths data/sudoku-extreme-full \
    --global_batch_size 256 \
    --epochs 4 \
    --hidden_size 512 --num_heads 8 \
    --num_memory_tokens 16 \
    --max_ponder_steps 18 \
    --ponder_lambda 0.0 \
    --router_init_bias -3.0 \
    --use_ema \
    --run_name utm-T16-deep-start
```

Reproducing the legacy shallow-start trap (for ablation):

```bash
python train.py ... --router_init_bias 0.0
```

## TPU deployment

Workflow: configure → create TPU → rsync code → run in tmux → upload checkpoints to GCS → tear down.

```bash
cp .env.example .env  # fill in GCP_PROJECT, GCS_CHECKPOINT_BUCKET

# Create a v6e-1 (smart zone fallback if primary zone is out of capacity)
./scripts/create_tpu.sh utm-run-1 v6e-1

# Push code and start training inside tmux
./scripts/sync_and_run.sh utm-run-1 us-south1-ai1b "python train.py --data_paths data/sudoku-extreme-full ..."

# After training (or on schedule), pull checkpoints into a GCS bucket
./scripts/upload_checkpoints.sh utm-run-1 us-south1-ai1b

# Done — release the TPU
./scripts/teardown_tpu.sh utm-run-1 us-south1-ai1b
```

The deploy scripts use `$GCP_PROJECT` from `.env` (or fall back to `gcloud config get-value project`), don't hardcode any zone, and try queued-resources flex-start first before falling back to a synchronous polling loop across zones that stock the requested accelerator.

See ADR 004 for the deployment design.

## Running tests

```bash
pytest tests/
```

## Reproducing paper results

The paper reports results across 3 seeds (0, 42, 123) for the memory-token sweep at hidden_size=512. To reproduce a single point:

```bash
for SEED in 0 42 123; do
    python train.py \
        --data_paths data/sudoku-extreme-full \
        --global_batch_size 256 --epochs 4 \
        --hidden_size 512 --num_heads 8 \
        --num_memory_tokens 16 \
        --max_ponder_steps 18 \
        --ponder_lambda 0.0 \
        --router_init_bias -3.0 \
        --use_ema --seed $SEED \
        --run_name utm-T16-S${SEED}
done
```

For the full sweep, vary `--num_memory_tokens` over `{0, 8, 16, 32, 64}`.

## Architecture Decision Records

The `docs/adr/` directory contains the design decisions in chronological order. Numbering has gaps where decisions were superseded or scoped out for v1 — the public set is curated to the choices that survived into the paper.

- [ADR 001 — ACT via bounded `lax.scan`](docs/adr/ADR_001_ACT.md)
- [ADR 002 — Norm-Free block + SwiGLU](docs/adr/ADR_002_NormFree_SwiGLU.md)
- [ADR 003 — Memory Tokens](docs/adr/ADR_003_Memory_Tokens.md)
- [ADR 004 — TPU Deployment](docs/adr/ADR_004_Deployment.md)
- [ADR 005 — RoPE with independent indices](docs/adr/ADR_005_RoPE.md)
- [ADR 006 — Baseline hyperparameters](docs/adr/ADR_006_Baseline_Hyperparameters.md)
- [ADR 008 — QK-norm, decoupled QKV, split-clip optimizer topology](docs/adr/ADR_008_QKNorm_and_Muon_Topology.md)
- [ADR 011 — Graves ACT gradient implementation](docs/adr/ADR_011_Graves_ACT_Gradient.md)
- [ADR 013 — ACT router initialization bias](docs/adr/ADR_013_Router_Init_Bias.md)
- [ADR 014 — Deep-start router default](docs/adr/ADR_014_Deep_Start_Default.md)

## Citation

```bibtex
@misc{utm_jax_2026,
  author = {Sapunov, Grigory},
  title  = {UTM-Jax: Universal Transformer with Memory Tokens},
  year   = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/che-shr-cat/utm-jax}}
}
```

## License

MIT.
