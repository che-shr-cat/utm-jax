# ADR 001: Adaptive Computation Time via Bounded `lax.scan`

## Status
Accepted

## Context
The Universal Transformer needs a dynamic-depth pondering mechanism so different tokens can stop processing at different steps. A naive implementation with a Python `while` loop or dynamic early break introduces variable shapes into the XLA compilation, which causes long compile times and breaks `jax.sharding`.

## Decision
We use `jax.lax.scan` bounded by a fixed `max_ponder_steps`.

- Every token executes the loop for the full `max_ponder_steps` iterations. Halting is handled analytically rather than by exiting the loop.
- Once a token's accumulated halting probability reaches `1 - epsilon`, a `jax.where` mask freezes its hidden state for all remaining steps.
- The remainder term `R = 1 - sum_{i<halt} p_i` is applied at the halting step only, providing the gradient path required by Graves' ACT formulation.

## Consequences

**Pros**
- Static compute graph, fast XLA compile.
- Trivially compatible with `jax.sharding` and `nnx.jit`.

**Cons**
- No wall-clock speedup from early halting on TPU — every step is physically executed. This is acceptable here because TPUs do not benefit from dynamic sparsity at the block level anyway, and we care about *learned* halting behavior more than runtime savings.
