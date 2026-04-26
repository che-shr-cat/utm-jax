# Architecture Decision Records

These ADRs document the load-bearing design choices behind UTM-Jax. They're ordered chronologically; gaps in numbering are intentional and signal decisions that were superseded or scoped out for the v1 paper.

| # | Title | Status |
|---|-------|--------|
| 001 | [ACT via bounded `lax.scan`](ADR_001_ACT.md) | Accepted |
| 002 | [Norm-Free block + SwiGLU](ADR_002_NormFree_SwiGLU.md) | Accepted |
| 003 | [Memory Tokens](ADR_003_Memory_Tokens.md) | Accepted |
| 004 | [TPU Deployment Pipeline](ADR_004_Deployment.md) | Accepted |
| 005 | [RoPE with independent indices](ADR_005_RoPE.md) | Accepted |
| 006 | [Baseline Hyperparameters](ADR_006_Baseline_Hyperparameters.md) | Accepted |
| 008 | [QK-Norm, Decoupled QKV, Split-Clip Optimizer Topology](ADR_008_QKNorm_and_Muon_Topology.md) | Accepted |
| 011 | [Graves ACT Gradient Implementation](ADR_011_Graves_ACT_Gradient.md) | Accepted |
| 013 | [ACT Router Initialization Bias](ADR_013_Router_Init_Bias.md) | Accepted |
| 014 | [Deep-Start Router Default](ADR_014_Deep_Start_Default.md) | Accepted |

## Why this ordering matters
The empirical claims in the paper are best understood after reading 011 (which invalidated some pre-2026-04-15 ablation results) and 013/014 (which identified the router init trap as the dominant cause of seed sensitivity). ADRs 001–006 give the architectural context.
