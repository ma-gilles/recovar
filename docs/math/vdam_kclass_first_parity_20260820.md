# VDAM K-class first parity boundary (2026-08-20)

## Contract

K-class InitialModel comparisons use a Hungarian class assignment selected by
normalized non-DC FSC-AUC.  Each matched class must have FSC-AUC at least
`0.999`; assigned-particle labels must agree at least `0.995` after applying
the same permutation.  Map correlation is neither computed nor gated.
Iteration zero has no assigned particles in RELION and is therefore map-only.

Both programs run sequentially on one physical Slurm GPU.  The reference is
the clean pinned RELION `f2c1a384400aec37dc6805856a5ba645650a44f1`
executable with SHA-256
`08d9151c976ec51e664060992db62d929e47282e0d8481d977f64e32f07fca39`.

## First K=2 result

The 5,000-particle, 128-pixel two-state PDB fixture first exposed a strict
CUDA preprocessing boundary failure: the split local-exact path did not pass
explicit identity normalization factors and integer shifts after those
operations had already been handled by its caller.  The backend correctly
failed closed.  The fixed boundary supplies typed identity operands only to
the strict RELION CUDA backend; general backends retain the prior path.

Same-GPU paired job `12767153` then passed iterations 0, 1, and 2:

| Iteration | Minimum matched FSC-AUC | Assignment agreement |
|---:|---:|---:|
| 0 | 1.000000000 | not assigned |
| 1 | 0.999995378 | 1.0000 |
| 2 | 0.999993747 | 0.9984 |

At iteration 2, eight of 5,000 hard class labels differ (four in each
direction), while both class maps remain above `0.9999937`.  This residual is
kept visible rather than described as exact particle-state equality.

The paired CLI wall times were 8.93 seconds for RELION and 51.83 seconds for
RECOVAR, a `5.80x` ratio.  Comparable-runtime work therefore remains open.
The immutable diagnostic report is:

`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k2_paired_evidence_retry_20260820T1745Z/pair/pair_report.json`

with SHA-256
`4c773b4fe1ef73e984e4afdf843b46cf68408603d861a618f8b2c3ea1d059724`.

The next qualification boundary is the clean-source K=2 default trajectory
at checkpoints 0, 1, 2, 4, and 8, followed by parameter variants and K=4.

## Current qualification tracker (2026-08-22)

The K-class path now passes the complete eight-iteration default trajectory
for both K=2 and K=4.  K=2 also passes the first parameter panel.  These
audits compare every written iteration (0 through 8), require exact artifact
topology, and retain the fixed `0.999` per-class FSC-AUC and `0.995` class
assignment thresholds.

| Case | Result | Minimum matched FSC-AUC | Minimum assignment accuracy | RELION wall time | RECOVAR wall time | Ratio |
|---|---|---:|---:|---:|---:|---:|
| K=2 default | PASS | 0.9999995522 | 1.0000 | 21.36 s | 119.97 s | 5.62x |
| K=4 default | PASS | 0.9999999947 | 1.0000 | 23.55 s | 168.76 s | 7.16x |
| K=2 seed 7 | PASS | 0.9999999996 | 1.0000 | 25.65 s | 313.97 s | 12.24x |
| K=2 Healpix 0 | PASS | 0.9999999995 | 1.0000 | 20.19 s | 366.77 s | 18.16x |
| K=2 tau2 fudge 2 | PASS | 0.9999999999 | 1.0000 | 28.59 s | 228.09 s | 7.98x |
| K=2 tau2 fudge 8 | PASS | 0.9999999999 | 1.0000 | 28.71 s | 217.94 s | 7.59x |
| K=2 offset range/step 4/1 | PASS | 0.9999999999 | 1.0000 | 28.93 s | 322.38 s | 11.14x |
| K=2 diameter 140 A | PASS | 0.9999999898 | 1.0000 | 28.59 s | 304.26 s | 10.64x |
| K=2 Healpix 2 | PASS | 0.9999996951 | 1.0000 | 30.76 s | 246.61 s | 8.02x |
| K=2 oversampling 2 | PASS | 0.9999999987 | 1.0000 | 31.44 s | 434.77 s | 13.83x |
| K=2 oversampling 0, latest EM stack | PASS | 0.9999999993 | 0.9998 | 28.61 s | 87.99 s | 3.07x |

The default and first parameter runs are Slurm jobs `12777536`, `12777537`,
`12777748`, `12777749`, `12778397`, `12778401`, `12778803`, `12778815`,
`12784677`, and `12784678`.  Job `12787780` is the clean post-rebase K=2
oversampling-zero qualification on top of the current PR 158 EM stack.  Its
immutable report is:

`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k2_os0_3cd31e91/pair/pair_report.json`

Correctness is qualified for this panel, but runtime is not: RECOVAR remains
3.07x to 18.16x slower on these small paired jobs.  The remaining tracker is:

- rerun the K=2 default and nonzero-oversampling panel after every material EM
  stack update;
- freeze and run K=4 parameter, scale, repeatability, 25-iteration, and
  real-particle panels;
- freeze and run K>1 larger-particle and larger-grid scaling panels;
- profile the qualified exact-local route and remove recompilation and
  per-iteration launch overhead without changing the parity thresholds;
- keep the public `recovar initial_model` CLI and GUI defaults as the single
  configurable entry point while exposing the important sampling, class,
  mask/CTF, batching, and diagnostic controls.
