# VDAM hybrid same-state iteration-35 boundary — H100 job 13358712

## Decision

The direct and certified-hybrid E-steps are decision-equivalent when they are
started from the exact same live VDAM iteration-34 state. All compared particle,
pose, translation, class, posterior, significance-count, and exact support-ID
outputs agree. The continuous reconstruction differences are the same order as
the differences between two repetitions of the same backend.

This closes a deterministic hybrid score/support omission as the cause of the
iteration-35 split in the full trajectory. It does **not** qualify the full
trajectory: ordinary CUDA atomic-order perturbations accumulated before
iteration 35 are still amplified into different VDAM basins.

## Experiment

| Item | Value |
|---|---|
| Slurm job | `13358712` (`COMPLETED`, exit `0:0`) |
| Source | `2fc852da7a408d32dd0142fb177371de0407c552` |
| GPU | H100, `GPU-9f98ccbf-3c62-c54f-7409-7eb58845ad4a` |
| Node / elapsed / MaxRSS | `della-h19g1` / `00:04:56` / `6,849,108 KiB` |
| Boundary | Direct checkpoint through iteration 34, then `direct / hybrid / hybrid / direct` for iteration 35 |
| Report | `science/report.json`, SHA-256 `a3809404cf10f5c4dd473c3a09795189cc07c543fb3b7d9d89b8fc892c3e7153` |
| Artifact root | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_hybrid_same_state_it34_2fc852da7/` |

Every arm received an independent deep copy whose model, particle, and
sampling-state manifests exactly matched the captured checkpoint.

## Exact science boundary

All six pairwise comparisons were exact for the recorded E-step decision
fields, including all four direct/hybrid comparisons:

- selected particle IDs;
- best rotation IDs and full pose assignments;
- translations and class assignments;
- maximum posterior values and significant counts;
- particle state and sampling state.

The exact-ID support audit was also identical in every arm: 200 images, 3,062
selected hypotheses, and aggregate support SHA-256
`dd63e10c2ea2e630af7d7b6c23e902274fa2138165e51ae53288556d58c94522`.
Both hybrid arms selected all 200 images without a fallback.

## Continuous reconstruction audit

CUDA reconstruction atomics are not repeat-bitwise even within one backend.
Cross-backend normalized-L2 deltas stayed at the same scale as those repeats:

| Field | Direct repeat | Hybrid repeat | Max direct/hybrid | Cross / max repeat |
|---|---:|---:|---:|---:|
| accumulator 0 data | `4.685e-8` | `4.300e-8` | `4.899e-8` | `1.046x` |
| accumulator 0 weight | `5.457e-8` | `4.588e-8` | `5.089e-8` | `0.933x` |
| accumulator 1 data | `7.569e-8` | `7.375e-8` | `7.633e-8` | `1.009x` |
| accumulator 1 weight | `4.479e-8` | `4.204e-8` | `4.440e-8` | `0.991x` |
| final `Iref` | `2.534e-10` | `2.968e-10` | `3.370e-10` | `1.136x` |

Other final-state fields were likewise tiny; the largest observed
cross/repeat ratio was `1.346x` for `data_vs_prior_class`. That is descriptive,
not a newly invented acceptance threshold.

## Timing

| Arm | Wall time (s) | Interpretation |
|---|---:|---|
| `direct_1` | `12.129` | cold direct compilation |
| `hybrid_1` | `7.068` | cold hybrid compilation |
| `hybrid_2` | `1.839` | warm hybrid |
| `direct_2` | `2.062` | warm direct |

At this modest iteration-35 grid the warm hybrid is `1.121x` faster. The
larger full-trajectory gain comes from later fine grids; the remaining runtime
work is the shared local/pass-2 executor and shape/JIT churn.

## Follow-up

The same-state harness now compares support audits nested inside per-halfset
profile summaries explicitly. The performance implementation should remain in
the shared EM fixed-capacity local path: generalize its sealed call-0 view to
all chronological calls, then carry the mature local score/reconstruction
primitives through one fixed-shape execution boundary.
