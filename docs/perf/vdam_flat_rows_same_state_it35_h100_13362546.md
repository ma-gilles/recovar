# VDAM packed fine rows at the production seam — H100 job 13362546

## Decision

Do not promote scorer-only packed rows. The real K=1 InitialModel call preserves
all discrete science and removes 38.89% of static score rows, but its warm
pass-2 and big-JIT times are neutral and its warm whole-iteration wall is
4.04% slower. The next candidate must also avoid dense projection and defer
M-step/noise work onto the already packed reconstruction support.

## Qualification

| Field | Value |
|---|---|
| Source | `1de2e224b2aed0659cf954e4763fb1ef260355f5` |
| Slurm | `13362546` (`COMPLETED`, exit `0:0`) |
| Hardware | `della-h19g1`, NVIDIA H100 80GB HBM3, `GPU-0d7b80c7-fef8-e346-6332-de36ae1af518` |
| Boundary | One exact in-memory GF46 iteration-34 state, iteration `34 -> 35` |
| Panel | direct / flat rows / flat rows / direct |
| Peak process RSS | 6,341,504 KiB |

Every arm begins from deep copies with identical model, particle, and sampling
state manifests. The candidate is controlled by
`RECOVAR_INITIAL_MODEL_FLAT_LOCAL_ROWS=1`; it remains off by default.

## Science result

- Selected particles, class assignments, pose assignments, best rotation IDs,
  best translations, Pmax values, and significant counts are exactly equal in
  every direct/candidate comparison.
- All 200 coarse-support audit rows and the aggregate support digest are exact.
- Reconstruction accumulator and final-state differences remain inside twice
  the maximum direct/direct or candidate/candidate repeat envelope. The worst
  normalized/envelope ratio is exactly 2.0; there is no candidate-specific
  decision or stability escape.

## Runtime result

| Metric | Warm direct | Warm flat rows | Change |
|---|---:|---:|---:|
| Whole iteration | 2.302587 s | 2.395506 s | +4.04% |
| Pass 1 | 0.591923 s | 0.617566 s | +4.33% |
| Pass 2 | 1.219563 s | 1.223849 s | +0.35% |
| Accounted shared EM | 1.174117 s | 1.177637 s | +0.30% |
| Local big JIT | 1.071022 s | 1.070994 s | -0.003% |

The packed plan reduces score rows from 62,208 to 38,016 (`-38.89%`) while
the logical union is 8,808 rows. That reduction produces no live speedup
because the current non-score path still projects the dense rectangular grid
and retains the same M-step/noise topology. This closes the scorer-only lane
as a performance explanation.

Cold timings are not used for the decision: direct was 13.162101 s and flat
rows was 5.750466 s because each arm compiled a different static program.

## Provenance

- Report JSON SHA-256: `0e255292085570ed3ee92a08450da083664bd31dd3794ecbae6dc4aaaf6536ee`
- CUDA binary SHA-256: `d49f79d04216657044064a21d4d37ad076b3fadced0214520b1fba907f68b5ba`
- Artifact manifest SHA-256: `9d07c95c3050316d56b21cd668dab66b15510a2b4cf56d4351e76f8a6cd21abe`
- Disposable artifact root:
  `/scratch/gpfs/GILLES/mg6942/vdam_runs/vdam_flat_rows_same_state_it34_1de2e224b_20260902T231827Z`

