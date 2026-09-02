# EMPIAR-10076 K=4 native coarse-score boundary

This diagnostic closes the ambiguity left by the shared-200 parent-count
counterfactual. It captures RELION's production coarse `diff2`, priors,
pre-exponent score, exponentiated weight, and significant-support decision for
the same 12 mismatch probes and four exact controls already captured by
RECOVAR. The capture is bounded, explicitly enabled, and is not a production
code path.

## Outcome

The first material difference is the raw likelihood/`diff2` surface, not the
class, orientation, or translation priors.

The two strongest causal score swaps are exact across all 16 particles:

- RELION raw likelihood plus RECOVAR's production prior contribution selects
  all 1,336 RELION parents exactly.
- RECOVAR raw likelihood plus RELION's captured priors selects exactly the same
  counterfactual parents as RECOVAR's ordinary combined score.

Thus copying RELION's parent count or changing RECOVAR's priors is not a valid
repair. The next boundary is inside coarse Gaussian likelihood evaluation,
principally the class/rotation-dependent term.

## Native capture qualification

Slurm job `13330906` requested and received exactly one H100, eight CPUs, and
192 GB without exclusivity. `ReqTRES` and `AllocTRES` were identical. The two
serial replay arms completed in 35 seconds; peak batch RSS was 2,782,180 KiB
for the control arm and 2,723,356 KiB for the instrumented arm.

The diagnostic RELION source is
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_coarse_score_capture_20260902`
at commit `a32cccccb8ba93b9dc471204d652236f67226282`. The exact binary is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k4_coarse_score_capture_build_20260902/build/bin/relion_refine`
with SHA-256
`f54e20924a99268a2df840390adef1738edf9a8d5ea2788a049a5bbc378b97e6`.
The submitted script is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_score_a32cccccb_20260902/jobs/run_native_coarse_score.sbatch`
with SHA-256
`93de36e19e78f2f67e41dcdbdb5b27594cf0836c7e1f239207bed8c0a8621984`.

The fail-closed capture validator checked 16 files and 1,069,056 candidates:

| Capture invariant | Result |
| --- | ---: |
| Expected particles | 16/16 |
| Candidate records | 1,069,056 |
| Active candidate records | 1,069,056 |
| Significant candidates | 1,336 |
| Prior/`diff2` algebra maximum absolute error | 0 |
| Exponent-shift maximum absolute error | 0 |
| Exponentiation maximum relative error | `2.525e-7` |

The authoritative validation is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_score_a32cccccb_20260902/analysis/capture_validation.json`
with SHA-256
`3d8913639736ec73d65ad486e3e1235b1a52d86244d4aa4e7967c591220bc1a9`.

## Capture inertness

The two independent RELION arms are not byte-identical. They do establish a
much tighter numerical inertness boundary:

- all iteration-0 map arrays and particle-table fields are exact;
- all iteration-1 angles, origins, classes, and significant-sample counts are
  exact;
- only `rlnLogLikeliContribution` (142/10,000 rows, maximum `7e-5`) and
  `rlnMaxValueProbDistribution` (107/10,000 rows, maximum `3.8e-5`) differ;
- iteration-1 class-map relative L2 is `6.11e-7`--`2.46e-6`; and
- class-map FSC-AUC is `0.999999999078`--`0.999999999651`.

The fixed capture-inertness gate therefore passes at FSC-AUC `0.999999` and
relative-L2 `1e-5`, while explicitly recording that bitwise inertness is
false. The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_score_a32cccccb_20260902/analysis/capture_inertness.json`
with SHA-256
`9aa9a484baabb36609ab65f15590925e4ddcc50a54c2f484e8c92e4aacb471f0`.

## Aligned score components

RELION's direction-major rotation IDs are converted to RECOVAR's psi-major
IDs before comparison. One additive constant per particle is removed because
the two raw score definitions differ by the image-only Xi2 contribution.
No scale, affine fit, or candidate filtering is used.

| Component over 1,069,056 values | Maximum centered abs. | Pooled centered RMS | Median centered relative L2 | Winners exact |
| --- | ---: | ---: | ---: | ---: |
| Orientation/class prior | 0 | 0 | 0 | 16/16 |
| Translation prior | `1.505e-6` | `2.475e-7` | `1.222e-7` | 16/16 |
| Total production prior contribution | `1.458e-5` | `2.827e-6` | `1.814e-6` | 7/16 |
| Raw likelihood score | `6.694` | `0.4141` | `0.01798` | 16/16 |
| Combined score | `6.694` | `0.4141` | `0.01799` | 16/16 |

The prior winner count is not an acceptance metric: the prior surfaces contain
many repeated values, so float32 addition-order noise can change an arbitrary
prior-only argmax. The exact support swaps below are the causal metric.

| Stable top-RELION-count surface | Exact particles | Aggregate support Jaccard |
| --- | ---: | ---: |
| Native combined score | 16/16 | 1.000000 |
| Native raw likelihood + RECOVAR prior | 16/16 | 1.000000 |
| RECOVAR combined score | 5/16 | 0.951790 |
| RECOVAR raw likelihood + native prior | 5/16 | 0.951790 |
| Native raw likelihood without priors | 1/16 | 0.582938 |
| RECOVAR raw likelihood without priors | 1/16 | 0.583877 |

There are no top-count boundary ties in any of these surfaces. The direct
native support also reproduces the earlier fine-expansion-derived support
metrics exactly, so the result does not depend on inferring parents from the
fine pass.

Across particles, the median fraction of centered raw-score squared difference
carried by a class/rotation-only mean is `0.83066` (range
`0.71548`--`0.92354`). The median within-class/rotation fraction is `0.16934`.
The translation-only main effect is just `0.000457` at the median. This is
consistent with the projection/reference-norm boundary carrying most of the
gap, with a smaller translation-dependent cross-term remainder; it does not
by itself prove which kernel operand is responsible.

The authoritative score report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_score_a32cccccb_20260902/analysis/native_coarse_score_boundary.json`
with SHA-256
`659adf64f0d451ac5b5eaf6dcd3a4faa6118cd435f485792dc16780510391e47`.
The compact checked-in result is
`docs/benchmarks/em/diagnostics/real-k4-native-coarse-score-a32cccccb-20260902.json`;
it pins the causal metrics, instrumentation floor, native resource envelope,
artifact hashes, and diagnostic-only admission boundary.

## Reproduction

Re-run both immutable post-hoc audits with fresh output names:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_harness_integrate_20260901
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1

.pixi/envs/default/bin/python -m scripts.analyze_em_real_k4_native_coarse_scores \
  --capture-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_score_a32cccccb_20260902/capture \
  --significance-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_significance_probe_r5_24317e40c_20260901/significance \
  --support-report /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_significance_probe_r5_24317e40c_20260901/analysis/coarse_score_support.json \
  --output-json /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_score_a32cccccb_20260902/analysis/native_coarse_score_boundary_replay.json

.pixi/envs/default/bin/python -m scripts.audit_relion_k4_coarse_score_capture_inertness \
  --control-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_score_a32cccccb_20260902/control/output \
  --instrumented-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_score_a32cccccb_20260902/instrumented/output \
  --capture-validation /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_score_a32cccccb_20260902/analysis/capture_validation.json \
  --n-classes 4 \
  --output-json /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_score_a32cccccb_20260902/analysis/capture_inertness_replay.json
```

Focused source checks are:

```bash
pixi run ruff check \
  scripts/analyze_em_real_k4_native_coarse_scores.py \
  scripts/audit_relion_k4_coarse_score_capture_inertness.py \
  tests/unit/test_analyze_em_real_k4_native_coarse_scores.py \
  tests/unit/test_audit_relion_k4_coarse_score_capture_inertness.py
pixi run pytest -q \
  tests/unit/test_analyze_em_real_k4_native_coarse_scores.py \
  tests/unit/test_audit_relion_k4_coarse_score_capture_inertness.py \
  tests/unit/test_validate_relion_coarse_score_capture.py \
  tests/unit/test_analyze_em_real_k4_coarse_score_support.py
```

This is a causal diagnostic, not an accepted real-data K=4 refinement result.
The rejected independent-half pilot remains rejected, and final admission still
requires stable class populations and matched per-class half-map quality.
