# EMPIAR-10076 K=4 native coarse-operand boundary

This diagnostic moves the frozen shared-200 K=4 boundary inside the
image-reference cross term. It compares RELION and RECOVAR projected
references, shifted images, correction arrays, Euler matrices, and translation
phases for the same 12 mismatch probes and four exact controls over all four
classes, 576 rotations, and 29 translations.

## Historical localization outcome

At source commit `24317e40c`, the projected reference was the first material
coarse-likelihood operand difference.

| Stable top-RELION-count surface | Exact particles | Support Jaccard | Fraction of baseline centered score energy remaining |
| --- | ---: | ---: | ---: |
| Native projected reference only | 16/16 | 1.000000 | `1.171e-9` |
| Native shifted image only | 5/16 | 0.951790 | `1.00000013` |
| Native correction only | 5/16 | 0.951790 | `1.00000047` |
| All native operands | 16/16 | 1.000000 | `1.123e-9` |
| RECOVAR operands | 5/16 | 0.951790 | `1.0` |

No tested support boundary contains a tie. The projected-reference relative L2
is `0.0175086`, whereas the shifted-image and correction relative-L2 maxima are
`6.30e-7` and `3.07e-7`. RELION Euler matrices agree with the transpose of the
RECOVAR matrices to `3.87e-7` maximum absolute error, and translation phases
agree to `1.49e-8`. These counterfactuals exclude the shifted image, correction,
Euler bridge, and translation phase at this boundary. They do not yet
distinguish a Projector::data construction difference from texture
interpolation arithmetic.

## Current-source closure

The historical boundary is closed on current source. The production defect
was a radius mismatch in the RELION texture projection launch: texture storage
correctly retained the padded PPref geometry, but the image-space projection
cutoff also used that radius instead of RELION's rounded image cutoff
`min(projector_max_r, current_size / 2)`. Commit `5f74755c2` applies the image
cutoff to all five texture launchers while leaving the texture geometry
unchanged. It also removes the redundant exact axis-aligned postmask, which
was not RELION's rounded radial support.

The H100 CUDA build/gate job `13336787` passed all four focused GPU tests. An
exact native-Euler discriminator (job `13336962`) then compared four classes,
576 rotations, and 171 Fourier pixels and found byte-identical projected
references: maximum relative L2 and maximum absolute error were both zero.
Using host-reconstructed Euler matrices instead would have changed 466
outer-shell values despite a matrix maximum difference of only `3.87e-7`, so
the replay now uses RELION's native CUDA Euler construction as part of the
qualified path.

Clean replay job `13337519` ran immutable commit
`b34b1ecdad7af248aa348c0d45ea85a03f0105a9` and inferred both projection and
reconstruction padding factor 1 directly from the RELION model. It requested
and received one H100, eight CPUs, and 32 GB without exclusivity, completed in
201 s, and reached peak batch RSS 8,107,444 KiB. The result closes the complete
coarse support boundary:

| Current-source gate | Result |
| --- | ---: |
| Projected-reference relative L2, maximum | `0` |
| Euler matrix maximum absolute error | `0` |
| Exact probe records | 16/16 |
| Exact selected candidates | 1,336/1,336 |
| Aggregate significant-support Jaccard | `1.0` |
| Equal-score cutoff ties | 0 |

The inferred-padding replay is independently repeatable against the explicit
padding replay: all 11 parity arrays are bitwise equal. The eight reconstructed
maps are not bitwise equal because the GPU atomic accumulation schedule is not
fixed, but their minimum FSC-AUC is `0.999999997335` and maximum relative L2 is
`1.57406e-7`, both inside the predeclared `0.999999` and `1e-5` gates.

The authoritative current-source reports are:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_inferred_padding_r3_b34b1ecda_20260902/analysis/ppref_projection_boundary.json`
  (SHA-256 `e4cd20c2b78c4b1dd478e455bb5e33fb3f847b31ba10db6d7b5a21d6e4b79f1f`);
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_inferred_padding_r3_b34b1ecda_20260902/analysis/native_coarse_operand_boundary_after_fix.json`
  (SHA-256 `3746663a5ece085ac23b5f01d8952466c66496722b8b69826479cf379ad9dbba`);
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_inferred_padding_r3_b34b1ecda_20260902/analysis/inferred_padding_repeatability_v1.json`
  (SHA-256 `aeb5ab3799a6d6974054b5318ab6699df56d500171fa909ccc012bba34e4eafc`).

This admits the bounded iteration-1 coarse projected-reference and
significant-support boundary only. It does not admit a final real-data K=4
refinement, resolution, or engine performance result; the next causal gate is
the matched fine-score/posterior boundary, followed by independent-half
multi-seed trajectories.

## Three-repeat observer qualification

The first class-resolved operand observer synchronized each class before the
remaining production kernels were launched. That changed otherwise
nondeterministic binary32 atomic scheduling: job `13334206` changed poses and
origins and produced class-map FSC-AUC `0.897397`--`0.934271`. Its evidence is
rejected.

RELION commit `b061776eae1ea67f2ab01da2c1575137d3d2221a` instead retains the
bounded device buffers until RELION's existing all-class synchronization and
only then copies them to the host. Independent jobs `13334583`, `13334584`,
and `13334585` each requested and received exactly one H100, eight CPUs, and
192 GB without exclusivity. All three operand-arithmetic validators pass, all
iteration-1 decision fields are exact, and every class map passes the
predeclared FSC-AUC `0.999999` and relative-L2 `1e-5` gate. Across the three
repeats, minimum FSC-AUC is `0.999999996391` and maximum relative L2 is
`1.984e-6`.

The executable is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k4_coarse_components_build_20260902/build/bin/relion_refine`
with SHA-256
`cce955d739c30dc0ac8be3a04c5da2008f252b6dd5d86e7b11120e5e2957cd67`.
The authoritative repeatability report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_operands_deferred_r1_b061776_20260902/analysis/operand_capture_repeatability.json`
with SHA-256
`429c0e072d66853e34196c2d13140f601eadecaba1f3fc9656f2189065608864`.

## Operand analysis

The analysis removes one per-particle additive score constant. It applies no
scale fit, affine fit, or candidate filtering. Exact significant-support
recovery and the fraction of centered squared-residual energy remaining are
the causal gates; operand relative L2 is descriptive.

The authoritative report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_operands_deferred_r1_b061776_20260902/analysis/native_coarse_operand_boundary.json`
with SHA-256
`6f5fd33411a9a8e929a55e232e16e1c044dd1f91cb6be5ef02607ed42fe292b6`.
The compact checked-in record is
`docs/benchmarks/em/diagnostics/real-k4-native-coarse-operands-b061776-20260902.json`.

## Reproduction

Re-run the immutable validators and analysis with fresh output names:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_harness_integrate_20260901
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1

.pixi/envs/default/bin/python scripts/validate_relion_k4_coarse_operand_capture.py \
  --capture-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_operands_deferred_r1_b061776_20260902/capture \
  --expected-stacks 126,133,308,439,605,832,858,1001,1013,1152,1224,1443,1579,2517,2791,2838 \
  --expected-iteration 1 \
  --output /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_operands_deferred_r1_b061776_20260902/analysis/operand_capture_validation_replay.json

.pixi/envs/default/bin/python scripts/audit_relion_k4_coarse_operand_repeatability.py \
  --run-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_operands_deferred_r1_b061776_20260902 \
  --run-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_operands_deferred_r2_b061776_20260902 \
  --run-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_operands_deferred_r3_b061776_20260902 \
  --output-json /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_operands_deferred_r1_b061776_20260902/analysis/operand_capture_repeatability_replay.json

.pixi/envs/default/bin/python -m scripts.analyze_em_real_k4_native_coarse_operands \
  --capture-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_operands_deferred_r1_b061776_20260902/capture \
  --significance-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_recovar_components_r1_24317e40c_20260902/significance \
  --support-report /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_significance_probe_r5_24317e40c_20260901/analysis/coarse_score_support.json \
  --operand-validation /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_operands_deferred_r1_b061776_20260902/analysis/operand_capture_validation.json \
  --repeatability-gate /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_operands_deferred_r1_b061776_20260902/analysis/operand_capture_repeatability.json \
  --full-image-size 256 \
  --output-json /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_operands_deferred_r1_b061776_20260902/analysis/native_coarse_operand_boundary_replay.json
```

The three exact GPU launchers remain under each run root's `jobs` directory;
each refuses to mix with pre-existing outputs. Focused source checks are:

```bash
pixi run ruff check \
  scripts/analyze_em_real_k4_native_coarse_operands.py \
  scripts/audit_relion_k4_coarse_operand_repeatability.py \
  scripts/validate_relion_k4_coarse_operand_capture.py \
  tests/unit/test_analyze_em_real_k4_native_coarse_operands.py \
  tests/unit/test_audit_relion_k4_coarse_operand_repeatability.py \
  tests/unit/test_validate_relion_k4_coarse_operand_capture.py
pixi run pytest -q \
  tests/unit/test_analyze_em_real_k4_native_coarse_operands.py \
  tests/unit/test_audit_relion_k4_coarse_operand_repeatability.py \
  tests/unit/test_validate_relion_k4_coarse_operand_capture.py \
  tests/unit/test_validate_em_benchmark_registry.py
```

This remains a bounded causal diagnostic. Real-data K=4 admission still
requires matched, stable, independent-half trajectories and common-mask
per-class half-map FSC.
