# EMPIAR-10076 K=4 native coarse-component boundary

This diagnostic moves the frozen shared-200 K=4 boundary one step inside the
coarse raw likelihood. It compares RELION and RECOVAR reference-norm and
image-reference cross terms for the same 12 mismatch probes and four exact
controls, over all four classes, 576 rotations, and 29 translations.

## Outcome

The cross term is the dominant component of the first material likelihood
difference.

| Stable top-RELION-count surface | Exact particles | Support Jaccard |
| --- | ---: | ---: |
| Native norm + native cross | 16/16 | 1.000000 |
| RECOVAR norm + RECOVAR cross | 5/16 | 0.951790 |
| RECOVAR norm + native cross | 11/16 | 0.989576 |
| Native norm + RECOVAR cross | 5/16 | 0.954645 |

The component swaps are unchanged when the native and RECOVAR priors are
exchanged, and no tested support boundary contains a tie. Replacing only the
cross term leaves 29.6% of the pooled centered raw-residual energy, whereas
replacing only the reference norm leaves 53.6%. This is causal localization,
not yet an implementation fix: the remaining cross-term boundary may be the
shifted image, CTF/weight operand, projected reference, or their reduction.

## Accepted capture qualification

The accepted RELION observer is an off-production-path replay. Production
`diff2` is computed exactly as in the control arm; only the 16 selected stacks
are evaluated a second time into disposable score and component buffers.
Slurm job `13332998` requested and received exactly one H100, eight CPUs, and
192 GB without exclusivity. It completed in 68 seconds. All iteration-0
numeric state and all iteration-1 pose, class, origin, and significant-support
decisions are exact. Iteration-1 class-map FSC-AUC is
`0.999999999073`--`0.999999999634`, and relative L2 is
`8.32e-7`--`2.85e-6`; the fixed FSC-AUC `0.999999` and relative-L2 `1e-5`
inertness gate passes.

The diagnostic RELION source is
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_coarse_components_20260902`
at commit `8f9ebc9221b9f8f3eb55c82f7d16377025ff3f46`. The executable is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k4_coarse_components_build_20260902/build/bin/relion_refine`
with SHA-256
`2f2eaaa4496c3c6dbca1aca57b4a1936603018531a5cca4983d50ce618e3800c`.
The exact Slurm launcher is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_components_replay_8f9ebc9_20260902/jobs/run_native_component_replay.sbatch`
with SHA-256
`9a907aafec3612ffebd006e471488fa106cec1d3a6aeea8e9f572b9e16b619f7`.

The paired component validator accepted 16 particles and 1,069,056 candidate
records. Its report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_components_replay_8f9ebc9_20260902/analysis/component_capture_validation.json`
with SHA-256
`bbcb54aa73cb4932cc663b8edfd9c7dff9a5c10b80b3493f47f60b78dde48df5`.
The inertness report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_components_replay_8f9ebc9_20260902/analysis/capture_inertness.json`
with SHA-256
`c5b6b415cc6bd1d8bb17975c5a02226c638d838a30d31f05341b08fabd48e32f`.

## RECOVAR component replay

RECOVAR job `13332392` requested and received exactly one H100, eight CPUs,
and 192 GB without exclusivity. It completed in 287 seconds with peak batch
RSS 8,068,208 KiB. It ran sealed source commit
`24317e40cc4b0b19406f75c917857c03869a9372` and retained norm and cross
surfaces with shape `(4, 576, 29)` for the same 16 particles. Its final parity
array is byte-identical to the earlier passive support run, with SHA-256
`5f469bf8eac9003608e7e90367139875f58e3742b9115a3763f66ae641d47b66`.
The launcher is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_recovar_components_r1_24317e40c_20260902/jobs/run_recovar_components.sbatch`
with SHA-256
`18bed076e3f011af60c5131cc4c46e3cacc6946a6b9459bf3aaeb694489ea94b`.

## Component metrics

One additive image-only constant is removed per particle. No scale, affine
fit, or candidate filtering is used.

| Component | Pooled centered RMS | Median centered relative L2 |
| --- | ---: | ---: |
| Reference norm | 0.225385 | 0.016900 |
| Cross term | 0.303162 | 0.015865 |
| Raw likelihood | 0.414104 | 0.017978 |

The centered residual identity closes with maximum absolute error
`1.030e-4` and median RMS `1.412e-5`, below the predeclared `1e-3` closure
ceiling. RELION production `diff2` remains binary32; the observer accumulates
the two diagnostic components in binary64, so exact component closure is not
claimed.

The authoritative report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_components_replay_8f9ebc9_20260902/analysis/native_coarse_component_boundary.json`
with SHA-256
`5b6b6bc1098b1703f68ec4261fc459be0ae1ddf20761f6bc54c69a7504ea4f83`.
The compact checked-in record is
`docs/benchmarks/em/diagnostics/real-k4-native-coarse-components-8f9ebc9-20260902.json`.

## Rejected observer

The first component observer at RELION commit `4847df3` accumulated binary64
diagnostics inside the production `diff2` kernel. Although the component files
were structurally valid, it changed one to three particle decisions and drove
control-to-capture class-map FSC-AUC down to `0.8974`--`0.9343`. Job
`13332319` and its outputs are rejected scientific evidence. The retained
inertness report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_components_4847df3a_20260902/analysis/capture_inertness.json`
with SHA-256
`eed7bf948508de81301824b353fccdb3c3e15e82b5c0cc4709c649db463cf9b6`.
This rejection is why the accepted observer replays selected operands only
after the production kernel has completed.

## Reproduction

Re-run the immutable validators and component analysis with fresh output
names:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_harness_integrate_20260901
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1

.pixi/envs/default/bin/python scripts/validate_relion_coarse_component_capture.py \
  --capture-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_components_replay_8f9ebc9_20260902/capture \
  --expected-stacks 126,133,308,439,605,832,858,1001,1013,1152,1224,1443,1579,2517,2791,2838 \
  --expected-iteration 1 \
  --output /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_components_replay_8f9ebc9_20260902/analysis/component_capture_validation_replay.json

.pixi/envs/default/bin/python -m scripts.analyze_em_real_k4_native_coarse_components \
  --capture-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_components_replay_8f9ebc9_20260902/capture \
  --significance-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_recovar_components_r1_24317e40c_20260902/significance \
  --support-report /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_significance_probe_r5_24317e40c_20260901/analysis/coarse_score_support.json \
  --component-validation /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_components_replay_8f9ebc9_20260902/analysis/component_capture_validation.json \
  --capture-inertness /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_components_replay_8f9ebc9_20260902/analysis/capture_inertness.json \
  --output-json /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_shared200_native_coarse_components_replay_8f9ebc9_20260902/analysis/native_coarse_component_boundary_replay.json
```

The full GPU producer launchers are retained and hashed above. Clone them to a
fresh run root before resubmission; they deliberately refuse to mix with an
existing evidence root.

Focused source checks are:

```bash
pixi run ruff check \
  scripts/analyze_em_real_k4_native_coarse_components.py \
  scripts/validate_relion_coarse_component_capture.py \
  tests/unit/test_analyze_em_real_k4_native_coarse_components.py \
  tests/unit/test_validate_relion_coarse_component_capture.py
pixi run pytest -q \
  tests/unit/test_analyze_em_real_k4_native_coarse_components.py \
  tests/unit/test_validate_relion_coarse_component_capture.py \
  tests/unit/test_validate_em_benchmark_registry.py
```

This remains a bounded causal diagnostic. A real-data K=4 admission still
requires matched, stable, independent-half trajectories and common-mask
per-class half-map FSC.
