# PR180 unit-checkpoint follow-ups, 2026-09-08

The original CPU checkpoint remains **failed**: Slurm `13623235` ran 6,817
cases on clean `42a3d6184c6d05a9f4f97bd00120e62d0081f1d3`, with 6,452 passed,
340 skipped and 25 failed. Every failed case now has an explicit passing
focused follow-up. These checks span revisions and execution environments;
they do not establish a fresh full-suite pass or current-source trajectory,
quality or performance qualification.

The [CPU summary](https://github.com/ma-gilles/recovar/blob/5c07fc3169d636dc3775b2bb41ae4c76d6829611/docs/development/evidence/pr180-unit-checkpoint-20260908/cpu_summary.json) preserves the original results. The
[failure reconciliation](https://github.com/ma-gilles/recovar/blob/5c07fc3169d636dc3775b2bb41ae4c76d6829611/docs/development/evidence/pr180-unit-checkpoint-20260908/failure_reconciliation.json) records each of the 25
failed cases, its follow-up, source revision, dirty fingerprint and artifact
hashes. Twenty-four retain their original case IDs. One diagnostic norm test
is explicitly replaced by two parametrized cases; it is not silently dropped.

## Source and test-contract changes

- `d50badd3d5f6a11adf7723b4af85c204c8a262cd` migrates an availability check to
  the module that owns the replay dependency. Its 122-case focused selection
  passes; the separate 38-case CPU fast guard also passes.
- `506c8e64926a71732aef31d05e1218901ff4db53` removes the sampling helper's
  reverse dependency on the iteration controller. Eight test grid stubs now
  accept the explicit dtype supplied by PR180; the precision test patches the
  dependency at its owner. A redundant monkeypatch that overwrote the intended
  replay fixture is removed. The ten affected cases pass, as does the
  overlapping 19-case sampling selection. Scientific assertions are unchanged.
- `d478137b60314e810ac5716f682d5170d2cc96cd` tests PR180's explicit diagnostic
  norm-reduction contract: float64 output retains low bits that a final
  float32 cast would lose. Both complex64 and complex128 inputs are covered.
  Separate default cases verify complex64-to-float32 and complex128-to-float64
  behavior. Nine norm/capture cases and two existing precision-default guards
  pass. Runtime code and tolerances are unchanged.

The [sampling comparison](https://github.com/ma-gilles/recovar/blob/5c07fc3169d636dc3775b2bb41ae4c76d6829611/docs/development/evidence/pr180-unit-checkpoint-20260908/sampling_comparison.json) records eight exact
original/new grid comparisons: HEALPix orders 0–3 in both precisions. It also
checks that the direct helper call leaves the controller unloaded and that
retained metadata computations and existing scientific assertions are
unchanged. This is structural equivalence evidence, not a speed measurement.

Failed intermediate attempts remain in the output root: `sampling_contracts_01`
exposed the overwritten replay fixture; `norm_contracts_01` exposed an inexact
new complex-valued test fixture, replaced with exactly representable real-valued
inputs; `norm_contracts_02` collected no cases because a requested node ID was
incorrect. No tolerance was widened.

## Identical-source GPU and tool follow-ups

Both jobs use the original clean CPU source, its exclusive frozen pixi
environment, and the previously qualified read-only CUDA library. Both ran on
`della-h20g2`, H100 80 GB, physical UUID
`GPU-35bc7e90-cea1-2c58-9092-aa2a3e6bcbc0`, driver `610.57.04`.
Scheduler-assigned visibility was preserved.

| Selection | Slurm job | Result | Job elapsed |
| --- | --- | --- | --- |
| Eight tiny GPU/native-CUDA cases | 13625819 | 8 passed, no skips | 15 s |
| Five dry-run tool checks and one float32 comparison | 13625952 | 6 passed, no skips | 6 s |

The [GPU records](https://github.com/ma-gilles/recovar/blob/5c07fc3169d636dc3775b2bb41ae4c76d6829611/docs/development/evidence/pr180-unit-checkpoint-20260908/gpu_followups.json) include exact test selections, commands,
environment overrides, submission records, import paths, source and library
hashes, and before/after identity checks. The second job supplies the existing
`RELION_REFINE_MPI` executable explicitly. Dry-run checks resolve and hash it;
they do not execute RELION or submit benchmarks. The shared build is unchanged.
Copied preparation metadata retains unused K1 header notes; `fixture_files` is
empty and these tests use in-memory fixtures, not curated K1 particles.

The CPU-versus-NumPy float32 assertion still fails on CPU at its original
tolerance. Its unchanged GPU test passes. This observation does not explain
the separate six-particle K1 Pmax residual, establish numerical-noise
attribution, or authorize a production precision change. Production EM remains
float32; double execution is diagnostic only.

## Reproduction and artifacts

Preparation root:
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/pr180_integration_20260908/`.

Output root:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr180_integration_20260908/`,
with `SAFE_TO_DELETE` present. Bulky outputs are disposable; these compact
versioned records preserve outcomes and provenance.

For the original exact local test commands, read each named output directory's
`outcome.json`. CPU XML is `cpu_checkpoint/unit-tests.xml`; focused XML files
are `<selection>/tests.xml`. The reconciliation records their hashes and paths.
Do not overwrite completed outputs or reuse an old label for a new run.

For GPU reproduction, start from the recorded frozen source and inspect
`inherited_gpu_cases/{inputs.json,run_tests.py,run.py,run.sbatch}` and
`inherited_tool_and_float32_cases/{inputs.json,run_tests.py,run.py,run.sbatch}`
under the preparation root. Clone the preparation with a new output/runtime
root, regenerate its provenance manifest, then submit its `run.sbatch` through
Slurm. Do not invoke the GPU runner on a login node or edit a sealed manifest.

The comparison and reconciliation scripts, original source copies, and a
human-readable checkpoint are under
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/pr180_sampling_contracts_20260908/`.

Current-source K1 and exact-K4 completion, real-data confirmation, and paired
performance evidence remain outstanding. See the [active status](../../em_status.md)
and the separate [K1 noise-state evidence](../pr180-k1-noise-state-20260908/README.md).
