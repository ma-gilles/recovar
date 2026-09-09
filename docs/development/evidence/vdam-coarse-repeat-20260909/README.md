# Frozen VDAM coarse-score repeat audit — September 9, 2026

This is diagnostic evidence from Slurm **13640613**, source
`fe8472947b8667f65498933fa4e9515f90ff33b7`, reviewed against shared source
`b3fbfef79ecfde69f53651f517c9cd6dcb5771f9`. It is not an accepted baseline,
a quality/performance qualification, or evidence about later trajectory flips.

## What was checked

A single compiled direct shared-pretranslated CUDA transaction was called 16
times on the same six operands. Every saved output has shape `(200,576,29)` and
dtype float32. Independent CPU recomputation verifies all 15 comparisons with
repeat 0 vary (maximum `0.0001220703125`, p95 `0.000030517578125`). All 200 raw
pre-prior winners remain unchanged; minimum represented-score winning margin
is `0.0013885498046875`.

The selected source uses float32 lane atomics; their actual execution order was
not captured. The frozen harness compiles once before its repeat loop and retains
distinct output buffers. This excludes recompilation between calls as necessary
for this variation. It does not identify the cause of every numerical difference.
Fixed-output negation/max controls are producer-reported, not independently rerun
here. Native source also uses atomic lane addition, but historical native
source-to-binary closure remains unavailable.

## Preserved records

- [review.json](review.json): independently recomputed metrics for every repeat,
  array file hashes, source/build identity and explicit limitations.
- [manifest.json](manifest.json): byte-identical producer manifest with 13 input,
  source and harness pins. The audit also checks 17 native inputs in both trees.
- [audit.py.txt](audit.py.txt): exact CPU audit source; SHA-256 `30fa0c32add89ea66629b65f185838eed5481851e69dc45e7633d2f2a6d4a4bc`.
  The text suffix keeps this historical script out of application entry points.

Original GPU artifacts (including the 16 score arrays and completion receipt):
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_coarse_atomic_repeat_20260909/`.
Independent review log:
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/coarse_repeat_review_20260909/audit.log`.
These compact records survive scratch cleanup; full recomputation still requires
all manifest-pinned inputs, frozen checkouts and the original arrays. Missing
artifacts invalidate reproduction and must not be silently omitted.

## Reproduce the CPU audit

Use the primary checkout's pinned pixi environment. Copy the archived script to
a fresh scratch directory so its report cannot overwrite this archived record:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_structural_cleanup_20260907
AUDIT_ROOT="$(mktemp -d /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/coarse-audit.XXXXXX)"
touch "$AUDIT_ROOT/SAFE_TO_DELETE"
cp docs/development/evidence/vdam-coarse-repeat-20260909/audit.py.txt "$AUDIT_ROOT/audit.py"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
.pixi/envs/default/bin/python "$AUDIT_ROOT/audit.py" > "$AUDIT_ROOT/audit.log"
```

This reads the original frozen evidence and verifies source/native hashes. It
launches no GPU work and makes no arithmetic or acceptance-policy changes.
