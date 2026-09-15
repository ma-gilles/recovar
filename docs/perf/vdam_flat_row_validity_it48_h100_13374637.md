# VDAM validity-aware packed fine rows at iteration 48 — H100 jobs 13374637/13374638

## Decision

The validity-aware CUDA path passes its primitive and same-state science
gates.  It preserves the physical packed shape but marks non-logical rows with
image ID `-1`; the existing exact fine scorer writes binary32 `+inf` for those
rows before loading pixels.  All valid-row arithmetic and the 256-lane
reduction tree remain unchanged.

Do not yet attribute an end-to-end speedup to this two-commit change.  The
same-state production panel compares direct against the complete certified
coarse + packed/deferred bundle, not the bundle with validity skipping toggled
off and on.  A focused alternating kernel benchmark is the remaining timing
gate.

## Qualification

| Field | Value |
|---|---|
| Source | `4dc4a79f7c5f6f31d7a8e04bfe43339b700b48c6` |
| CPU contract | `tests/unit/test_flat_local_rows.py`: 10 passed |
| CUDA primitive | Job `13374638`: 1 passed, 71 deselected |
| Same-state production | Job `13374637` (`COMPLETED`, exit `0:0`, `00:08:25`) |
| Hardware | `della-h19g1`, H100, `GPU-75c2d200-95d1-ef57-fb52-1698386c756c` |
| Boundary | Exact in-memory GF46 state, iteration `47 -> 48` |
| Panel | direct / combined+validity / combined+validity / direct |

The primitive compares valid rows bitwise with a compact call containing only
those rows and requires every invalid output bit pattern to be positive
infinity.  The production harness deep-copies identical model, particle, and
sampling state into all four arms and records complete support identities.

## Science result

- Every tracked pose, rotation, translation, class, posterior, significant
  count, particle-state, sampling-state, and support-audit field is exact in
  all six pairwise comparisons.
- Direct/direct accumulator normalized L2 reaches `9.93e-8`;
  candidate/candidate reaches `9.13e-8`; the largest cross-mode value is
  `9.69e-8`.
- Direct/direct final-state normalized L2 reaches `1.33e-8`; the largest
  cross-mode value is `1.98e-8`.

Thus the early exit introduces no detected scientific displacement at the
first hard transition of the full GF46 sentinel.  Remaining continuous
differences are inside the observed repeated-execution envelope.

## Runtime result

The warmed second arms show the complete combined backend remaining material,
but these numbers must not be credited solely to validity skipping:

| Metric | Warm direct | Warm combined+validity | Change | Speedup |
|---|---:|---:|---:|---:|
| Whole transition | 5.907805 s | 2.474943 s | -58.11% | 2.387x |
| Expectation | 5.679418 s | 2.237250 s | -60.61% | 2.539x |
| Coarse pass 1 | 3.797733 s | 0.680404 s | -82.08% | 5.582x |
| Fine pass 2 | 1.340418 s | 1.020668 s | -23.85% | 1.313x |
| Local big JIT | 1.162960 s | 0.735781 s | -36.73% | 1.581x |

The packed scorer has 19,008 physical rows, of which 5,584 are logical; the
new early exit avoids pixel traversal for 13,424 rows (`70.62%`) without
changing the static shape.  The original dense physical total is 55,296 rows.
Peak GPU memory over the whole job was 17,595 MiB.

Against job `13372936` on a different H100/run, candidate pass 2 changes
`1.068328 -> 1.020668 s` while direct pass 2 changes only
`1.347488 -> 1.340418 s`.  That cross-job signal is encouraging but is not an
admissible isolated timing result.  Other stages and both whole-arm times also
shift substantially between the jobs.

## Provenance

- Same-state report SHA-256:
  `da4f9e4e3146a862e714e9531f44fbbe2c48e6a466dcd0a1c9547cc11dab4e8d`
- Qualified same-state CUDA SHA-256:
  `d8253dc6dcd8420bca3e1935ffe4318df301fc83a96da932cb46dc91f31aa851`
- Same-state disposable root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_flat_row_validity_it47_4dc4a79f7_20260903T0500Z`
- Primitive disposable root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_flat_row_validity_gpu_test_4dc4a79f7_20260903T0500Z`
