# Six-particle K1 replay and score-dump check, 2026-09-08

The six-particle replay exactly reproduces the production Pmax values, saved
Euler angles and translations from the full 5,000-particle replay. The existing
score dump changes Pmax by at most `3.0517578125e-5`, with saved poses unchanged.
It passes the diagnostic comparison bound of `1e-4`, but is not an identical
execution: enabling the dump disables dense big-JIT. This distinction must
remain attached to any conclusions drawn from its scores.

Slurm **13627156** completed both stages in one H100 80 GB allocation on
`della-h20g2`, physical UUID `GPU-35bc7e90-cea1-2c58-9092-aa2a3e6bcbc0`, driver
`610.57.04`. Job elapsed time was 92 seconds, not a performance benchmark.
Both stages use clean frozen PR180 source
`42a3d6184c6d05a9f4f97bd00120e62d0081f1d3`, the same frozen pixi environment,
inputs, state and native libraries as the preceding noise-state diagnostic.
Source, inputs and libraries remain unchanged. Production execution is float32
in both stages; double scoring, projection and M-step flags are all disabled.

## Particle comparison

Rows are zero-based original stack indices; each selected half retains its
original particle order. The subset and capture use the existing
`--keep_stack_indices` option with rows `901,1257,1300,1414,3694,4568` and the
same iteration-3-to-4 state with `--continuous-relion-noise-state`.

| Stack row | Full run and subset Pmax | Dump-path Pmax | Saved RELION Pmax |
| --- | --- | --- | --- |
| 901 | 0.846456766129 | 0.846440911293 | 0.844761 |
| 1257 | 0.619761049747 | 0.619757413864 | 0.618619 |
| 1300 | 0.503266811371 | 0.503236293793 | 0.502258 |
| 1414 | 0.563377559185 | 0.563375353813 | 0.561925 |
| 3694 | 0.514434397221 | 0.514445841312 | 0.512997 |
| 4568 | 0.784231245518 | 0.784208059311 | 0.785391 |

The [subset audit](subset_audit.json) and [capture audit](capture_audit.json)
retain full values, deltas, half membership, pose comparisons and consumed
artifact hashes. The audit also rejects missing particles. The production
residuals remain unresolved. In particular, the dump moves row 1300 across the
`1e-3` discrepancy threshold; that does not close its original production gap.

## Captured score surface

Row 901, the largest original Pmax gap, has 38 pre-prior and 38 post-prior
blocks. Their 992-by-29 layout covers 36,864 rotations and 29 translations.
The post-prior padding is verified as negative infinity. Of 1,069,056 grid
candidates, 1,057,920 have finite combined scores; every finite stored value
round-trips exactly through float32. The legacy dump stores these values in
float64 arrays, which does not change their original arithmetic precision.

The [score summary](score_summary.json) records the leading eight candidates
and hashes every consumed score block. The winner/runner-up score margin is
`1.7069549560546875`. Normalizing the captured float32 scores in float64 gives
Pmax `0.8464409107915538`, just `-5.01476e-10` from the dump-path Pmax. Its gap to
the saved RELION value remains `0.0016799107915538292`.

Thus, higher-precision final normalization does not close the discrepancy on
this captured score surface. This does not identify the cause of the original
production gap. The next comparison needs matched score operands, priors and
candidate geometry. RELION's live candidate scores are still absent, and the
historical oracle's generating commit/build is unknown. Candidate indices in
this report refer to the captured RECOVAR grid; correspondence to RELION's
internal candidate indices is not established.

## Reproduction and evidence

The [run record](run_record.json) contains both exact argument lists, precision
and capture settings, source/native-library identities, import paths, commands,
submission and hardware records. Preflight checked 1,794 source files and 155
input/result files plus instrumentation and libraries. Audit self-checks
accepted exact values and rejected a perturbed Pmax and a missing particle.

Preparation root:
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/pr180_integration_20260908/k1_targeted_capture/`.

Output root, with `SAFE_TO_DELETE`:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr180_integration_20260908/k1_targeted_capture/`.

The submitted command was `sbatch --parsable <preparation-root>/run.sbatch`.
The paired runner executes `subset/run.py`, then `capture/run.py` only after
the subset audit passes. Exact runtime commands are in each stage's
`outcome.json`; source/fixture manifests and wrappers are in its preparation
directory. `replay.log`, `diagnostic_audit.json`, `effective_precision.json` and
loaded-library records are under each stage's output directory. Raw score
blocks are in `capture/scores/`; `score_summary.json` is at the output root.

For reproduction, create new output/runtime roots and regenerate manifests
for the intended source. Preserve Slurm-assigned visibility. Do not overwrite
completed outputs or edit their sealed preparation. The standalone
`analyze_scores.py` uses the frozen pixi Python on CPU and performs only the
labeled diagnostic float64 normalization; it does not rerun EM.

This is a diagnostic transition, not autonomous convergence, map-quality,
real-data, K4 completion or performance qualification. Six-particle output maps
are not comparable quality estimates to the 5,000-particle oracle maps. The
current later cleanup source is also unqualified by this frozen-source run.
See the [preceding noise-state comparison](../pr180-k1-noise-state-20260908/README.md)
and [active status](../../em_status.md) for the remaining gates.
