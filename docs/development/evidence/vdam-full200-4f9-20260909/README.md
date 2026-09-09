# Frozen VDAM full201 map evidence — September 9, 2026

This historical benchmark is **one GF43 synthetic 3,000-particle, 128-pixel,
K1/C1, seed29 InitialModel cell** at RECOVAR
`4f9a194923b084c649c7d9ce929eec7ae9f78902`. Native and candidate each completed
200 iterations naturally on the same physical H100 in Slurm **13653485**, with
201 saved checkpoints including initialization. Both existing map conditions
pass at every checkpoint. Strict state parity still fails. This does not qualify
later cleanup commits, ordinary auto-refine, other data/seeds, exactly K4,
100k/256 completion, or all-stage float32 execution. No baseline is replaced.

## Map evidence

| Quantity | Value | Existing condition |
| --- | ---: | --- |
| Minimum cross-engine FSC-AUC | 0.9997253944709251 at193 | >=0.999, all201 pass |
| Minimum registered-GT AUC delta, candidate minus native | -0.0002512155449481135 at173 | >=-0.002, all201 pass |
| Final cross-engine FSC-AUC | 0.9997294862648998 | >=0.999 |
| Final registered-GT AUC, candidate / native | 0.38897697468181186 / 0.38907639621865686 | Delta -0.00009942153684500132 |

All1,005 shellwise curves and saved integrals survive in [shellwise.npz](shellwise.npz)
(456KB; numerical metrics only, no particle images or volumes). The labels are
`new_candidate`, `native`; all201 integer checkpoint IDs and63 shells are explicit.
There are201 cross-engine curves,402 raw-GT curves and402 registered-GT curves.
The fixed proper transform fitted on the earlier native1 final map was applied
unchanged to both arms: no fit per arm or checkpoint. Shell0/DC is excluded by the
pinned canonical AUC function. Raw-GT curves remain diagnostic; registration does
not change the independent cross-engine condition.

The archived read-only audit recomputes all1,005 AUCs exactly, checks finite values,
shapes, complete inventory, hashes, extrema and unchanged thresholds. VDAM's
separate parent audit additionally checked2,451 input pins and recomputed the worst
cross/GT curves from raw MRCs; its independent final-map calculation agrees exactly.
This admission rechecks the saved curves and evidence hashes, not another raw-map
or trajectory run.

## State, precision and performance limits

There are3,726 selected coarse-cutoff differences among112,400 updated rows;
first at32/image313 (6vs5). The first selected Pmax gap>=1e-3 is61/image2605,
and the maximum reaches0.699197 at154/image1332. Maximum saved pose/origin gaps
are157.589 degrees/10.198 Angstrom; class assignments match. Fine support and
competing-score margins are absent, so no tie/roundoff waiver follows.

Counter187 differs (candidate0/native5). The source-bound
[counter audit](late_counter_scope.md) establishes that it is a monitor in this
fixed200 gradient InitialModel configuration, bypassed for sampling/termination.
The state difference remains; this does **not** waive ordinary auto-refine, where
the counter can affect sampling and convergence. Four native fields remain
uncaptured: sampling_accuracy_estimated, sampling_has_fine_enough_angular_sampling,
current_resolution_shell and n_rotations. An explicit +infinity in a historical
STAR resolution diagnostic is retained in the run record; all required map arrays
and AUC values are finite.

Scoring/projector operands and the explicit M/state route use F32/C64. Existing
higher-precision construction, noise and prior stages remain. This is not proof
of all-stage F32 execution and does not adopt private precision907.

Whole-process candidate/native time is452.295/303.905s, ratio1.4882765197.
Candidate CLI time is439.875s; whole-child timing includes import/loader/setup and
postvalidation overhead, cold compilation and iteration I/O. Native post-run
controller validation is outside its process timer. One ordering/pair is not a
repeated speed qualification. Peak GPU memory is unmeasured. A development-host
2.62GB filesystem read overlapped the candidate; storage contention is unmeasured.
No isolated-stage, current100k or newer-tip speed claim follows.

## Reproduce and inspect

[run_record.json](run_record.json) preserves the exact native and RECOVAR CLI
arguments, manifest/input pins, binaries/libraries, physical GPU identity,
producer/analysis commands, timing boundaries and limitations.
[quality_summary.json](quality_summary.json) preserves the producer's summary.
[manifest.json](manifest.json) binds the archived files to original evidence and
the canonical AUC source. [audit.py.txt](audit.py.txt) contains that exact function
and the archive checks; the text suffix keeps this historical audit out of runtime
entry points. Run from the primary checkout with its pinned pixi environment:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u VIRTUAL_ENV \
  CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .pixi/envs/default/bin/python \
  docs/development/evidence/vdam-full200-4f9-20260909/audit.py.txt
```

This is read-only and needs no original scratch files or GPU. Hash/inventory
failures stop the audit. Original raw-map/state reproduction still needs every
pinned input, frozen source and binary under:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_integrated_full200_4f9a19492_20260909/`.
Follow its recorded analysis command with a fresh output directory; never restart
or overwrite the completed arms. Missing original assets block raw reproduction.
The archived curves alone cannot reconstruct particles, maps or hidden state.

The [benchmark contract](../../benchmarks.md) and [current EM status](../../em_status.md)
define the remaining milestone scope. This historical evidence is not an expected
result to be adjusted when a future implementation disagrees.
