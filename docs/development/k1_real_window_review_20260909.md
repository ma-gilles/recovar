# 10073 global-window result review — September 9, 2026

The frozen full-data K1 run improves final masked halfmap FSC-AUC over both
modern RECOVAR repeats, but remains below both RELION repeats. The high-shell
deficit is smaller and still present. This measures the historical global-window
correction; it does not qualify the source currently on PR179 or close strict
trajectory parity. See the [current EM status](em_status.md) for active work.

## Source, run and scoring identities

Producer 13610518 completed 0:0 on H100 node `della-h20g1`, with whole-child wall
65,596.708s (about18.22h). Its frozen clean source is
`a0a86f19e9b3b8292140479c748abea426227cb3` in
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_global_preupdate_coarse_20260907`.
It changes global coarse Fourier sizing to use the incoming angular order;
its modern control is `6d1b21646`. The adapted global-window correction already
on PR179 is `0219caa35`; later cleanup/integration history is not covered by this run.

Inputs comprise all 138,899 EMPIAR 10073 particles at 380×380, halves 69,449/69,450,
C1 and seed 42. The recipe retains native I0 state, mt19937 physical shuffle,
firstiter_cc, coarse order 2/local entry 4, adaptive oversampling 1, image batch 250,
rotation block 8192, offset 5/2, initial lowpass 60 Å and tau2 fudge 1. No later native
trajectory replay, fitted alignment, restart or forced finalization is used.

The saved state and logs confirm **20 numbered iterations followed by final iteration 21**,
matching both modern RECOVAR repeats. The requested cap 999 is not the measured
endpoint. Final-all-data gridding correction remains **off**, preserving the
frozen implementation policy; this does not satisfy the strict on-policy target.
Per-image trajectory/tie agreement and native convergence equality were not
established by this endpoint review.

Scorer 13610539 and high-band analysis 13629099 completed 0:0 on CPU. The fixed
common mask, RELION phase randomization at FSC 0.8/seed 42, voxel 1.40001094 Å,
masked shells 1–128 and unmasked shells 1–80 remain unchanged. These are each
engine's final halfmap FSCs, not GT scores or direct cross-engine map FSCs.
High-band splits 1–80 and 81–128 were prespecified descriptive bands, not new gates.

## Final comparisons

Signed normalized AUC is window-corrected minus comparator. Positive is better
for that metric. Resolution is the recorded RELION-postprocess scalar, for which
smaller is better. The window-corrected masked AUC is 0.716196559055118 and its
masked resolution is 4.189009 Å.

| Comparator | Masked AUC delta, shells 1–128 | Unmasked AUC delta, shells 1–80 | Comparator resolution (Å) | Resolution direction |
| --- | ---: | ---: | ---: | --- |
| Modern RECOVAR 13494560 | +0.001741976378 | +0.000775199732 | 4.156282 | worse |
| Modern RECOVAR 13497843 | +0.002335401575 | +0.000849806620 | 4.189009 | same |
| RELION repeat 1 | −0.000768271654 | +0.000775312633 | 4.124063 | worse |
| RELION repeat 2 | −0.000604661417 | +0.000941966664 | 4.124063 | worse |

The corrected masked high-band AUC deltas versus the two native repeats are
−0.004986531915 and −0.004443223404 over shells 81–128, with 75% and 77.08% of those
shells lower. Lower-band deltas are positive. Both raw masked and phase-randomized
corrected curves retain the high-band deficit. Relative to the two modern
RECOVAR repeats, corrected high-band AUC improves +0.003713563830 and +0.005548765957.
This is a mixed result across quality summaries, not a parity pass.

The two native repeats differ by −0.000163610236 in full-band masked AUC. That
single observed shift is descriptive; it is not a variance estimate, confidence
interval or tolerance. One corrected run versus historical same-seed controls
cannot establish independent-seed robustness or causal attribution of every delta.
No matched current-source runtime ratio or peak-memory qualification follows
from the single 18.22-hour producer time.

## Independent audit and reproduction

The integrator independently recomputed **388 numeric fields**, with zero
numeric discrepancy, from the saved masked/unmasked curves and all raw/corrected
high-band comparisons. Five STAR corrected curves match their NPZ arrays exactly.
The sealed natural-endpoint predicate was re-executed on saved metadata/history
and the log. The complete half partition and first 100 accuracy-trial row IDs
were also checked. **92 named artifact hashes** match before/after, including
all three final MRC file hashes; no FFT, E/M or native executable was rerun.

The first audit attempt assumed fully qualified repeat-pair names and stopped
on the producer's abbreviated `repeat1` label. Its script/log are preserved.
The corrected schema mapping handles only the two declared within-engine labels;
it does not change metrics, scientific thresholds or producer artifacts.

Run the review from the primary checkout with its pixi Python:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u VIRTUAL_ENV \
  CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .pixi/envs/default/bin/python \
  /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/k1_real_window_review_20260909/review.py
```

Audit script, output, exact pins and failed attempt:
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/k1_real_window_review_20260909/`.
Audit completed in 10.40s CPU. Producer/scorer artifacts:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_10073_global_window_full_20260908/`.
Read `launch.json`, `endpoint.json`, `runtime.json`, `analysis/summary.json`,
`modern_comparison.json` and `refine.log` there. High-band report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_10073_window_highband_20260908/comparison-13629099.json`.
The preparation README/manifest still describe an unsubmitted packet; terminal
accounting and immutable producer/scorer receipts establish subsequent execution.
Those historical preparation records are preserved, not overwritten.

Historical RELION dynamic-library closure and complete content hashes for the
2,348 backing particle stacks remain missing. The prior six shared GPU-fast
failures are unwaived. This audit verifies named evidence, not every dependency
or raw particle byte. It closes the previously missing final-FSC measurement for
this frozen real-data experiment; current-primary K1, exactly K4, broader datasets,
state/pose parity and representative performance remain open.
