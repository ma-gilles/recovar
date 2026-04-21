# OS1 1000-Image Checkpoint After Child-Grid Fix

Date: 2026-04-21

## What changed

- Fixed adaptive pass-2 child orientation generation in `recovar/em/sampling.py`.
  - Root cause: `get_oversampled_rotation_grid_from_samples()` was treating RELION coarse orientation indices as if their HEALPix direction component were in recovar's older RING-style convention.
  - The RELION-parity coarse grid is direction-fast over RELION/NEST pixels, so the old child generator produced child directions that were wrong by tens to more than one hundred degrees.
  - The function now generates child directions in RELION/NEST order and also applies the per-iteration RELION perturbation to the oversampled children.
- Fixed GT-pose reporting in `scripts/run_multi_iter_parity.py`.
  - The script was double-subtracting the stack index when mapping `poses.pkl` back onto `particles.star` order.
  - This did **not** affect refinement itself, but it made the saved `pose_comparison_iter*.npz` GT-angle arrays wrong.

## Regression checks

- Direct parity tests for `get_oversampled_rotation_grid_from_samples()` now match RELION's `get_oversampled_orientations()` for both:
  - unperturbed children
  - perturbed children
- Targeted test added:
  - `tests/unit/test_run_multi_iter_parity.py`

## 1000-image adaptive benchmark

Dataset:
- `/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_parity_1000_noise0p01_os1`
- 1000 particles
- noise level `0.01`
- RELION replay dir:
  - `/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_parity_1000_noise0p01_os1/relion_ref_os1`

Output:
- `_agent_scratch/multi_iter_1000_noise0p01_os1_v3_childfix_2iter`

### Map quality vs GT

Iter 1 (RELION iter 4):
- RECOVAR regularized merged corr vs GT: `0.743562`
- RELION merged corr vs GT: `0.771827`
- RECOVAR regularized merged FSC<0.5 shell: `35`
- RELION merged FSC<0.5 shell: `28`
- RECOVAR regularized merged FSC<0.143 shell: `43`
- RELION merged FSC<0.143 shell: `40`

Iter 2 (RELION iter 5):
- RECOVAR regularized merged corr vs GT: `0.757285`
- RELION merged corr vs GT: `0.771748`
- RECOVAR regularized merged FSC<0.5 shell: `40`
- RELION merged FSC<0.5 shell: `31`
- RECOVAR regularized merged FSC<0.143 shell: `41`
- RELION merged FSC<0.143 shell: `42`

Observation:
- This is no longer the catastrophic `~0.53`-corr failure seen before the child-grid fix.
- RECOVAR still trails RELION on merged-map real-space correlation for this benchmark, even though the FSC thresholds are competitive or better depending on the threshold.

### Pose quality vs GT

These numbers use the **corrected** GT mapping.

Iter 1:
- RECOVAR vs GT full-angle mean: `3.903°`
- RECOVAR vs GT view-direction mean: `3.193°`
- RECOVAR vs GT in-plane mean: `1.800°`
- RECOVAR vs GT translation mean: `0.430 px`
- RELION vs GT full-angle mean: `16.951°`
- RELION vs GT view-direction mean: `11.558°`
- RELION vs GT in-plane mean: `10.416°`
- RELION vs GT translation mean: `0.672 px`

Iter 2:
- RECOVAR vs GT full-angle mean: `3.386°`
- RECOVAR vs GT view-direction mean: `2.751°`
- RECOVAR vs GT in-plane mean: `1.584°`
- RECOVAR vs GT translation mean: `0.454 px`
- RELION vs GT full-angle mean: `15.789°`
- RELION vs GT view-direction mean: `10.242°`
- RELION vs GT in-plane mean: `10.179°`
- RELION vs GT translation mean: `0.638 px`

### RECOVAR vs RELION pose gap

Iter 1:
- full-angle mean: `14.928°`
- `74.4%` of particles are within `10°`
- `66` particles are more than `90°` away from RELION

Important interpretation:
- Those large RECOVAR-vs-RELION outliers are mostly cases where RELION is wrong on this synthetic set.
- For the `66` particles with RECOVAR-vs-RELION angle error `> 90°` at iter 1:
  - RECOVAR mean full-angle error vs GT: `2.256°`
  - RELION mean full-angle error vs GT: `141.853°`

So direct RECOVAR-vs-RELION angular disagreement is not, by itself, evidence that RECOVAR is in the wrong basin on this dataset.

## Tau2 status on this benchmark

Checked from:
- `_agent_scratch/multi_iter_1000_noise0p01_os1_v3_childfix_2iter/intermediates/it000_tau2.npy`
- RELION `run_it004_half1_model.star`

Shell-averaged tau2 parity is already tight:
- median `recovar / RELION` ratio over the first 20 shells: `0.986`
- shells 0-11 are all within about `1-4%`

Interpretation:
- On this benchmark, the remaining adaptive gap is **not** a gross tau2 scaling bug.
- The child-grid fix moved the run out of the catastrophic failure mode; the residual map-quality gap is elsewhere.

## Larger-run context

The corrected pose metric bug also affects earlier saved `pose_comparison_iter*.npz` outputs from the 5k benchmark.

After rewriting those with the corrected mapping, the existing 5k hp4 checkpoint
`_agent_scratch/multi_iter_full_hp4_v3_gt` remains essentially tied with RELION at the end:

Iter 4:
- RECOVAR vs GT full-angle mean: `2.208°`
- RELION vs GT full-angle mean: `2.225°`
- RECOVAR vs RELION full-angle mean: `0.659°`
- RECOVAR merged corr vs GT: `0.962886`
- RELION merged corr vs GT: `0.961718`

That confirms the GT reporting bug was in metrics only, not in the already-good 5k hp4 geometry.
