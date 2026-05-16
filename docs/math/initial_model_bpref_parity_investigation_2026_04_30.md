# InitialModel BPref parity investigation, 2026-04-30

This note records the current diagnosis for the RELION InitialModel iter-1
E-step to BPref parity gap.  The short version is that the observed ~0.73 BPref
correlation is not explained by the outer edge of the reconstruction radius.
The dominant remaining difference is that the InitialModel dense path evaluates
the full oversampled Cartesian pose grid, while RELION first prunes to a
particle-specific significant support and only then evaluates oversampled child
poses for the M-step.

## Reproducer inputs

- RECOVAR checkout: `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar`
- RELION run fixture: `/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_estep_run_small`
- RELION E-step dump: `/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_estep_dump_small`
- Particles STAR:
  `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/.tmp/slurm_7178672/pytest-of-mg6942/pytest-0/test_pipeline_spa_gpu0/gpu_spa/test_dataset/particles.star`

For this fixture, do not rely on `run_it000_optimiser.star --i`: it stores a
relative `particles.star`, while the coherent dataset is the explicit STAR path
above.

## Findings

1. The half-volume M-step / outer-`rmax` explanation is only a small effect.
   With coherent inputs, switching the M-step accumulator from the full-volume
   layout to the RELION half-volume layout changed BPref data CC only from about
   0.737/0.746 to about 0.746/0.755 for halfsets 0/1.  Weight CC stayed around
   0.484/0.491.  This is not large enough to explain the parity gap.

2. The RELION projector setup is not the culprit.  Reimplementing RELION
   `Projector::computeFourierTransformMap` from `iref_c0_pre_setup.bin` matches
   `ppref_c0_data_post_setup.bin` at CC 0.999997 after gridding correction.
   The separate `scripts/diag_p0_diff2_relion_proj.py` path also shows that
   RELION's projected reference, image, CTF, and `Minvsigma2` reproduce RELION
   particle-0 score deltas at CC 0.999960, slope 0.999506 after applying the
   `ori_size` scale needed by the relion-bind projector diagnostic.

3. The production dense score ingredients have a scale/sign mismatch under the
   previous InitialModel diagnostic defaults.  For particle 0:
   - RECOVAR's processed FFT image is `N^2` larger than RELION `Fimg`.
   - RECOVAR CTF is the negative of RELION's local CTF.
   - RELION `Minvsigma2` is `1/sigma2` on nonzero shells, with DC zeroed.
   - Therefore the dense path needs the reference scale/sign/noise convention
     equivalent to `--iref-scale-mode one --iref-sign 1 --noise-scale-mode n4`
     for the raw RECOVAR image convention.

4. With that corrected score convention, dense RECOVAR matches RELION within
   RELION's retained significant support for particle 0:
   - Unmasked full dense grid: score deltas on RELION nonzero cells match at
     CC 0.999852, slope 1.000888, but the global best poses include many cells
     for which RELION has zero posterior because they were pruned.
   - Applying the RELION particle-0 support mask to dense `run_em` gives score
     delta CC 0.999903, slope 0.999985, posterior-probability CC 0.998260,
     and the same argmax flat pose index, 37060.

5. RELION's support is particle-specific.  In the dump:
   - `p0_exp_Mweight.bin` has 736 nonzero oversampled child poses from 23
     coarse cells.
   - `p1_exp_Mweight.bin` has 2112 nonzero child poses from 66 coarse cells.
   - `p2_exp_Mweight.bin` has 416 nonzero child poses from 13 coarse cells.
   These masks differ by particle even though the oversampled Euler and
   translation grids are shared.

## Why normal EM can reach ~0.999 while InitialModel is ~0.73

The normal RELION-parity EM path already has code for per-image local/search
support: significance pruning, exact local layouts, and bucketed sparse pass-2
candidate masks.  Those paths evaluate only the image-specific candidate support
that survives pass 1.

The InitialModel GPU path currently drives dense `run_em` directly over the full
oversampled grid.  That can match score algebra inside RELION's support, but it
does not reproduce RELION's adaptive pruning.  Extra full-grid poses with good
dense scores receive posterior mass and backproject into BPref, which explains
why BPref remains far below the normal EM parity even after fixing projector and
half-volume details.

## Diagnostic outputs

- Coherent full-vs-half BPref layout:
  `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/_agent_scratch/initial_model_bpref_diag_coherent_mrc_20260430_124357/summary.json`
- RELION-projector particle-0 diff2 parity:
  `/scratch/gpfs/GILLES/mg6942/_agent_scratch/p0_diff2_diag_relion_proj_scaled_20260430/summary.json`
- Dense score dump with corrected score convention, no support mask:
  `/scratch/gpfs/GILLES/mg6942/_agent_scratch/dense_score_dump_p0_scaled_20260430`
- Dense score dump with corrected score convention and RELION p0 support mask:
  `/scratch/gpfs/GILLES/mg6942/_agent_scratch/dense_score_dump_p0_scaled_masked_20260430/summary.json`
- BPref diagnostic with corrected score convention but no per-image support:
  `/scratch/gpfs/GILLES/mg6942/_agent_scratch/initial_model_bpref_diag_score_scaled_20260430/summary.json`

## Next implementation step

InitialModel should not call the dense full-grid `run_em` for iter-1 parity.
It should either:

1. Reuse `helpers.significance._compute_significance_batched` to produce
   per-image significant coarse samples, then call
   `helpers.oversampling.compute_pass2_stats_sparse` /
   `helpers.sparse_pass2_bucketed.compute_pass2_stats_sparse_bucketed` for the
   oversampled pass and M-step; or
2. Extend the dense path to accept per-image `(B, R, T)` candidate masks and
   generate those masks from the same coarse significance pass.

The first option is closer to the existing normal EM implementation and should
be the lower-risk route to near-perfect InitialModel BPref parity.

## 2026-04-30 implementation attempt

`run_iter_gpu_vdam` now has an opt-in `estep_mode="relion_adaptive"` path.
The old dense grid remains available as `estep_mode="dense"` and is still the
default because the adaptive path is not parity-correct yet.

Changes made:

- Added explicit `sorted_particle_ids` halfset routing so the wrapper can match
  RELION's `sorted_idx_iter001.bin` split instead of only micrograph sorting.
- Added adaptive-grid inference so legacy fine-grid callers can be replayed as
  RELION coarse pass-1 plus oversampled pass-2.
- Routed adaptive mode through the same exact-local sparse pass-2 wrapper used
  by normal EM.

Validation on the coherent 500-particle fixture:

- Adaptive wrapper with RELION sorted halfsets:
  `/scratch/gpfs/GILLES/mg6942/_agent_scratch/initial_model_vdam_adaptive_local_sorted_20260430/summary.json`
- Resulting BPref CC stayed poor: data CC about 0.464/0.458 and weight CC
  about 0.158/0.168.
- Particle-0 generated coarse support contains all 23 RELION final coarse
  support cells after converting RELION's direction-major order to the internal
  psi-major order.
- Particle-0 exact-local pass-2 scores still differ from the dense score path
  that matched RELION: score-delta CC 0.945, slope 1.005 on the same candidate
  rotations/translations.

Current conclusion: adaptive support generation and halfset routing are no
longer the leading suspect.  The next bug is inside the exact-local pass-2
score/preprocess path for this InitialModel convention.  The dense path score
matches RELION on dumped support, while exact-local does not.
