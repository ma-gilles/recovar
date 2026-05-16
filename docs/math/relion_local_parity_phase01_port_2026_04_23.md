# RELION Local-Parity Port Notes (2026-04-23)

This branch ports the confirmed late-iteration RELION parity fixes onto
`origin/codex/em-phase01-sparse-pass2` and applies them to both local-search
engine paths:

- `grouped_union`
- `exact_v1` (per-image local rotations)

## Final branch state

Both local-search engines now reproduce the late solved baseline on this
branch:

- `grouped_union`
  - run: job `7299927`
  - output:
    `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_em_phase01_sparse_pass2/_agent_scratch/late_it014_full5k_grouped_union_local_v6_skipfinal`
  - RECOVAR `ave_Pmax = 0.3245294973`
  - RELION `ave_Pmax = 0.3245866848`
  - gap `= -5.7187589e-05`
  - mean abs per-particle `Pmax` diff `= 0.0055527547`
  - per-particle `Pmax` corr `= 0.9973253170`
  - RECOVAR vs RELION merged-map corr `= 0.999900`
  - elapsed `= 666.50 s`

- `exact_v1`
  - run: job `7301876`
  - output:
    `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_em_phase01_sparse_pass2/_agent_scratch/late_it014_full5k_exact_v1_local_v8_skipfinal`
  - RECOVAR `ave_Pmax = 0.3245294973`
  - RELION `ave_Pmax = 0.3245866848`
  - gap `= -5.7187589e-05`
  - mean abs per-particle `Pmax` diff `= 0.0055527547`
  - per-particle `Pmax` corr `= 0.9973253170`
  - RECOVAR vs RELION merged-map corr `= 0.999900`
  - elapsed `= 725.95 s`

The exact engine now reaches the same parity endpoint as the grouped engine.

## Confirmed fixes

Two changes were required to close the late `iter 13 -> 14` RELION mismatch:

1. **RELION local translation prior must use the coarse RELION grid**
   - RELION evaluates the local translation prior on the unperturbed coarse
     translation grid, even when the scored candidate translations are the
     perturbed replay grid.
   - The grouped local engine previously built its prior directly on the
     perturbed candidate translations.
   - The exact local engine layout also built per-image translation priors on
     the perturbed candidate translations.
   - The port therefore adds
     `translation_prior_reference_translations` plumbing so both engines can
     score perturbed translations while evaluating the prior on the RELION
     coarse grid.

2. **RELION masked-image scoring uses soft-mask background fill, not plain multiplication**
   - RECOVAR previously scored masked images as `image * mask`.
   - RELION's `softMaskOutsideMap(..., Mnoise=NULL)` replaces the exterior with
     the weighted average background under `1 - mask` and blends the cosine
     edge toward that background.
   - The port adds `apply_relion_soft_image_mask(...)` and threads it through
     the particle image backend via `image_mask_mode="relion_background_fill"`.

3. **`exact_v1` must currently reuse the parity-clean grouped local scorer**
   - The new custom exact local score path on this branch was not RELION-clean
     on the late benchmark.
   - The failing exact run on this branch was:
     - output:
       `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_em_phase01_sparse_pass2/_agent_scratch/late_it014_full5k_exact_v1_local_v6_skipfinal`
     - RECOVAR `ave_Pmax = 0.3262620007`
     - RELION `ave_Pmax = 0.3245866848`
     - gap `= +0.0016753159`
     - mean abs per-particle `Pmax` diff `= 0.0067942814`
     - per-particle `Pmax` corr `= 0.9905568597`
   - The regression also reproduced on a 2-particle outlier subset, where old
     `exact_v1` gave `ave_Pmax = 0.4465` versus RELION `0.3393`.
   - The parity-safe fix is to make `exact_v1` a per-image wrapper around the
     proven grouped RELION-mode scorer with `image_batch_size=1`. This keeps
     true per-image local neighborhoods while reusing the same scoring math
     that matches RELION.
   - On the same 2-particle outlier subset, the fixed `exact_v1` dropped to
     `ave_Pmax = 0.3416` versus RELION `0.3393` before the skipped final
     all-data step.

## Stored late-parity baseline from the solved branch

Source artifacts on the solved branch:

- bad late full replay:
  `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/_agent_scratch/local_profile_full5k_it014_late_v1`
- fixed late full replay:
  `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/_agent_scratch/late_it014_full5k_maskfix_v1`
- fixed late replay log:
  `/scratch/gpfs/GILLES/mg6942/slurmo/recovar-late-replay-7288799.out`

### Before fix

- RECOVAR `ave_Pmax = 0.4758841119`
- RELION `ave_Pmax = 0.3245866848`
- gap `= +0.1512974271`
- mean abs per-particle `Pmax` diff `= 0.1602249668`
- per-particle `Pmax` corr `= 0.7546786006`

### After fix

- RECOVAR `ave_Pmax = 0.3245294973`
- RELION `ave_Pmax = 0.3245866848`
- gap `= -0.0000571875`
- mean abs per-particle `Pmax` diff `= 0.0055527548`
- median abs diff `= 0.0044897006`
- max abs diff `= 0.0282227426`
- per-particle `Pmax` corr `= 0.9973253170`
- RECOVAR vs RELION merged-map corr `= 0.999900`
- half1 corr `= 0.999838`
- half2 corr `= 0.999842`

### FSC / map metrics from the solved late replay

- `recovar_half1` corr `0.959933`
- `relion_half1` corr `0.959874`
- `recovar_half2` corr `0.959693`
- `relion_half2` corr `0.959656`
- `recovar_merged` corr `0.966514`
- `relion_merged` corr `0.966607`
- half-map `FSC<0.5` shell `36`, `res=15.11 A` on both
- half-map `FSC<0.143` shell `43`, `res=12.65 A` on both
- merged-map `FSC<0.5` shell `39`, `res=13.95 A` on both
- merged-map `FSC<0.143` shell `43`, `res=12.65 A` on both

## Port validation performed on this branch

Targeted unit validation passed after the port:

- `tests/unit/test_image_backends.py`
- `tests/unit/test_relion_bind/test_p2_image_mask.py`
- `tests/unit/test_refine_relion_mode.py`

The exact local engine now has explicit unit coverage for the coarse-grid
translation-prior reference path, and `exact_v1` now has wrapper coverage that
it forces per-image local neighborhoods by dispatching through the grouped
RELION-mode scorer with `image_batch_size=1`.

## Branch-specific benchmark gate

This gate is now closed:

1. reran the late full 5k replay with `local_engine=grouped_union`
2. reran the late full 5k replay with `local_engine=exact_v1`
3. confirmed both reproduce the stored late-parity baseline above to within
   negligible numerical drift
