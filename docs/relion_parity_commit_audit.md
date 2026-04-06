# RELION-parity commit audit (after FFT convention bug discovery)

The FFT convention bug was introduced in `c0cf265` (2026-04-01) and lived in
the codebase until `4be563e`, `31f2671`, and `5eb5e4b` (2026-04-06). About
half of the ~50 commits in this window were wasted chasing symptoms of this
bug.

## The bug, in brief
- `prepare_relion_parity_benchmark.py` saved volumes via raw
  `np.fft.ifftn(np.fft.ifftshift(centered_FT))` + `mrcfile`. This is **not**
  the inverse of recovar's canonical `ftu.get_dft3 / get_idft3` helpers.
- `run_full_refinement.py` then loaded volumes via raw `mrcfile.open` +
  `np.fft.fftn(np.fft.ifftshift(...))`, which is **also** not the canonical
  idiom.
- The two ad-hoc pairs do **not** round-trip through `slice_volume`. The
  resulting projections were corrupted (correlation 0.0004 with simulated
  particles instead of the expected high correlation), with DC reading as
  Nyquist and ~2400× lower amplitude at low frequencies.
- Symptoms: iter-1 Pmax 0.66 (vs RELION 0.59), iter-2 Pmax → 0.90, iter-3
  collapse to 1.0; recovar's converged volume was uncorrelated with both
  the GT and RELION's volume.

## Commit categories

Legend: 🟢 Real RELION-parity work (still useful) · 🔴 Wasted (chasing the
FFT bug) · 🔵 The actual fix · ⚪ Cleanup or revert

### 🔵 Fix commits (the cure)
- `5eb5e4b` Fix prepare_relion_parity_benchmark.py FFT convention with `save_volume`
- `4be563e` Fix FFT convention bug in run_full_refinement.py (init_vol_ft was wrong)
- `31f2671` Reset tau2_fudge to 1.0 (RELION default) now that FFT bug is fixed

### 🟢 Real RELION-parity infrastructure (still useful, ~22 commits)
- `c0cf265` RELION-parity dense single-volume EM: complete implementation *(introduced the bug, but most of the new code is real)*
- `e3c6315` Fix 3 gaps and run head-to-head comparison vs RELION
- `d56f1d9` Add high-SNR head-to-head comparison vs RELION
- `7df73fa` Add RELION convention helpers and document convention mismatch
- `55e5fe7` Document RELION volume convention in EM CLAUDE.md
- `f1db4c8` Remove obsolete tests for superseded engine wrappers
- `4703c63` Revert helpers.py and dataset files to clean origin/dev state
- `3f2c45a` Add final recovar vs RELION comparison on simulated data
- `604f7c6` Add RELION-parity mode to refine_single_volume
- `a6889b0` Add Gaussian prior weighting to local angular search + intermediate saving
- `20ac334` Optimize local search: deduplicate prior rotations before distance computation
- `685f874` Replace brute-force local search with fast HEALPix query_disc approach
- `ede9325` Cap local search grid to 20K rotations for consistent speed
- `0ba0df6` Fix noise estimation: use all images from both half-sets
- `b0a2586` Add padding factor 2 and adaptive oversampling for RELION parity
- `6180b35` Fix OOM at high HEALPix orders + relax convergence for adaptive OS
- `422e23e` Disable padding_factor=2 (broken zero_pad_fourier_volume)
- `09f7320` Cap full rotation grid at order 4 to prevent OOM at order 5+
- `d1aac63` Fix critical bug: PADDING_FACTOR=2 override was clobbering PADDING_FACTOR=1
- `a81b5ea` Raise max_union_pixels cap from 200 to 5000 in adaptive oversampling
- `0549c5d` Add RELION-parity status & handoff doc
- `0713039` RELION-parity: half-spectrum scoring, posterior-weighted noise, shell statistics *(scaffolds for proper noise update; the per-shell ablations on top were wasted)*
- `a9c8b22` Clean up dead padding code in compute_relion_prior_from_reconstruction_stats

### 🔴 Wasted on FFT-bug symptoms (~25 commits)
**Noise floors and inflation experiments** (all chasing iter-3 collapse from corrupted projections):
- `49122ce` Use running-maximum noise floor to prevent posterior collapse
- `52c9c37` Document noise inflation experiment: 5x fudge enables convergence
- `fce30a9` Add mask-factor noise inflation to prevent posterior collapse
- `9288a24` Use per-shell noise inflation based on mask geometry
- `f9df580` Tune per-shell noise inflation to 2.3x at DC (0.15 * area_ratio)
- `54b8d16` Increase per-shell inflation to 3.1x at DC (0.25 * area_ratio)
- `6f20ba8` Use annealing schedule for per-shell noise inflation
- `27cc084` Inflate hard-assignment noise by mask area ratio (9.4x)
- ⚪ `e9c3d7b` Revert v25 ad-hoc mask-ratio noise inflation

**Tau2_fudge experiments**:
- `6b1148a` Apply tau2_fudge=0.5 to over-regularize Wiener filter
- `51b70bd` Apply tau2_fudge=0.5 to volume reconstruction (not just prior update)
- `061fe52` Reduce tau2_fudge to 0.25 for stronger Wiener regularization
- `b9b298b` Reduce tau2_fudge to 0.1 - even stronger Wiener prior

**Soft-mask init volume experiments**:
- `76b1bfe` Apply RELION-style soft mask to initial volume
- ⚪ `7f4492b` Revert v31 soft mask on initial volume (made things worse)
- `1773a58` Apply soft mask to initial volume with correct FFT convention
- ⚪ `9d3e5e5` Revert v32 soft mask: still collapses due to over-aggressive init lowpass

**Score temperature experiments**:
- `e6bf304` Restore tau2_fudge=0.25 (v29 config - best stable result)
- `31583e5` Apply 6x score temperature to match RELION iter 1 Pmax (0.59)
- `7e6faa6` Use score_temperature=6 only at iter 1 (schedule)
- `cc0752a` Reduce iter-1 score temperature from 6x to 2x
- ⚪ `73a7cf8` Revert score temperature: distorts M-step reconstruction

**RELION mask + noise fill (tested on corrupted volumes, gave wrong conclusions)**:
- `99a1f43` Add noise_fill_outside_mask infrastructure (disabled: makes Pmax worse) *(infrastructure is real; the "makes worse" conclusion was on the bugged volume)*
- `34c5c4d` Apply RELION softMaskOutsideMap with noise fill to scoring images
- ⚪ `b7d2f7f` Revert v37 noise fill: empirically makes iter 2 Pmax worse

**Workaround for FSC issues caused by corrupted projections**:
- `47fff64` Increase incr_size from 10 to 12 to prevent current_size collapse *(should be re-tested with fixed volumes; if RELION uses 10, we should too)*

## Summary
- ~25 commits (≈ half of all RELION-parity work since 2026-04-01) were
  effectively wasted chasing symptoms of one missing `np.fft.fftshift`.
- The user diagnosed this bug class previously (see commits `7df73fa`
  "Add RELION convention helpers and document convention mismatch" and
  `55e5fe7` "Document RELION volume convention in EM CLAUDE.md") and the
  documentation existed in the EM-specific CLAUDE.md, but the
  `prepare_relion_parity_benchmark.py` and `run_full_refinement.py`
  scripts went around the canonical helpers.
- The fix is small (one helper call per script) and the bug was caused
  entirely by reinventing the FFT/MRC pair instead of using
  `recovar.output.output.save_volume` and
  `recovar.core.fourier_transform_utils.get_dft3(load_mrc(path))`.
- The CLAUDE.md instructions in `~/CLAUDE.md` have been updated to make
  this convention permanently visible at the top of every recovar session
  to prevent the bug from recurring.

## Recommended actions
1. **Re-run all wasted experiments** with the fixed volumes to confirm:
   - `tau2_fudge=1.0` is correct (no need for the 0.25 workaround)
   - `incr_size=10` is correct (no need to bump to 12)
   - The "noise inflation" patterns we measured were all artifacts
2. **Re-evaluate the noise_fill_outside_mask infrastructure** with the
   fixed volumes — the previous "makes Pmax worse" conclusion may have
   been an artifact of the bug.
3. **Re-test the soft-mask init volume** experiment with the fixed
   volumes — it may now match RELION's behavior.
