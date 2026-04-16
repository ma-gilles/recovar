# RELION Auto-Refine Feature Audit

Living checklist of every RELION `relion_refine --auto_refine` sub-operation,
whether recovar has ported it, and whether a pybind11 binding has validated
parity on identical inputs.

**Maintained alongside `recovar/relion_bind/`.**
Update this file every time a binding validates or disproves parity.

## Legend

| Symbol | Meaning |
|--------|---------|
| Yes | recovar has an implementation |
| Partial | Implementation exists but known incomplete |
| Broken | Implementation exists but produces wrong results |
| No | No recovar implementation |
| N/A | Not applicable (e.g., C1 symmetry) |
| ✓ | Binding confirmed numerical parity |
| **DIVERGENT** | Binding showed measurable difference (see Notes) |
| — | Not yet tested with binding |

---

## Pre-processing & Image I/O

| # | RELION operation | RELION source | Ported? | recovar location | Validated? | Notes |
|---|-----------------|---------------|---------|------------------|------------|-------|
| 1 | Read image + apply gain/defect | ml_optimiser.cpp:5840 | Partial | `data_io/` handles gain | — | recovar skips defect correction |
| 2 | Soft circular mask (particle_diameter) | ml_optimiser.cpp:5893 | No | — | — | recovar uses unmasked images for E+M |
| 3 | FFT of masked image (for E-step) | ml_optimiser.cpp:5920 | Yes | `engine_v2.py` FFT in preprocess | — | |
| 4 | FFT of unmasked image (for M-step) | ml_optimiser.cpp:5927 | No | — | — | recovar uses same image for both |
| 5 | selfTranslate (beam-tilt phase ramp) | ml_optimiser.cpp:5935 | No | — | — | Zero effect for SPA w/o beam tilt |
| 6 | demodulatePhase (aberration correction) | ml_optimiser.cpp:5940 | No | — | — | Zero for standard optics |
| 7 | divideByMtf (detector MTF correction) | ml_optimiser.cpp:5945 | No | — | — | Not provided in simulated data |
| 8 | CTF image (2× padded box → downsample) | ctf.cpp:314 | No | recovar computes at image size directly | — | **Known divergence** — binding P1 |
| 9 | 1/σ² noise weighting per pixel | ml_optimiser.cpp:5960 | Yes | `engine_v2.py` sigma2_noise weighting | — | |
| 10 | windowFourierTransform (current_size crop) | fftw.h:809 | Yes | `fourier_window.py` index-based | — | Different strategy — binding P3 |

## Projector Setup

| # | RELION operation | RELION source | Ported? | recovar location | Validated? | Notes |
|---|-----------------|---------------|---------|------------------|------------|-------|
| 11 | Zero-pad volume to pf×N | projector.cpp:340 | Broken | `relion_functions.py:24` | — | pf=2 disabled, using pf=1 |
| 12 | FFT + CenterFFTbySign → projector storage | projector.cpp:360 | No | recovar uses centered full-complex | ✓ | Binding E1: projector data matches numpy ref at rtol=1e-12 |
| 13 | Trilinear interpolation (3D→2D slice) | projector.cpp:630 | Yes | `slicing.py:slice_volume:221` | ✓ | Binding E2: identity projection matches projector kz=0 exactly |

## E-step: Scoring

| # | RELION operation | RELION source | Ported? | recovar location | Validated? | Notes |
|---|-----------------|---------------|---------|------------------|------------|-------|
| 14 | Phase-shift images (translations) | fftw.cpp:762 | Yes | `geometry.py:translate_images:167` | — | RELION uses tab sin/cos — binding E3 |
| 15 | ‖img − CTF·proj‖²/σ² per shell | ml_optimiser.cpp:7098 | Yes | `engine_v2.py:_e_step_block_scores` | — | binding E4 |
| 16 | First-iter CC scoring (hard assignment) | ml_optimiser.cpp:7414 | No | — | — | recovar uses soft Bayesian iter 1 |
| 17 | exp(−diff2) × priors → posteriors | ml_optimiser.cpp:7704 | Yes | `engine_v2.py:_update_logsumexp` | — | binding E5 |
| 18 | pdf_orientation prior term | ml_optimiser.cpp:7780 | Yes | `engine_v2.py` rotation_log_prior | — | |
| 19 | pdf_offset prior term | ml_optimiser.cpp:7780 | Yes | `engine_v2.py` translation_log_prior | — | |
| 20 | Significance pruning (adaptive OS pass 2) | ml_optimiser.cpp:7850 | Yes | `refine.py` adaptive_oversampling | — | Union-of-images, not per-image |

## E-step: Accumulation (storeWeightedSums)

| # | RELION operation | RELION source | Ported? | recovar location | Validated? | Notes |
|---|-----------------|---------------|---------|------------------|------------|-------|
| 21 | Σ w_i · CTF·img → Fourier accumulators | ml_optimiser.cpp:8241 | Yes | `engine_v2.py:_m_step_block` | — | binding E6 |
| 22 | Σ w_i · ‖img−proj‖² per shell → σ²_noise | ml_optimiser.cpp:8241 | Partial | hard-assignment only | — | **Known divergence** — binding E7 |
| 23 | Per-image norm_correction accumulation | ml_optimiser.cpp:8350 | No | — | — | No-op single optics group |
| 24 | Per-image scale_correction accumulation | ml_optimiser.cpp:8370 | No | — | — | No-op single optics group |

## M-step: Reconstruction

| # | RELION operation | RELION source | Ported? | recovar location | Validated? | Notes |
|---|-----------------|---------------|---------|------------------|------------|-------|
| 25 | Symmetrise Fourier accumulators | backprojector.cpp:1200 | N/A | C1 symmetry only | — | |
| 26 | Enforce Hermitian symmetry | backprojector.cpp:2207 | Yes | `half_volume_to_full_volume` | — | binding M8 |
| 27 | Blob deconvolution (iterative) | backprojector.cpp:1400 | No | direct Wiener instead | — | Different strategy — binding M2 |
| 28 | IFFT → crop to ori_size | backprojector.cpp:2589 | Yes | iDFT + unpad | — | binding M7 |
| 29 | Gridding correction (radial sinc²) | projector.cpp:595 | Yes | `relion_functions.py:64` | ✓ | Validated Phase 1: max diff 1.6e-5 (float k-coords vs int) |
| 30 | Flatten solvent (mask volume exterior) | ml_optimiser.cpp:5100 | No | — | — | |
| 31 | Zero mask (zero exterior instead of noise) | ml_optimiser.cpp:5110 | No | — | — | |

## M-step: Statistics & Resolution

| # | RELION operation | RELION source | Ported? | recovar location | Validated? | Notes |
|---|-----------------|---------------|---------|------------------|------------|-------|
| 32 | Gold-standard FSC between half-maps | backprojector.cpp:998 | Yes | `regularization.py:get_fsc_gpu:132` | — | binding M3 |
| 33 | updateSSNRarrays (tau2, data_vs_prior) | backprojector.cpp:1044 | Yes | `regularization.py:compute_data_vs_prior` | — | binding M4 |
| 34 | updateCurrentResolution | ml_optimiser.cpp:5579 | Yes | `regularization.py:resolution_from_data_vs_prior:624` | — | binding M5 |
| 35 | Noise update (σ² per shell) | ml_optimiser.cpp:5090 | Partial | hard-assignment in `refine.py` | — | See #22 |
| 36 | tau2_fudge multiplier | ml_optimiser.cpp:5050 | Yes | `refine.py` kwarg, default=4 | — | |
| 37 | Low-resolution halves joining (40 Å) | ml_optimiser.cpp:5030 | Yes | `regularization.py:812-910` | — | |

## Sampling

| # | RELION operation | RELION source | Ported? | recovar location | Validated? | Notes |
|---|-----------------|---------------|---------|------------------|------------|-------|
| 38 | HEALPix orientation grid | healpix_sampling.cpp:1832 | Yes | `sampling.py:get_rotation_grid_at_order:530` | — | binding S1 |
| 39 | Oversampled sub-grid (psi children) | healpix_sampling.cpp:1850 | Yes | `sampling.py` adaptive OS | — | |
| 40 | Translation grid | healpix_sampling.cpp:1724 | Yes | `sampling.py:get_translation_grid:100` | — | binding S2 |
| 41 | Perturbation (per-iter rigid rotation) | healpix_sampling.cpp:1909 | Yes | `sampling.py:apply_relion_rotation_perturbation` | — | binding S3 |
| 42 | 3σ cone prior filtering | healpix_sampling.cpp:695 | Yes | `sampling.py` sigma_cutoff=3.0 | — | binding S4 |

## Convergence & Loop Control

| # | RELION operation | RELION source | Ported? | recovar location | Validated? | Notes |
|---|-----------------|---------------|---------|------------------|------------|-------|
| 43 | Angular step refinement trigger | ml_optimiser.cpp:5200 | Yes | `convergence.py:710-790` | — | |
| 44 | Auto-termination (no improvement) | ml_optimiser.cpp:5250 | Yes | `convergence.py:544-620` | — | |
| 45 | ave_Pmax tracking | ml_optimiser.cpp:5230 | Yes | `convergence.py` | — | |
| 46 | Assignment change tracking | ml_optimiser.cpp:5220 | Yes | `convergence.py` | — | |

---

## Summary

| Status | Count | IDs |
|--------|-------|-----|
| Ported (full) | 30 | 1,3,9,10,13,14,15,17,18,19,20,21,26,28,29,32,33,34,36,37,38,39,40,41,42,43,44,45,46 |
| Ported (partial/broken) | 4 | 4,11,22,35 |
| Not ported | 11 | 2,5,6,7,8,16,23,24,27,30,31 |
| Not applicable | 1 | 25 |
| **Binding-validated** | **3** | **12 (projector storage), 13 (trilinear projection), 29 (gridding correction)** |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-04-16 | Initial audit. Phase 0+1 complete: layout conversions validated (61 tests), M1 gridding bound and validated (14 tests). |
| 2026-04-15 | Phase 2b complete: E1 projector storage + E2 projection bound and validated (10 tests). Key finding: CenterFFTbySign uses (-1)^(k+i+j) on physical indices (fftw.h:400), not linear-index parity. RELION forward FFT normalizes by 1/N^3 (fftw.cpp:358). |
