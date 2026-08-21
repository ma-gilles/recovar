# RELION d476e6f versus RECOVAR image-preprocessing source audit

Date: 2026-07-14

RECOVAR checkout: `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_em_parity_20260711/recovar`, commit `adf61bb48903b62f3b8effdafa802ef4f25d6439`

RELION source: `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_d476e6f_clean_binding_20260713/src`, commit d476e6f

Scope: supplied-map EM particle-image preprocessing for iteration-1 normalized CC and later local/final Gaussian scoring, with case 22's `_rlnDoZeroMask=1`. This is a source audit, not a claim of bitwise parity between cuFFT and JAX/NumPy.

## Bottom line

No executable semantic preprocessing bug was found in the production case-22 path. Do not change production preprocessing based on the currently observed translation-delta residuals (`0.000569` and `-0.008679`). The two implementations agree on operation order and signs after accounting for algebraically redistributed scalar factors:

1. multiply the particle by `avg_norm / normcorr`;
2. apply RELION's half-away-from-zero rounded old offset as a zero-filled real-space shift, `out[y+dy,x+dx] = in[y,x]`;
3. preserve the shifted unmasked image for reconstruction and background-fill soft-mask the scoring image;
4. center, transform, and crop/window the Fourier image;
5. apply CTF, noise precision, scale correction, and candidate translation phase with `exp(-2 pi i k dot shift)`;
6. use the rectangular current-size half-image, including DC, for iteration-1 normalized CC, and the radial/noise-weighted support excluding DC for Gaussian scoring.

The strongest remaining source-level explanation for small residuals is arithmetic route, not different mathematics. The production RELION executable reports `Precision: BASE=double, CUDA-ACC=single`. Its accelerated path casts the image to `XFLOAT=float`, reduces the mask background on the GPU in float32, uses cuFFT, and scales the forward FFT by `1/Npix`. RECOVAR's ordinary host background-fill path reduces in float64, runs NumPy FFT, and casts to complex64. RECOVAR also distributes normalization/scale factors into Fourier score operands instead of mutating the real image/reference at identical instruction boundaries. These paths should agree closely, but not bitwise.

One real source divergence exists outside the supplied case: `recovar.core.mask.relion_soft_image_mask` clamps a positive mask radius to `D/2`, while RELION's accelerated non-helical path only substitutes `D/2` when the radius is negative. Case 22 uses radius 23.53 px in a 128 box, so this clamp is inactive. It should get a separate oversized-diameter fixture before any code change.

The corrected executable replay is Slurm job `11163725`. It produced the complete operand dump before a stale diagnostic filename assertion set the job's final status to failed; the scientific output is valid, while jobs `11163615` and `11163657` failed during setup. The exact replay analysis is recorded at `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case16_recovar_relstate_valid_mask_probe_20260714_022000/valid_mask_exact_cross_analysis.txt`.

## Exact RELION path

All line numbers below refer to the d476e6f source root above.

### Common metadata and ordering

- `ml_optimiser.cpp:5857-5858` reads the per-particle norm correction.
- `ml_optimiser.cpp:5871-5879` reads the old offsets.
- `ml_optimiser.cpp:6081` executes `my_old_offset.selfROUND()`; `macros.h:197` implements ties away from zero.
- The CPU/general path multiplies by `avg_norm/normcorr` at `ml_optimiser.cpp:6221-6237`, then calls `selfTranslate(..., DONT_WRAP)` at `ml_optimiser.cpp:6239-6246`.
- In the accelerated production path, `acc/acc_ml_optimiser_impl.h:432-439` passes the scalar and rounded offsets to `TranslateAndNormCorrect`. `acc/utilities_impl.h:374-405` casts input pixels to `XFLOAT` and applies the scalar; `acc/utilities_impl.h:411-435` runs the zero-filled translate kernel.
- The concrete sign is explicit in `acc/cuda/cuda_kernels/helper.cuh:245-275` and `acc/cpu/cpu_kernels/helper.cpp:255-281`: `xp=x+dx`, `yp=y+dy`, followed by `out[yp,xp]=in[y,x]` if in bounds.

Thus the old offset is applied before masking and FFT, with no wrapping. RECOVAR's zero-fill implementation has the same sign.

### Unmasked versus masked images

- RELION transforms and stores the unmasked reconstruction image before masking at `acc/acc_ml_optimiser_impl.h:488-542`.
- It derives the mask radius as `particle_diameter/(2*pixel_size)` at `acc/acc_ml_optimiser_impl.h:549-552`.
- For `_rlnDoZeroMask=1`, it reduces the raised-cosine-weighted background at `acc/acc_ml_optimiser_impl.h:618-653` and blends with that background at `acc/acc_ml_optimiser_impl.h:660-668`.
- The accelerated code only replaces a negative radius with `D/2` (`acc/acc_ml_optimiser_impl.h:611-616`); it does not clamp a positive radius.
- The masked image is transformed at `acc/acc_ml_optimiser_impl.h:683-700`.

Case-22 source metadata is explicit in `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_small_stress_relion_20260711_042025_22010/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/relion_ref/run_it000_optimiser.star:22-24`: particle diameter 200 A, mask edge 5 px, and `_rlnDoZeroMask=1`. The resolved 23.53 px radius is logged in `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_controlflow_fix_20260713_230835/run.log:13`. It is far below `D/2=64`, so the RECOVAR clamp is not active.

### FFT convention and windowing

- `acc/utilities_impl.h:438-484` centers the real image, performs the forward accelerated FFT, multiplies every Fourier element by `1/reals.getSize()`, and then calls `windowFourierTransform2`.
- The CPU/general path expresses the equivalent sequence as Fourier transform, `windowFourierTransform`, and `CenterFFTbySign` (`ml_optimiser.cpp:6273-6283` for reconstruction and `ml_optimiser.cpp:6374-6417` for scoring).
- First-iteration scoring crops `Fimg` at `ml_optimiser.cpp:6826`, computes `sqrtXi2=sqrt(sum |Fimg|^2)` over that cropped rectangular half-array at `ml_optimiser.cpp:6843-6852`, and crops CTF identically at `ml_optimiser.cpp:6854-6857`.
- Gaussian scoring maps inverse noise by shell with DC excluded at `ml_optimiser.cpp:6867-6876`. Although the stored array is rectangular, pixels outside the modeled radial support have zero inverse-noise weight.

RECOVAR's centered rFFT is `fftshift(real, both axes) -> rfft2 -> fftshift(non-packed y axis)` (`recovar/data_io/image_backends.py:78-88`), which is the packed-half equivalent of RELION's centering. RECOVAR uses an unnormalized forward FFT (`recovar/core/fourier_transform_utils.py:4,424-434`), unlike RELION's explicit `1/Npix`. This is a global representation convention: RECOVAR's references and estimated noise use the same convention. It cancels exactly from normalized CC and is accounted for consistently in Gaussian residual/noise units; it must not be “fixed” by dividing only particle images.

### CTF, noise, scale, and candidate translation

- RELION generates the CTF at full size and handles premultiplication at `ml_optimiser.cpp:6455-6488` before the score-sized crop.
- CPU first-iteration CC uses `-Re(Frefctf * conj(Fimg_shift)) / (sqrt(suma2)*sqrtXi2)` at `ml_optimiser.cpp:7409-7424`.
- CPU Gaussian scoring starts with `highres_Xi2/2` and adds `0.5*Minvsigma2*|Frefctf-Fimg_shift|^2` at `ml_optimiser.cpp:7429-7437`.
- The accelerated candidate phase is `-2*pi*xshift/full_size` (and y/z) at `acc/acc_ml_optimiser_impl.h:1239-1241`.
- The accelerated path divides its stored `Fimg_` by scale and CTF at `acc/acc_ml_optimiser_impl.h:1251-1267`, then `buildCorrImage` restores CTF-squared and scale-squared weights (`acc/acc_helper_functions_impl.h:164-196`). This is algebraically the same cross/norm split used by RECOVAR; comparing raw `Fimg_` or `corr_img` individually without recomposing the score is invalid.

RECOVAR constructs the candidate phase as `exp(-2j*pi*dot(translation,k))` in `recovar/em/dense_single_volume/helpers/preprocessing.py:251-262`. Iteration-1 preprocessing directly forms `Fimg*CTF*phase` and image power in `helpers/preprocessing.py:175-248`; `em_engine.py:1282-1297` applies `1/Xi2` to the image cross operand and CTF-squared reference norm. Gaussian preprocessing forms `Fimg*CTF/noise*phase` and `CTF^2/noise` in `helpers/preprocessing.py:87-136`. The exact-local path repeats the same construction in `local_em_engine.py:1438-1598`; its fused path is in `local_big_jit.py:703-793`.

## RECOVAR operation trace

### Old-offset shift

- `helpers/orientation_priors.py:15-19` implements RELION half-away-from-zero rounding.
- `helpers/orientation_priors.py:72-79` constructs the integer search base.
- `helpers/image_shifts.py:36-75` implements zero-filled `out[y+dy,x+dx]=in[y,x]`.
- Dense scoring applies it before preprocessing at `em_engine.py:1090-1108`.
- Exact-local scoring applies it before either masked or unmasked transform at `local_em_engine.py:1463-1496`.

`helpers/image_shifts.py:27-33` uses `np.rint` only to verify that an already RELION-rounded input is integral. The half-away rounding occurs upstream, so the tie behavior is not replaced by banker's rounding in production.

### Norm/scale correction order

RELION multiplies the real image by `avg_norm/normcorr` before shift/mask. RECOVAR applies the corresponding image correction to Fourier operands in `em_engine.py:1215-1246` and applies scale squared to CTF weights at `em_engine.py:1259-1268`. This commutes mathematically with the zero-filled shift, FFT, and RELION background-fill mask because the mask background is a linear weighted mean. It changes rounding locations, so exact float32 identity is not expected. The fused local path performs the same redistribution at `local_big_jit.py:738-764`.

### Mask dtype

- Host path: `image_backends.py:35-62` casts mask and images to float64, performs the background reduction/blend in float64, and returns at least float32. `image_backends.py:294-320` then runs NumPy rFFT and casts to the backend complex64 dtype.
- Fused local path: `local_big_jit.py:63-82` uses the JAX mask and rFFT path. `recovar/core/mask.py:630-647` currently requests float64 for the background reduction, then casts back to the image dtype.
- RELION production path: `acc/settings.h:6-15` defines `XFLOAT=float` unless `ACC_DOUBLE_PRECISION`; the installed production binary prints CUDA-ACC single. Its mask background partial sums are `XFLOAT` in `acc/cuda/cuda_kernels/helper.cu:355-405` and subsequent reduction/blend remains `XFLOAT`.

This dtype/reduction distinction is the leading preprocessing explanation for tiny score deltas. It is not by itself a quality bug; changing RECOVAR to emulate CUDA reduction order would be fragile and GPU-specific unless an exact fixture demonstrates a material posterior/FSC effect.

## Existing tests and what they do not prove

- `tests/unit/test_relion_bind/test_p2_image_mask.py:129-187` compares RECOVAR mask geometry/background fill against the CPU double-precision RELION binding. Production float32 is accepted at `atol=5e-6`.
- `tests/unit/test_relion_bind/test_p3_window_fourier.py` compares exact C++ `windowFourierTransform` layouts.
- `tests/unit/test_relion_bind/test_e3_shift.py:1-135` proves candidate phase sign/layout against RELION, apart from the documented non-integral Nyquist ambiguity.
- `tests/unit/test_relion_bind/test_e4_scores.py` and `test_estep_composite_parity.py` prove score algebra using already-preprocessed Fourier operands.
- `tests/unit/test_refine_relion_mode.py:9292-9330` checks half-away rounding, integral selection, and zero-filled shift convention.

The current binding explicitly describes itself as CPU-only double precision (`recovar/relion_bind/module.cpp:1-8`). It exposes the CPU mask, Fourier window, candidate shift, CTF, projections, and score-related primitives, but not accelerated `TranslateAndNormCorrect + mask reduction + center/cuFFT + crop` as one operation. Therefore it cannot reproduce the exact production CUDA arithmetic boundary.

## Minimal decisive fixture

Add one debug-only accelerated RELION binding or executable dump that accepts:

- 2-4 deterministic float32 images (impulse, ramp, seeded noise, constant-plus-noise), preferably `D=32`;
- norm corrections including one non-unit value;
- old offsets covering `(0,0)`, positive/negative integers, and `+/-0.5` before RELION rounding;
- mask radius below `D/2`, width 5, and `do_zero_mask=true`;
- current sizes `D` and one smaller even size;
- one deterministic CTF and three candidate translations.

Dump boundaries, in production `XFLOAT`, immediately after:

1. `TranslateAndNormCorrect` real image;
2. background-value reduction and masked real image;
3. normalized centered full Fourier image;
4. current-size cropped Fourier image;
5. `sqrtXi2`, `Fimg_`, `corr_img`, and recomposed per-pose cross/norm/score.

The RECOVAR test should convert conventions explicitly (`RELION_F = RECOVAR_F/Npix`) and compare, in this order:

- exact support/index and sign invariants;
- max absolute/relative error at each boundary with float32-scaled tolerances;
- recomposed per-pose scores and translation deltas, not raw distributed operands alone;
- posterior `log_Z`, `Pmax`, and expected sufficient statistics;
- both host and fused-local RECOVAR preprocessing routes.

Pass criterion should be error consistent with float32 GPU reductions and no systematic translation-dependent bias. Only if the first divergent boundary exceeds that envelope should production code change.

Also add a separate CPU binding regression with radius `>D/2`. It will expose RECOVAR's positive-radius clamp and establish whether strict parity should remove it. Do not mix this out-of-case issue into the case-22 replay.

## Audit commands

Read-only source searches and line-numbered inspections used `rg -n` and `nl -ba ... | sed -n ...` over the two source roots above. Production precision was verified with:

```bash
/projects/MOLBIO/local/relion-5.0.1-gcc-11.5.0-cuda-12.6-rhel9-arch90/bin/relion_refine --version
```

Output:

```text
RELION version: 5.0.1-commit-d476e6
Precision: BASE=double, CUDA-ACC=single
```

No production code was edited and no GPU or Slurm job was launched for this source audit.
