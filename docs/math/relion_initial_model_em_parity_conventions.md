# RELION InitialModel and EM parity conventions

This note documents the practical differences that mattered when making the
InitialModel replay match RELION and explains the shared RELION-style EM
algorithm. It is intended as the first checklist for future RELION parity work.

## Scope

The discussion covers the dense single-volume RELION-parity code paths:

- `recovar/em/initial_model/gpu_pipeline.py`
- `recovar/em/dense_single_volume/`
- `scripts/run_k_class_parity.py`
- `tests/unit/initial_model/test_estep_fixture.py`

It does not describe the older generic homogeneous EM API or the low-rank
heterogeneity pipeline except where those paths share conventions.

## Executive summary

The main mistake was assuming that InitialModel could reuse the normal dense EM
full-grid convention with a few scale conversions. It could not. RELION's
InitialModel iteration is an adaptive RELION pass:

1. score a coarse RELION grid,
2. select image-specific significant coarse samples,
3. score only oversampled children of that support,
4. backproject in RELION's normalized FFT / `Minvsigma2` frame.

Normal RELION replay EM already had most of this machinery. InitialModel needed
to enter that same adaptive/sparse path and use the exact InitialModel-specific
units, signs, and metadata.

The dense full-grid path is still useful and intentionally remains available as
`estep_mode="dense"` for diagnostics. It is not the parity path for RELION
InitialModel.

## Fixes that closed the InitialModel gap

### 1. Adaptive sparse support instead of dense full grid

RELION does not backproject all oversampled poses in InitialModel. It first
evaluates a coarse pass, keeps a particle-specific significant support, then
evaluates oversampled child poses only for those retained parents.

The old InitialModel RECOVAR path evaluated the full oversampled Cartesian grid.
That can match RELION scores inside RELION's support, but it assigns posterior
mass to extra poses that RELION pruned away. Those extra poses pollute BPref and
were the main reason BPref map/weight correlations stayed poor even when
single-particle score algebra looked correct.

Current behavior:

- `estep_mode="relion_adaptive"` is the InitialModel parity mode.
- `estep_mode="dense"` remains an explicit diagnostic option.
- Adaptive mode routes through the same significance and sparse pass-2 style
  used by the normal RELION EM path.

### 2. RELION projector frame, not RECOVAR full Fourier frame

RELION InitialModel scores projections from `Projector::data` / PPref, not from
RECOVAR's centered full Fourier volume grid.

The parity path therefore:

- converts the input real-space reference to RELION's volume frame,
- calls the RELION projector binding to build PPref,
- uses the returned RELION half-spectrum projector data for scoring.

One important implementation detail is that the binding returns PPref with a
normalization relative to RELION's dumped amplitudes. Multiplying the returned
projector data by `ori_size` reproduces the RELION `PPref` / `Fref` amplitudes
used by the InitialModel dump.

### 3. Normal EM and InitialModel use different FFT/noise frames

The historical RECOVAR dense EM path uses a RECOVAR image/noise convention:

- images are in RECOVAR's centered Fourier frame,
- the noise spectrum is converted through the `N^4` frame,
- converting the final accumulator to RELION BPref needs explicit scale/sign
  factors.

The RELION-adaptive InitialModel path uses RELION's native convention:

- particle/reference Fourier arrays are normalized like RELION FFTW rows,
- `sigma2_noise` is consumed directly as RELION `Minvsigma2`,
- the signed CTF/image convention is already handled before accumulation,
- the final BPref output needs no extra `-N^2` / `N^4` conversion.

This is encoded in `gpu_pipeline.py` as:

- adaptive mode: `noise_scale = 1.0`
- dense mode: `noise_scale = N^4`
- adaptive BPref frame scale: data `1.0`, weight `1.0`
- dense BPref frame scale: data `-N^2`, weight `N^4`

This was a major source of misleading partial parity: one can make score deltas
look good under one frame and still reconstruct the wrong BPref if the
accumulator is converted under the other frame.

### 4. Signed image correction in adaptive InitialModel

RELION's local image/CTF convention differs from the historical RECOVAR dense
image convention. For InitialModel adaptive replay, the per-image correction is
the RELION-normalized signed scale:

```text
image_correction = -1 / N^2
```

With this correction, the score path and the BPref accumulator are already in
the RELION frame. Do not apply an additional BPref sign flip in adaptive mode.

### 5. Translation prior units

The translation prior was another large hidden mismatch. The relevant metadata
is the previous model's `rlnSigmaOffsetsAngst`, not a stale CLI or optimiser
value from a different stage.

For the coherent InitialModel fixture, reading the previous model gives:

```text
rlnSigmaOffsetsAngst = 10 A
```

The adaptive code stores phase-shift translations in pixels, while RELION's
InitialModel prior path stores offsets in Angstroms and still multiplies by an
additional pixel-size factor inside its weight conversion. Replaying the RELION
prior from pixel-space translations therefore needs:

```text
log_prior(t_px) = -0.5 * ||t_px||^2 * pixel_size^4 / sigma_offset_A^2
```

This `pixel_size^4` factor looks suspicious if viewed from a clean mathematical
model, but it is the source-level behavior needed for RELION replay parity.
Do not replace it with the cleaner `pixel_size^2` expression unless RELION-side
dumps prove that a different code path is being replayed.

### 6. Pass-1 current size

The first adaptive attempt changed the pass-1 Fourier window size. RELION's
InitialModel dump for the iter-1 fixture keeps the same effective `current_size`
across the coarse and oversampled passes. The parity path preserves RELION's
reported `current_size` by default.

If a future RELION fixture shows a different InitialModel rule, record the
source line and dump metadata before changing this.

### 7. RELION projector rows must be centered for local buckets

The exact-local EM path can feed RELION projector rows into bucketed local
scoring. Those rows must be returned in the centered-row layout expected by the
local engine.

The relevant fix is:

```python
centered_rows=True
```

when `_project_local_bucket` calls the RELION projector helper. Without this,
the values are close enough to hide in some score summaries but wrong enough to
damage posterior and BPref parity.

### 8. Dataset and halfset ordering must be coherent

InitialModel replay is sensitive to the exact particle ordering used by RELION.
For coherent replay:

- read the same `particles.star` used by the RELION run,
- align RELION `run_itNNN_data.star` rows back to dataset order by stack index,
- use RELION sorted particle ids when reproducing RELION halfsets,
- read previous-iteration model metadata from `run_it000_model.star` for an
  iter-1 target.

Do not rely on relative STAR paths stored in old optimiser files when a fixture
has been moved or copied.

## Normal RELION EM versus RELION InitialModel

The two paths share the same mathematical EM structure, but the replay
contracts are different.

### Shared structure

Both paths estimate a volume from hidden orientation/translation variables. For
image `i`, rotation `r`, and translation `t`:

```text
y_i ~= S_t C_i P_r mu + noise
```

The E-step computes posterior weights over hidden states, and the M-step
accumulates weighted backprojection sufficient statistics:

```text
Ft_y   = sum gamma_i,r,t * P_r^* C_i Sigma_i^-1 S_t^* y_i
Ft_ctf = sum gamma_i,r,t * P_r^* C_i Sigma_i^-1 C_i P_r
```

The reconstructed map is a regularized filtered backprojection using `Ft_y`,
`Ft_ctf`, and RELION's tau/noise model.

### Normal RELION replay EM

Normal replay starts from a previous RELION optimiser/model state. It must
preserve:

- current resolution and `current_size`,
- healpix order and translation search metadata,
- random perturbation,
- half-set state,
- previous maps, noise spectra, tau2 spectra, FSC, and convergence counters,
- previous best poses/translations for local search.

The normal path already had code for per-image local/adaptive support,
bucketed sparse pass 2, and RELION half-volume M-step conventions. Most of the
later parity work was about exact low-level details: shell boundaries, tau/noise
update order, local candidate layouts, and convergence state.

### RELION InitialModel replay

InitialModel starts from low-pass initial reference(s) rather than a mature
refinement state. Its first useful parity target is usually the transition from
`run_it000_*` to `run_it001_*`.

Important InitialModel-specific differences:

- the reference is fed through RELION `Projector::data` / PPref,
- the E-step must use RELION adaptive support, not a full dense fine grid,
- the image/noise convention is RELION's normalized FFT / `Minvsigma2` frame,
- translation priors must use the previous model sigma offset and RELION's
  pixel-size behavior,
- BPref output is already in RELION frame in adaptive mode,
- dense full-grid mode remains a diagnostic, not the parity implementation.

## Algorithmic flow for InitialModel parity

This is the current intended algorithm for one InitialModel replay iteration.

### Inputs

Read these from the RELION fixture and the dataset:

- particle STAR in the exact RELION run order,
- previous model STAR, e.g. `run_it000_model.star`,
- target model/data/sampling STAR, e.g. `run_it001_*`,
- initial reference map(s),
- `sigma2_noise`, `rlnSigmaOffsetsAngst`, `rlnCurrentImageSize`,
  particle diameter, pixel size, class distributions, and direction priors.

### Reference preparation

1. Start from the RECOVAR-frame real-space reference.
2. Convert to RELION real-space frame.
3. Build RELION projector PPref with `compute_fourier_transform_map`.
4. Multiply the returned projector data by `ori_size` to match RELION dumped
   amplitudes.
5. Use the returned `r_max` for RELION projector scoring.

### Grid preparation

1. Build the RELION coarse HEALPix rotation grid for the target sampling order.
2. Apply RELION's random perturbation to rotations and translations.
3. Build the base translation grid in pixels.
4. Build oversampled child rotations/translations from retained parent samples.
5. Preserve the RELION target `current_size` unless fixture metadata proves
   RELION used a different coarse-pass window.

### Prior preparation

1. Direction priors come from `model_pdf_orient_class_N`.
2. Class priors come from `model_classes`, for K-class runs.
3. Translation priors come from the previous model's `rlnSigmaOffsetsAngst`.
4. Pixel-space translation prior replay uses:

```text
-0.5 * ||t_px||^2 * pixel_size^4 / sigma_offset_A^2
```

### E-step pass 1

For each image batch:

1. load images in dataset order,
2. apply RELION image mask/background behavior,
3. compute RELION-frame Fourier rows,
4. apply CTF/noise weighting in the `Minvsigma2` frame,
5. score the coarse rotation/translation grid in the RELION Fourier window,
6. add direction, translation, and class priors,
7. compute per-image normalization statistics,
8. retain image-specific significant coarse samples.

The important output of pass 1 is not just a best pose. It is the significant
support mask that determines which oversampled child poses are legal in pass 2.

### E-step pass 2 and M-step accumulation

For each image-specific support bucket:

1. expand retained coarse rotations/translations to oversampled child samples,
2. score only those children,
3. normalize against the pass-1/pass-2 log normalizer,
4. accumulate weighted image terms into `Ft_y`,
5. accumulate weighted CTF/noise terms into `Ft_ctf`,
6. preserve RELION centered-row half-spectrum layout when using RELION projector
   rows.

### BPref conversion and reconstruction

After accumulation:

1. convert `Ft_y` / `Ft_ctf` to RELION BPref slab layout,
2. in adaptive InitialModel mode, apply no additional frame scale,
3. reconstruct with RELION-style tau2/tau2-fudge/minres/grid-correction rules,
4. compare against RELION maps after converting RELION MRCs into RECOVAR frame.

For dense diagnostic mode only, apply the historical dense conversion:

```text
bp_data   *= -N^2
bp_weight *=  N^4
```

## K-class extension

The K-class replay is the same algorithm with a class dimension added to the
hidden state:

```text
gamma(i, class, rotation, translation)
```

Class-specific inputs:

- one previous mean/reference per class,
- one direction prior table per class,
- one tau2 spectrum per class,
- class priors/distributions from RELION model metadata.

The same convention checklist applies:

- RELION projector frame,
- centered rows,
- RELION image/noise frame,
- image-specific support,
- correct translation prior units,
- no adaptive-mode BPref rescale.

Class labels are permutation-invariant. Always compare maps and assignments
after selecting the best class permutation.

## Validation status

The small coherent InitialModel fixture now reaches near-perfect BPref parity:

- `h0 bp_data` CC about `+0.99965`
- `h1 bp_data` CC about `+0.99968`
- `h0 bp_weight` CC about `+0.99961`
- `h1 bp_weight` CC about `+0.99963`

The larger 5k / 256 M-step fixture also passed the targeted parity tests, with
key BPref correlations at `+1.000000` and final converter-chain relative error
around `1.6e-7`.

The K=2 5k / 256 Class3D replay reached:

- class assignment accuracy after permutation: `1.0000`
- map mean correlation: about `0.999989`
- Pmax mean absolute difference: about `2.7e-4`

## Debugging checklist for future parity gaps

When InitialModel or EM parity regresses, check these in order:

1. Are the dataset STAR, RELION data STAR, and internal dataset order coherent?
2. Are previous-model metadata values read from the correct iteration?
3. Is the path using adaptive support when comparing to RELION adaptive output?
4. Does the score path use RELION PPref/projector data or RECOVAR full Fourier
   projections?
5. Is `noise_scale` correct for the selected mode?
6. Is `image_correction = -1/N^2` applied in RELION-adaptive InitialModel?
7. Is the translation prior using `pixel_size^4 / sigma_offset_A^2` for this
   replay path?
8. Are RELION projector rows returned in centered-row layout?
9. Is BPref frame scaling disabled in adaptive mode and enabled only in dense
   diagnostic mode?
10. Are RELION maps converted with `load_relion_volume` before comparison?

If score parity is high but BPref parity is poor, suspect support membership,
posterior normalization, or accumulator frame conversion before suspecting the
projection operator.
