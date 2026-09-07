# RECOVAR Source Code Conventions

## Fourier and volume conventions

Use the existing transform and volume-I/O helpers. Centering, axis order and
sign are part of the scientific contract, not interchangeable file conventions.

### FFT / MRC volume I/O
Volumes live FLAT and in **CENTERED Fourier space** (DC at array center).
Real-space volumes are **CENTERED** (origin at `[N/2, N/2, N/2]`).

Use these helpers in `recovar.utils.helpers` and `recovar.core.fourier_transform_utils`:
- `load_mrc(path)` / `write_mrc(path, vol)` — recovar/cryoSPARC/cryoDRGN frame
- `load_relion_volume(path)` — load a RELION MRC, convert to recovar frame
- `save_volume(flat_ft, path, ...)` — write a flat centered-FT volume
- `ftu.get_dft3(real)` / `ftu.get_idft3(ft)` — centered 3D FFT pair
- `ftu.get_dft2` / `ftu.get_idft2` — centered 2D FFT pair

Do not replace these helpers with raw `np.fft.fftn(np.fft.ifftshift(...))`
or `mrcfile.open(...).data` for 3D volumes. Those expressions do not implement
the required axis/frame conversion and centered-transform contract.

### RELION ↔ recovar volume axis flip
recovar and RELION use different real-space axis conventions:
```python
vol_recovar = -np.transpose(vol_relion, (2, 1, 0))   # negate + swap X<->Z
```
The negation is paired with `R_to_relion` / `R_from_relion`; both are
correct as written. Do NOT "fix" them.

When loading a **RELION-produced** MRC for FSC against a recovar
reconstruction, use `load_relion_volume(path)`, NOT `load_mrc(path)` —
the latter is for recovar/cryoSPARC frame and leaves RELION volumes in
the wrong frame, producing FSC ≈ 0 against the matching recovar volume.

[Volume-convention tests](../tests/unit/test_relion_volume_convention.py)
check these conversions. Run them when changing the helpers or their callers;
resolve failures before interpreting map comparisons.

## Scoped developer guides

Read the applicable guides explicitly before working across these boundaries.
Do not assume a particular agent automatically loaded a guide:
- `recovar/em/CLAUDE.md` — EM module: RELION-parity plan, engine
  performance, more on the volume conventions, test rules
- `recovar/cuda/CLAUDE.md` — CUDA kernel coordinate convention (k0=row,
  k1=col), build via pixi, JAX FFI headers
- `recovar/gui_v2/CLAUDE.md` — GUI v2 architecture
- `tests/CLAUDE.md` — test conventions, baseline rules

## JAX / Equinox Patterns

### Static vs Dynamic
- `ForwardModelConfig` and option modules declare compile-time fields with
  `eqx.field(static=True)`. Changing static configuration can trigger recompilation.
- `ModelState` contains dynamic array leaves (mean, mask, basis, eigenvalues);
  being an Equinox module does not make every field static.
- Keep image data, poses, CTF parameters and evolving numerical state dynamic.
  Put compile-time choices in configuration; inspect the actual JIT boundary
  before changing static/dynamic ownership.

### Float64
JAX is configured with `jax_enable_x64 = True` globally (`jax_config.py`). Float64 is required for numerical stability in covariance estimation and eigendecomposition. Do not disable this.

### Half-Spectrum Layout
Images and volumes can use rfft-packed layouts for ~50% memory savings. The `slicing.py` and `forward.py` modules handle both full and half layouts. When adding new operations, check whether inputs are half or full spectrum.

## Module Boundaries

- **`core/`** — Low-level JAX ops. Keep numerical kernels independent of pipeline orchestration. This package contains JIT kernels and host-side geometry/configuration helpers; preserve their actual compilation boundaries.
- **`data_io/`** — File formats, loading, indexing. `CryoEMDataset` is the single entry point for all downstream code. Never bypass it to load data directly.
- **`heterogeneity/`** — The science. Covariance estimation, PCA, embedding, volume generation. Operates on batches from `CryoEMDataset`.
- **`reconstruction/`** — Classical 3D reconstruction (mean, noise, regularization). Used by `heterogeneity/` and `commands/`.
- **`commands/`** — CLI entry points. Each is a standalone argparse module. `pipeline.py` orchestrates the full workflow.
- **`output/`** — Results serialization. `PipelineOutput` for reading results. `ResultPaths` for output directory structure.

## Numerical Stability

### Covariance estimation
The covariance estimation in `covariance_estimation.py` uses half-set cross-validation to remove noise bias. The RHS involves subtraction of nearly-equal large numbers (outer products minus regularized terms). Small floating-point errors get amplified ~1e6x by this cancellation. Changes to this code require careful validation against baselines.

### Volume normalization
When comparing eigenvectors between pipeline output and ground truth:
```python
vol_norm = np.sqrt(np.prod(volume_shape))
u_est = load_u_real_for_metrics(po, n_pcs)  # real-space, properly normalized
u_est = np.array(u_est.reshape(n_pcs, -1)).T * vol_norm
```
Never load eigenvectors from MRC files directly — wrong normalization/space.

### Noise shells
Pipeline stores `grid_size//2 - 1` shells (63 for 128^3). Ground truth has `grid_size - 1` shells (127). Always compare only the first `grid_size//2 - 1` shells.

## Key Patterns

### Batch iteration
```python
for (
    images, rotation_matrices, translations, ctf_params,
    noise_variance, particle_indices, image_indices,
) in dataset.iter_batches(batch_size, by_image=True):
    original_ids = dataset.original_image_indices_from_local(image_indices)
    ...
```

`CryoEMDataset.iter_batches` yields seven explicit fields. Image indices are
local to the dataset; use its mapper when a diagnostic needs original image
identity. `by_image=False` selects particle-grouped iteration for tilt series;
inspect the caller's group and noise-indexing requirements before changing it.

### Forward model
```python
from recovar.core.configs import ForwardModelConfig
from recovar.core.forward import forward_model, adjoint_forward_model

config = ForwardModelConfig.from_dataset(dataset, disc_type="linear_interp")
projected_ft = forward_model(config, volume_ft, ctf_params, rotation_matrices)
backprojected_ft = adjoint_forward_model(
    config, projected_ft, ctf_params, rotation_matrices,
)
```

These default calls use full, centered Fourier arrays and apply the CTF at the
supplied rotations. Image preprocessing and translation correction belong to
the calling workflow. Inspect `half_image` and `half_volume` when using packed
layouts; do not pass raw real-space images as Fourier slices.
