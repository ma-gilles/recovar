# RECOVAR Source Code Conventions

## ⚠ Critical conventions — always loaded, do NOT reinvent

Two convention pitfalls that have repeatedly cost days of debugging in
this repo:

### FFT / MRC volume I/O
Volumes live FLAT and in **CENTERED Fourier space** (DC at array center).
Real-space volumes are **CENTERED** (origin at `[N/2, N/2, N/2]`).

Use these helpers in `recovar.utils.helpers` and `recovar.core.fourier_transform_utils`:
- `load_mrc(path)` / `write_mrc(path, vol)` — recovar/cryoSPARC/cryoDRGN frame
- `load_relion_volume(path)` — load a RELION MRC, convert to recovar frame
- `save_volume(flat_ft, path, ...)` — write a flat centered-FT volume
- `ftu.get_dft3(real)` / `ftu.get_idft3(ft)` — centered 3D FFT pair
- `ftu.get_dft2` / `ftu.get_idft2` — centered 2D FFT pair

NEVER write raw `np.fft.fftn(np.fft.ifftshift(...))` or raw
`mrcfile.open(...).data` for 3D volumes — these omit the
`(2, 1, 0)` axis transpose AND/OR an outer `fftshift`, silently
corrupting projections (DC reads as Nyquist, ~2400× amplitude error
at low frequencies).

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

`tests/unit/test_relion_volume_convention.py` pins all these helpers and
will fail loudly if any future PR removes one of them. **Do not skip
fixing the test if it breaks** — that test exists because the helpers
were silently deleted in commit 4703c634 (2026-04-01) and the
documentation in `recovar/em/CLAUDE.md` was left referencing them, which
cost ~25 wasted commits over the next year (see
`docs/relion_parity_commit_audit.md`).

## Subdirectory developer guides (loaded lazily on file read)

The recovar repo has module-specific CLAUDE.md files that load only
when Claude reads a file in the corresponding subtree. Read them
explicitly if a task spans multiple modules without touching their
files:
- `recovar/em/CLAUDE.md` — EM module: RELION-parity plan, engine
  performance, more on the volume conventions, test rules
- `recovar/cuda/CLAUDE.md` — CUDA kernel coordinate convention (k0=row,
  k1=col), build via pixi, JAX FFI headers
- `recovar/gui_v2/CLAUDE.md` — GUI v2 architecture
- `tests/CLAUDE.md` — test conventions, baseline rules

## JAX / Equinox Patterns

### Static vs Dynamic
- `ForwardModelConfig`, `CTFEvaluator`, `ModelState`, `EmbeddingOpts`, `CovarianceOpts` are **Equinox modules** (static, immutable). Changing any field triggers JIT recompilation.
- Image data, poses, CTF parameters are **dynamic** JAX arrays passed as function arguments.
- Rule: if it doesn't change between batches, put it in a config struct.

### Float64
JAX is configured with `jax_enable_x64 = True` globally (`jax_config.py`). Float64 is required for numerical stability in covariance estimation and eigendecomposition. Do not disable this.

### Half-Spectrum Layout
Images and volumes can use rfft-packed layouts for ~50% memory savings. The `slicing.py` and `forward.py` modules handle both full and half layouts. When adding new operations, check whether inputs are half or full spectrum.

## Module Boundaries

- **`core/`** — Low-level JAX ops. No knowledge of datasets, pipelines, or file I/O. Everything here is JIT-compiled.
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
for batch in dataset.iter_batches(batch_size):
    images, metadata_batch = batch
    # metadata_batch has .rotation_matrices, .ctf_params, .translations, etc.
```

### Forward model
```python
config = ForwardModelConfig(image_shape=..., volume_shape=..., ctf=ctf_eval, ...)
projected = forward_model(config, volume, batch_data)        # volume → images
backprojected = adjoint_forward_model(config, images, batch_data)  # images → volume
```
