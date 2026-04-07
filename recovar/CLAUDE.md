# RECOVAR Source Code Conventions

## ⚠ READ FIRST: subdirectory developer guides

The recovar repo has critical conventions documented in subdirectory
`CLAUDE.md` files. Claude Code only loads them lazily (on file read), so
import them all at startup to make sure they're always in context:

@em/CLAUDE.md
@cuda/CLAUDE.md
@gui_v2/CLAUDE.md

(`tests/CLAUDE.md` is loaded by the parent project as well.)

The most commonly missed convention is the **RELION ↔ recovar volume axis
flip** (negate + transpose(2,1,0)) — see `recovar/em/CLAUDE.md`. The
canonical helpers are in `recovar/utils/helpers.py`:
- `load_mrc(path)` / `write_mrc(path, vol)` — cryosparc/cryoDRGN frame
- `load_relion_volume(path)` — converts on load to recovar's frame
- `relion_volume_to_recovar(vol)` / `recovar_volume_to_relion(vol)` — explicit conversion
- `R_to_relion(R)` / `R_from_relion(euler)` — rotation Euler conversion

`tests/unit/test_relion_volume_convention.py` pins these helpers so they
cannot be silently removed.

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
