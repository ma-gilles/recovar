# Troubleshooting

## Data loading errors

### "Cannot find MRC file(s)"

```
FileNotFoundError: Cannot find 15 MRC file(s) referenced in the metadata.
  First missing: /old/path/to/Micrographs/image.mrcs
```

RECOVAR automatically tries extension swaps (`.mrc` ↔ `.mrcs`) and flat-directory fallbacks before failing. If it still can't find files:

1. **Diagnose** with `recovar check_paths` to see exactly what paths are tried:

    ```bash
    recovar check_paths particles.cs --datadir /your/data
    ```

2. **Fix** with `--datadir` and/or `--strip-prefix`:

    ```bash
    recovar pipeline particles.star -o output --mask mask.mrc \
        --datadir /correct/path/to/data \
        --strip-prefix old/path/prefix
    ```

See [Fixing broken paths](guide/input-data.md#fixing-broken-file-paths).

!!! tip
    The [web GUI](guide/gui.md) validates input files when you select them — it checks that paths resolve and shows the particle count before you submit a job.

### "CS file has no alignments3D/pose field"

Your `.cs` file doesn't contain pose information (e.g., it's a passthrough file or import job). Use the `*_particles.cs` file from a refinement job, not a passthrough or import file.

### "Must provide --poses and --ctf for .mrcs input"

When using `.mrcs` files directly, you must provide pose and CTF pickle files:

```bash
recovar pipeline particles.mrcs -o output \
    --poses poses.pkl --ctf ctf.pkl --mask mask.mrc
```

Use `.star` or `.cs` files instead for automatic extraction.

## GPU and memory

### "No GPU found"

```
ValueError: No GPU found. Set --accept-cpu if you really want to run on CPU
```

JAX can't see your GPU. Check:

```python
import jax
print(jax.devices())
```

If only CPU shows up, reinstall JAX with CUDA:

```bash
pip install "recovar[gpu]"
```

On clusters, you may need to load CUDA modules first:

```bash
module load cudatoolkit/12.3
```

### Out of GPU memory

Try these in order:

1. **Downsample**: `--downsample 128` reduces memory by ~4x vs 256
2. **Plan for less GPU memory**: `--gpu-budget-gb 8` to shrink RECOVAR batches
3. **Low memory mode**: `--low-memory-option` or `--very-low-memory-option`
4. **Lazy loading**: `--lazy` to avoid loading full dataset into RAM
5. **Fewer images**: `--n-images 50000` for initial exploration

### JAX pre-allocation

By default, JAX pre-allocates most GPU memory. On shared machines:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false recovar pipeline ...
```

### Custom CUDA extension is unavailable or `nvcc` is missing

RECOVAR prefers its custom CUDA backproject/project extension on GPU because it
is substantially faster than the pure JAX fallback. Installing `recovar[gpu]`
or `.[gpu]` gives you CUDA-enabled JAX wheels, but a working JAX GPU install
alone is not enough to build that extension.

If RECOVAR stops with a custom CUDA build/load error, make sure a local CUDA
toolkit/compiler is available through one of these mechanisms:

- `NVCC=/full/path/to/nvcc`
- `CUDACXX=/full/path/to/nvcc`
- `nvcc` available on `PATH`
- `LOCAL_CUDA_PATH`, `CUDA_HOME`, or `CUDA_PATH` pointing at a toolkit root

Then either rerun RECOVAR so it can auto-build the shared library, or build it
explicitly first:

```bash
recovar build_custom_cuda
recovar ...
```

If you need to get unblocked temporarily, force the slower JAX GPU path:

```bash
RECOVAR_DISABLE_CUDA=1 recovar ...
```

That workaround is supported, but it is not the preferred configuration.

### `JaxRuntimeError: INTERNAL: Autotuning failed for HLO ... NOT_FOUND: No valid config found!`

This is XLA's GPU autotuner — the part of JAX that picks an optimal CUDA
kernel at JIT time — failing because it has no tuning data for your GPU.
Symptoms include the message above, sometimes preceded by repeated
`Allocator (GPU_X_bfc) ran out of memory` warnings (those are autotuner
candidate-kernel probes failing, not real OOM).

This happens on **GPUs newer than the JAX version was tuned for** —
most often Blackwell (sm_100, sm_120: B100/B200, RTX 50-series, RTX
PRO Blackwell) on JAX versions cut before Blackwell support landed.

Workaround — disable autotuning so XLA falls back to default heuristic
kernel selection:

```bash
export XLA_FLAGS="--xla_gpu_autotune_level=0"
recovar ...
```

You can stack this with `RECOVAR_DISABLE_CUDA=1` if you also need to
avoid recovar's custom kernel:

```bash
export RECOVAR_DISABLE_CUDA=1
export XLA_FLAGS="--xla_gpu_autotune_level=0"
recovar ...
```

If `--xla_gpu_autotune_level=0` doesn't help, two more knobs to try:

```bash
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
# or
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer="
```

If none of these work, your GPU is past what your JAX version's XLA can
lower at all. The fix is upgrading JAX once a release with tuning data
for your hardware is available — there is no recovar-side workaround.
This is a JAX/XLA limitation, not a recovar bug.

## Pipeline issues

### Mean looks wrong

1. Check your mask isn't inverted or too tight: `--mask-dilate-iter 10`
2. Run with `--only-mean` first to quickly verify setup
3. Try `--mask=sphere` to rule out mask issues
4. Check that poses are from a good consensus refinement
5. Use the GUI's slice viewer (`recovar gui`) to inspect the mean volume and mask side-by-side

### Results differ between runs

This can happen if:

- Half-set splits differ (use `--halfsets` to fix a specific split)
- Image ordering changed (use `--ind` to fix image selection)
- JAX non-determinism on different hardware

### Pipeline is slow

1. **Downsample**: `--downsample 128` is the biggest speedup
2. **Multi-GPU**: `--multi-gpu` for parallel processing
3. **Fewer PCs**: `--zdim=4,10` instead of `1,2,4,10,20`
4. **Skip analysis steps**: Use `--only-mean` for quick checks

## Analysis issues

### UMAP is slow

For large datasets (>200k particles), UMAP can take a long time:

```bash
recovar analyze output --zdim=10 --skip-umap
```

You can run UMAP separately later on a subset of particles.

### Density estimation runtime

Runtime scales as O(N^pca_dim). Keep `--pca_dim` at 4 or below:

```bash
recovar estimate_conformational_density output --pca_dim 3
```

Check `all_densities.png` and `Lcurve.png` to verify the optimal regularization was selected correctly.

## Installation issues

### Native fast-marching extension build failure

The in-tree C++ fast-marching extension is optional. Published Linux and macOS
wheels include it on supported builds, while source installs compile it locally
when a C++ toolchain is available. If that build fails, installation still
succeeds and RECOVAR uses the pure-Python fallback.

If you want the native backend, install a working C++ toolchain and then
reinstall RECOVAR.

### JAX version conflicts

RECOVAR requires JAX 0.9.0.1. Pin the version:

```bash
pip install "recovar[gpu]"
```

### Multiple recovar installations

If you have multiple editable installs, the wrong one may be imported:

```bash
# Check which recovar is active
python -c "import recovar; print(recovar.__file__)"

# Force isolation
export PYTHONNOUSERSITE=1
```
