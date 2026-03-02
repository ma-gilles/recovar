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
pip install "jax[cuda12]"==0.9.0.1
```

On clusters, you may need to load CUDA modules first:

```bash
module load cudatoolkit/12.3
```

### Out of GPU memory

Try these in order:

1. **Downsample**: `--downsample 128` reduces memory by ~4x vs 256
2. **Limit memory**: `--gpu-gb 8` to control allocation
3. **Low memory mode**: `--low-memory-option` or `--very-low-memory-option`
4. **Lazy loading**: `--lazy` to avoid loading full dataset into RAM
5. **Fewer images**: `--n-images 50000` for initial exploration

### JAX pre-allocation

By default, JAX pre-allocates most GPU memory. On shared machines:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false recovar pipeline ...
```

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

### scikit-fmm build failure

Install from source:

```bash
pip install git+https://github.com/scikit-fmm/scikit-fmm.git
```

### JAX version conflicts

RECOVAR requires JAX 0.4.23+. Pin the version:

```bash
pip install "jax[cuda12]"==0.9.0.1
```

### Multiple recovar installations

If you have multiple editable installs, the wrong one may be imported:

```bash
# Check which recovar is active
python -c "import recovar; print(recovar.__file__)"

# Force isolation
export PYTHONNOUSERSITE=1
```
