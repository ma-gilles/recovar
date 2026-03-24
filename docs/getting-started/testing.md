# Test Your Installation

## Quick test

Run the built-in test on a GPU-capable machine:

```bash
recovar run_test_dataset
```

This generates a small synthetic dataset and runs the full pipeline. It should complete in a few minutes and confirms that JAX, CUDA, and all dependencies are working.

## Development test suite

If you installed from source, run the test suite:

```bash
# Fast unit tests (no GPU required, ~30 seconds)
./scripts/run_pytests.sh fast

# Pipeline smoke test (CPU only, ~2 min)
./scripts/run_pytests.sh smoke

# GPU tests (requires GPU)
./scripts/run_pytests.sh gpu

# Full suite (unit + integration + GPU + slow, ~1-2h)
./scripts/run_pytests.sh full

# Quick quality check (small synthetic data, ~30 min, GPU)
./scripts/run_pytests.sh tiny-metrics

# Full quality regression (synthetic data, 6-12h, GPU)
./scripts/run_pytests.sh long-test
```

Or via pixi:

```bash
pixi run test-fast          # unit tests only
pixi run test-full          # full suite
pixi run test-parallel      # parallel Slurm submission
pixi run test-parallel-long # parallel including long regression
```

All test data is generated synthetically — no external downloads required.

See `tests/README.md` for test layout, markers, baselines, and guidelines.

## Common issues

### JAX not finding GPU

```
No GPU found. Set --accept-cpu if you really want to run on CPU
```

Check that JAX can see your GPU:

```python
import jax
print(jax.devices())  # Should show CudaDevice
```

If it shows only CPU devices, reinstall JAX with CUDA support:

```bash
pip install "jax[cuda12]"==0.9.0.1
```

### CUDA driver version mismatch

If you get CUDA errors, ensure your driver version is compatible with the CUDA toolkit version used by JAX. Check with:

```bash
nvidia-smi  # Shows driver version
python -c "import jax; print(jax.lib.xla_bridge.get_backend().platform_version)"
```
