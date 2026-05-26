"""JAX runtime configuration and global constants.

Initializes JAX with x64 precision, sets GPU memory fraction,
and defines numerical constants used throughout the codebase.
"""

import logging
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".90")

# Robustness: when XLA picks a Triton GEMM and its autotuner can't find
# a valid config (observed on MIG 3g.40gb slices for the
# (P=20100, n_pcs=200, n_pcs=200) float32 cross_terms einsum in
# compute_projected_covariance, slurm 8279410), the run errors out
# with "No valid config found!".
#
# ``--xla_gpu_cublas_fallback=true`` was tried (slurm 8287256) and did
# NOT recover — that flag only kicks in for *performance* fallback
# (cuBLAS when faster), not for *failure* fallback. To handle the
# autotuner failure we have to disable Triton GEMM entirely so XLA
# emits cuBLAS calls directly.
#
# Triton GEMM is typically 0-20% faster than cuBLAS for recovar's
# matmul shapes on full A100/H100; we trade that small win for
# cross-device robustness (MIG slices, smaller GPUs, exotic shapes).
# Recovar's primary bottleneck is FFTs and the custom CUDA backproject
# kernel, not the float32 gemm in proj-cov, so the perf cost is small.
#
# Users on full GPUs who want the Triton speed-up can re-enable it
# via ``XLA_FLAGS=--xla_gpu_enable_triton_gemm=true`` before launching.
#
# References:
#   https://github.com/NVIDIA/JAX-Toolbox/issues/317
#   https://github.com/google-deepmind/alphafold3/issues/240
#   https://docs.jax.dev/en/latest/gpu_performance_tips.html
_existing_flags = os.environ.get("XLA_FLAGS", "")
if "xla_gpu_enable_triton_gemm" not in _existing_flags:
    triton_off = "--xla_gpu_enable_triton_gemm=false"
    os.environ["XLA_FLAGS"] = f"{_existing_flags} {triton_off}".strip() if _existing_flags else triton_off

# Persistent JAX compilation cache. Set RECOVAR_JAX_CACHE_DIR or
# JAX_COMPILATION_CACHE_DIR before import to use a custom path. Disable with
# RECOVAR_DISABLE_JAX_CACHE=1. Cache contents are large XLA modules keyed by
# compute graph hash — safe to delete; recovar will rebuild on next run.
# Measured 2026-05-11: cut K=1 50k/256 4-iter wall from 171s → 75s (56%) on
# second run by skipping repeated iter-1 JIT compile.
_jax_cache_disabled = os.environ.get("RECOVAR_DISABLE_JAX_CACHE", "").lower() in {"1", "true", "yes", "on"}
if not _jax_cache_disabled and not os.environ.get("JAX_COMPILATION_CACHE_DIR"):
    _default_cache_dir = os.environ.get(
        "RECOVAR_JAX_CACHE_DIR",
        os.path.expanduser("~/.cache/recovar/jax_compile"),
    )
    os.environ["JAX_COMPILATION_CACHE_DIR"] = _default_cache_dir
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    try:
        os.makedirs(_default_cache_dir, exist_ok=True)
    except OSError:
        pass

import jax

jax.config.update("jax_enable_x64", True)
logger = logging.getLogger(__name__)

try:
    devices = jax.devices()
    logger.info("Devices found: %s", ",".join([d.device_kind for d in devices]))
except Exception as e:
    logger.warning("---------------------------------------------------")
    logger.warning("JAX device query failed: %s", e)
    logger.warning("Falling back to CPU-only mode.")
    logger.warning("---------------------------------------------------")


# CPU device helper — use instead of np.array(jax_arr) to stay in JAX
_CPU_DEVICE = jax.devices("cpu")[0]


## TODO remove? this should probably be elsewhere
def _to_cpu(x):
    """Transfer a JAX array to the CPU device, returning a JAX CPU array."""
    return jax.device_put(x, _CPU_DEVICE)


# Numerical constants
EPSILON = 1e-16
ROOT_EPSILON = 1e-8
REG_INIT_MULTIPLIER = 1e-2
FSC_ZERO_THRESHOLD = 0.001  # Values below this are considered zero
