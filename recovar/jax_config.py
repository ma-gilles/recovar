"""JAX runtime configuration and global constants.

Initializes JAX with x64 precision, sets GPU memory fraction,
and defines numerical constants used throughout the codebase.
"""

import logging
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".90")

# Robustness: when XLA picks a Triton GEMM and its autotuner can't find a valid
# config, the run errors out with "No valid config found!". Observed on MIG
# slices for the float32 cross_terms einsum in compute_projected_covariance,
# and on box-256 kernel-regression deconvolution where the per-image reduce
# ``weights @ per_image`` is a (n_weight_sets, ~67M) GEMM that Triton rejects.
#
# ``--xla_gpu_cublas_fallback=true`` does NOT recover — that flag only kicks in
# for *performance* fallback (cuBLAS when faster), not *failure* fallback. To
# handle the autotuner failure we disable Triton GEMM entirely so XLA emits
# cuBLAS calls directly.
#
# Triton GEMM is typically 0-20% faster than cuBLAS for recovar's matmul shapes
# on full A100/H100; we trade that small win for cross-device robustness.
# Recovar's primary bottleneck is FFTs and the custom CUDA backproject kernel,
# not the float32 gemm, so the perf cost is small. Re-enable via
# ``XLA_FLAGS=--xla_gpu_enable_triton_gemm=true`` before launching.
#
# References:
#   https://github.com/NVIDIA/JAX-Toolbox/issues/317
#   https://docs.jax.dev/en/latest/gpu_performance_tips.html
_existing_flags = os.environ.get("XLA_FLAGS", "")
if "xla_gpu_enable_triton_gemm" not in _existing_flags:
    triton_off = "--xla_gpu_enable_triton_gemm=false"
    os.environ["XLA_FLAGS"] = f"{_existing_flags} {triton_off}".strip() if _existing_flags else triton_off

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
