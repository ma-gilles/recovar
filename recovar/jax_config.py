"""JAX runtime configuration and global constants.

Initializes JAX with x64 precision, sets GPU memory fraction,
and defines numerical constants used throughout the codebase.
"""

import logging
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".90")

# Robustness: when XLA picks a Triton GEMM and its autotuner can't find a
# valid config (common on MIG slices, smaller GPUs, and uncommon einsum
# shapes), fall back to cuBLAS automatically instead of failing with
# "No valid config found". Triton GEMM is otherwise faster on full A100/
# H100, so we keep it enabled and only force the fallback.
#
# See:
#   https://github.com/NVIDIA/JAX-Toolbox/issues/317
#   https://github.com/google-deepmind/alphafold3/issues/240
#   https://docs.jax.dev/en/latest/gpu_performance_tips.html
#
# Triggered by recovar's compute_projected_covariance:
#   jit(_reduce_covariance_inner_explicit)/bpk,bpl->pkl/dot_general
# which produces a (P, n_pcs, n_pcs) float32 tensor (P = n_pcs(n_pcs+1)/2).
_existing_flags = os.environ.get("XLA_FLAGS", "")
if "xla_gpu_cublas_fallback" not in _existing_flags:
    fallback_flag = "--xla_gpu_cublas_fallback=true"
    os.environ["XLA_FLAGS"] = f"{_existing_flags} {fallback_flag}".strip() if _existing_flags else fallback_flag

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
