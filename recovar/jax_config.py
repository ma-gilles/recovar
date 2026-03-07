"""JAX runtime configuration and global constants.

Initializes JAX with x64 precision, sets GPU memory fraction,
and defines numerical constants used throughout the codebase.
"""
import logging
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".90")
import jax
jax.config.update("jax_enable_x64", True)
logger = logging.getLogger(__name__)

try:
    devices = jax.devices()
    logger.info("Devices found: %s", ','.join([d.device_kind for d in devices]))
except Exception as e:
    logger.warning("---------------------------------------------------")
    logger.warning("JAX device query failed: %s", e)
    logger.warning("Falling back to CPU-only mode.")
    logger.warning("---------------------------------------------------")


# CPU device helper — use instead of np.array(jax_arr) to stay in JAX
_CPU_DEVICE = jax.devices('cpu')[0]

## TODO remove? this should probably be elsewhere
def _to_cpu(x):
    """Transfer a JAX array to the CPU device, returning a JAX CPU array."""
    return jax.device_put(x, _CPU_DEVICE)


# Numerical constants
EPSILON = 1e-16
ROOT_EPSILON = 1e-8
REG_INIT_MULTIPLIER = 1e-2
FSC_ZERO_THRESHOLD = 0.001  # Values below this are considered zero
