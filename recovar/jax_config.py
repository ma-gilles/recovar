"""JAX runtime configuration and global constants.

Initializes JAX with x64 precision, sets GPU memory fraction,
and defines numerical constants used throughout the codebase.
"""

import logging
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".90")

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
