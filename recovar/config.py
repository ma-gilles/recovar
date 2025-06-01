import logging
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
# import tensorflow as tf
# tf.config.experimental.set_visible_devices([], "GPU")
import jax
jax.config.update("jax_enable_x64", True)
logger = logging.getLogger(__name__)

try:
    devices = jax.devices()
    logger.info(f"Devices found: {','.join([d.device_kind for d in devices])}")
except RuntimeError as e:
    logger.warning("---------------------------------------------------")
    logger.warning("---------------------------------------------------")
    logger.warning("No JAX devices found! Falling back to CPU-only mode.")
    logger.warning("---------------------------------------------------")
    logger.warning("---------------------------------------------------")

