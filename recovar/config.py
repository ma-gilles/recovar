import logging
# import logging.config
# logging.config.dictConfig(
#     {
#         "version": 1,
#         "formatters": {
#             "standard": {
#                 "format": "(%(levelname)s) (%(filename)s) (%(asctime)s) %(message)s",
#                 "datefmt": "%d-%b-%y %H:%M:%S",
#             }
#         },
#         "handlers": {
#             "default": {
#                 "level": "NOTSET",
#                 "formatter": "standard",
#                 "class": "logging.StreamHandler",
#                 "stream": "ext://sys.stdout",
#             }
#         },
#         "loggers": {"": {"handlers": ["default"], "level": "INFO"}},
#     }
# )

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".75"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")



import jax
jax.config.update("jax_enable_x64", True)
# Interestingly, nothing works if I don't do this print statement :)))) 
# Something weird with JAX not finding devices?
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

