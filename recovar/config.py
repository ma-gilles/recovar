import logging
import logging.config

logging.config.dictConfig(
    {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "(%(levelname)s) (%(filename)s) (%(asctime)s) %(message)s",
                "datefmt": "%d-%b-%y %H:%M:%S",
            }
        },
        "handlers": {
            "default": {
                "level": "NOTSET",
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {"": {"handlers": ["default"], "level": "INFO"}},
    }
)

logger = logging.getLogger(__name__)
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
from jax.config import config
config.update("jax_enable_x64", True)
import jax
# Interestingly, nothing works if I don't do this print statement :)))) 
# Something weird with JAX not finding devices?
logger.info(f"Devices found: {','.join([l.device_kind for l in jax.devices()])}")