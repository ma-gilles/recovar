"""Backward-compatibility shim — constants now live in :mod:`recovar.jax_config`.

Import from ``recovar.jax_config`` directly in new code.
"""
from recovar.jax_config import EPSILON, ROOT_EPSILON, REG_INIT_MULTIPLIER, FSC_ZERO_THRESHOLD  # noqa: F401
