import os
import pytest

pytest.importorskip("jax")
import recovar.jax_config as jax_config

pytestmark = pytest.mark.unit


def test_constants_sanity():
    assert jax_config.EPSILON > 0
    assert jax_config.ROOT_EPSILON > 0
    assert jax_config.REG_INIT_MULTIPLIER > 0
    assert jax_config.FSC_ZERO_THRESHOLD > 0


def test_config_side_effects_and_import():
    # jax_config import sets these once; verify expected configuration side effects exist.
    assert os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") == ".90"
    assert hasattr(jax_config, "logger")
