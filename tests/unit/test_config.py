import os

import pytest

pytest.importorskip("jax")
import jax

from recovar import jax_config

pytestmark = pytest.mark.unit


def test_config_sets_expected_jax_defaults():
    # Import side-effects in recovar.jax_config should set these defaults.
    assert os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") == ".90"
    assert jax.config.read("jax_enable_x64") is True
    assert jax_config is not None
