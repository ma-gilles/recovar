import os
import pytest

import recovar.constants as constants
pytest.importorskip("jax")
import recovar.config as config

pytestmark = pytest.mark.unit


def test_constants_sanity():
    assert constants.EPSILON > 0
    assert constants.ROOT_EPSILON > 0
    assert constants.REG_INIT_MULTIPLIER > 0
    assert constants.FSC_ZERO_THRESHOLD > 0


def test_config_side_effects_and_import():
    # config import sets these once; verify expected configuration side effects exist.
    assert os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") == ".90"
    assert hasattr(config, "logger")
