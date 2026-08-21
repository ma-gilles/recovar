import os
import subprocess
import sys

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
    # jax_config import sets this once, unless the parent shell made an
    # explicit choice before import.
    assert os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") is not None
    assert hasattr(jax_config, "logger")


def test_config_clean_process_gets_default_mem_fraction():
    env = dict(os.environ)
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import recovar.jax_config; print(os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ".90"
