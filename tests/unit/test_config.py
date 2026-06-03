import os
import subprocess
import sys

import pytest

pytest.importorskip("jax")
import jax

from recovar import jax_config

pytestmark = pytest.mark.unit


def test_config_sets_expected_jax_defaults():
    # Import side-effects in recovar.jax_config should set these defaults,
    # but must not clobber an explicit parent shell override.
    assert os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") is not None
    assert jax.config.read("jax_enable_x64") is True
    assert jax_config is not None


def test_config_sets_mem_fraction_default_in_clean_process():
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
