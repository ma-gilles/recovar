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


@pytest.mark.parametrize(
    ("explicit_threshold", "expected_threshold"),
    [(None, "0.01"), ("0.25", "0.25")],
)
def test_config_sets_cache_threshold_without_clobbering_override(
    tmp_path, explicit_threshold, expected_threshold
):
    env = dict(os.environ)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    env.pop("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", None)
    env.pop("RECOVAR_DISABLE_JAX_CACHE", None)
    env["RECOVAR_JAX_CACHE_DIR"] = str(tmp_path / "jax-cache")
    if explicit_threshold is not None:
        env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = explicit_threshold

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import recovar.jax_config; "
            "print(os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'])",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == expected_threshold
