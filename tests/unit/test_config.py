import importlib
import os

import pytest

pytest.importorskip("jax")
import jax

from recovar import jax_config

pytestmark = pytest.mark.unit


def test_config_sets_expected_jax_defaults(monkeypatch):
    # Import side-effects in recovar.jax_config use ``os.environ.setdefault``,
    # so they only fire when the env var is unset on import. Reload with the
    # var cleared to verify the default propagates as expected. This makes the
    # test robust against shells/CI envs that pre-set XLA_PYTHON_CLIENT_MEM_FRACTION
    # (e.g. via an sbatch wrapper that wanted a smaller fraction for a slice GPU).
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    importlib.reload(jax_config)
    assert os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") == ".90"
    assert jax.config.read("jax_enable_x64") is True
    assert jax_config is not None
