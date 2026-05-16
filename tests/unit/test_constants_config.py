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


def test_config_side_effects_and_import(monkeypatch):
    # ``os.environ.setdefault`` in jax_config only fires when the env var is
    # unset on import. Reload with the var cleared to verify the default
    # propagates — robust against shells/CI envs that pre-set
    # XLA_PYTHON_CLIENT_MEM_FRACTION to a smaller fraction.
    import importlib

    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    importlib.reload(jax_config)
    assert os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") == ".90"
    assert hasattr(jax_config, "logger")
