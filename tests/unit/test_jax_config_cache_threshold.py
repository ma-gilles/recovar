"""The persistent-cache compile-time threshold must apply when the caller sets the cache directory."""

import os
import subprocess

from conftest import repo_python_command, repo_subprocess_env


def _threshold_with_env(extra_env):
    env = {k: v for k, v in os.environ.items() if not k.startswith("JAX_") and k != "RECOVAR_JAX_CACHE_DIR"}
    env.update({"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""})
    env.update(extra_env)
    code = "import recovar.jax_config, os; print(os.environ.get('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', 'unset'))"
    out = subprocess.run(
        repo_python_command("-c", code), env=repo_subprocess_env(env), capture_output=True, text=True, check=True
    )
    return out.stdout.strip().splitlines()[-1]


def test_threshold_applies_with_explicit_cache_dir(tmp_path):
    assert _threshold_with_env({"JAX_COMPILATION_CACHE_DIR": str(tmp_path)}) == "0.01"


def test_threshold_applies_with_recovar_default_dir(tmp_path):
    assert _threshold_with_env({"RECOVAR_JAX_CACHE_DIR": str(tmp_path)}) == "0.01"


def test_explicit_threshold_is_preserved(tmp_path):
    assert _threshold_with_env({"JAX_COMPILATION_CACHE_DIR": str(tmp_path), "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0"}) == "0"


def test_disabled_cache_sets_no_threshold():
    assert _threshold_with_env({"RECOVAR_DISABLE_JAX_CACHE": "1"}) == "unset"
