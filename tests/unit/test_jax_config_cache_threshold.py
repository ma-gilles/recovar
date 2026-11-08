"""The persistent JAX compilation cache: its compile-time threshold and when a default directory is chosen."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env_after_import(extra_env, name, home):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("JAX_") and k not in ("RECOVAR_JAX_CACHE_DIR", "RECOVAR_EM_XLA_DEFAULTS", "RECOVAR_DISABLE_JAX_CACHE")
    }
    env.update({"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT), "HOME": str(home)})
    env.update(extra_env)
    code = f"import recovar.jax_config, os; print(os.environ.get({name!r}, 'unset'))"
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def _threshold_with_env(extra_env, home):
    return _env_after_import(extra_env, "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", home)


def test_threshold_applies_with_explicit_cache_dir(tmp_path):
    assert _threshold_with_env({"JAX_COMPILATION_CACHE_DIR": str(tmp_path)}, tmp_path) == "0.01"


def test_threshold_applies_with_recovar_default_dir(tmp_path):
    assert _threshold_with_env({"RECOVAR_JAX_CACHE_DIR": str(tmp_path)}, tmp_path) == "0.01"


def test_explicit_threshold_is_preserved(tmp_path):
    env = {"JAX_COMPILATION_CACHE_DIR": str(tmp_path), "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0"}
    assert _threshold_with_env(env, tmp_path) == "0"


def test_disabled_cache_sets_no_threshold(tmp_path):
    assert _threshold_with_env({"RECOVAR_DISABLE_JAX_CACHE": "1", "RECOVAR_EM_XLA_DEFAULTS": "1"}, tmp_path) == "unset"


def test_plain_import_chooses_no_cache_directory(tmp_path):
    assert _env_after_import({}, "JAX_COMPILATION_CACHE_DIR", tmp_path) == "unset"
    assert _threshold_with_env({}, tmp_path) == "unset"
    assert not (tmp_path / ".cache").exists()


def test_em_entry_point_gets_the_default_directory(tmp_path):
    directory = _env_after_import({"RECOVAR_EM_XLA_DEFAULTS": "1"}, "JAX_COMPILATION_CACHE_DIR", tmp_path)
    assert directory == str(tmp_path / ".cache" / "recovar" / "jax_compile")
    assert _threshold_with_env({"RECOVAR_EM_XLA_DEFAULTS": "1"}, tmp_path) == "0.01"


def test_em_marker_off_chooses_no_cache_directory(tmp_path):
    assert _env_after_import({"RECOVAR_EM_XLA_DEFAULTS": "0"}, "JAX_COMPILATION_CACHE_DIR", tmp_path) == "unset"
