"""End-to-end smoke tests for ``recovar run_test_dataset``.

Background — issue #131. A user on a Blackwell GPU was forced onto the
JAX-native fallback path (``RECOVAR_DISABLE_CUDA=1``) because his cached
``libcuda_backproject.so`` predated the Blackwell-arch fix. He then hit a
separate OOM in ``analyze`` because the heterogeneity-kernel auto-batch
formula was calibrated for the custom-CUDA path's memory budget.

These tests guard both paths end-to-end via the actual ``recovar
run_test_dataset`` wrapper (which does pipeline → analyze →
estimate_conformational_density) so a future regression in either:

  - the custom CUDA backproject kernel (e.g. wrong arch list), or
  - the JAX-native fallback path (e.g. heterogeneity batch overshoot),

would fail this test loudly instead of silently producing OOMs in
production for downstream users.

The DISABLE_CUDA case in particular would have caught issue #131 if it
existed earlier — it exercises exactly the code path that OOM'd Yann.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.io,
    pytest.mark.tiny_metrics,
]


def _run_recovar_test_dataset(out_dir, env_overrides):
    """Invoke `recovar run_test_dataset` in a subprocess with the given env.

    Returns the completed process. Caller asserts on returncode and any
    fields it cares about.
    """
    from conftest import gpu_subprocess_env

    env = gpu_subprocess_env()
    env.update(env_overrides)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    cmd = [
        sys.executable,
        "-m",
        "recovar.commands.run_test_dataset",
        "--output-dir",
        str(out_dir),
        "--no-delete",  # keep outputs for assertion + debugging
    ]

    return subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=20 * 60,  # 20 min cap; H100 typically takes ~6 min
    )


def _result_text(result):
    return result.stdout + result.stderr


def test_run_test_dataset_custom_cuda_path(tmp_path):
    """Default path: custom CUDA backproject kernel + JAX autotuner.

    This is the production code path users hit when their GPU is in the
    Makefile's ``CUDA_ARCH`` list (sm_70..sm_90 + sm_100/120 on nvcc>=12.8).
    Should complete pipeline + analyze + density estimation cleanly.
    """
    result = _run_recovar_test_dataset(tmp_path, env_overrides={})

    if result.returncode != 0:
        pytest.fail(
            "recovar run_test_dataset failed on the custom-CUDA path "
            f"(rc={result.returncode}).\n--- stdout (last 50 lines) ---\n"
            f"{chr(10).join(result.stdout.splitlines()[-50:])}\n"
            f"--- stderr (last 50 lines) ---\n"
            f"{chr(10).join(result.stderr.splitlines()[-50:])}"
        )

    # Check the wrapper announced full success.
    assert "All functions completed successfully!" in _result_text(result), (
        "wrapper did not report all functions as successful — check stdout"
    )


def test_run_test_dataset_jax_fallback_path(tmp_path):
    """Issue #131 regression: with RECOVAR_DISABLE_CUDA=1 forcing the JAX
    fallback, the heterogeneity-kernel auto-batch must be small enough to
    fit. Without the fix in kernel_regression_reconstruction.py (which
    scales the budget down 3x when DISABLE_CUDA is set), this OOMs on
    every GPU regardless of VRAM size.
    """
    env = {
        "RECOVAR_DISABLE_CUDA": "1",
        # JAX 0.9 autotuner has known issues on the fallback transpose path
        # for some GPUs; disable autotuning to keep this test stable across
        # GPU architectures.
        "XLA_FLAGS": "--xla_gpu_autotune_level=0",
    }
    result = _run_recovar_test_dataset(tmp_path, env_overrides=env)

    if result.returncode != 0:
        # Surface the heterogeneity-kernel batch line and the OOM (if any)
        # so the failure message points the next reader at the right place.
        lines = result.stdout.splitlines() + result.stderr.splitlines()
        het_lines = [ln for ln in lines if "batch size in heterogeneity kernel" in ln]
        oom_lines = [ln for ln in lines if "RESOURCE_EXHAUSTED" in ln or "ran out of memory" in ln]
        pytest.fail(
            "recovar run_test_dataset failed on the JAX-fallback path "
            f"(RECOVAR_DISABLE_CUDA=1, rc={result.returncode}). "
            "Issue #131 regression: the heterogeneity-kernel batch size is "
            "probably overshooting the JAX-fallback path's memory budget. "
            "Check the 3x scale-down logic in "
            "recovar/heterogeneity/kernel_regression_reconstruction.py.\n"
            f"--- batch-size lines ---\n{chr(10).join(het_lines)}\n"
            f"--- OOM lines ---\n{chr(10).join(oom_lines[:5])}\n"
            f"--- last 30 lines stdout ---\n"
            f"{chr(10).join(result.stdout.splitlines()[-30:])}\n"
            f"--- last 30 lines stderr ---\n"
            f"{chr(10).join(result.stderr.splitlines()[-30:])}"
        )

    assert "All functions completed successfully!" in _result_text(result), (
        "wrapper did not report all functions as successful on the JAX-fallback path"
    )

    # Verify the scale-down log line actually fired — otherwise the env-var
    # detection is broken and the test is passing for the wrong reason.
    assert "scaling heterogeneity-kernel memory budget" in _result_text(result), (
        "RECOVAR_DISABLE_CUDA=1 was set but the heterogeneity-kernel "
        "scale-down log line did not fire. Either the env-var check in "
        "kernel_regression_reconstruction.py is broken, or some upstream "
        "consumer bypassed the kernel entirely."
    )
