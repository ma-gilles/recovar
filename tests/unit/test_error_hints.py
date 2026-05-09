"""Unit tests for the error_hints classifier."""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.unit]


def _ctx_with(monkeypatch, **overrides):
    """Return a DiagnosticContext patched with whatever the test cares about."""
    from recovar.utils import error_hints

    base = error_hints.DiagnosticContext()
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_oom_hint_suggests_recovery_flags():
    from recovar.utils import error_hints

    text = "XlaRuntimeError: RESOURCE_EXHAUSTED: failed to allocate 12.34 GiB"
    hint = error_hints.classify_text(text, error_hints.DiagnosticContext())
    assert hint is not None
    assert hint.category == "gpu_oom"
    blob = "\n".join(hint.suggestions)
    assert "--gpu-budget-gb" in blob
    assert "--adaptive-n-pcs" in blob
    assert "--low-memory-option" in blob
    assert "--very-low-memory-option" in blob


def test_oom_under_disable_cuda_mentions_jax_fallback(monkeypatch):
    from recovar.utils import error_hints

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    ctx = error_hints.DiagnosticContext(
        backend="jax_fallback",
        custom_cuda_disabled=True,
    )
    hint = error_hints.classify_text("CUDA_ERROR_OUT_OF_MEMORY", ctx)
    assert hint is not None
    text = " ".join(hint.suggestions) + " " + hint.likely_cause
    assert "RECOVAR_DISABLE_CUDA" in text
    assert "build_custom_cuda" in text


def test_conflicting_process_when_free_is_tiny(monkeypatch):
    from recovar.utils import error_hints

    ctx = error_hints.DiagnosticContext(
        physical_total_gb=80.0,
        physical_free_gb=10.0,
        physical_processes=[{"pid": 99999, "name": "other.py", "used_mb": 60_000}],
        last_memory_plan={"budget": {"requested_gb": 70.0}},
    )
    hint = error_hints.classify_text(
        "some unrelated runtime error",  # not OOM, but conflict is obvious
        ctx,
    )
    assert hint is not None
    assert hint.category == "conflicting_gpu_process"
    blob = " ".join(hint.suggestions)
    assert "CUDA_VISIBLE_DEVICES" in blob


def test_custom_cuda_unavailable():
    from recovar.utils import error_hints

    text = (
        "RuntimeError: RECOVAR's preferred custom CUDA backproject/project "
        "extension is unavailable. Last error: cannot find libcuda_backproject.so"
    )
    hint = error_hints.classify_text(text, error_hints.DiagnosticContext())
    assert hint is not None
    assert hint.category == "custom_cuda_unavailable"
    blob = " ".join(hint.suggestions)
    assert "build_custom_cuda" in blob


def test_cpu_fallback():
    from recovar.utils import error_hints

    text = "No GPU/TPU found, falling back to CPU."
    hint = error_hints.classify_text(text, error_hints.DiagnosticContext())
    assert hint is not None
    assert hint.category == "cpu_fallback_needed"


def test_dataset_path_error():
    from recovar.utils import error_hints

    exc = FileNotFoundError("foo.mrcs")
    ctx = error_hints.DiagnosticContext()
    hint = error_hints.classify_exception(exc, ctx)
    assert hint is not None
    assert hint.category == "dataset_path_error"
    # Must NOT suggest GPU memory flags for a path error.
    blob = "\n".join(hint.suggestions)
    assert "--gpu-budget-gb" not in blob
    assert "--adaptive-n-pcs" not in blob
    assert "check_paths" in blob


def test_unrecognized_returns_none_when_no_conflict():
    from recovar.utils import error_hints

    ctx = error_hints.DiagnosticContext()  # no physical info
    hint = error_hints.classify_text("totally unrelated random error string", ctx)
    assert hint is None


def test_format_hint_emits_delimiter_and_recover_section():
    from recovar.utils import error_hints

    hint = error_hints.classify_text("RESOURCE_EXHAUSTED: out of memory", error_hints.DiagnosticContext())
    out = error_hints.format_error_hint(hint)
    assert "═" * 10 in out
    assert "TO RECOVER" in out
    assert hint.summary in out


def test_write_hint_log(tmp_path):
    from recovar.utils import error_hints

    hint = error_hints.classify_text("RESOURCE_EXHAUSTED", error_hints.DiagnosticContext())
    p = error_hints.write_hint_log(hint, tmp_path)
    assert p.exists()
    payload = json.loads(p.read_text())
    assert payload["category"] == "gpu_oom"


def test_classify_subprocess_failure_uses_combined_text():
    from recovar.utils import error_hints

    stdout = "ok\nstill ok\n"
    stderr = "Error: CUDA_ERROR_OUT_OF_MEMORY at line 99"
    hint = error_hints.classify_subprocess_failure(stderr, stdout, error_hints.DiagnosticContext())
    assert hint is not None
    assert hint.category == "gpu_oom"


def test_oom_hint_includes_preallocate_workaround():
    """OOM hint should mention XLA_PYTHON_CLIENT_PREALLOCATE=false as a
    recovery for workstation / shared-GPU users where JAX's default
    preallocation is the actual cause of the OOM. Orthogonal to
    --gpu-budget-gb (which is RECOVAR's batch-size hint, not a JAX cap)."""
    from recovar.utils import error_hints

    hint = error_hints.classify_text("RESOURCE_EXHAUSTED: out of memory", error_hints.DiagnosticContext())
    assert hint is not None
    blob = " ".join(hint.suggestions)
    assert "XLA_PYTHON_CLIENT_PREALLOCATE" in blob
