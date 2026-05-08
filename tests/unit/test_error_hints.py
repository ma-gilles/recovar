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
    assert "--gpu-gb" in blob
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
    assert "--gpu-gb" not in blob
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


def test_oom_hint_surfaces_cap_not_applied():
    """When the user passed --gpu-gb but the bootstrap couldn't apply
    the cap (no nvidia-smi), the OOM hint must say so explicitly.
    Without this, the user sees `--gpu-gb 24` and an OOM that exceeded
    24 GB and assumes the cap is broken when in fact the cap was
    silently skipped."""
    from recovar.utils import error_hints

    ctx = error_hints.DiagnosticContext(
        backend="custom_cuda",
        custom_cuda_disabled=False,
        last_memory_plan={
            "budget": {
                "requested_gb": 24.0,
                "effective_budget_gb": 24.0,
                "hard_cap_applied": False,
                "hard_cap_skip_reason": "no_nvml_or_nvidia_smi",
            }
        },
    )
    hint = error_hints.classify_text("RESOURCE_EXHAUSTED: out of memory", ctx)
    assert hint is not None
    assert hint.category == "gpu_oom"
    blob = hint.likely_cause + " ".join(hint.suggestions)
    assert "cap was NOT applied" in hint.likely_cause or "NOT applied" in hint.likely_cause
    assert "no_nvml_or_nvidia_smi" in hint.likely_cause
    assert any("nvidia-smi" in s or "pynvml" in s for s in hint.suggestions)


def test_oom_hint_does_not_mention_cap_skip_when_applied():
    """If the cap fired, the OOM hint should NOT mention skip reasons."""
    from recovar.utils import error_hints

    ctx = error_hints.DiagnosticContext(
        backend="custom_cuda",
        last_memory_plan={
            "budget": {
                "requested_gb": 24.0,
                "hard_cap_applied": True,
                "hard_cap_skip_reason": None,
            }
        },
    )
    hint = error_hints.classify_text("RESOURCE_EXHAUSTED", ctx)
    assert hint is not None
    assert "NOT applied" not in hint.likely_cause
    assert not any("nvidia-smi" in s for s in hint.suggestions)
