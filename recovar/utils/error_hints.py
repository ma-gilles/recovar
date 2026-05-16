"""Classify exceptions / subprocess failures into actionable hints.

Used in two places:

1. ``recovar/command_line.py`` wraps ``mod.main()`` and, on any
   non-``SystemExit`` exception, prints the captured traceback FIRST,
   then a structured hint via :func:`format_error_hint`. Order matters:
   the hint must be the LAST thing on stderr so the user's eye lands
   on actionable suggestions, not on XLA internals.

2. ``recovar/commands/run_test_dataset.py`` (and siblings) capture each
   subprocess's stderr/stdout and run them through
   :func:`classify_subprocess_failure` to re-emit the hint at the wrapper
   level when an inner pipeline call dies.

Categories targeted:

  * ``gpu_oom`` — JAX/XLA out-of-memory traces.
  * ``conflicting_gpu_process`` — physical free << requested budget,
    other compute apps visible.
  * ``custom_cuda_unavailable`` — RECOVAR's optional CUDA extension
    failed to load or build.
  * ``cpu_fallback_needed`` — no GPU detected and ``--accept-cpu`` not
    set.
  * ``dataset_path_error`` — ``FileNotFoundError`` / ``OSError`` on a
    dataset file.

The classifier is best-effort: when it cannot recognize the failure it
returns ``None`` and the wrapper just re-raises with the original
traceback, no hint added. We never swallow exceptions or alter exit
codes.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticContext:
    """Snapshot of runtime state used to enrich hints.

    All fields are optional — the classifier handles ``None`` gracefully
    so the hint can fire even when the planner never ran (early
    crashes).
    """

    backend: str | None = None
    custom_cuda_disabled: bool | None = None
    last_memory_plan: dict[str, Any] | None = None
    physical_total_gb: float | None = None
    physical_free_gb: float | None = None
    physical_processes: list[dict[str, Any]] = field(default_factory=list)
    env_recovar_disable_cuda: str | None = None
    env_recovar_cuda_disable_typo: str | None = None
    env_cuda_visible_devices: str | None = None
    last_trace_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ErrorHint:
    """Structured advice the wrapper formats and prints to stderr."""

    category: str
    summary: str
    likely_cause: str
    suggestions: list[str]
    diagnostic_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Context collection
# ---------------------------------------------------------------------------


def collect_context(*, last_memory_plan: dict[str, Any] | None = None) -> DiagnosticContext:
    ctx = DiagnosticContext()

    try:
        from recovar.utils import cuda_env

        ctx.backend = cuda_env.detect_backend()
        disabled, _ = cuda_env.custom_cuda_disabled_from_env()
        ctx.custom_cuda_disabled = disabled
    except Exception:
        pass

    ctx.env_recovar_disable_cuda = os.environ.get("RECOVAR_DISABLE_CUDA")
    ctx.env_recovar_cuda_disable_typo = os.environ.get("RECOVAR_CUDA_DISABLE")
    ctx.env_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    try:
        from recovar.utils.gpu_preflight import get_physical_gpu_memory_info

        info = get_physical_gpu_memory_info(0)
        if info is not None:
            ctx.physical_total_gb = info.total_gb
            ctx.physical_free_gb = info.free_gb
            ctx.physical_processes = [dataclasses.asdict(p) for p in info.processes]
    except Exception:
        pass

    if last_memory_plan is not None:
        ctx.last_memory_plan = last_memory_plan

    return ctx


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


_OOM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"RESOURCE_EXHAUSTED",
        r"CUDA_ERROR_OUT_OF_MEMORY",
        r"out of memory",
        r"CUBLAS_STATUS_ALLOC_FAILED",
        r"XlaRuntimeError.*[Aa]llocation",
        r"failed to allocate.*on device",
    )
]

_CUDA_BUILD_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"libcuda_backproject\.so",
        r"recovar build_custom_cuda",
        r"custom CUDA backproject",
        r"_cuda_unavailable_message",
    )
]

_CPU_FALLBACK_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"No GPU/TPU found",
        r"No GPU device found",
        r"No CUDA GPUs are available",
    )
]

_PATH_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\.star['\"]?\s*$",
        r"\.mrcs?['\"]?\s*$",
        r"\.cs['\"]?\s*$",
        r"_rlnImageName",
    )
]

_XLA_AUTOTUNER_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"Autotuning failed.*No valid config found",
        r"gemm_fusion_autotuner.*No valid config",
    )
]


def _looks_like_oom(text: str) -> bool:
    return any(p.search(text) for p in _OOM_PATTERNS)


def _looks_like_xla_autotuner(text: str) -> bool:
    return any(p.search(text) for p in _XLA_AUTOTUNER_PATTERNS)


def _looks_like_cuda_build(text: str) -> bool:
    return any(p.search(text) for p in _CUDA_BUILD_PATTERNS)


def _looks_like_cpu_fallback(text: str) -> bool:
    return any(p.search(text) for p in _CPU_FALLBACK_PATTERNS)


def _looks_like_path_error(text: str) -> bool:
    if "FileNotFoundError" in text or "No such file or directory" in text:
        return True
    return any(p.search(text) for p in _PATH_PATTERNS)


def _conflicting_process_present(ctx: DiagnosticContext, requested_gb: float | None) -> bool:
    """Heuristic: physical free much less than requested AND someone else is using the GPU."""
    if ctx.physical_total_gb is None or ctx.physical_free_gb is None:
        return False
    if requested_gb is not None and ctx.physical_free_gb >= requested_gb * 0.9:
        return False
    if ctx.physical_total_gb - ctx.physical_free_gb < ctx.physical_total_gb * 0.10:
        return False  # under 10% used: probably not a conflict, just preallocation
    # Filter out our own process
    own_pid = os.getpid()
    others = [p for p in ctx.physical_processes if p.get("pid") != own_pid]
    return len(others) > 0


# ---------------------------------------------------------------------------
# Hint constructors
# ---------------------------------------------------------------------------


def _format_memory_context(ctx: DiagnosticContext) -> dict[str, Any]:
    plan = ctx.last_memory_plan or {}
    budget = plan.get("budget", {}) if isinstance(plan, dict) else {}
    return {
        "backend": ctx.backend,
        "custom_cuda_disabled": ctx.custom_cuda_disabled,
        "requested_gpu_gb": budget.get("requested_gb"),
        "effective_budget_gb": budget.get("effective_budget_gb"),
        "predicted_peak_gb": plan.get("predicted_peak_gb_total") if isinstance(plan, dict) else None,
        "physical_total_gb": ctx.physical_total_gb,
        "physical_free_gb": ctx.physical_free_gb,
    }


def _hint_gpu_oom(ctx: DiagnosticContext) -> ErrorHint:
    """OOM hint: diagnose the SPECIFIC cause and recommend ONE fix.

    Decision tree (first match wins, most specific to most generic):

      1. Another process is using a significant chunk of the GPU.
         → fix: switch GPUs (CUDA_VISIBLE_DEVICES) or lower --gpu-gb.

      2. The planner already warned this would OOM (predicted_peak >
         budget, or covariance fixed allocations > 50% of budget).
         → fix: --n-pcs-adaptive (the warning that fired pre-launch).

      3. Generic OOM. Suggest lowering --gpu-gb and/or --n-pcs-adaptive.
         If RECOVAR_DISABLE_CUDA=1 is set, mention it as informational
         (the fallback path uses more memory), but DO NOT tell the user
         to unset it — they set it deliberately.
    """
    plan = ctx.last_memory_plan or {}
    budget = plan.get("budget", {}) if isinstance(plan, dict) else {}
    predicted_peak = plan.get("predicted_peak_gb_total") if isinstance(plan, dict) else None
    effective_budget = budget.get("effective_budget_gb")
    grid_size = plan.get("grid_size") if isinstance(plan, dict) else None
    desired_n_pcs = plan.get("desired_n_pcs") if isinstance(plan, dict) else None
    adaptive_already = bool(plan.get("adaptive_n_pcs")) if isinstance(plan, dict) else False

    # Case 1: another process is using the GPU.
    # Only fire on CONFIRMED PID evidence — i.e. nvidia-smi enumerated
    # at least one compute process whose PID != ours. Don't infer
    # "another process" from "GPU is partially occupied but PID list
    # is empty": that signal is common when nvidia-smi can't enumerate
    # under cgroup / slurm isolation and recovar itself is the holder.
    # Falling through to case 3 (generic) is the right behavior there.
    own_pid = os.getpid()
    other_procs = [p for p in ctx.physical_processes if p.get("pid") != own_pid]
    other_used_gb = sum(p.get("used_mb", 0) / 1024.0 for p in other_procs)
    if other_procs and other_used_gb >= 1.0:
        proc_str = ", ".join(
            f"PID {p['pid']} ({p.get('name', '?')}, {p.get('used_mb', 0) / 1024:.1f} GB)" for p in other_procs[:3]
        )
        cause = (
            f"Another process is using {other_used_gb:.1f} GB on this GPU "
            f"({proc_str}). That likely left RECOVAR without enough headroom "
            f"for its requested workload."
        )
        suggestions = [
            "CUDA_VISIBLE_DEVICES=<idx-of-free-gpu> recovar pipeline ...   (run on a different GPU)",
            "OR lower the budget so RECOVAR fits beside the other process: --gpu-gb <smaller-N>",
            "OR stop / wait for the other GPU process (check `nvidia-smi`)",
        ]
        return ErrorHint(
            category="gpu_oom",
            summary="RECOVAR ran out of GPU memory — another process is using this GPU.",
            likely_cause=cause,
            suggestions=suggestions,
            diagnostic_context=_format_memory_context(ctx),
        )

    # Case 2: planner predicted this OOM (warning was logged pre-launch).
    prelaunch_warned = False
    if predicted_peak is not None and effective_budget is not None and predicted_peak > effective_budget * 0.95:
        prelaunch_warned = True
    if not adaptive_already and grid_size is not None and desired_n_pcs is not None and effective_budget is not None:
        basis_gb = desired_n_pcs * (grid_size**3) * 8 / 1e9
        packed = desired_n_pcs * (desired_n_pcs + 1) // 2
        lhs_gb = packed * packed * 8 / 1e9
        if basis_gb + lhs_gb > effective_budget * 0.50:
            prelaunch_warned = True

    if prelaunch_warned:
        peak_str = f"{predicted_peak:.1f} GB" if predicted_peak is not None else "the configured workload"
        budget_str = f"{effective_budget:.1f} GB" if effective_budget is not None else "the budget"
        cause = (
            f"Before launch the planner warned that {peak_str} would not fit in {budget_str}. "
            f"The run was launched anyway (--adaptive-n-pcs was not passed) and hit the "
            f"predicted OOM."
        )
        suggestions = [
            "recovar pipeline ... --adaptive-n-pcs   (planner auto-shrinks n_pcs to fit budget)",
            "OR raise --gpu-gb if a larger GPU is available, or use a smaller --grid-size",
        ]
        return ErrorHint(
            category="gpu_oom",
            summary="RECOVAR ran out of GPU memory — the planner already warned this would happen.",
            likely_cause=cause,
            suggestions=suggestions,
            diagnostic_context=_format_memory_context(ctx),
        )

    # Case 3 (generic): no clear external cause, no pre-launch warning.
    cause_parts = [
        "JAX/XLA reported an out-of-memory allocation during the run. The planner did not "
        "predict this ahead of time, so the workload was heavier in practice than the "
        "calibration table suggested."
    ]
    if ctx.custom_cuda_disabled:
        cause_parts.append(
            "Note: RECOVAR_DISABLE_CUDA=1 is set, which routes RECOVAR through the JAX fallback "
            "path. The fallback consumes more GPU memory than the custom kernel, so the same "
            "budget that worked with the kernel may be too tight here."
        )
    cause = " ".join(cause_parts)
    suggestions = [
        "recovar pipeline ... --gpu-gb <smaller-N>   (tighter budget → planner shrinks batches)",
        "OR recovar pipeline ... --adaptive-n-pcs   (if a pre-launch warning suggested it)",
    ]
    return ErrorHint(
        category="gpu_oom",
        summary="RECOVAR ran out of GPU memory.",
        likely_cause=cause,
        suggestions=suggestions,
        diagnostic_context=_format_memory_context(ctx),
    )


def _hint_conflicting_process(ctx: DiagnosticContext) -> ErrorHint:
    others = [p for p in ctx.physical_processes if p.get("pid") != os.getpid()]
    proc_str = ", ".join(
        f"PID {p['pid']} ({p.get('name', '?')}, {p.get('used_mb', 0) / 1024:.1f} GB)" for p in others[:5]
    )
    return ErrorHint(
        category="conflicting_gpu_process",
        summary="The GPU appears to be partially occupied by another process.",
        likely_cause=(
            f"GPU 0: total {ctx.physical_total_gb:.1f} GB, free {ctx.physical_free_gb:.1f} GB. "
            f"Other compute processes: {proc_str or '(none reported by nvidia-smi)'}."
        ),
        suggestions=[
            "CUDA_VISIBLE_DEVICES=<idx-of-free-gpu> recovar pipeline ...",
            "recovar pipeline ... --gpu-budget-gb <smaller-N> --adaptive-n-pcs   (fit into the free portion of this GPU)",
            "stop the other GPU process (nvidia-smi -> kill PID)",
        ],
        diagnostic_context=_format_memory_context(ctx),
    )


def _hint_xla_autotuner_failed(ctx: DiagnosticContext, raw_text: str) -> ErrorHint:
    """XLA's autotuner couldn't find a valid kernel config for some fusion.

    The Triton GEMM subtype is already disabled by default in
    ``recovar.jax_config`` (``--xla_gpu_enable_triton_gemm=false``), so
    a remaining hit here is the non-Triton autotuner failing on a
    particular shape — observed on full A100 at the (P, P) Cholesky
    transpose in compute_projected_covariance for certain n_pcs values
    (e.g. P=19306 at n_pcs=196). This is an XLA/JAX upstream gap, not a
    recovar bug.
    """
    # Try to surface the failing op + shape for the user.
    op_match = re.search(r'op_name="([^"]+)"', raw_text)
    shape_match = re.search(r"f32\[[\d,\s]+\]|f16\[[\d,\s]+\]|bf16\[[\d,\s]+\]", raw_text)
    op = op_match.group(1) if op_match else "(unknown op)"
    shape = shape_match.group(0) if shape_match else "(unknown shape)"
    return ErrorHint(
        category="xla_autotuner_failed",
        summary="JAX/XLA autotuner could not find a valid kernel config — known JAX upstream bug.",
        likely_cause=(
            f"XLA tried to autotune a fusion at op `{op}` with shape "
            f"`{shape}` and reported `No valid config found!`. This is "
            "not an OOM or a recovar bug — XLA's autotuner has gaps at "
            "specific shape values that happen to be hit by RECOVAR's "
            "(P, P) packed-covariance matrix where P = n_pcs(n_pcs+1)/2. "
            "Adjacent n_pcs values (and adjacent budgets) work fine — "
            "only this specific shape triggers the gap. See: "
            "https://github.com/NVIDIA/JAX-Toolbox/issues/317 "
            "https://github.com/google-deepmind/alphafold3/issues/240"
        ),
        suggestions=[
            'export XLA_FLAGS="$XLA_FLAGS --xla_gpu_autotune_level=0" && recovar pipeline ... '
            "(disables ALL XLA autotuning; ~5-15% slower but eliminates this class of failure)",
        ],
        diagnostic_context=_format_memory_context(ctx),
    )


def _hint_custom_cuda_unavailable(ctx: DiagnosticContext) -> ErrorHint:
    return ErrorHint(
        category="custom_cuda_unavailable",
        summary="RECOVAR could not use its custom CUDA extension.",
        likely_cause=(
            "The custom CUDA backproject/project shared library failed to "
            "build or load. RECOVAR prefers this kernel by default because it "
            "is substantially faster than the JAX fallback path."
        ),
        suggestions=[
            "recovar build_custom_cuda   (rebuild the kernel; check NVCC is on PATH)",
            'PYTHON="$(pixi run which python)" make -C recovar/cuda clean all',
            "RECOVAR_DISABLE_CUDA=1 recovar pipeline ...   (TEMPORARY: forces JAX "
            "fallback; expect ~2-3x slower runs and reduce --gpu-budget-gb accordingly)",
        ],
        diagnostic_context=_format_memory_context(ctx),
    )


def _hint_cpu_fallback(ctx: DiagnosticContext) -> ErrorHint:
    return ErrorHint(
        category="cpu_fallback_needed",
        summary="No JAX GPU device was detected.",
        likely_cause=(
            "JAX could not enumerate any GPU devices. Common causes: missing "
            "NVIDIA driver, mismatched CUDA toolkit, CUDA_VISIBLE_DEVICES "
            "filtering everything out, running on a login node without GPUs."
        ),
        suggestions=[
            f"echo $CUDA_VISIBLE_DEVICES   (currently: {ctx.env_cuda_visible_devices!r}; set to a valid index)",
            "pip install -U 'jax[cuda12]'   (verify JAX CUDA install)",
            "recovar pipeline ... --accept-cpu   (run on CPU; expect 100x+ slower)",
        ],
        diagnostic_context=_format_memory_context(ctx),
    )


def _hint_dataset_path(ctx: DiagnosticContext, raw_text: str) -> ErrorHint:
    return ErrorHint(
        category="dataset_path_error",
        summary="RECOVAR could not find a dataset file referenced by the input.",
        likely_cause=(
            "A path in the .star / .cs metadata or a CLI argument did not "
            "resolve. .star files often contain relative paths that need "
            "--datadir / --strip-prefix to be resolved correctly."
        ),
        suggestions=[
            "recovar check_paths <particles-or-star> [--datadir <dir>] [--strip-prefix <prefix>]",
            "use absolute paths for --particles / --poses / --ctf",
            "verify the file exists and is readable: ls -la <path>",
        ],
        diagnostic_context={"raw_excerpt": raw_text[-400:] if raw_text else ""},
    )


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------


def classify_text(text: str, context: DiagnosticContext | None = None) -> ErrorHint | None:
    """Classify a textual error blob (e.g. captured stderr)."""
    if not text:
        return None
    ctx = context or collect_context()

    # Check XLA autotuner failure BEFORE OOM — the autotuner error
    # message sometimes contains substrings that match OOM patterns
    # (e.g. "fusion" near "allocation"). Specific pattern wins.
    if _looks_like_xla_autotuner(text):
        return _hint_xla_autotuner_failed(ctx, text)
    if _looks_like_oom(text):
        return _hint_gpu_oom(ctx)
    if _looks_like_cuda_build(text):
        return _hint_custom_cuda_unavailable(ctx)
    if _looks_like_cpu_fallback(text):
        return _hint_cpu_fallback(ctx)
    if _looks_like_path_error(text):
        return _hint_dataset_path(ctx, text)

    # Even without a known signature, surface a conflict-process hint
    # if the GPU was clearly oversubscribed.
    if _conflicting_process_present(
        ctx,
        requested_gb=(ctx.last_memory_plan or {}).get("budget", {}).get("requested_gb")
        if isinstance(ctx.last_memory_plan, dict)
        else None,
    ):
        return _hint_conflicting_process(ctx)
    return None


def classify_exception(exc: BaseException, context: DiagnosticContext | None = None) -> ErrorHint | None:
    """Classify a live exception by inspecting its repr + str."""
    text = f"{type(exc).__name__}: {exc!s}"
    if isinstance(exc, FileNotFoundError):
        return _hint_dataset_path(context or collect_context(), text)
    return classify_text(text, context)


def classify_subprocess_failure(
    stderr_text: str,
    stdout_text: str,
    context: DiagnosticContext | None = None,
) -> ErrorHint | None:
    """Used by ``run_test_dataset`` after a non-zero subprocess exit."""
    combined = (stderr_text or "") + "\n" + (stdout_text or "")
    return classify_text(combined, context)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


_DELIMITER = "═" * 70


def format_error_hint(hint: ErrorHint) -> str:
    """Render the hint with a clear delimiter and TO RECOVER section.

    Designed to be the LAST thing printed on stderr so the user's eye
    lands on the suggestions instead of the XLA traceback above it.
    """
    suggestions = "\n".join(f"  - {s}" for s in hint.suggestions)
    ctx_lines = []
    for k, v in hint.diagnostic_context.items():
        if v is None:
            continue
        if isinstance(v, float):
            ctx_lines.append(f"    {k}: {v:.2f}")
        else:
            ctx_lines.append(f"    {k}: {v}")
    ctx_block = "\n".join(ctx_lines) if ctx_lines else "    (no context available)"

    cause_block = textwrap.indent(hint.likely_cause, "    ")

    lines = [
        _DELIMITER,
        f"RECOVAR error hint  [category: {hint.category}]",
        _DELIMITER,
        "",
        hint.summary,
        "",
        "Likely cause:",
        cause_block,
        "",
        "Diagnostic context:",
        ctx_block,
        "",
        "TO RECOVER, try one of:",
        suggestions,
        _DELIMITER,
        "",
    ]
    return "\n".join(lines)


def write_hint_log(hint: ErrorHint, outdir: str | Path) -> Path:
    """Persist the hint as JSON next to memory_plan.json for diagnostics."""
    import json

    out_path = Path(outdir) / "error_hint.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(hint.to_dict(), fh, indent=2, default=str)
    return out_path
