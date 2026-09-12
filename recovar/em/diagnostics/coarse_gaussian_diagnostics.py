"""Diagnostic scopes and dumps of the K-class significance pass.

The GEMM diagnostic and streaming scopes with their manifests and sealing,
the coarse runtime-prefix operand dumps, the tree-rescore and K-class
significance batch dumps and the env-gated stop after a dump. None of this
changes production arithmetic.
"""

import json
import logging
import os
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from recovar.em.diagnostics.coarse_score_diagnostics import (
    _coarse_gaussian_direct_macro_diagnostics,
    _coarse_gaussian_qualification_decision,
)
from recovar.em.helpers.batch_fetch import original_image_indices
from recovar.em.helpers.env_flags import parse_env_int_set
from recovar.em.scoring.coarse_gaussian_gemm import CoarseGaussianGemmHybridBatchResult, CoarseGaussianGemmResources
from recovar.em.scoring.coarse_gemm_hybrid import SOURCE_ROTATION_BLOCK_SIZE
from recovar.em.scoring.coarse_gemm_streaming import (
    COARSE_GEMM_STREAMING_SCHEMA,
    aggregate_coarse_gemm_streaming_summaries,
)

logger = logging.getLogger("recovar.em.dense_single_volume.helpers.coarse_gaussian_diagnostics")


_COARSE_RUNTIME_PREFIX_DUMP_DIR_ENV = "RECOVAR_COARSE_RUNTIME_PREFIX_DUMP_DIR"


_COARSE_RUNTIME_PREFIX_DUMP_INDICES_ENV = (
    "RECOVAR_COARSE_RUNTIME_PREFIX_DUMP_ORIGINAL_INDICES"
)


_COARSE_RUNTIME_PREFIX_DUMP_LABEL_ENV = "RECOVAR_COARSE_RUNTIME_PREFIX_DUMP_LABEL"


_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR"
)


_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_INDICES_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_ORIGINAL_INDICES"
)


_COARSE_GAUSSIAN_GEMM_STREAM_DIAGNOSTIC_DIR_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_STREAM_DIAGNOSTIC_DIR"
)


_COARSE_GAUSSIAN_GEMM_STREAM_TOPK_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_STREAM_TOPK"
)


_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET_ENV = (
    "RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET"
)


class SignificanceDumpComplete(RuntimeError):
    """Raised after an explicitly targeted coarse-significance dump is durable."""

    def __init__(self, *, dump_path: str):
        self.dump_path = str(dump_path)
        super().__init__(
            "requested RECOVAR coarse-significance target was written "
            f"(dump_path={self.dump_path})"
        )


def _maybe_stop_after_significance_dump(
    dump_path: str,
    *,
    dump_dir: str,
    target_original_indices: set[int],
    current_size: int | None,
    debug_iteration: int | None,
) -> None:
    """Stop an explicit diagnostic only after its complete target set exists."""

    if os.environ.get(_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET_ENV) != "1":
        return
    if not os.path.isfile(dump_path):
        raise RuntimeError(
            "RECOVAR significance stop target is missing its dump file: "
            f"{dump_path}"
        )
    target_iteration = os.environ.get("RECOVAR_SIGNIFICANCE_DUMP_ITERATION")
    iteration_suffix = (
        ""
        if not target_iteration
        else f"_it{int(debug_iteration):03d}"
    )
    current_size_label = -1 if current_size is None else int(current_size)
    expected_paths = [
        os.path.join(
            dump_dir,
            f"significance_orig{int(original_index):06d}{iteration_suffix}_cs"
            f"{current_size_label:03d}.npz",
        )
        for original_index in sorted(target_original_indices)
    ]
    missing_paths = [path for path in expected_paths if not os.path.isfile(path)]
    if missing_paths:
        logger.info(
            "RECOVAR coarse-significance stop target progress: %d/%d files written",
            len(expected_paths) - len(missing_paths),
            len(expected_paths),
        )
        return
    raise SignificanceDumpComplete(dump_path=dump_path)


class CoarseGaussianGemmDiagnosticScope(NamedTuple):
    """Deterministic identity and completion contract for one diagnostic call.

    InitialModel invokes the shared significance engine once per non-empty
    pseudo-halfset/group.  ``expected_call_ids`` names that complete run, and
    exactly one (the final call) sets ``finalize=True`` so an aggregate
    manifest can prove that every globally requested particle was captured
    exactly once across the disjoint call scopes.
    """

    run_id: str
    call_id: str
    expected_call_ids: tuple[str, ...]
    finalize: bool


def _validate_coarse_gaussian_gemm_diagnostic_identifier(
    value: str,
    *,
    field: str,
) -> str:
    """Return one filename-safe deterministic diagnostic identifier."""

    token = str(value)
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    )
    if not token or len(token) > 160 or any(character not in allowed for character in token):
        raise ValueError(
            f"coarse GEMM diagnostic {field} must be 1--160 filename-safe "
            f"characters, got {token!r}",
        )
    return token


def _resolve_coarse_gaussian_gemm_diagnostic_scope(
    scope: CoarseGaussianGemmDiagnosticScope | None,
    *,
    debug_iteration: int | None,
    current_size: int | None,
) -> tuple[CoarseGaussianGemmDiagnosticScope, bool]:
    """Resolve an explicit multi-call scope or a strict single-call default."""

    if scope is None:
        iteration_label = -1 if debug_iteration is None else int(debug_iteration)
        size_label = -1 if current_size is None else int(current_size)
        iteration_token = (
            f"m{-iteration_label:04d}"
            if iteration_label < 0
            else f"{iteration_label:04d}"
        )
        size_token = f"m{-size_label:04d}" if size_label < 0 else f"{size_label:04d}"
        run_id = f"shared_it{iteration_token}_cs{size_token}"
        return (
            CoarseGaussianGemmDiagnosticScope(
                run_id=run_id,
                call_id="call0000_global",
                expected_call_ids=("call0000_global",),
                finalize=True,
            ),
            False,
        )
    run_id = _validate_coarse_gaussian_gemm_diagnostic_identifier(
        scope.run_id,
        field="run_id",
    )
    call_id = _validate_coarse_gaussian_gemm_diagnostic_identifier(
        scope.call_id,
        field="call_id",
    )
    expected_call_ids = tuple(
        _validate_coarse_gaussian_gemm_diagnostic_identifier(
            value,
            field="expected_call_id",
        )
        for value in scope.expected_call_ids
    )
    if not expected_call_ids or len(set(expected_call_ids)) != len(expected_call_ids):
        raise ValueError(
            "coarse GEMM diagnostic expected_call_ids must be non-empty and unique",
        )
    if call_id not in expected_call_ids:
        raise ValueError(
            "coarse GEMM diagnostic call_id must occur in expected_call_ids",
        )
    if bool(scope.finalize) != (call_id == expected_call_ids[-1]):
        raise ValueError(
            "coarse GEMM diagnostic finalize must be true exactly for the last "
            "expected call",
        )
    return (
        CoarseGaussianGemmDiagnosticScope(
            run_id=run_id,
            call_id=call_id,
            expected_call_ids=expected_call_ids,
            finalize=bool(scope.finalize),
        ),
        True,
    )


def _coarse_gaussian_gemm_diagnostic_request() -> tuple[str | None, set[int] | None]:
    """Resolve the default-off paired production-score diagnostic.

    Capture deliberately executes both the direct-square and expanded-square
    scorers on the selected production operands.  It therefore doubles score
    work for those blocks and is never a valid timed-runtime arm.  Runtime
    qualification must use separate diagnostic-off runs; capture artifacts and
    returned statistics report that exclusion explicitly.
    """

    directory = os.environ.get(_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR_ENV, "").strip()
    target_token_present = bool(
        os.environ.get(_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_INDICES_ENV, "").strip()
    )
    if not directory:
        if target_token_present:
            raise ValueError(
                f"{_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_INDICES_ENV} requires "
                f"{_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR_ENV}",
            )
        return None, None
    targets = parse_env_int_set(_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_INDICES_ENV)
    if not targets:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR_ENV} requires a non-empty "
            f"{_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_INDICES_ENV}",
        )
    if any(int(target) < 0 for target in targets):
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_INDICES_ENV} must contain "
            "non-negative original image indices",
        )
    return os.path.abspath(os.path.expanduser(directory)), {int(target) for target in targets}


def _coarse_gaussian_gemm_streaming_diagnostic_request(
    *,
    max_significants: int,
) -> tuple[str | None, int | None]:
    """Resolve the all-particle bounded exact-rescore diagnostic.

    The diagnostic retains ``topk + 1`` paired candidates per image (the extra
    row certifies cutoff-tie and band coverage) and reduces error statistics
    across every candidate block on device.  It never serializes a score cube.
    """

    directory = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_STREAM_DIAGNOSTIC_DIR_ENV,
        "",
    ).strip()
    topk_token = os.environ.get(_COARSE_GAUSSIAN_GEMM_STREAM_TOPK_ENV, "").strip()
    if not directory:
        if topk_token:
            raise ValueError(
                f"{_COARSE_GAUSSIAN_GEMM_STREAM_TOPK_ENV} requires "
                f"{_COARSE_GAUSSIAN_GEMM_STREAM_DIAGNOSTIC_DIR_ENV}",
            )
        return None, None
    default_topk = max(
        2048,
        int(max_significants) + 64 if int(max_significants) > 0 else 2048,
    )
    try:
        retained_topk = int(topk_token) if topk_token else default_topk
    except ValueError as error:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_STREAM_TOPK_ENV} must be a positive integer, "
            f"got {topk_token!r}",
        ) from error
    if retained_topk <= 0:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_STREAM_TOPK_ENV} must be a positive integer, "
            f"got {retained_topk}",
        )
    return os.path.abspath(os.path.expanduser(directory)), retained_topk


def _coarse_gaussian_gemm_scope_manifest_path(
    directory: str,
    scope: CoarseGaussianGemmDiagnosticScope,
) -> str:
    return os.path.join(
        directory,
        f"coarse_gemm_scope_{scope.run_id}_{scope.call_id}.json",
    )


def _write_json_exclusive(path: str, payload: dict) -> None:
    """Write immutable diagnostic metadata without hiding path collisions."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite a coarse GEMM diagnostic manifest: {path}",
        ) from error


def _seal_coarse_gaussian_gemm_diagnostic_scope(
    directory: str,
    *,
    scope: CoarseGaussianGemmDiagnosticScope,
    selection_policy: str,
    requested_targets: set[int],
    targets_in_scope: set[int],
    captured_target_counts: dict[int, int],
    artifact_paths: list[str],
) -> tuple[str, str | None]:
    """Seal one call record and, on the final call, its aggregate manifest."""

    requested = sorted(int(value) for value in requested_targets)
    in_scope = sorted(int(value) for value in targets_in_scope)
    counts = {
        str(int(target)): int(captured_target_counts.get(int(target), 0))
        for target in in_scope
    }
    bad_scope_counts = {
        target: count
        for target, count in counts.items()
        if count != 1
    }
    if bad_scope_counts:
        raise RuntimeError(
            "coarse GEMM diagnostic targets must be captured exactly once "
            f"within call {scope.call_id}: {bad_scope_counts}",
        )
    scope_record = {
        "schema_version": 1,
        "run_id": scope.run_id,
        "call_id": scope.call_id,
        "expected_call_ids": list(scope.expected_call_ids),
        "selection_policy": str(selection_policy),
        "requested_original_indices": requested,
        "targets_in_scope": in_scope,
        "targets_explicitly_out_of_scope": sorted(
            set(requested) - set(in_scope),
        ),
        "captured_target_counts": counts,
        "artifact_paths": [os.path.basename(path) for path in artifact_paths],
    }
    scope_path = _coarse_gaussian_gemm_scope_manifest_path(directory, scope)
    _write_json_exclusive(scope_path, scope_record)
    if not scope.finalize:
        return scope_path, None

    scope_records = []
    missing_call_ids = []
    for expected_call_id in scope.expected_call_ids:
        expected_scope = scope._replace(
            call_id=expected_call_id,
            finalize=expected_call_id == scope.expected_call_ids[-1],
        )
        expected_path = _coarse_gaussian_gemm_scope_manifest_path(
            directory,
            expected_scope,
        )
        if not os.path.isfile(expected_path):
            missing_call_ids.append(expected_call_id)
            continue
        with open(expected_path, encoding="utf-8") as stream:
            record = json.load(stream)
        if (
            record.get("run_id") != scope.run_id
            or record.get("call_id") != expected_call_id
            or record.get("expected_call_ids") != list(scope.expected_call_ids)
            or record.get("requested_original_indices") != requested
        ):
            raise RuntimeError(
                "coarse GEMM diagnostic scope manifest does not match the "
                f"aggregate contract: {expected_path}",
            )
        scope_records.append(record)
    if missing_call_ids:
        raise RuntimeError(
            "coarse GEMM diagnostic aggregate is missing expected calls: "
            f"{missing_call_ids}",
        )

    aggregate_counts = {str(target): 0 for target in requested}
    for record in scope_records:
        for target, count in record["captured_target_counts"].items():
            if target not in aggregate_counts:
                raise RuntimeError(
                    "coarse GEMM scope captured an unrequested target: "
                    f"{target}",
                )
            aggregate_counts[target] += int(count)
    missing_targets = [
        int(target)
        for target, count in aggregate_counts.items()
        if count == 0
    ]
    duplicate_targets = [
        int(target)
        for target, count in aggregate_counts.items()
        if count > 1
    ]
    if missing_targets or duplicate_targets:
        raise RuntimeError(
            "coarse GEMM diagnostic aggregate requires every requested target "
            "exactly once; "
            f"missing={missing_targets}, duplicate={duplicate_targets}",
        )
    aggregate_record = {
        "schema_version": 1,
        "run_id": scope.run_id,
        "expected_call_ids": list(scope.expected_call_ids),
        "requested_original_indices": requested,
        "captured_target_counts": aggregate_counts,
        "all_requested_captured_exactly_once": True,
        "scope_records": scope_records,
    }
    aggregate_path = os.path.join(
        directory,
        f"coarse_gemm_manifest_{scope.run_id}.json",
    )
    _write_json_exclusive(aggregate_path, aggregate_record)
    return scope_path, aggregate_path


def _coarse_gaussian_gemm_stream_scope_manifest_path(
    directory: str,
    scope: CoarseGaussianGemmDiagnosticScope,
) -> str:
    return os.path.join(
        directory,
        f"coarse_gemm_rescore_scope_{scope.run_id}_{scope.call_id}.json",
    )


def _seal_coarse_gaussian_gemm_streaming_scope(
    directory: str,
    *,
    scope: CoarseGaussianGemmDiagnosticScope,
    retained_topk: int,
    artifact_paths: list[str],
    original_indices: list[int],
) -> tuple[str, str | None]:
    """Seal compact all-particle summaries across explicit call scopes."""

    particle_ids = [int(value) for value in original_indices]
    if len(particle_ids) != len(set(particle_ids)):
        raise RuntimeError(
            "coarse GEMM streaming diagnostic captured a particle more than once "
            f"inside call {scope.call_id}",
        )
    artifact_particle_ids = []
    for artifact_path in artifact_paths:
        if not os.path.isfile(artifact_path):
            raise RuntimeError(
                "coarse GEMM streaming diagnostic artifact is missing: "
                f"{artifact_path}",
            )
        with np.load(artifact_path, allow_pickle=False) as artifact:
            if (
                artifact.get("schema", np.asarray("")).item()
                != COARSE_GEMM_STREAMING_SCHEMA
                or artifact.get("diagnostic_run_id", np.asarray("")).item()
                != scope.run_id
                or artifact.get("diagnostic_call_id", np.asarray("")).item()
                != scope.call_id
                or bool(artifact.get("stores_score_cube", np.asarray(True)).item())
            ):
                raise RuntimeError(
                    "coarse GEMM streaming diagnostic artifact differs from its "
                    f"scope contract: {artifact_path}",
                )
            artifact_particle_ids.extend(
                int(value) for value in np.asarray(artifact["original_indices"])
            )
    if artifact_particle_ids != particle_ids:
        raise RuntimeError(
            "coarse GEMM streaming diagnostic artifacts do not cover the call's "
            "particle stream in order",
        )
    record = {
        "schema_version": 2,
        "run_id": scope.run_id,
        "call_id": scope.call_id,
        "expected_call_ids": list(scope.expected_call_ids),
        "retained_topk": int(retained_topk),
        "particle_count": len(particle_ids),
        "original_indices": particle_ids,
        "artifact_paths": [os.path.basename(path) for path in artifact_paths],
        "stores_score_cube": False,
    }
    scope_path = _coarse_gaussian_gemm_stream_scope_manifest_path(
        directory,
        scope,
    )
    _write_json_exclusive(scope_path, record)
    if not scope.finalize:
        return scope_path, None

    scope_records = []
    aggregate_particle_ids = []
    for expected_call_id in scope.expected_call_ids:
        expected_scope = scope._replace(
            call_id=expected_call_id,
            finalize=expected_call_id == scope.expected_call_ids[-1],
        )
        expected_path = _coarse_gaussian_gemm_stream_scope_manifest_path(
            directory,
            expected_scope,
        )
        if not os.path.isfile(expected_path):
            raise RuntimeError(
                "coarse GEMM streaming diagnostic aggregate is missing call "
                f"{expected_call_id}: {expected_path}",
            )
        with open(expected_path, encoding="utf-8") as stream:
            expected_record = json.load(stream)
        if (
            expected_record.get("schema_version") != 2
            or expected_record.get("run_id") != scope.run_id
            or expected_record.get("call_id") != expected_call_id
            or expected_record.get("expected_call_ids") != list(scope.expected_call_ids)
            or expected_record.get("retained_topk") != int(retained_topk)
            or expected_record.get("stores_score_cube") is not False
        ):
            raise RuntimeError(
                "coarse GEMM streaming scope manifest does not match the "
                f"aggregate contract: {expected_path}",
            )
        scope_records.append(expected_record)
        aggregate_particle_ids.extend(
            int(value) for value in expected_record["original_indices"]
        )
    if len(aggregate_particle_ids) != len(set(aggregate_particle_ids)):
        raise RuntimeError(
            "coarse GEMM streaming diagnostic captured duplicate particles "
            "across call scopes",
        )
    aggregate_artifact_paths = [
        os.path.join(directory, artifact_name)
        for record in scope_records
        for artifact_name in record["artifact_paths"]
    ]
    aggregate = {
        "schema_version": 2,
        "run_id": scope.run_id,
        "expected_call_ids": list(scope.expected_call_ids),
        "retained_topk": int(retained_topk),
        "particle_count": len(aggregate_particle_ids),
        "all_particles_captured_exactly_once": True,
        "stores_score_cube": False,
        "scope_records": scope_records,
        "summary": aggregate_coarse_gemm_streaming_summaries(
            aggregate_artifact_paths,
        ),
    }
    aggregate_path = os.path.join(
        directory,
        f"coarse_gemm_rescore_manifest_{scope.run_id}.json",
    )
    _write_json_exclusive(aggregate_path, aggregate)
    return scope_path, aggregate_path


def _write_coarse_gaussian_gemm_diagnostic(
    output_path: str,
    *,
    direct_scores_pre_prior,
    macro_scores_pre_prior,
    direct_scores_with_prior,
    macro_scores_with_prior,
    direct_support,
    macro_support,
    original_indices,
    local_indices,
    actual_batch_size: int,
    padded_batch_size: int,
    adaptive_fraction: float,
    max_significants: int,
    resource_estimate: CoarseGaussianGemmResources,
    diagnostic_scope: CoarseGaussianGemmDiagnosticScope,
    diagnostic_selection_policy: str,
    debug_iteration: int | None,
    current_size: int | None,
) -> None:
    """Write one immutable paired score surface for repeat-envelope analysis."""

    if os.path.exists(output_path):
        raise FileExistsError(
            "refusing to overwrite a coarse GEMM diagnostic artifact: "
            f"{output_path}",
        )
    diagnostics = _coarse_gaussian_direct_macro_diagnostics(
        direct_scores_with_prior,
        macro_scores_with_prior,
        direct_support=direct_support,
        macro_support=macro_support,
    )
    direct_pre_prior = np.asarray(direct_scores_pre_prior)
    macro_pre_prior = np.asarray(macro_scores_pre_prior)
    macro_negative_implied_diff2 = macro_pre_prior > 0.0
    direct_negative_implied_diff2 = direct_pre_prior > 0.0
    qualification = _coarse_gaussian_qualification_decision(
        exact_arithmetic_equivalent=True,
        repeat_stable=None,
        unbiased_non_directional=None,
        bounded_non_growing=None,
        discrete_choices_equal=bool(
            np.all(diagnostics["argmax_equal"])
            and np.all(diagnostics["support_equal"])
        ),
        final_basin_quality_equal=None,
        material_runtime_win=None,
        scale_amplified=None,
        negative_implied_diff2=bool(
            np.any(macro_negative_implied_diff2 & ~direct_negative_implied_diff2)
        ),
        nonfinite_scores=bool(np.any(~np.isfinite(macro_pre_prior))),
        exact_zero_cancellation_drift=bool(
            np.any(diagnostics["exact_zero_direct_nonzero_macro_per_image"])
        ),
    )
    payload = {
        "layout": np.asarray("image,class,rotation,translation"),
        "numerical_policy": np.asarray(
            "exact-arithmetic-equivalent_expanded-square_cancellation-sensitive_qualification-only"
        ),
        "qualification_status": np.asarray(qualification["status"]),
        "qualification_policy": np.asarray(
            "allow_repeat-stable_unbiased_bounded_non-growing_noise_never_lone-epsilon_promotion"
        ),
        "automatic_no_go_reasons": np.asarray(
            qualification["failure_reasons"],
            dtype=np.str_,
        ),
        "pending_qualification_gates": np.asarray(
            qualification["pending_gates"],
            dtype=np.str_,
        ),
        "requires_bitwise_score_identity": np.asarray(
            qualification["requires_bitwise_score_identity"],
        ),
        "requires_exact_discrete_identity": np.asarray(
            qualification["requires_exact_discrete_identity"],
        ),
        "repeat_spread_assessment": np.asarray(
            "NO_GO_until_same-hardware_repeat_artifacts_establish_native-relative_spread"
        ),
        "scale_growth_assessment": np.asarray(
            "NO_GO_until_multiscale_artifacts_exclude_scale-amplified_drift"
        ),
        "paired_capture_active": np.asarray(True),
        "clean_timing_eligible": np.asarray(False),
        "timing_policy": np.asarray(
            "paired_capture_executes_both_scorers_use_separate_diagnostic-off_timing_arm"
        ),
        "diagnostic_run_id": np.asarray(diagnostic_scope.run_id),
        "diagnostic_call_id": np.asarray(diagnostic_scope.call_id),
        "diagnostic_selection_policy": np.asarray(diagnostic_selection_policy),
        "debug_iteration": np.asarray(
            -1 if debug_iteration is None else int(debug_iteration),
            dtype=np.int64,
        ),
        "current_size": np.asarray(
            -1 if current_size is None else int(current_size),
            dtype=np.int64,
        ),
        "direct_scores_pre_prior": direct_pre_prior,
        "macro_scores_pre_prior": macro_pre_prior,
        "direct_scores_with_prior": np.asarray(direct_scores_with_prior),
        "macro_scores_with_prior": np.asarray(macro_scores_with_prior),
        "original_indices": np.asarray(original_indices, dtype=np.int64),
        "local_indices": np.asarray(local_indices, dtype=np.int64),
        "actual_batch_size": np.asarray(actual_batch_size, dtype=np.int64),
        "padded_batch_size": np.asarray(padded_batch_size, dtype=np.int64),
        "adaptive_fraction": np.asarray(adaptive_fraction, dtype=np.float64),
        "max_significants": np.asarray(max_significants, dtype=np.int64),
        "direct_negative_implied_diff2_count": np.asarray(
            np.count_nonzero(direct_negative_implied_diff2),
            dtype=np.int64,
        ),
        "macro_negative_implied_diff2_count": np.asarray(
            np.count_nonzero(macro_negative_implied_diff2),
            dtype=np.int64,
        ),
        "macro_only_negative_implied_diff2_count": np.asarray(
            np.count_nonzero(
                macro_negative_implied_diff2 & ~direct_negative_implied_diff2,
            ),
            dtype=np.int64,
        ),
    }
    payload.update(diagnostics)
    payload.update(
        {
            f"resource_{field}": np.asarray(value, dtype=np.int64)
            for field, value in resource_estimate._asdict().items()
        }
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **payload)


def _significance_debug_dump_enabled() -> bool:
    return bool(os.environ.get("RECOVAR_SIGNIFICANCE_DUMP_DIR"))


def _significance_debug_dump_matches(*, current_size, debug_iteration) -> bool:
    """Return whether significance capture applies at this scoring boundary."""

    if not _significance_debug_dump_enabled():
        return False
    if not parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES"):
        return False
    target_current_size = os.environ.get("RECOVAR_SIGNIFICANCE_DUMP_CURRENT_SIZE")
    if target_current_size and (
        current_size is None or int(current_size) != int(target_current_size)
    ):
        return False
    target_iteration = os.environ.get("RECOVAR_SIGNIFICANCE_DUMP_ITERATION")
    if target_iteration and (
        debug_iteration is None or int(debug_iteration) != int(target_iteration)
    ):
        return False
    return True


def _coarse_runtime_prefix_dump_request() -> tuple[str | None, set[int], str]:
    """Resolve a target-only dump of the exact compact coarse-score operands."""

    directory = os.environ.get(_COARSE_RUNTIME_PREFIX_DUMP_DIR_ENV, "").strip()
    targets = parse_env_int_set(_COARSE_RUNTIME_PREFIX_DUMP_INDICES_ENV) or set()
    label = os.environ.get(_COARSE_RUNTIME_PREFIX_DUMP_LABEL_ENV, "unlabeled").strip()
    if bool(directory) != bool(targets):
        raise ValueError(
            f"{_COARSE_RUNTIME_PREFIX_DUMP_DIR_ENV} and "
            f"{_COARSE_RUNTIME_PREFIX_DUMP_INDICES_ENV} must be set together",
        )
    if any(int(target) < 0 for target in targets):
        raise ValueError(
            f"{_COARSE_RUNTIME_PREFIX_DUMP_INDICES_ENV} must contain "
            "non-negative original image indices",
        )
    safe_label = "".join(
        character if character.isalnum() or character in "_.-" else "_"
        for character in label
    )
    return (
        os.path.abspath(os.path.expanduser(directory)) if directory else None,
        {int(target) for target in targets},
        safe_label or "unlabeled",
    )


def _maybe_dump_coarse_runtime_prefix_operands(
    *,
    dump_dir: str | None,
    dump_label: str,
    target_original_indices: set[int],
    experiment_dataset,
    indices,
    compact_result: CoarseGaussianGemmHybridBatchResult,
    support_pose_ids,
    batch_weights,
    batch_sig_mask,
    batch_n_sig,
    batch_cutoff_count,
    batch_sum_weight,
    batch_significant_weight,
    projection_cache,
    shifted_corrected,
    pixel_weight,
    initial_diff2,
    full_to_compact,
    logical_full_pixel_count: int,
    class_log_prior,
    rotation_log_prior,
    translation_log_prior,
    current_size,
    physical_current_size,
    debug_iteration,
) -> None:
    """Persist exact production source-16 operands after support is decided.

    The selected diff2 buffer is the same buffer consumed by compact score
    assembly.  Host materialization happens only after the posterior/support
    decision, so this diagnostic cannot alter the atomic order under study.
    """

    if dump_dir is None:
        return
    selected_diff2 = compact_result.diagnostic_selected_diff2
    compact = compact_result.compact_scores
    if selected_diff2 is None or compact is None:
        raise RuntimeError(
            "coarse runtime-prefix operand capture requires compact selected rescoring",
        )
    local_indices = np.asarray(indices, dtype=np.int64)
    original_indices = original_image_indices(experiment_dataset, local_indices)
    target_rows = np.flatnonzero(
        np.isin(
            original_indices,
            np.fromiter(target_original_indices, dtype=np.int64),
        ),
    )
    if not target_rows.size:
        return

    os.makedirs(dump_dir, exist_ok=True)
    source_block_ids_all = np.asarray(compact.source_block_ids, dtype=np.int32)
    block_counts = np.asarray(compact.block_count, dtype=np.int32)
    selected_diff2_all = np.asarray(selected_diff2, dtype=np.float32)
    posterior_scores_all = np.asarray(compact.posterior_scores_flat, dtype=np.float32)
    weights_all = np.asarray(batch_weights, dtype=np.float32)
    support_mask_all = np.asarray(batch_sig_mask, dtype=bool)
    raw_score_max_all = np.asarray(compact.raw_score_max, dtype=np.float32)
    min_diff2_offsets_all = np.asarray(compact.min_diff2_offsets, dtype=np.float32)
    best_score_all = np.asarray(compact.best_score, dtype=np.float32)
    best_pose_all = np.asarray(compact.best_pose, dtype=np.int32)
    n_sig_all = np.asarray(batch_n_sig, dtype=np.int32)
    cutoff_count_all = np.asarray(batch_cutoff_count, dtype=np.int32)
    sum_weight_all = np.asarray(batch_sum_weight, dtype=np.float32)
    significant_weight_all = np.asarray(batch_significant_weight, dtype=np.float32)
    shifted = jnp.asarray(shifted_corrected)
    weight = jnp.asarray(pixel_weight)
    initial = jnp.asarray(initial_diff2)
    cache = jnp.asarray(projection_cache)
    n_translations = int(shifted.shape[1])
    physical_pixel_count = int(shifted.shape[2])
    capacity = int(source_block_ids_all.shape[1])

    rotation_prior = (
        np.empty((0,), dtype=np.float32)
        if rotation_log_prior is None
        else np.asarray(rotation_log_prior, dtype=np.float32)
    )
    translation_prior_all = (
        None
        if translation_log_prior is None
        else np.asarray(translation_log_prior, dtype=np.float32)
    )
    for row in target_rows.tolist():
        block_count = int(block_counts[row])
        if block_count <= 0 or block_count > capacity:
            raise RuntimeError(
                "coarse runtime-prefix operand capture found an invalid block count",
            )
        source_block_ids = source_block_ids_all[row, :block_count]
        rotation_ids = (
            source_block_ids[:, None] * np.int32(SOURCE_ROTATION_BLOCK_SIZE)
            + np.arange(SOURCE_ROTATION_BLOCK_SIZE, dtype=np.int32)[None, :]
        )
        candidate_pose_ids = (
            rotation_ids[:, :, None] * np.int32(n_translations)
            + np.arange(n_translations, dtype=np.int32)[None, None, :]
        )
        active_candidate_count = (
            block_count * SOURCE_ROTATION_BLOCK_SIZE * n_translations
        )
        support_mask = support_mask_all[row, :active_candidate_count].reshape(
            block_count,
            SOURCE_ROTATION_BLOCK_SIZE,
            n_translations,
        )
        observed_support = np.sort(candidate_pose_ids[support_mask]).astype(
            np.int32,
            copy=False,
        )
        expected_support = np.asarray(support_pose_ids[row], dtype=np.int32)
        if not np.array_equal(observed_support, expected_support):
            raise RuntimeError(
                "coarse runtime-prefix dump support mapping differs from production",
            )
        original_index = int(original_indices[row])
        iteration_label = -1 if debug_iteration is None else int(debug_iteration)
        size_label = -1 if current_size is None else int(current_size)
        output_path = os.path.join(
            dump_dir,
            f"coarse_runtime_prefix_{dump_label}_orig{original_index:06d}_"
            f"it{iteration_label:03d}_cs{size_label:03d}.npz",
        )
        translation_prior = (
            np.empty((0,), dtype=np.float32)
            if translation_prior_all is None
            else (
                translation_prior_all
                if translation_prior_all.ndim == 1
                else translation_prior_all[row]
            )
        )
        with open(output_path, "xb") as stream:
            np.savez(
                stream,
                schema=np.asarray("recovar.coarse_runtime_prefix_operands.v1"),
                capture_policy=np.asarray(
                    "same_selected_diff2_buffer_materialized_after_support_decision",
                ),
                original_index=np.int64(original_index),
                local_index=np.int64(local_indices[row]),
                debug_iteration=np.int64(iteration_label),
                current_size=np.int64(size_label),
                physical_current_size=np.int64(physical_current_size),
                logical_full_pixel_count=np.int64(logical_full_pixel_count),
                physical_pixel_count=np.int64(physical_pixel_count),
                n_translations=np.int64(n_translations),
                source_block_ids=source_block_ids,
                source_rotation_ids=rotation_ids,
                candidate_pose_ids=candidate_pose_ids,
                selected_reference=np.asarray(
                    cache[0, rotation_ids.reshape(-1)],
                    dtype=np.complex64,
                ).reshape(
                    block_count,
                    SOURCE_ROTATION_BLOCK_SIZE,
                    physical_pixel_count,
                ),
                shifted_corrected=np.asarray(shifted[row], dtype=np.complex64),
                pixel_weight=np.asarray(weight[row], dtype=np.float32),
                initial_diff2=np.asarray(initial[row], dtype=np.float32),
                full_to_compact=np.asarray(full_to_compact, dtype=np.int32),
                selected_diff2=selected_diff2_all[row, :block_count],
                posterior_scores=posterior_scores_all[
                    row,
                    :active_candidate_count,
                ].reshape(
                    block_count,
                    SOURCE_ROTATION_BLOCK_SIZE,
                    n_translations,
                ),
                posterior_weights=weights_all[
                    row,
                    :active_candidate_count,
                ].reshape(
                    block_count,
                    SOURCE_ROTATION_BLOCK_SIZE,
                    n_translations,
                ),
                support_mask=support_mask,
                support_pose_ids=observed_support,
                n_significant=n_sig_all[row],
                cutoff_count=cutoff_count_all[row],
                sum_weight=sum_weight_all[row],
                significant_weight=significant_weight_all[row],
                raw_score_max=raw_score_max_all[row],
                min_diff2_offset=min_diff2_offsets_all[row],
                best_score=best_score_all[row],
                best_pose=best_pose_all[row],
                class_log_prior=np.asarray(class_log_prior, dtype=np.float32),
                rotation_log_prior=rotation_prior,
                translation_log_prior=translation_prior,
            )
        logger.warning(
            "wrote post-support coarse runtime-prefix operand capture: %s",
            output_path,
        )


def _maybe_dump_tree_rescore_batch(
    *,
    experiment_dataset,
    indices,
    ambiguous_rows,
    candidate_pose_ids,
    original_best_pose,
    original_best_score,
    original_second_pose,
    original_second_score,
    rescored_scores,
    rescored_winner_slot,
    shifted_candidates,
    score_weight_candidates,
    numerator_weight_candidates,
    rotation_matrices,
    translation_angles,
    n_trans,
    half_weights,
    packed_to_compact,
    projector_full,
    current_size,
    padding_factor,
    projector_max_r,
    debug_iteration,
):
    """Persist exact bounded-rescore operands for selected pass-1 particles."""

    if not _significance_debug_dump_matches(
        current_size=current_size,
        debug_iteration=debug_iteration,
    ):
        return
    target_original_indices = parse_env_int_set(
        "RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES"
    )
    batch_original_indices = original_image_indices(experiment_dataset, indices)
    ambiguous_original_indices = batch_original_indices[
        np.asarray(ambiguous_rows, dtype=np.int64)
    ]
    dump_dir = os.environ["RECOVAR_SIGNIFICANCE_DUMP_DIR"]
    os.makedirs(dump_dir, exist_ok=True)
    candidate_pose_ids = np.asarray(candidate_pose_ids, dtype=np.int32)
    original_best_pose = np.asarray(original_best_pose, dtype=np.int32)
    original_best_score = np.asarray(original_best_score, dtype=np.float32)
    original_second_pose = np.asarray(original_second_pose, dtype=np.int32)
    original_second_score = np.asarray(original_second_score, dtype=np.float32)
    rescored_scores = np.asarray(rescored_scores, dtype=np.float32)
    rescored_winner_slot = np.asarray(rescored_winner_slot, dtype=np.int32)
    for row, original_index in enumerate(ambiguous_original_indices):
        if int(original_index) not in target_original_indices:
            continue
        original_scores_by_candidate = np.where(
            candidate_pose_ids[row] == original_best_pose[row],
            original_best_score[row],
            original_second_score[row],
        ).astype(np.float32, copy=False)
        out_path = os.path.join(
            dump_dir,
            f"tree_rescore_orig{int(original_index):06d}_it"
            f"{int(debug_iteration):03d}_cs{int(current_size):03d}.npz",
        )
        np.savez_compressed(
            out_path,
            original_index=np.int64(original_index),
            candidate_pose_ids=candidate_pose_ids[row],
            candidate_rotation_ids=(candidate_pose_ids[row] // int(n_trans)),
            candidate_translation_ids=(candidate_pose_ids[row] % int(n_trans)),
            original_best_pose=original_best_pose[row],
            original_second_pose=original_second_pose[row],
            original_scores_by_candidate=original_scores_by_candidate,
            direct_texture_scores=rescored_scores[row],
            direct_texture_winner_slot=rescored_winner_slot[row],
            image_candidates=np.asarray(shifted_candidates[row], dtype=np.complex64),
            image_candidates_are_unshifted=np.asarray(True, dtype=np.bool_),
            translation_angles=np.asarray(translation_angles[row], dtype=np.float32),
            score_weight_candidates=np.asarray(
                score_weight_candidates[row], dtype=np.float32
            ),
            numerator_weight_candidates=np.asarray(
                numerator_weight_candidates[row], dtype=np.float32
            ),
            rotation_matrices=np.asarray(rotation_matrices[row], dtype=np.float32),
            half_weights=np.asarray(half_weights, dtype=np.float32),
            packed_to_compact=np.asarray(packed_to_compact, dtype=np.int32),
            projector_full=np.asarray(projector_full, dtype=np.complex64),
            current_size=np.int64(current_size),
            padding_factor=np.int64(padding_factor),
            projector_max_r=np.int64(projector_max_r),
        )


def _maybe_dump_k_class_significance_batch(
    *,
    experiment_dataset,
    indices,
    n_classes: int,
    rotations,
    translations,
    class_weight_mats,
    batch_sig_mask,
    batch_n_sig,
    hard_assignment_batch,
    class_assignment_batch,
    global_log_z,
    class_log_z_values,
    best_score,
    max_posterior,
    rotation_log_prior_padded,
    batch_translation_log_prior,
    class_log_priors,
    current_size,
    adaptive_fraction,
    max_significants,
    target_local_positions=None,
    target_scores_pre_prior_per_class=None,
    target_scores_with_prior_per_class=None,
    projected_reference_rotation_ids=None,
    projected_reference_per_class=None,
    projected_reference_norm_score_per_class=None,
    projected_cross_score_per_class=None,
    shifted_data=None,
    ctf2_data=None,
    window_indices=None,
    half_weights_used=None,
    coarse_gaussian_shifted_corrected=None,
    coarse_gaussian_unshifted_corrected=None,
    coarse_gaussian_pixel_weight=None,
    coarse_gaussian_initial_diff2=None,
    coarse_gaussian_score_indices=None,
    translation_phase_source=None,
    relion_projector_half=None,
    relion_projector_r_max=None,
    projection_padding_factor=None,
    relion_f32_sum_weight=None,
    relion_f32_significant_weight=None,
    relion_f32_cutoff_count=None,
    score_capture_mode="intrusive_per_block_host_materialization",
    debug_iteration=None,
):
    """Env-gated debug dump for the K-class significance pass.

    File naming matches the single-class dump so existing diff tooling works.
    The payload extends the K=1 schema with per-class fields and an explicit
    ``n_classes`` scalar so the user can decode the joint candidate space.
    """

    if not _significance_debug_dump_matches(
        current_size=current_size,
        debug_iteration=debug_iteration,
    ):
        return
    dump_dir = os.environ["RECOVAR_SIGNIFICANCE_DUMP_DIR"]
    target_original_indices = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
    target_iteration = os.environ.get("RECOVAR_SIGNIFICANCE_DUMP_ITERATION")

    local_indices = np.asarray(indices, dtype=np.int64)
    original_indices = original_image_indices(experiment_dataset, local_indices)

    os.makedirs(dump_dir, exist_ok=True)
    n_rot = int(rotations.shape[0])
    n_trans = int(translations.shape[0])

    weights_per_class = np.stack(
        [np.asarray(mat, dtype=np.float64) for mat in class_weight_mats],
        axis=1,
    )
    sig_mask_array = np.asarray(batch_sig_mask, dtype=bool)
    sig_mask_full = sig_mask_array.reshape(
        sig_mask_array.shape[0],
        n_classes,
        n_rot * n_trans,
    )[: local_indices.shape[0]]
    class_log_z_stack = np.stack(
        [np.asarray(class_log_z, dtype=np.float64) for class_log_z in class_log_z_values],
        axis=1,
    )

    flat_indices = np.arange(n_classes * n_rot * n_trans, dtype=np.int32)
    class_indices_flat = (flat_indices // (n_rot * n_trans)).astype(np.int32)
    rot_indices_flat = ((flat_indices % (n_rot * n_trans)) // n_trans).astype(np.int32)
    trans_indices_flat = (flat_indices % n_trans).astype(np.int32)

    # Build a map from local_pos to dump-target index (row in
    # target_scores_pre_prior_per_class[c]) so we can pick the right
    # per-class raw-score slab for each saved particle.
    target_pos_to_dump_row = None
    if target_local_positions is not None:
        target_pos_to_dump_row = {int(p): row for row, p in enumerate(np.asarray(target_local_positions).tolist())}

    projector_half_per_class = None
    if relion_projector_half is not None:
        projector_half_per_class = np.stack(
            [np.asarray(value, dtype=np.complex64) for value in relion_projector_half],
            axis=0,
        )
        if projector_half_per_class.shape[0] != n_classes:
            raise ValueError(
                "RELION projector dump class count differs from significance class count: "
                f"{projector_half_per_class.shape[0]} != {n_classes}",
            )

    for local_pos, original_idx in enumerate(original_indices):
        if int(original_idx) not in target_original_indices:
            continue
        weights_full = weights_per_class[local_pos].reshape(-1)
        sig_mask = sig_mask_full[local_pos].reshape(-1)
        sig_indices = np.flatnonzero(sig_mask).astype(np.int32)
        trans_prior = None
        if batch_translation_log_prior is not None:
            prior_arr = np.asarray(batch_translation_log_prior)
            trans_prior = prior_arr if prior_arr.ndim == 1 else prior_arr[local_pos]
        rot_prior_arr = (
            np.asarray(rotation_log_prior_padded, dtype=np.float64)[:, :n_rot]
            if rotation_log_prior_padded is not None
            else None
        )

        # Per-class raw scores (pre-prior and with-prior) for this image,
        # if the engine collected them. Shape per class: (n_rot, n_trans).
        scores_pre_prior_per_class = None
        scores_with_prior_per_class = None
        if target_pos_to_dump_row is not None and target_scores_pre_prior_per_class is not None:
            dump_row = target_pos_to_dump_row.get(int(local_pos))
            if dump_row is not None:
                scores_pre_prior_per_class = np.stack(
                    [np.asarray(arr[dump_row], dtype=np.float64) for arr in target_scores_pre_prior_per_class],
                    axis=0,
                )
                scores_with_prior_per_class = np.stack(
                    [np.asarray(arr[dump_row], dtype=np.float64) for arr in target_scores_with_prior_per_class],
                    axis=0,
                )

        image_rows = slice(local_pos * n_trans, (local_pos + 1) * n_trans)
        shifted_target = None
        if shifted_data is not None:
            shifted_target = np.asarray(shifted_data[image_rows], dtype=np.complex128)
        ctf2_target = None
        if ctf2_data is not None:
            ctf2_arr = np.asarray(ctf2_data)
            ctf2_target = (
                ctf2_arr[local_pos : local_pos + 1]
                if ctf2_arr.shape[0] == local_indices.shape[0]
                else ctf2_arr[image_rows]
            )

        iteration_suffix = "" if not target_iteration else f"_it{int(debug_iteration):03d}"
        out_path = os.path.join(
            dump_dir,
            f"significance_orig{int(original_idx):06d}{iteration_suffix}_cs"
            f"{(-1 if current_size is None else int(current_size)):03d}.npz",
        )
        save_kwargs = dict(
            original_index=np.int64(original_idx),
            local_index=np.int64(local_indices[local_pos]),
            debug_iteration=np.int64(-1 if debug_iteration is None else int(debug_iteration)),
            one_based_iteration=np.int64(-1 if debug_iteration is None else int(debug_iteration)),
            current_size=np.int64(-1 if current_size is None else int(current_size)),
            adaptive_fraction=np.float64(adaptive_fraction),
            max_significants=np.int64(max_significants),
            n_classes=np.int64(n_classes),
            n_rot=np.int64(n_rot),
            n_trans=np.int64(n_trans),
            weights_full=weights_full,
            weights_per_class=weights_per_class[local_pos],
            significant_mask=sig_mask,
            significant_indices=sig_indices,
            n_significant=np.int64(batch_n_sig[local_pos]),
            hard_assignment=np.int64(hard_assignment_batch[local_pos]),
            class_assignment=np.int64(class_assignment_batch[local_pos]),
            normalization_log_z=np.float64(global_log_z[local_pos]),
            class_log_z=class_log_z_stack[local_pos],
            best_score=np.float64(best_score[local_pos]),
            max_posterior=np.float64(max_posterior[local_pos]),
            rotations=np.asarray(rotations),
            translations=np.asarray(translations),
            class_indices=class_indices_flat,
            rot_indices=rot_indices_flat,
            trans_indices=trans_indices_flat,
            class_log_priors=np.asarray(class_log_priors, dtype=np.float64),
            rotation_log_prior=(rot_prior_arr if rot_prior_arr is not None else np.empty((0,), dtype=np.float64)),
            translation_log_prior=(
                np.asarray(trans_prior, dtype=np.float64)
                if trans_prior is not None
                else np.empty((0,), dtype=np.float64)
            ),
            shifted_data=(
                shifted_target
                if shifted_target is not None
                else np.empty((0,), dtype=np.complex128)
            ),
            ctf2_data=(
                np.asarray(ctf2_target, dtype=np.float64)
                if ctf2_target is not None
                else np.empty((0,), dtype=np.float64)
            ),
            window_indices=(
                np.asarray(window_indices, dtype=np.int32)
                if window_indices is not None
                else np.empty((0,), dtype=np.int32)
            ),
            half_weights=(
                np.asarray(half_weights_used, dtype=np.float64)
                if half_weights_used is not None
                else np.empty((0,), dtype=np.float64)
            ),
            coarse_gaussian_unshifted_corrected=(
                np.asarray(coarse_gaussian_unshifted_corrected[local_pos])
                if coarse_gaussian_unshifted_corrected is not None
                else np.empty((0,), dtype=np.complex64)
            ),
            coarse_gaussian_shifted_corrected=(
                np.asarray(coarse_gaussian_shifted_corrected[local_pos], dtype=np.complex64)
                if coarse_gaussian_shifted_corrected is not None
                else np.empty((0,), dtype=np.complex64)
            ),
            coarse_gaussian_pixel_weight=(
                np.asarray(coarse_gaussian_pixel_weight[local_pos])
                if coarse_gaussian_pixel_weight is not None
                else np.empty((0,), dtype=np.float32)
            ),
            coarse_gaussian_initial_diff2=(
                np.asarray(coarse_gaussian_initial_diff2[local_pos])
                if coarse_gaussian_initial_diff2 is not None
                else np.empty((0,), dtype=np.float32)
            ),
            coarse_gaussian_score_indices=(
                np.asarray(coarse_gaussian_score_indices, dtype=np.int32)
                if coarse_gaussian_score_indices is not None
                else np.empty((0,), dtype=np.int32)
            ),
            translation_phase_source=(
                np.asarray(translation_phase_source)
                if translation_phase_source is not None
                else np.empty((0, 2), dtype=np.float64)
            ),
            relion_projector_half_per_class=(
                projector_half_per_class
                if projector_half_per_class is not None
                else np.empty((0,), dtype=np.complex64)
            ),
            relion_projector_r_max=np.int64(
                -1 if relion_projector_r_max is None else int(relion_projector_r_max)
            ),
            projection_padding_factor=np.int64(
                -1 if projection_padding_factor is None else int(projection_padding_factor)
            ),
            relion_f32_sum_weight=(
                np.float32(np.asarray(relion_f32_sum_weight)[local_pos])
                if relion_f32_sum_weight is not None
                else np.float32(np.nan)
            ),
            relion_f32_significant_weight=(
                np.float32(np.asarray(relion_f32_significant_weight)[local_pos])
                if relion_f32_significant_weight is not None
                else np.float32(np.nan)
            ),
            relion_f32_cutoff_count=(
                np.int32(np.asarray(relion_f32_cutoff_count)[local_pos])
                if relion_f32_cutoff_count is not None
                else np.int32(-1)
            ),
            score_capture_mode=np.asarray(str(score_capture_mode)),
        )
        if scores_pre_prior_per_class is not None:
            # Per-class raw recovar score (= -0.5 * residual in
            # `_e_step_block_scores`; differs from RELION's diff2 by the
            # per-image Xi2/2 constant which cancels in relative pose
            # comparisons). Shape (n_classes, n_rot, n_trans).
            save_kwargs["scores_pre_prior_per_class"] = scores_pre_prior_per_class
            save_kwargs["scores_with_prior_per_class"] = scores_with_prior_per_class
        if projected_reference_per_class is not None:
            projection_values = np.asarray(projected_reference_per_class)
            projection_ids = np.asarray(projected_reference_rotation_ids, dtype=np.int32)
            if projection_values.shape[:2] != (n_classes, projection_ids.size):
                raise ValueError(
                    "projected-reference dump must have shape "
                    f"({n_classes}, {projection_ids.size}, n_pixels), got {projection_values.shape}",
                )
            save_kwargs["projected_reference_rotation_ids"] = projection_ids
            save_kwargs["projected_reference_per_class"] = projection_values.astype(np.complex128)
            norm_scores = np.asarray(projected_reference_norm_score_per_class)
            cross_scores = np.asarray(projected_cross_score_per_class)
            expected_component_shape = (
                n_classes,
                local_indices.shape[0],
                projection_ids.size,
                n_trans,
            )
            if (
                norm_scores.shape != expected_component_shape
                or cross_scores.shape != expected_component_shape
            ):
                raise ValueError(
                    "projected score components must both have shape "
                    f"{expected_component_shape}, got "
                    f"{norm_scores.shape} and {cross_scores.shape}",
                )
            save_kwargs["projected_reference_norm_score_per_class"] = norm_scores[
                :, local_pos
            ].astype(np.float64)
            save_kwargs["projected_cross_score_per_class"] = cross_scores[
                :, local_pos
            ].astype(np.float64)
        np.savez_compressed(out_path, **save_kwargs)
        _maybe_stop_after_significance_dump(
            out_path,
            dump_dir=dump_dir,
            target_original_indices=target_original_indices,
            current_size=current_size,
            debug_iteration=debug_iteration,
        )
