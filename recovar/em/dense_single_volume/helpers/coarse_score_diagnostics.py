"""Host-only coarse-score diagnostics and qualification reporting.

Keep NumPy evidence summaries separate from JAX score/M-step kernels.
These reports do not establish scientific or performance acceptance by themselves.
"""

import hashlib
import operator

import numpy as np

from .significant_samples import significant_sample_ids

_COARSE_SELECTOR_WRAPPER_TARGETS = {
    "relion_coarse_diff2_projector_f32": "cuda_relion_coarse_diff2_projector_f32",
    "relion_coarse_diff2_projector_multistream_f32": (
        "cuda_relion_coarse_diff2_projector_multistream_f32"
    ),
}


def _coarse_gaussian_direct_macro_diagnostics(
    direct_scores,
    macro_scores,
    *,
    direct_support=None,
    macro_support=None,
):
    """Summarize paired direct-square and expanded-GEMM score surfaces.

    Inputs use the public coarse layout ``[image, class, rotation,
    translation]``.  This helper deliberately reports raw deltas, signed bias,
    winner margins, and exact discrete differences without defining a
    promotion tolerance.  Repeated H100 artifacts can therefore establish a
    native/repeat envelope without silently turning one observed delta into an
    acceptance threshold.
    """

    direct = np.asarray(direct_scores)
    macro = np.asarray(macro_scores)
    if direct.shape != macro.shape or direct.ndim != 4:
        raise ValueError(
            "paired coarse score diagnostics require equal "
            "[image, class, rotation, translation] arrays, got "
            f"{direct.shape} and {macro.shape}",
        )
    if direct.dtype != macro.dtype or direct.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(
            "paired coarse score diagnostics require one shared float32 or "
            f"float64 dtype, got {direct.dtype} and {macro.dtype}",
        )
    direct_flat = direct.reshape(direct.shape[0], -1)
    macro_flat = macro.reshape(macro.shape[0], -1)
    delta = macro_flat.astype(np.float64) - direct_flat.astype(np.float64)
    absolute_delta = np.abs(delta)
    relative_delta_to_direct = np.full(delta.shape, np.nan, dtype=np.float64)
    np.divide(
        delta,
        np.abs(direct_flat.astype(np.float64)),
        out=relative_delta_to_direct,
        where=direct_flat != 0,
    )

    def ordered_float_bits(values):
        values = np.asarray(values).copy()
        values[values == 0] = 0
        if values.dtype == np.float32:
            unsigned = values.view(np.uint32)
            sign_mask = np.uint32(1 << 31)
            ordered = np.where(
                (unsigned & sign_mask) != 0,
                ~unsigned,
                unsigned ^ sign_mask,
            ).astype(np.uint64)
        else:
            unsigned = values.view(np.uint64)
            sign_mask = np.uint64(1 << 63)
            ordered = np.where(
                (unsigned & sign_mask) != 0,
                ~unsigned,
                unsigned ^ sign_mask,
            )
        return ordered

    direct_ordered = ordered_float_bits(direct_flat)
    macro_ordered = ordered_float_bits(macro_flat)
    ulp_delta = np.maximum(direct_ordered, macro_ordered) - np.minimum(
        direct_ordered,
        macro_ordered,
    )
    nonfinite = ~np.isfinite(direct_flat) | ~np.isfinite(macro_flat)
    ulp_delta = np.where(nonfinite, np.iinfo(np.uint64).max, ulp_delta)

    def winner_and_margin(values):
        winner = np.argmax(values, axis=1).astype(np.int64)
        if values.shape[1] == 1:
            margin = np.full(values.shape[0], np.inf, dtype=np.float64)
        else:
            largest_two = np.partition(values.astype(np.float64), -2, axis=1)[:, -2:]
            margin = largest_two[:, 1] - largest_two[:, 0]
        return winner, margin

    direct_winner, direct_margin = winner_and_margin(direct_flat)
    macro_winner, macro_margin = winner_and_margin(macro_flat)
    diagnostics = {
        "score_delta": delta.reshape(direct.shape),
        "absolute_score_delta": absolute_delta.reshape(direct.shape),
        "relative_score_delta_to_direct": relative_delta_to_direct.reshape(direct.shape),
        "ulp_score_delta": ulp_delta.reshape(direct.shape),
        "score_precision_bits": np.asarray(direct.dtype.itemsize * 8, dtype=np.int64),
        "direct_max_abs_score_per_image": np.max(
            np.abs(direct_flat.astype(np.float64)),
            axis=1,
        ),
        "macro_max_abs_score_per_image": np.max(
            np.abs(macro_flat.astype(np.float64)),
            axis=1,
        ),
        "signed_mean_delta_per_image": np.mean(delta, axis=1),
        "rms_delta_per_image": np.sqrt(np.mean(delta * delta, axis=1)),
        "max_abs_delta_per_image": np.max(np.abs(delta), axis=1),
        "positive_delta_count_per_image": np.count_nonzero(delta > 0.0, axis=1),
        "negative_delta_count_per_image": np.count_nonzero(delta < 0.0, axis=1),
        "zero_delta_count_per_image": np.count_nonzero(delta == 0.0, axis=1),
        "exact_zero_direct_nonzero_macro_per_image": np.all(
            direct_flat == 0.0,
            axis=1,
        )
        & np.any(macro_flat != 0.0, axis=1),
        "direct_argmax": direct_winner,
        "macro_argmax": macro_winner,
        "argmax_equal": direct_winner == macro_winner,
        "direct_winner_margin": direct_margin,
        "macro_winner_margin": macro_margin,
    }
    if (direct_support is None) != (macro_support is None):
        raise ValueError("direct_support and macro_support must be supplied together")
    if direct_support is not None:
        direct_support = np.asarray(direct_support, dtype=bool)
        macro_support = np.asarray(macro_support, dtype=bool)
        if direct_support.shape != direct_flat.shape or macro_support.shape != direct_flat.shape:
            raise ValueError(
                "coarse support diagnostics must match flattened score surfaces, got "
                f"{direct_support.shape}, {macro_support.shape}, and {direct_flat.shape}",
            )
        diagnostics.update(
            direct_support=direct_support.reshape(direct.shape),
            macro_support=macro_support.reshape(direct.shape),
            support_equal=np.all(direct_support == macro_support, axis=1),
            support_symmetric_difference_count=np.count_nonzero(
                direct_support != macro_support,
                axis=1,
            ).astype(np.int64),
        )
    return diagnostics


def _coarse_gaussian_repeat_spread_diagnostics(score_delta_repeats):
    """Report repeat-to-repeat drift without defining an acceptance epsilon.

    ``score_delta_repeats`` has layout
    ``[repeat,image,class,rotation,translation]`` and should be assembled from
    immutable same-hardware paired captures.  The caller must compare this raw
    spread with native/direct repeats, scale panels, discrete outcomes, and
    final quality; this helper cannot promote the GEMM path by itself.
    """

    deltas = np.asarray(score_delta_repeats, dtype=np.float64)
    if deltas.ndim != 5 or deltas.shape[0] < 2:
        raise ValueError(
            "coarse GEMM repeat diagnostics require at least two "
            "[repeat,image,class,rotation,translation] score-delta surfaces",
        )
    flattened = deltas.reshape(deltas.shape[0], deltas.shape[1], -1)
    signed_mean_per_repeat_image = np.mean(flattened, axis=2)
    max_abs_per_repeat_image = np.max(np.abs(flattened), axis=2)
    return {
        "repeat_count": np.asarray(deltas.shape[0], dtype=np.int64),
        "signed_mean_delta_per_repeat_image": signed_mean_per_repeat_image,
        "max_abs_delta_per_repeat_image": max_abs_per_repeat_image,
        "signed_mean_repeat_spread_per_image": np.ptp(
            signed_mean_per_repeat_image,
            axis=0,
        ),
        "max_abs_repeat_spread_per_image": np.ptp(
            max_abs_per_repeat_image,
            axis=0,
        ),
        "elementwise_delta_repeat_spread": np.ptp(deltas, axis=0),
        "qualification_status": np.asarray(
            "NO_GO_raw_repeat_spread_requires_native_scale_discrete_quality_runtime_context"
        ),
    }


def _coarse_gaussian_scale_panel_diagnostics(
    operand_scales,
    score_deltas_by_scale,
    *,
    precision_bits: int,
):
    """Classify strictly growing multiscale cancellation drift as NO-GO.

    This is an ordering test over raw observations, not an epsilon threshold.
    A non-growing panel remains unqualified until same-hardware repeats,
    discrete outcomes, final quality, and clean runtime all pass.
    """

    scales = np.asarray(operand_scales, dtype=np.float64).reshape(-1)
    deltas = np.asarray(score_deltas_by_scale, dtype=np.float64)
    if scales.size < 2 or deltas.ndim != 5 or deltas.shape[0] != scales.size:
        raise ValueError(
            "coarse GEMM scale diagnostics require matching scale and "
            "[scale,image,class,rotation,translation] arrays with at least two scales",
        )
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0) or np.any(np.diff(scales) <= 0.0):
        raise ValueError("coarse GEMM operand scales must be finite, positive, and increasing")
    if int(precision_bits) not in (32, 64):
        raise ValueError("coarse GEMM precision_bits must be 32 or 64")
    flattened = deltas.reshape(scales.size, -1)
    max_abs_by_scale = np.max(np.abs(flattened), axis=1)
    signed_mean_by_scale = np.mean(flattened, axis=1)
    growing_steps = max_abs_by_scale[1:] > max_abs_by_scale[:-1]
    scale_amplified = bool(np.any(growing_steps))
    return {
        "operand_scales": scales,
        "precision_bits": np.asarray(precision_bits, dtype=np.int64),
        "signed_mean_delta_by_scale": signed_mean_by_scale,
        "max_abs_delta_by_scale": max_abs_by_scale,
        "scale_growth_steps": growing_steps,
        "scale_amplified": np.asarray(scale_amplified),
        "qualification_status": np.asarray(
            (
                "NO_GO_scale-amplified_drift"
                if scale_amplified
                else "NO_GO_unqualified_non-growing_scale_panel"
            )
        ),
    }


def _coarse_gaussian_qualification_decision(
    *,
    exact_arithmetic_equivalent: bool,
    repeat_stable: bool | None,
    unbiased_non_directional: bool | None,
    bounded_non_growing: bool | None,
    discrete_choices_equal: bool | None,
    final_basin_quality_equal: bool | None,
    material_runtime_win: bool | None,
    scale_amplified: bool | None,
    negative_implied_diff2: bool,
    nonfinite_scores: bool,
    exact_zero_cancellation_drift: bool,
):
    """Apply the coarse GEMM promotion policy without a numeric tolerance.

    Mathematically equivalent reduction-order noise may pass without bitwise
    score equality only after independent evidence establishes repeat
    stability, non-directional bias, bounded/non-growing scale and iteration
    behavior, exact discrete decisions, unchanged final basin/quality, and a
    material clean-run speedup.  Cancellation evidence, non-finite scores, or
    a negative implied diff2 is an unconditional NO-GO.  ``None`` means the
    corresponding empirical gate has not run and therefore cannot promote.
    """

    failures = []
    pending = []
    if not exact_arithmetic_equivalent:
        failures.append("not_exact-arithmetic-equivalent")
    if scale_amplified is True:
        failures.append("scale-amplified_drift")
    elif scale_amplified is None:
        pending.append("multiscale_growth")
    if negative_implied_diff2:
        failures.append("negative_implied_diff2")
    if nonfinite_scores:
        failures.append("nonfinite_scores")
    if exact_zero_cancellation_drift:
        failures.append("exact-zero_cancellation_drift")
    for name, value in (
        ("repeat_stability", repeat_stable),
        ("unbiased_non-directional", unbiased_non_directional),
        ("bounded_non-growing", bounded_non_growing),
        ("discrete_choices", discrete_choices_equal),
        ("final_basin_quality", final_basin_quality_equal),
        ("material_runtime_win", material_runtime_win),
    ):
        if value is None:
            pending.append(name)
        elif not value:
            failures.append(name)
    if failures:
        status = "NO_GO"
    elif pending:
        status = "NO_GO_UNQUALIFIED"
    else:
        status = "GO_STABLE_BOUNDED_MATHEMATICALLY_EQUIVALENT_NOISE"
    return {
        "status": status,
        "failure_reasons": tuple(failures),
        "pending_gates": tuple(pending),
        "requires_bitwise_score_identity": False,
        "requires_exact_discrete_identity": True,
    }


def _validate_coarse_selector_audit(audit: dict) -> dict:
    """Validate and normalize one host-observed coarse selector audit.

    A configured fused selector is not evidence that its wrapper ran.  The
    wrapper name, XLA target, and positive call/row counters are therefore
    required whenever the fused path is effective.  An inactive control is
    represented explicitly by ``None`` wrapper/target values and zero counts.
    """

    if not isinstance(audit, dict):
        raise TypeError("coarse selector audit must be a dict")
    required = {
        "score_mode",
        "translation_count",
        "requested_fused",
        "effective_fused",
        "requested_workers",
        "effective_workers",
        "requested_atomic",
        "effective_atomic",
        "wrapper",
        "target",
        "counts",
    }
    missing = sorted(required.difference(audit))
    if missing:
        raise ValueError(
            "coarse selector audit is missing fields: " + ", ".join(missing)
        )

    score_mode = audit["score_mode"]
    if score_mode not in {"gaussian", "normalized_cc"}:
        raise ValueError(
            f"coarse selector audit has unsupported score_mode={score_mode!r}"
        )

    integer_fields = {
        "translation_count": audit["translation_count"],
        "requested_workers": audit["requested_workers"],
        "effective_workers": audit["effective_workers"],
    }
    normalized_integers = {}
    for name, value in integer_fields.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise TypeError(f"coarse selector audit {name} must be an integer")
        normalized_integers[name] = int(value)
    translation_count = normalized_integers["translation_count"]
    requested_workers = normalized_integers["requested_workers"]
    effective_workers = normalized_integers["effective_workers"]
    if translation_count <= 0:
        raise ValueError("coarse selector audit translation_count must be positive")
    if requested_workers not in {0, 8} or effective_workers not in {0, 8}:
        raise ValueError(
            "coarse selector audit requested/effective workers must be 0 or 8"
        )

    prehalf_fields = {"requested_prehalf", "effective_prehalf"}
    present_prehalf_fields = prehalf_fields.intersection(audit)
    if present_prehalf_fields and present_prehalf_fields != prehalf_fields:
        raise ValueError(
            "coarse selector audit must provide requested/effective prehalf together"
        )

    normalized_booleans = {}
    for name in (
        "requested_fused",
        "effective_fused",
        "requested_atomic",
        "effective_atomic",
        *(sorted(prehalf_fields) if present_prehalf_fields else ()),
    ):
        value = audit[name]
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"coarse selector audit {name} must be boolean")
        normalized_booleans[name] = bool(value)
    requested_fused = normalized_booleans["requested_fused"]
    effective_fused = normalized_booleans["effective_fused"]
    requested_atomic = normalized_booleans["requested_atomic"]
    effective_atomic = normalized_booleans["effective_atomic"]
    requested_prehalf = normalized_booleans.get("requested_prehalf", False)
    effective_prehalf = normalized_booleans.get("effective_prehalf", False)
    if effective_fused and not requested_fused:
        raise ValueError("effective fused coarse selector was not requested")
    if effective_workers and requested_workers != effective_workers:
        raise ValueError("effective coarse workers do not match the request")
    if effective_atomic and not requested_atomic:
        raise ValueError("effective native-atomic coarse reduction was not requested")
    if effective_prehalf and not requested_prehalf:
        raise ValueError("effective coarse prehalf weight was not requested")
    if effective_workers and not effective_fused:
        raise ValueError("effective coarse workers require the fused selector")
    if effective_atomic and not effective_fused:
        raise ValueError("effective native-atomic reduction requires the fused selector")
    if effective_prehalf and not effective_atomic:
        raise ValueError("effective coarse prehalf weight requires native-atomic reduction")
    if effective_fused and score_mode != "gaussian":
        raise ValueError("the fused coarse selector is Gaussian-only")
    if effective_workers and score_mode != "gaussian":
        raise ValueError("coarse worker streams are Gaussian-only")
    if effective_atomic and (
        score_mode != "gaussian" or translation_count != 29
    ):
        raise ValueError(
            "effective native-atomic reduction requires the Gaussian T=29 gate"
        )

    counts = audit["counts"]
    if not isinstance(counts, dict):
        raise TypeError("coarse selector audit counts must be a dict")
    required_counts = {
        "fused_calls",
        "actual_rows",
        "multistream_calls",
        "native_atomic_selected_calls",
    }
    if present_prehalf_fields:
        required_counts.add("prehalf_selected_calls")
    missing_counts = sorted(required_counts.difference(counts))
    if missing_counts:
        raise ValueError(
            "coarse selector audit counts are missing fields: "
            + ", ".join(missing_counts)
        )
    normalized_counts = {}
    for name in sorted(required_counts):
        value = counts[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise TypeError(f"coarse selector audit count {name} must be an integer")
        value = int(value)
        if value < 0:
            raise ValueError(f"coarse selector audit count {name} must be non-negative")
        normalized_counts[name] = value

    wrapper = audit["wrapper"]
    target = audit["target"]
    if wrapper is not None and not isinstance(wrapper, str):
        raise TypeError("coarse selector audit wrapper must be a string or None")
    if target is not None and not isinstance(target, str):
        raise TypeError("coarse selector audit target must be a string or None")
    fused_calls = normalized_counts["fused_calls"]
    actual_rows = normalized_counts["actual_rows"]
    multistream_calls = normalized_counts["multistream_calls"]
    native_atomic_calls = normalized_counts["native_atomic_selected_calls"]
    prehalf_calls = normalized_counts.get("prehalf_selected_calls", 0)
    if effective_fused:
        expected_wrapper = (
            "relion_coarse_diff2_projector_multistream_f32"
            if effective_workers
            else "relion_coarse_diff2_projector_f32"
        )
        expected_target = _COARSE_SELECTOR_WRAPPER_TARGETS[expected_wrapper]
        if wrapper != expected_wrapper or target != expected_target:
            raise ValueError(
                "coarse selector audit observed the wrong wrapper/target: "
                f"{wrapper!r}/{target!r} != {expected_wrapper!r}/{expected_target!r}"
            )
        if fused_calls <= 0:
            raise ValueError("effective fused coarse selector recorded zero calls")
        if actual_rows <= 0:
            raise ValueError("effective fused coarse selector recorded zero actual rows")
        if actual_rows < fused_calls:
            raise ValueError("coarse selector actual rows cannot be smaller than calls")
        expected_multistream_calls = fused_calls if effective_workers else 0
        if multistream_calls != expected_multistream_calls:
            raise ValueError(
                "coarse selector multistream call count does not match the effective wrapper"
            )
        expected_atomic_calls = fused_calls if effective_atomic else 0
        if native_atomic_calls != expected_atomic_calls:
            raise ValueError(
                "coarse selector native-atomic call count does not match the effective reduction"
            )
        expected_prehalf_calls = fused_calls if effective_prehalf else 0
        if prehalf_calls != expected_prehalf_calls:
            raise ValueError(
                "coarse selector prehalf call count does not match the effective specialization"
            )
    else:
        if wrapper is not None or target is not None:
            raise ValueError("inactive coarse selector must not report a wrapper/target")
        if effective_workers or effective_atomic or effective_prehalf:
            raise ValueError(
                "inactive coarse selector cannot report effective workers/atomic/prehalf"
            )
        if any(normalized_counts.values()):
            raise ValueError("inactive coarse selector must report zero execution counts")

    normalized = dict(audit)
    normalized.update(normalized_integers)
    normalized.update(normalized_booleans)
    normalized["counts"] = normalized_counts
    return normalized



def _build_coarse_significance_support_audit(
    significant_sample_indices,
    *,
    samples_per_class: int,
    include_ids: bool = False,
) -> dict:
    """Hash every ordered class/image coarse support without changing it.

    The canonical byte stream for one row is four little-endian int64 header
    values ``(class, image, samples_per_class, selected_count)`` followed by
    the strictly increasing selected sample IDs as little-endian int64. Each
    length-prefixed row enters the aggregate digest in class-major/image-major
    order. Per-row digests and counts make any mismatch localizable while the
    aggregate digest provides a compact direct/hybrid equality gate.
    """

    try:
        total_size = operator.index(samples_per_class)
    except TypeError as error:
        raise ValueError("samples_per_class must be an integer") from error
    if total_size <= 0:
        raise ValueError("samples_per_class must be positive")
    if not isinstance(significant_sample_indices, (tuple, list)) or not significant_sample_indices:
        raise ValueError("support audit requires at least one class")
    n_images = None
    aggregate = hashlib.sha256()
    per_class_image_sha256: list[list[str]] = []
    per_class_counts: list[list[int]] = []
    per_class_ids: list[list[list[int]]] = []
    for class_index, rows in enumerate(significant_sample_indices):
        if not isinstance(rows, (tuple, list)):
            raise TypeError("support audit class rows must be a sequence")
        if n_images is None:
            n_images = len(rows)
            if n_images <= 0:
                raise ValueError("support audit requires at least one image")
        elif len(rows) != n_images:
            raise ValueError("support audit classes must cover the same images")
        row_digests = []
        row_counts = []
        row_ids: list[list[int]] = []
        for image_index, samples in enumerate(rows):
            ids = np.asarray(
                significant_sample_ids(samples, total_size),
                dtype=np.int64,
            ).reshape(-1)
            if (
                np.any(ids < 0)
                or np.any(ids >= total_size)
                or (ids.size > 1 and np.any(np.diff(ids) <= 0))
            ):
                raise ValueError(
                    "support audit requires unique, strictly increasing in-range IDs",
                )
            header = np.asarray(
                (class_index, image_index, total_size, ids.size),
                dtype="<i8",
            )
            ids_le = np.ascontiguousarray(ids.astype("<i8", copy=False))
            row_bytes = header.tobytes(order="C") + ids_le.tobytes(order="C")
            row_digests.append(hashlib.sha256(row_bytes).hexdigest())
            row_counts.append(int(ids.size))
            if include_ids:
                row_ids.append([int(value) for value in ids])
            aggregate.update(
                np.asarray((len(row_bytes),), dtype="<u8").tobytes(order="C"),
            )
            aggregate.update(row_bytes)
        per_class_image_sha256.append(row_digests)
        per_class_counts.append(row_counts)
        if include_ids:
            per_class_ids.append(row_ids)

    counts = np.ascontiguousarray(np.asarray(per_class_counts, dtype="<i8"))
    result = {
        "schema": "recovar.coarse_significance_support_audit.v2",
        "classification": "diagnostic_only",
        "canonical_encoding": (
            "class-major/image-major; uint64 row-byte-length; "
            "int64-le header(class,image,total,count); int64-le sorted IDs"
        ),
        "n_classes": len(per_class_counts),
        "n_images": int(n_images),
        "samples_per_class": total_size,
        "selected_count_sum": int(np.sum(counts, dtype=np.int64)),
        "selected_count_min": int(np.min(counts)),
        "selected_count_max": int(np.max(counts)),
        "per_class_image_selected_counts": per_class_counts,
        "per_class_image_selected_counts_sha256": hashlib.sha256(
            counts.tobytes(order="C"),
        ).hexdigest(),
        "support_ids_included": bool(include_ids),
        "per_class_image_support_sha256": per_class_image_sha256,
        "aggregate_support_sha256": aggregate.hexdigest(),
    }
    if include_ids:
        # Explicit opt-in avoids retaining a potentially enormous full-support
        # diagnostic in other geometries. GF46 iteration 181 has only ~5,900
        # selected IDs across all 1,000 images.
        result["per_class_image_support_ids"] = per_class_ids
    return result



def _coarse_selector_audit_from_full_stats(full_stats: dict) -> dict:
    """Require a valid execution audit at the coarse-score boundary."""

    if not isinstance(full_stats, dict):
        raise RuntimeError("K-class significance did not return coarse full_stats")
    if "coarse_selector_audit" not in full_stats:
        raise RuntimeError(
            "K-class significance did not return a coarse selector execution audit"
        )
    try:
        return _validate_coarse_selector_audit(full_stats["coarse_selector_audit"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("K-class significance returned an invalid coarse selector audit") from error



def _with_coarse_selector_audit(result, audit: dict | None):
    """Seal the validated coarse audit into a result profile summary."""

    if audit is None:
        return result
    try:
        validated = _validate_coarse_selector_audit(audit)
    except (TypeError, ValueError) as error:
        raise RuntimeError("cannot propagate an invalid coarse selector audit") from error
    profile_summary = dict(result.profile_summary or {})
    profile_summary["coarse_selector_audit"] = validated
    return result._replace(profile_summary=profile_summary)



def _with_coarse_significance_diagnostics(
    result,
    *,
    selector_audit: dict | None,
    support_audit: dict | None,
    hybrid_stats: dict | None,
    exact_coarse_operand_assembly: dict | None = None,
):
    """Propagate exact coarse-support and hybrid telemetry to InitialModel."""

    result = _with_coarse_selector_audit(result, selector_audit)
    additions = {
        key: dict(value)
        for key, value in (
            ("coarse_significance_support_audit", support_audit),
            ("coarse_gaussian_gemm_hybrid", hybrid_stats),
            (
                "exact_coarse_operand_assembly",
                exact_coarse_operand_assembly,
            ),
        )
        if value is not None
    }
    if not additions:
        return result
    profile_summary = dict(result.profile_summary or {})
    profile_summary.update(additions)
    return result._replace(profile_summary=profile_summary)
