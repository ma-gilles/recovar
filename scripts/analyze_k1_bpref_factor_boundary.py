#!/usr/bin/env python3
"""Localize a selected K=1 RELION/RECOVAR BPref factor boundary.

This diagnostic consumes passive RELION factor captures and RECOVAR sparse
pass-2 dumps from the same physical iteration.  It never changes production
state.  Map acceptance remains signed shellwise FSC/FSC-AUC; the intermediate
comparisons here use exact equality and relative L2 only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .validate_relion_bpref_factor_capture import fnv1a64, load_factor_capture
else:
    from validate_relion_bpref_factor_capture import (  # type: ignore[no-redef]
        fnv1a64,
        load_factor_capture,
    )


SELECTION_SCHEMA = "recovar.em.k1_bpref_factor_panel.v1"
INERTNESS_SCHEMA = "recovar.em.k1_bpref_factor_capture_inertness.v1"
REPORT_SCHEMA = "recovar.em.k1_bpref_factor_boundary.v1"
# The qualified case-4 passive factor observer differs from RELION's live
# pre-scatter numerator by at most 3.51e-8 relative L2 and is exact for the
# denominator.  Keep margin above that observer floor without classifying the
# demonstrated 1.1e-7--2.7e-7 operand residual as closed.
RELATIVE_L2_BOUND = 1.0e-7


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    _require(left.shape == right.shape and left.size > 0, "comparison shape changed or is empty")
    delta = right.astype(np.complex128) - left.astype(np.complex128)
    denominator = max(float(np.linalg.norm(left.astype(np.complex128))), np.finfo(np.float64).tiny)
    return {
        "shape": list(left.shape),
        "reference_dtype": str(left.dtype),
        "candidate_dtype": str(right.dtype),
        "exact_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "relative_l2_over_reference": float(np.linalg.norm(delta) / denominator),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
    }


def _distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.size > 0 and np.all(np.isfinite(array)), "invalid metric distribution")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(np.max(array)),
    }


def _rotation_map(
    factor_rotations: np.ndarray,
    recovar_rotations: np.ndarray,
) -> tuple[np.ndarray, float]:
    relion = np.asarray(factor_rotations["matrix"], dtype=np.float32).reshape(-1, 3, 3)
    relion = relion.transpose(0, 2, 1)
    recovar = np.asarray(recovar_rotations, dtype=np.float32).reshape(-1, 3, 3)
    distance = np.max(np.abs(relion[:, None] - recovar[None]), axis=(2, 3))
    nearest = np.argmin(distance, axis=1)
    error = distance[np.arange(relion.shape[0]), nearest]
    _require(
        np.all(error <= 1.0e-6) and np.unique(nearest).size == nearest.size,
        "RELION/RECOVAR fine rotation matrices do not map one-to-one",
    )
    return nearest.astype(np.int64), float(np.max(error, initial=0.0))


def _translation_map(
    factor_translations: np.ndarray,
    recovar_translations: np.ndarray,
    *,
    physical_image_size: int,
) -> tuple[np.ndarray, float]:
    relion = np.column_stack((factor_translations["x"], factor_translations["y"])).astype(np.float64)
    recovar = (
        -2.0
        * np.pi
        * np.asarray(recovar_translations, dtype=np.float64).reshape(-1, 2)
        / physical_image_size
    )
    distance = np.max(np.abs(relion[:, None] - recovar[None]), axis=2)
    nearest = np.argmin(distance, axis=1)
    error = distance[np.arange(relion.shape[0]), nearest]
    _require(
        np.all(error <= 1.0e-6) and np.unique(nearest).size == nearest.size,
        "RELION/RECOVAR fine translations do not map one-to-one",
    )
    return nearest.astype(np.int64), float(np.max(error, initial=0.0))


def _pixel_coordinates(centered_packed_indices: np.ndarray, physical_image_size: int) -> list[tuple[int, int]]:
    indices = np.asarray(centered_packed_indices, dtype=np.int64).reshape(-1)
    half_width = physical_image_size // 2 + 1
    return [
        (int(index % half_width), int(index // half_width - physical_image_size // 2))
        for index in indices
    ]


def _device_sums(
    probs: np.ndarray,
    shifted: np.ndarray,
    ctf2_over_nv: np.ndarray,
    *,
    require_h100: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    from recovar.em.dense_single_volume.local_backprojection import compute_local_mstep_sums

    devices = jax.devices()
    _require(bool(devices), "JAX reports no devices")
    device = devices[0]
    device_kind = str(getattr(device, "device_kind", device))
    if require_h100:
        _require(device.platform == "gpu" and "H100" in device_kind.upper(), "scientific comparison requires H100")
    inputs = (
        jnp.asarray(np.asarray(probs, dtype=np.float32)[None]),
        jnp.asarray(np.asarray(shifted, dtype=np.complex64)[None]),
        jnp.asarray(np.asarray(ctf2_over_nv, dtype=np.float32)[None]),
    )
    highest_num, highest_den = compute_local_mstep_sums(
        *inputs,
        relion_x_half=True,
        sequential_translation_reduction=False,
    )
    sequential_num, sequential_den = compute_local_mstep_sums(
        *inputs,
        relion_x_half=True,
        sequential_translation_reduction=True,
    )
    highest_num.block_until_ready()
    sequential_num.block_until_ready()
    return (
        np.asarray(jax.device_get(highest_num))[0],
        np.asarray(jax.device_get(highest_den))[0],
        np.asarray(jax.device_get(sequential_num))[0],
        np.asarray(jax.device_get(sequential_den))[0],
        {
            "platform": device.platform,
            "device_kind": device_kind,
            "devices": [str(item) for item in devices],
            "jax_backend": jax.default_backend(),
        },
    )


def _classify_localization(
    *,
    capture_self_closes: bool,
    same_posterior_operands_close: bool,
    sequential_summary_closes: bool,
    highest_summary_closes: bool,
) -> str:
    if not capture_self_closes:
        return "relion_factor_capture_does_not_reproduce_relion_summary"
    if not same_posterior_operands_close:
        return "bpref_operand_mismatch_before_translation_reduction"
    if sequential_summary_closes and not highest_summary_closes:
        return "recovar_translation_reduction_order_mismatch"
    if sequential_summary_closes and highest_summary_closes:
        return "particle_prescatter_boundary_closes"
    return "translation_reduction_or_unmeasured_operand_mismatch"


def _first_primitive_boundary(comparisons: dict[str, dict[str, Any]]) -> str:
    for name in (
        "ctf_with_scale",
        "inverse_noise",
        "weighted_ctf",
        "translated_fourier_image",
        "same_posterior_numerator_terms",
        "same_posterior_denominator_terms",
    ):
        if float(comparisons[name]["relative_l2_over_reference"]) > RELATIVE_L2_BOUND:
            return name
    return "particle_prescatter_boundary_closes"


def _first_cross_engine_boundary(particles: list[dict[str, Any]]) -> str:
    """Report the earliest unequal boundary represented by the fixed panel."""

    if any(not bool(particle["support_exact"]) for particle in particles):
        return "support"
    if any(
        float(
            particle["comparisons"]["posterior_common_support"][
                "relative_l2_over_reference"
            ]
        )
        > RELATIVE_L2_BOUND
        for particle in particles
    ):
        return "posterior"
    if any(not bool(particle["same_posterior_operands_close"]) for particle in particles):
        boundaries = sorted(
            {
                str(particle["first_primitive_boundary"])
                for particle in particles
                if not bool(particle["same_posterior_operands_close"])
            }
        )
        return "bpref_primitive:" + ",".join(boundaries)
    if any(
        not bool(particle["sequential_summary_closes"])
        and not bool(particle["highest_summary_closes"])
        for particle in particles
    ):
        return "translation_reduction_or_unmeasured_operand"
    if any(
        bool(particle["sequential_summary_closes"])
        and not bool(particle["highest_summary_closes"])
        for particle in particles
    ):
        return "translation_reduction_order"
    return "accumulator_destination_and_inter_particle_reduction"


def _recovar_capture_path(directory: Path, original_index: int, current_size: int) -> Path:
    matches = sorted(directory.glob(f"pass2_orig{original_index:06d}_cs{current_size:03d}.npz"))
    matches += sorted(
        directory.glob(
            f"pass2_orig{original_index:06d}_class001_cs{current_size:03d}.npz"
        )
    )
    _require(len(matches) == 1, f"expected one RECOVAR pass-2 capture for source row {original_index}")
    return matches[0]


def _capture_stack_indices(selection: dict[str, Any], target_stacks: list[int]) -> list[int]:
    """Return the full native capture set when analyzing a qualified subset."""
    raw = selection.get("capture_stack_indices_one_based", target_stacks)
    capture_stacks = [int(value) for value in raw]
    _require(capture_stacks, "native capture stack set is empty")
    _require(
        len(capture_stacks) == len(set(capture_stacks)) and all(value > 0 for value in capture_stacks),
        "native capture stack set must contain unique positive identities",
    )
    _require(
        set(target_stacks).issubset(capture_stacks),
        "analysis targets are not a subset of the native capture stack set",
    )
    return capture_stacks


def _compare_particle(
    *,
    target: dict[str, Any],
    factor,
    recovar: dict[str, np.ndarray],
    physical_image_size: int,
    current_size: int,
    require_h100: bool,
) -> dict[str, Any]:
    original_index = int(target["original_index_zero_based"])
    stack_index = int(target["stack_index_one_based"])
    _require(not factor.geometry_only, f"stack {stack_index}: factor capture is geometry-only")
    _require(factor.stack_index == stack_index, f"stack {stack_index}: native identity changed")
    _require(
        int(factor.header[9]) == int(target["physical_iteration"]),
        f"stack {stack_index}: native physical iteration changed",
    )
    _require(
        int(factor.header[16]) == current_size // 2 + 1
        and int(factor.header[17]) == current_size
        and int(factor.header[18]) == 1,
        f"stack {stack_index}: native current-size Fourier layout changed",
    )
    _require(int(recovar["original_index"]) == original_index, f"stack {stack_index}: RECOVAR identity changed")
    _require(int(recovar["current_size"]) == current_size, f"stack {stack_index}: current size changed")

    rotation_map, rotation_error = _rotation_map(factor.rotations, recovar["rotations"])
    translation_map, translation_error = _translation_map(
        factor.translations,
        recovar["fine_translations"],
        physical_image_size=physical_image_size,
    )
    recovar_prob = np.asarray(recovar["reconstruction_probs"], dtype=np.float32)
    recovar_mask = np.asarray(recovar["reconstruction_mask"], dtype=bool)
    shifted = np.asarray(recovar["shifted_recon"], dtype=np.complex64)
    ctf2 = np.asarray(recovar["ctf2_over_nv_recon"], dtype=np.float32)
    _require(recovar_prob.shape == recovar_mask.shape, f"stack {stack_index}: posterior shape changed")
    _require(
        recovar_prob.shape == (np.asarray(recovar["rotations"]).shape[0], shifted.shape[0]),
        f"stack {stack_index}: rotation/translation shape changed",
    )

    coordinates = _pixel_coordinates(recovar["recon_window_indices"], physical_image_size)
    _require(len(set(coordinates)) == len(coordinates), f"stack {stack_index}: duplicate RECOVAR pixels")
    native_pixel_lookup = {
        (int(x), int(y)): row
        for row, (x, y) in enumerate(zip(factor.pixels["x"], factor.pixels["y"]))
    }
    _require(
        all(coordinate in native_pixel_lookup for coordinate in coordinates),
        f"stack {stack_index}: native pixel panel does not contain RECOVAR reconstruction window",
    )
    pixel_rows = np.asarray([native_pixel_lookup[coordinate] for coordinate in coordinates], dtype=np.int64)
    score_coordinates = _pixel_coordinates(recovar["window_indices"], physical_image_size)
    score_coordinate_lookup = {
        coordinate: row for row, coordinate in enumerate(score_coordinates)
    }
    _require(
        all(coordinate in score_coordinate_lookup for coordinate in coordinates),
        f"stack {stack_index}: reconstruction window is not contained in the score window",
    )
    score_rows = np.asarray(
        [score_coordinate_lookup[coordinate] for coordinate in coordinates],
        dtype=np.int64,
    )

    accepted_flat = np.flatnonzero((factor.hypotheses["flags"] & 1) != 0)
    _require(accepted_flat.size > 0, f"stack {stack_index}: native accepted support is empty")
    accepted = factor.hypotheses[accepted_flat]
    mapped_rotation = rotation_map[np.asarray(accepted["orientation_local"], dtype=np.int64)]
    mapped_translation = translation_map[np.asarray(accepted["translation"], dtype=np.int64)]
    mapped_keys = np.column_stack((mapped_rotation, mapped_translation))
    _require(
        np.unique(mapped_keys, axis=0).shape[0] == mapped_keys.shape[0],
        f"stack {stack_index}: native hypotheses do not map one-to-one",
    )
    posterior = np.asarray(accepted["posterior_over_weight_norm"], dtype=np.float32)
    _require(np.all(posterior > 0), f"stack {stack_index}: accepted normalized posterior is not positive")
    native_prob = np.zeros(recovar_prob.shape, dtype=np.float32)
    native_mask = np.zeros(recovar_mask.shape, dtype=bool)
    native_prob[mapped_rotation, mapped_translation] = posterior
    native_mask[mapped_rotation, mapped_translation] = True

    terms = factor.terms.reshape(accepted_flat.size, factor.pixels.size)[:, pixel_rows]
    _require(
        np.array_equal(terms[:, 0]["orientation_local"], accepted["orientation_local"])
        and np.array_equal(terms[:, 0]["translation"], accepted["translation"]),
        f"stack {stack_index}: native factor term/hypothesis order changed",
    )
    scale_num = np.float32(physical_image_size**2)
    scale_den = np.float32(physical_image_size**4)
    native_term_num = -(terms["term_re"] + 1j * terms["term_im"]) / scale_num
    native_term_den = terms["weight_term"] / scale_den
    recovar_term_num = posterior[:, None] * shifted[mapped_translation]
    recovar_term_den = posterior[:, None] * ctf2[None]
    native_base_num = native_term_num / posterior[:, None]
    native_base_den = native_term_den / posterior[:, None]

    native_ctf = factor.pixels["ctf"][pixel_rows].astype(np.float32)
    native_inverse_noise = factor.pixels["minvsigma2"][pixel_rows].astype(np.float32)
    recovar_ctf = (
        np.asarray(recovar["direct_ctf_rfloat_score"], dtype=np.float64)[score_rows]
        * np.float64(np.asarray(recovar["batch_scale_correction"]).item())
    ).astype(np.float32)
    recovar_inverse_noise = np.asarray(
        recovar["direct_inverse_noise_score"], dtype=np.float32
    )[score_rows]
    recovar_inverse_noise_native = (recovar_inverse_noise * scale_den).astype(np.float32)
    native_weighted_ctf = (
        -terms["weighted_ctf"] / posterior[:, None] / scale_den
    ).astype(np.float32)
    recovar_weighted_ctf = (recovar_ctf * recovar_inverse_noise).astype(np.float32)
    recovar_weighted_ctf_grid = np.broadcast_to(
        recovar_weighted_ctf,
        native_weighted_ctf.shape,
    )
    native_translated = (
        terms["translated_re"] + np.complex64(1j) * terms["translated_im"]
    ).astype(np.complex64) * scale_num
    recovar_shifted = shifted[mapped_translation]
    valid_weight = np.abs(recovar_weighted_ctf_grid) > np.float32(1.0e-20)
    _require(np.any(valid_weight), f"stack {stack_index}: weighted CTF is zero on the full window")
    recovar_translated = np.zeros_like(recovar_shifted)
    recovar_translated[valid_weight] = (
        recovar_shifted[valid_weight] / recovar_weighted_ctf_grid[valid_weight]
    ).astype(np.complex64)

    native_term_num_grid = np.zeros((recovar_prob.shape[0], shifted.shape[1]), dtype=np.complex64)
    native_term_den_grid = np.zeros((recovar_prob.shape[0], shifted.shape[1]), dtype=np.float32)
    for row in range(accepted_flat.size):
        native_term_num_grid[mapped_rotation[row]] += native_term_num[row]
        native_term_den_grid[mapped_rotation[row]] += native_term_den[row]

    native_summary_num = np.zeros_like(native_term_num_grid)
    native_summary_den = np.zeros_like(native_term_den_grid)
    native_summary_support = np.zeros_like(native_term_den_grid, dtype=bool)
    coordinate_to_recovar = {coordinate: row for row, coordinate in enumerate(coordinates)}
    outside_summary_rows = []
    for summary in factor.summaries:
        coordinate = (int(summary["x"]), int(summary["y"]))
        recovar_pixel = coordinate_to_recovar.get(coordinate)
        if recovar_pixel is None:
            outside_summary_rows.append((int(summary["orientation_local"]), *coordinate))
            continue
        recovar_rotation = int(rotation_map[int(summary["orientation_local"])])
        _require(
            not native_summary_support[recovar_rotation, recovar_pixel],
            f"stack {stack_index}: duplicate mapped native summary row",
        )
        native_summary_support[recovar_rotation, recovar_pixel] = True
        native_summary_num[recovar_rotation, recovar_pixel] = -np.complex64(
            complex(float(summary["source_re"]), float(summary["source_im"]))
        ) / scale_num
        native_summary_den[recovar_rotation, recovar_pixel] = np.float32(summary["source_weight"]) / scale_den
    _require(not outside_summary_rows, f"stack {stack_index}: native summary contains pixels outside RECOVAR window")
    support = native_summary_support
    _require(np.any(support), f"stack {stack_index}: native summary support is empty")

    highest_num, highest_den, sequential_num, sequential_den, device = _device_sums(
        native_prob,
        shifted,
        ctf2,
        require_h100=require_h100,
    )
    common_support = native_mask & recovar_mask
    _require(np.any(common_support), f"stack {stack_index}: native/RECOVAR support intersection is empty")
    comparisons = {
        "posterior_common_support": _metric(native_prob[common_support], recovar_prob[common_support]),
        "ctf_with_scale": _metric(native_ctf, -recovar_ctf),
        "inverse_noise": _metric(native_inverse_noise, recovar_inverse_noise_native),
        "weighted_ctf": _metric(native_weighted_ctf, recovar_weighted_ctf_grid),
        "translated_fourier_image": _metric(
            native_translated[valid_weight], recovar_translated[valid_weight]
        ),
        "same_posterior_numerator_terms": _metric(native_term_num, recovar_term_num),
        "same_posterior_denominator_terms": _metric(native_term_den, recovar_term_den),
        "base_numerator_operand": _metric(native_base_num, shifted[mapped_translation]),
        "base_denominator_operand": _metric(native_base_den, np.broadcast_to(ctf2, native_base_den.shape)),
        "relion_terms_to_relion_summary_numerator": _metric(
            native_summary_num[support], native_term_num_grid[support]
        ),
        "relion_terms_to_relion_summary_denominator": _metric(
            native_summary_den[support], native_term_den_grid[support]
        ),
        "relion_summary_to_recovar_highest_numerator": _metric(
            native_summary_num[support], highest_num[support]
        ),
        "relion_summary_to_recovar_highest_denominator": _metric(
            native_summary_den[support], highest_den[support]
        ),
        "relion_summary_to_recovar_sequential_numerator": _metric(
            native_summary_num[support], sequential_num[support]
        ),
        "relion_summary_to_recovar_sequential_denominator": _metric(
            native_summary_den[support], sequential_den[support]
        ),
        "native_translated_with_recovar_weighted_ctf": _metric(
            native_base_num,
            (native_translated * recovar_weighted_ctf_grid).astype(np.complex64),
        ),
        "recovar_translated_with_native_weighted_ctf": _metric(
            native_base_num,
            (recovar_translated * native_weighted_ctf).astype(np.complex64),
        ),
        "native_internal_numerator": _metric(
            native_base_num,
            (native_translated * native_weighted_ctf).astype(np.complex64),
        ),
    }
    capture_self_closes = all(
        comparisons[name]["relative_l2_over_reference"] <= RELATIVE_L2_BOUND
        for name in (
            "relion_terms_to_relion_summary_numerator",
            "relion_terms_to_relion_summary_denominator",
        )
    )
    same_posterior_operands_close = all(
        comparisons[name]["relative_l2_over_reference"] <= RELATIVE_L2_BOUND
        for name in ("same_posterior_numerator_terms", "same_posterior_denominator_terms")
    )
    sequential_summary_closes = all(
        comparisons[name]["relative_l2_over_reference"] <= RELATIVE_L2_BOUND
        for name in (
            "relion_summary_to_recovar_sequential_numerator",
            "relion_summary_to_recovar_sequential_denominator",
        )
    )
    highest_summary_closes = all(
        comparisons[name]["relative_l2_over_reference"] <= RELATIVE_L2_BOUND
        for name in (
            "relion_summary_to_recovar_highest_numerator",
            "relion_summary_to_recovar_highest_denominator",
        )
    )
    return {
        "original_index_zero_based": original_index,
        "stack_index_one_based": stack_index,
        "role": str(target["role"]),
        "native_part_id": int(factor.header[11]),
        "native_mpi_rank": int(factor.header[14]),
        "native_thread_id": int(factor.header[15]),
        "factor_path": str(factor.path.resolve()),
        "factor_sha256": factor.sha256,
        "recovar_capture_path": str(Path(str(recovar["_path"])).resolve()),
        "recovar_capture_sha256": _sha256(Path(str(recovar["_path"]))),
        "rotation_count_native": int(factor.rotations.size),
        "rotation_count_recovar": int(recovar_prob.shape[0]),
        "translation_count_native": int(factor.translations.size),
        "accepted_hypothesis_count_native": int(accepted_flat.size),
        "accepted_hypothesis_count_recovar": int(np.count_nonzero(recovar_mask)),
        "support_intersection_count": int(np.count_nonzero(common_support)),
        "support_union_count": int(np.count_nonzero(native_mask | recovar_mask)),
        "support_exact": bool(np.array_equal(native_mask, recovar_mask)),
        "rotation_map_max_abs": rotation_error,
        "translation_map_max_abs": translation_error,
        "comparisons": comparisons,
        "capture_self_closes": capture_self_closes,
        "same_posterior_operands_close": same_posterior_operands_close,
        "sequential_summary_closes": sequential_summary_closes,
        "highest_summary_closes": highest_summary_closes,
        "first_primitive_boundary": _first_primitive_boundary(comparisons),
        "classification": _classify_localization(
            capture_self_closes=capture_self_closes,
            same_posterior_operands_close=same_posterior_operands_close,
            sequential_summary_closes=sequential_summary_closes,
            highest_summary_closes=highest_summary_closes,
        ),
        "device": device,
    }


def analyze(
    *,
    factor_directory: Path,
    recovar_capture_directory: Path,
    selection_json: Path,
    capture_inertness_json: Path,
    require_h100: bool = True,
) -> dict[str, Any]:
    selection = json.loads(selection_json.read_text())
    _require(selection.get("schema") == SELECTION_SCHEMA, "selection schema changed")
    case_id = int(selection["case_id"])
    physical_iteration = int(selection["physical_iteration"])
    physical_image_size = int(selection["physical_image_size"])
    current_size = int(selection["current_size"])
    _require(case_id > 0 and physical_iteration > 0, "invalid case or iteration")
    _require(physical_image_size > 0 and physical_image_size % 2 == 0, "invalid physical image size")
    _require(0 < current_size <= physical_image_size and current_size % 2 == 0, "invalid current size")
    targets = selection.get("targets")
    _require(isinstance(targets, list) and targets, "factor panel is empty")
    target_stacks = [int(target["stack_index_one_based"]) for target in targets]
    _require(len(set(target_stacks)) == len(target_stacks), "factor panel stack identities are duplicated")
    capture_stacks = _capture_stack_indices(selection, target_stacks)
    _require(
        all(int(target["expected_mpi_rank"]) > 0 for target in targets),
        "factor panel is missing a positive expected follower rank",
    )

    inertness = json.loads(capture_inertness_json.read_text())
    _require(inertness.get("schema") == INERTNESS_SCHEMA, "capture inertness schema changed")
    _require(inertness.get("qualified") is True, "factor capture inertness did not qualify")
    _require(int(inertness["case_id"]) == case_id, "capture inertness case changed")
    _require(int(inertness["physical_iteration"]) == physical_iteration, "capture inertness iteration changed")
    _require(
        sorted(int(value) for value in inertness["target_stack_indices_one_based"])
        == sorted(target_stacks),
        "capture inertness target identities changed",
    )

    factor_paths = sorted(factor_directory.glob("*.bpre-v2.bin"))
    _require(len(factor_paths) == len(capture_stacks), "full factor artifact count changed")
    factors = [load_factor_capture(path) for path in factor_paths]
    factors_by_stack = {factor.stack_index: factor for factor in factors}
    _require(set(factors_by_stack) == set(capture_stacks), "native factor stack identities changed")
    selected_set_hash = fnv1a64(",".join(str(value) for value in capture_stacks))
    target_set_hash = fnv1a64(",".join(str(value) for value in target_stacks))
    expected_rank_by_stack = {
        int(target["stack_index_one_based"]): int(target["expected_mpi_rank"])
        for target in targets
    }
    _require(
        all(factor.header[36] == selected_set_hash for factor in factors),
        "native factor selected-set hash changed",
    )
    _require(
        all(
            factors_by_stack[stack].header[14] == expected_rank_by_stack[stack]
            for stack in target_stacks
        ),
        "native factor MPI rank changed",
    )

    particles = []
    for target in targets:
        target = {**target, "physical_iteration": physical_iteration}
        original_index = int(target["original_index_zero_based"])
        stack_index = int(target["stack_index_one_based"])
        recovar_path = _recovar_capture_path(recovar_capture_directory, original_index, current_size)
        with np.load(recovar_path, allow_pickle=False) as archive:
            recovar = {name: archive[name] for name in archive.files}
        recovar["_path"] = str(recovar_path)
        particles.append(
            _compare_particle(
                target=target,
                factor=factors_by_stack[stack_index],
                recovar=recovar,
                physical_image_size=physical_image_size,
                current_size=current_size,
                require_h100=require_h100,
            )
        )

    classifications = [particle["classification"] for particle in particles]
    support_exact_count = sum(bool(particle["support_exact"]) for particle in particles)
    posterior_close_count = sum(
        float(particle["comparisons"]["posterior_common_support"]["relative_l2_over_reference"])
        <= RELATIVE_L2_BOUND
        for particle in particles
    )
    if all(value == "particle_prescatter_boundary_closes" for value in classifications):
        aggregate = "fixed_panel_particle_prescatter_boundary_closes"
    else:
        aggregate = next(value for value in classifications if value != "particle_prescatter_boundary_closes")
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "metric_policy": (
            "exact and relative-L2 intermediate metrics; no correlation; "
            "signed shellwise FSC/FSC-AUC remains the map acceptance metric"
        ),
        "relative_l2_bound": RELATIVE_L2_BOUND,
        "case_id": case_id,
        "physical_iteration": physical_iteration,
        "physical_image_size": physical_image_size,
        "current_size": current_size,
        "selected_stack_text": ",".join(str(value) for value in target_stacks),
        "capture_stack_text": ",".join(str(value) for value in capture_stacks),
        "selected_stack_fnv1a64": target_set_hash,
        "capture_stack_fnv1a64": selected_set_hash,
        "selection_json": str(selection_json.resolve()),
        "selection_sha256": _sha256(selection_json),
        "capture_inertness_json": str(capture_inertness_json.resolve()),
        "capture_inertness_sha256": _sha256(capture_inertness_json),
        "factor_directory": str(factor_directory.resolve()),
        "recovar_capture_directory": str(recovar_capture_directory.resolve()),
        "particle_count": len(particles),
        "support_exact_count": support_exact_count,
        "posterior_common_support_close_count": posterior_close_count,
        "particle_classification_counts": {
            value: classifications.count(value) for value in sorted(set(classifications))
        },
        "posterior_common_support_relative_l2": _distribution(
            [
                float(particle["comparisons"]["posterior_common_support"]["relative_l2_over_reference"])
                for particle in particles
            ]
        ),
        "classification": aggregate,
        "first_cross_engine_boundary": _first_cross_engine_boundary(particles),
        "next_boundary": (
            "conditional_on_matched_posterior__accumulator_destination_and_inter_particle_reduction"
            if aggregate == "fixed_panel_particle_prescatter_boundary_closes"
            else "resolve_reported_first_unequal_particle_boundary"
        ),
        "production_authorized": False,
        "fixed_scorecard_changed": False,
        "particles": particles,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-directory", type=Path, required=True)
    parser.add_argument("--recovar-capture-directory", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--capture-inertness-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--allow-non-h100", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        factor_directory=args.factor_directory,
        recovar_capture_directory=args.recovar_capture_directory,
        selection_json=args.selection_json,
        capture_inertness_json=args.capture_inertness_json,
        require_h100=not args.allow_non_h100,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "next_boundary": report["next_boundary"],
                "particle_count": report["particle_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
