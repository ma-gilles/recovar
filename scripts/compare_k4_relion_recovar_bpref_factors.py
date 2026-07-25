#!/usr/bin/env python3
"""Compare exact-boundary K4 RELION and RECOVAR BPref factor operands."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar.core.ctf import _compute_spa_ctf
from recovar.cuda_backproject import relion_preprocess_real_f32
from recovar.data_io.image_backends import _centered_rfft2_jax, _centered_rfft2_numpy
from recovar.em.dense_single_volume.helpers.image_shifts import (
    apply_relion_integer_pre_shifts,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _half_translation_phase_table_for_indices,
)

if __package__:
    from .validate_relion_bpref_factor_capture import (
        FactorCapture,
        load_factor_capture,
        validate_directory,
    )
else:
    from validate_relion_bpref_factor_capture import (  # type: ignore[no-redef]
        FactorCapture,
        load_factor_capture,
        validate_directory,
    )

PHYSICAL_IMAGE_SIZE = 256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _float32_from_bits(value: int) -> np.float32:
    return np.float32(struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0])


def _recovar_exp50_weight_normalizer(
    values: dict[str, np.ndarray],
    particle: int,
) -> np.float32:
    normalized_sum = np.float32(values["candidate_normalized_sum_exp"][particle])
    return np.float32(normalized_sum * np.exp(np.float32(50.0), dtype=np.float32))


def _metric(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, object]:
    left = np.asarray(lhs)
    right = np.asarray(rhs)
    _require(left.shape == right.shape, f"factor shape changed: {left.shape} != {right.shape}")
    promoted_left = left.astype(np.complex128, copy=False)
    promoted_right = right.astype(np.complex128, copy=False)
    delta = promoted_right - promoted_left
    denominator = max(float(np.linalg.norm(promoted_left)), np.finfo(np.float64).tiny)
    return {
        "shape": list(left.shape),
        "lhs_dtype": str(left.dtype),
        "rhs_dtype": str(right.dtype),
        "exact_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "relative_l2_over_relion": float(np.linalg.norm(delta) / denominator),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "median_abs": float(np.median(np.abs(delta))) if delta.size else 0.0,
        "p95_abs": float(np.quantile(np.abs(delta), 0.95)) if delta.size else 0.0,
    }


def _compact_indices(values: dict[str, np.ndarray]) -> np.ndarray:
    image_shape = tuple(int(value) for value in values["image_shape"])
    height, width = image_shape
    half_width = width // 2 + 1
    window = np.asarray(values["window_indices"], dtype=np.int32)
    fftw_rows = window // half_width
    centered = (((fftw_rows - height // 2) % height) * half_width + window % half_width).astype(
        np.int32,
        copy=False,
    )
    _require(
        np.array_equal(centered // half_width, (fftw_rows - height // 2) % height)
        and np.array_equal(centered % half_width, window % half_width),
        "centered-packed index conversion changed",
    )
    return centered


def _scalar_rotation_records(path: Path, stacks: list[int]) -> dict[int, tuple[tuple[int, int], ...]]:
    report = json.loads(path.read_text())
    _require(
        report.get("classification") == "pixel_varying_source_difference_not_explained_by_per_rotation_scalar",
        "prescatter scalar classification changed",
    )
    records: dict[int, tuple[tuple[int, int], ...]] = {}
    for particle in report.get("particles", []):
        stack = int(particle["stack_index_one_based"])
        if stack not in stacks:
            continue
        fits = particle["rotation_scalar_fits"]
        _require(bool(fits), f"stack {stack}: factor panel has no matched class-2 contributor")
        records[stack] = tuple(
            (
                int(fit["recovar_global_rotation_index"]),
                int(fit["relion_rotation_local_row"]),
            )
            for fit in fits
        )
        _require(len(set(records[stack])) == len(records[stack]), f"stack {stack}: duplicate contributor rotation")
    _require(set(records) == set(stacks), "factor-panel rotations are incomplete in the scalar report")
    return records


def _contribution_locations(directory: Path, stacks: list[int]) -> tuple[dict[int, tuple[Path, int]], dict[str, str]]:
    locations: dict[int, tuple[Path, int]] = {}
    hashes: dict[str, str] = {}
    wanted = set(stacks)
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            shard_stacks = np.asarray(archive["stack_indices_1based"], dtype=np.int64)
        matched = wanted.intersection(int(value) for value in shard_stacks)
        if not matched:
            continue
        hashes[path.name] = _sha256(path)
        for stack in matched:
            rows = np.flatnonzero(shard_stacks == stack)
            _require(rows.size == 1 and stack not in locations, f"stack {stack}: duplicate contribution shard")
            locations[stack] = (path, int(rows[0]))
    _require(set(locations) == wanted, "factor-panel contribution shards are incomplete")
    return locations, hashes


def _processed_reconstruction_inputs(
    values: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values["raw_real_images"], dtype=np.float32)
    normalization = np.asarray(
        values["relion_preprocess_normalization_factors"],
        dtype=np.float32,
    )
    integer_shifts = np.asarray(values["integer_pre_shifts"], dtype=np.int32)
    relion_cuda = bool(np.asarray(values["relion_cuda_preprocess"]).item())
    backend = str(np.asarray(values["preprocess_backend"]).item())
    _require(
        relion_cuda == (backend == "relion_cuda"),
        "RECOVAR preprocessing flag and backend disagree",
    )
    if relion_cuda:
        _, preprocessed = relion_preprocess_real_f32(
            jnp.asarray(raw),
            jnp.asarray(normalization),
            jnp.asarray(integer_shifts),
            1.0,
            1.0,
            False,
        )
        processed = _centered_rfft2_jax(preprocessed)
        reconstruction_correction = np.asarray(
            values["scale_corrections"],
            dtype=np.float32,
        )
    else:
        _require(backend == "dataset_native", f"unsupported RECOVAR preprocessing backend {backend!r}")
        _require(
            np.array_equal(normalization, np.ones_like(normalization)),
            "dataset-native capture unexpectedly stored active RELION normalization factors",
        )
        shifted = apply_relion_integer_pre_shifts(raw, integer_shifts)
        processed = _centered_rfft2_numpy(shifted)
        reconstruction_correction = np.asarray(
            values["image_corrections"],
            dtype=np.float32,
        )
    return (
        np.asarray(processed.reshape(raw.shape[0], -1), dtype=np.complex64),
        reconstruction_correction,
    )


def _production_factors(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    _require(jax.default_backend() == "gpu", "K4 factor comparison requires a JAX GPU")
    image_shape = tuple(int(value) for value in values["image_shape"])
    compact_np = _compact_indices(values)
    compact = jnp.asarray(compact_np, dtype=jnp.int32)
    processed, reconstruction_correction = _processed_reconstruction_inputs(values)
    ctf = _compute_spa_ctf(
        jnp.asarray(values["ctf_params"], dtype=jnp.float32),
        image_shape,
        float(np.asarray(values["voxel_size"]).item()),
        half_image=True,
    ).astype(jnp.float32)
    noise = jnp.asarray(values["noise_variance_half"], dtype=jnp.float32)
    translations = jnp.asarray(values["fine_translations"], dtype=jnp.float32)
    phases = _half_translation_phase_table_for_indices(translations, image_shape, compact)
    return {
        "compact_indices": compact_np,
        "processed": np.asarray(processed[:, compact_np]),
        "ctf": np.asarray(ctf[:, compact]),
        "noise": np.asarray(noise[compact]),
        "phases": np.asarray(phases),
        "reconstruction_correction": reconstruction_correction,
    }


def _pixel_rows(capture: FactorCapture, compact_indices: np.ndarray) -> np.ndarray:
    half_width = PHYSICAL_IMAGE_SIZE // 2 + 1
    centered_rows = compact_indices // half_width
    coordinates = tuple(
        (int(index % half_width), int(row - PHYSICAL_IMAGE_SIZE // 2))
        for index, row in zip(compact_indices, centered_rows)
    )
    lookup = {(int(x), int(y)): row for row, (x, y) in enumerate(zip(capture.pixels["x"], capture.pixels["y"]))}
    _require(len(set(coordinates)) == len(coordinates), f"stack {capture.stack_index}: compact pixels are duplicated")
    _require(
        all(coordinate in lookup for coordinate in coordinates),
        f"stack {capture.stack_index}: compact pixel support changed",
    )
    return np.asarray([lookup[coordinate] for coordinate in coordinates], dtype=np.int64)


def _translation_map(capture: FactorCapture, fine_translations: np.ndarray) -> dict[int, int]:
    relion = np.column_stack((capture.translations["x"], capture.translations["y"])).astype(np.float64)
    recovar = -2 * np.pi * np.asarray(fine_translations, dtype=np.float64) / PHYSICAL_IMAGE_SIZE
    distance = np.max(np.abs(relion[:, None, :] - recovar[None, :, :]), axis=2)
    nearest = np.argmin(distance, axis=1)
    nearest_error = distance[np.arange(relion.shape[0]), nearest]
    _require(
        np.all(nearest_error <= 1e-6) and np.unique(nearest).size == nearest.size,
        f"stack {capture.stack_index}: translation-vector alignment changed",
    )
    return {row: int(candidate) for row, candidate in enumerate(nearest)}


def _target_terms(capture: FactorCapture, orientation: int, translation: int, pixel_rows: np.ndarray) -> np.ndarray:
    selected = capture.terms[
        (capture.terms["orientation_local"] == orientation) & (capture.terms["translation"] == translation)
    ]
    _require(selected.size == capture.pixels.size, f"stack {capture.stack_index}: accepted term panel is incomplete")
    _require(
        np.array_equal(selected["pixel"], np.arange(capture.pixels.size)),
        f"stack {capture.stack_index}: accepted term pixel order changed",
    )
    return selected[pixel_rows]


def _relion_rotation_matrix(capture: FactorCapture, orientation: int) -> np.ndarray:
    """Convert RELION's column-major capture record to RECOVAR matrix layout."""

    return capture.rotations["matrix"][orientation].reshape(3, 3).T


def _append(operands: dict[str, list[np.ndarray]], name: str, relion: np.ndarray, recovar: np.ndarray) -> None:
    operands[f"{name}_relion"].append(np.asarray(relion).reshape(-1))
    operands[f"{name}_recovar"].append(np.asarray(recovar).reshape(-1))


def compare(
    capture_directory: Path,
    selection_json: Path,
    contribution_directory: Path,
    scalar_json: Path,
) -> dict[str, object]:
    validation = validate_directory(capture_directory, selection_json)
    selection = json.loads(selection_json.read_text())
    stacks = [int(record["stack_index_1based"]) for record in selection["selected"]]
    rotations = _scalar_rotation_records(scalar_json, stacks)
    locations, contribution_hashes = _contribution_locations(contribution_directory, stacks)
    captures = {
        capture.stack_index: capture
        for capture in (load_factor_capture(path) for path in capture_directory.glob("*.bpre-v2.bin"))
    }
    _require(set(captures) == set(stacks), "validated factor capture set changed")

    operands: dict[str, list[np.ndarray]] = {
        f"{name}_{engine}": []
        for name in (
            "processed_fft",
            "ctf",
            "inverse_noise",
            "translation_phase_increment",
            "posterior",
            "raw_posterior_weight",
            "posterior_weight_normalizer",
            "log_posterior",
            "shifted_image",
            "weighted_ctf",
            "term",
            "weight_term",
            "source_sum",
            "term_with_relion_posterior",
            "weight_term_with_relion_posterior",
            "source_sum_with_relion_posterior",
        )
        for engine in ("relion", "recovar")
    }
    particles: list[dict[str, object]] = []
    for path in sorted({location[0] for location in locations.values()}):
        selected_stacks = [stack for stack in stacks if locations[stack][0] == path]
        with np.load(path, allow_pickle=False) as archive:
            values = {name: np.asarray(archive[name]) for name in archive.files}
        production = _production_factors(values)
        for stack in selected_stacks:
            particle = locations[stack][1]
            capture = captures[stack]
            pixel_rows = _pixel_rows(capture, production["compact_indices"])
            translation_map = _translation_map(capture, values["fine_translations"])
            processed = production["processed"][particle]
            ctf = production["ctf"][particle]
            inverse_noise = 1.0 / production["noise"]
            scale = np.float32(values["scale_corrections"][particle])
            reconstruction_correction = np.float32(production["reconstruction_correction"][particle])
            relion_processed = (
                capture.pixels["image_re"][pixel_rows] + 1j * capture.pixels["image_im"][pixel_rows]
            ) * PHYSICAL_IMAGE_SIZE**2
            relion_ctf = -capture.pixels["ctf"][pixel_rows] / scale
            relion_inverse_noise = capture.pixels["minvsigma2"][pixel_rows] / PHYSICAL_IMAGE_SIZE**4
            _append(operands, "processed_fft", relion_processed, processed)
            _append(operands, "ctf", relion_ctf, ctf)
            _append(operands, "inverse_noise", relion_inverse_noise, inverse_noise)
            relion_weight_normalizer = _float32_from_bits(capture.header[26])
            recovar_weight_normalizer = _recovar_exp50_weight_normalizer(
                values,
                particle,
            )
            _append(
                operands,
                "posterior_weight_normalizer",
                np.asarray([relion_weight_normalizer]),
                np.asarray([recovar_weight_normalizer]),
            )

            contributors: list[dict[str, object]] = []
            for global_rotation, relion_orientation in rotations[stack]:
                candidate_rows = np.flatnonzero(values["oversampled_rotation_indices"][particle] == global_rotation)
                _require(
                    candidate_rows.size == 1,
                    f"stack {stack}: RECOVAR global rotation {global_rotation} is not unique",
                )
                recovar_orientation = int(candidate_rows[0])
                active_rows = np.flatnonzero(
                    (values["active_particle_rows"] == particle)
                    & (values["active_global_rotation_indices"] == global_rotation)
                )
                _require(
                    active_rows.size == 1,
                    f"stack {stack}: RECOVAR active contributor {global_rotation} is not unique",
                )
                active = int(active_rows[0])
                geometry = _metric(
                    _relion_rotation_matrix(capture, relion_orientation),
                    values["active_rotations"][active],
                )
                _require(
                    geometry["max_abs"] == 0,
                    f"stack {stack}: exact contributor geometry {global_rotation} changed",
                )

                relion_hypotheses = capture.hypotheses[capture.hypotheses["orientation_local"] == relion_orientation]
                relion_accepted = relion_hypotheses[(relion_hypotheses["flags"] & 1) != 0]
                recovar_probabilities = np.asarray(
                    values["reconstruction_probs"][particle, recovar_orientation],
                    dtype=np.float32,
                )
                recovar_accepted = np.flatnonzero(recovar_probabilities != 0)
                mapped_relion = np.asarray(
                    [translation_map[int(row["translation"])] for row in relion_accepted],
                    dtype=np.int64,
                )
                _require(
                    np.array_equal(np.sort(mapped_relion), recovar_accepted),
                    f"stack {stack}: accepted translation support for {global_rotation} changed",
                )

                relion_term_sum = np.zeros_like(processed, dtype=np.complex64)
                recovar_term_sum = np.zeros_like(processed, dtype=np.complex64)
                relion_posterior_term_sum = np.zeros_like(
                    processed,
                    dtype=np.complex64,
                )
                per_translation: list[dict[str, object]] = []
                for hypothesis, recovar_translation in zip(relion_accepted, mapped_relion):
                    relion_translation = int(hypothesis["translation"])
                    probability = recovar_probabilities[recovar_translation]
                    terms = _target_terms(
                        capture,
                        relion_orientation,
                        relion_translation,
                        pixel_rows,
                    )
                    relion_phase_increment = np.asarray(
                        [
                            capture.translations["x"][relion_translation],
                            capture.translations["y"][relion_translation],
                        ],
                        dtype=np.float32,
                    )
                    recovar_phase_increment = (
                        -2
                        * np.pi
                        * np.asarray(
                            values["fine_translations"][recovar_translation],
                            dtype=np.float32,
                        )
                        / PHYSICAL_IMAGE_SIZE
                    )
                    relion_probability = np.float32(hypothesis["posterior_over_weight_norm"])
                    relion_raw_weight = np.float32(hypothesis["posterior"])
                    recovar_raw_weight = np.float32(
                        values["candidate_raw_exp_weights_f32"][
                            particle,
                            recovar_orientation,
                            recovar_translation,
                        ]
                    )
                    relion_log_probability = np.log(np.float64(relion_probability))
                    recovar_log_probability = np.float64(
                        values["candidate_combined_scores"][
                            particle,
                            recovar_orientation,
                            recovar_translation,
                        ]
                        - values["candidate_log_z"][particle]
                    )
                    shifted = (processed * production["phases"][recovar_translation]).astype(np.complex64)
                    weighted_ctf = (probability * ctf * inverse_noise * reconstruction_correction).astype(np.float32)
                    term = (shifted * weighted_ctf).astype(np.complex64)
                    weight_term = (probability * ctf**2 * inverse_noise * scale**2).astype(np.float32)
                    relion_posterior_weighted_ctf = (
                        relion_probability * ctf * inverse_noise * reconstruction_correction
                    ).astype(np.float32)
                    relion_posterior_term = (shifted * relion_posterior_weighted_ctf).astype(np.complex64)
                    relion_posterior_weight_term = (relion_probability * ctf**2 * inverse_noise * scale**2).astype(
                        np.float32
                    )
                    relion_shifted = (terms["translated_re"] + 1j * terms["translated_im"]) * PHYSICAL_IMAGE_SIZE**2
                    relion_weighted_ctf = -terms["weighted_ctf"] / PHYSICAL_IMAGE_SIZE**4
                    relion_term = -(terms["term_re"] + 1j * terms["term_im"]) / PHYSICAL_IMAGE_SIZE**2
                    relion_weight_term = terms["weight_term"] / PHYSICAL_IMAGE_SIZE**4
                    _append(
                        operands,
                        "translation_phase_increment",
                        relion_phase_increment,
                        recovar_phase_increment,
                    )
                    _append(
                        operands,
                        "posterior",
                        np.asarray([relion_probability]),
                        np.asarray([probability]),
                    )
                    _append(
                        operands,
                        "raw_posterior_weight",
                        np.asarray([relion_raw_weight]),
                        np.asarray([recovar_raw_weight]),
                    )
                    _append(
                        operands,
                        "log_posterior",
                        np.asarray([relion_log_probability]),
                        np.asarray([recovar_log_probability]),
                    )
                    _append(operands, "shifted_image", relion_shifted, shifted)
                    _append(
                        operands,
                        "weighted_ctf",
                        relion_weighted_ctf,
                        weighted_ctf,
                    )
                    _append(operands, "term", relion_term, term)
                    _append(
                        operands,
                        "weight_term",
                        relion_weight_term,
                        weight_term,
                    )
                    _append(
                        operands,
                        "term_with_relion_posterior",
                        relion_term,
                        relion_posterior_term,
                    )
                    _append(
                        operands,
                        "weight_term_with_relion_posterior",
                        relion_weight_term,
                        relion_posterior_weight_term,
                    )
                    relion_term_sum += relion_term.astype(np.complex64)
                    recovar_term_sum += term
                    relion_posterior_term_sum += relion_posterior_term
                    per_translation.append(
                        {
                            "relion_translation": relion_translation,
                            "recovar_translation": int(recovar_translation),
                            "posterior": _metric(
                                np.asarray([relion_probability]),
                                np.asarray([probability]),
                            ),
                            "raw_posterior_weight": _metric(
                                np.asarray([relion_raw_weight]),
                                np.asarray([recovar_raw_weight]),
                            ),
                            "log_posterior": _metric(
                                np.asarray([relion_log_probability]),
                                np.asarray([recovar_log_probability]),
                            ),
                            "shifted_image": _metric(relion_shifted, shifted),
                            "weighted_ctf": _metric(
                                relion_weighted_ctf,
                                weighted_ctf,
                            ),
                            "term": _metric(relion_term, term),
                            "weight_term": _metric(
                                relion_weight_term,
                                weight_term,
                            ),
                            "term_with_relion_posterior": _metric(
                                relion_term,
                                relion_posterior_term,
                            ),
                            "weight_term_with_relion_posterior": _metric(
                                relion_weight_term,
                                relion_posterior_weight_term,
                            ),
                        }
                    )
                _append(
                    operands,
                    "source_sum",
                    relion_term_sum,
                    values["active_summed"][active],
                )
                _append(
                    operands,
                    "source_sum_with_relion_posterior",
                    relion_term_sum,
                    relion_posterior_term_sum,
                )
                contributors.append(
                    {
                        "recovar_global_rotation_index": global_rotation,
                        "relion_orientation_local": relion_orientation,
                        "recovar_orientation_local": recovar_orientation,
                        "accepted_translation_count": len(per_translation),
                        "geometry": geometry,
                        "source_sum_relion_terms_vs_recovar_captured": _metric(
                            relion_term_sum,
                            values["active_summed"][active],
                        ),
                        "source_sum_recovar_terms_vs_recovar_captured": _metric(
                            recovar_term_sum,
                            values["active_summed"][active],
                        ),
                        "source_sum_with_relion_posterior": _metric(
                            relion_term_sum,
                            relion_posterior_term_sum,
                        ),
                        "translations": per_translation,
                    }
                )
            particles.append(
                {
                    "stack_index_1based": stack,
                    "matched_contributor_count": len(contributors),
                    "accepted_hypothesis_count": sum(
                        int(contributor["accepted_translation_count"]) for contributor in contributors
                    ),
                    "processed_fft": _metric(relion_processed, processed),
                    "ctf": _metric(relion_ctf, ctf),
                    "inverse_noise": _metric(relion_inverse_noise, inverse_noise),
                    "posterior_weight_normalizer": _metric(
                        np.asarray([relion_weight_normalizer]),
                        np.asarray([recovar_weight_normalizer]),
                    ),
                    "contributors": contributors,
                }
            )

    aggregate = {
        name: _metric(
            np.concatenate(operands[f"{name}_relion"]),
            np.concatenate(operands[f"{name}_recovar"]),
        )
        for name in (
            "processed_fft",
            "ctf",
            "inverse_noise",
            "translation_phase_increment",
            "posterior",
            "raw_posterior_weight",
            "posterior_weight_normalizer",
            "log_posterior",
            "shifted_image",
            "weighted_ctf",
            "term",
            "weight_term",
            "source_sum",
            "term_with_relion_posterior",
            "weight_term_with_relion_posterior",
            "source_sum_with_relion_posterior",
        )
    }
    return {
        "schema": "k4-relion-recovar-bpref-factor-comparison-v3",
        "metric_policy": "exact and scale-aware array metrics only; no correlation",
        "counterfactual_policy": (
            "RELION posterior substituted only into RECOVAR term and weight factors on the exact accepted support"
        ),
        "status": "complete",
        "factor_validation": validation,
        "selection_sha256": _sha256(selection_json),
        "prescatter_scalar_sha256": _sha256(scalar_json),
        "contribution_artifact_sha256": contribution_hashes,
        "device": str(jax.devices()[0]),
        "device_kind": str(jax.devices()[0].device_kind),
        "particle_count": len(particles),
        "matched_contributor_count": sum(int(particle["matched_contributor_count"]) for particle in particles),
        "accepted_hypothesis_count": sum(int(particle["accepted_hypothesis_count"]) for particle in particles),
        "aggregate": aggregate,
        "particles": particles,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--selection-json", required=True, type=Path)
    parser.add_argument("--contribution-directory", required=True, type=Path)
    parser.add_argument("--prescatter-scalar-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite comparison artifact: {args.output_json}")
    report = compare(
        args.capture_directory,
        args.selection_json,
        args.contribution_directory,
        args.prescatter_scalar_json,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
