#!/usr/bin/env python3
"""Compare native RELION and RECOVAR candidate weights feeding one scale-AA sum."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
    make_frequency_coords_half_np,
)
from scripts.analyze_k1_bpref_contributor_membership import match_rotations
from scripts.analyze_k1_scale_aa_boundary import _native_aa_shells


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _flat(path: Path, dtype: np.dtype) -> np.ndarray:
    payload = path.read_bytes()
    _require(len(payload) >= 4, f"truncated flat array: {path}")
    count = struct.unpack_from("<i", payload)[0]
    values = np.frombuffer(payload, dtype=dtype, offset=4).copy()
    _require(values.size == count, f"flat-array size mismatch: {path}")
    return values


def _real(path: Path) -> np.ndarray:
    return _flat(path, np.dtype("<f8"))


def _scalar(path: Path) -> float:
    payload = path.read_bytes()
    _require(len(payload) == 8, f"scalar size mismatch: {path}")
    return float(struct.unpack("<d", payload)[0])


def _f32_key(x: float, y: float) -> tuple[int, int]:
    return (
        int(np.asarray(x, dtype=np.float32).view(np.uint32).item()),
        int(np.asarray(y, dtype=np.float32).view(np.uint32).item()),
    )


def _metric(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    _require(candidate.shape == reference.shape and candidate.size > 0, "metric topology mismatch")
    residual = candidate - reference
    denominator = max(float(np.linalg.norm(reference)), float(np.finfo(np.float64).tiny))
    return {
        "count": int(candidate.size),
        "relative_l2": float(np.linalg.norm(residual) / denominator),
        "median_abs": float(np.median(np.abs(residual))),
        "max_abs": float(np.max(np.abs(residual))),
    }


def _entropy(probabilities: np.ndarray) -> float:
    positive = np.asarray(probabilities, dtype=np.float64)
    positive = positive[positive > 0.0]
    return float(-np.sum(positive * np.log(positive), dtype=np.float64))


def analyze(
    recovar_capture: Path,
    native_directory: Path,
    *,
    native_prefix: str,
    image_size: int,
    native_components: Path | None = None,
    recovar_term_divisor: float = float(128**4),
) -> dict[str, object]:
    _require(image_size > 0 and image_size % 2 == 0, "image size must be positive and even")
    with np.load(recovar_capture, allow_pickle=False) as payload:
        _require(
            str(payload["schema"].item())
            in {
                "recovar-k1-scale-aa-chunked-v1",
                "recovar-k1-scale-xa-aa-chunked-v2",
                "recovar-k1-scale-xa-aa-chunked-v3",
                "recovar-k1-scale-xa-aa-chunked-v4",
            },
            "unsupported RECOVAR capture schema",
        )
        recovar_probabilities = np.asarray(payload["candidate_posterior_probs"], dtype=np.float32)
        recovar_rotations = np.asarray(payload["candidate_rotation_matrices"], dtype=np.float32)
        fine_translations = np.asarray(payload["fine_translations"], dtype=np.float32)
        scale_mask = np.asarray(payload["scale_correction_pixel_mask"], dtype=bool)
        ctf_probs_raw_sum = np.asarray(
            payload["ctf_probs_raw_sum_per_pixel"],
            dtype=np.float64,
        )
        identity = {
            "iteration": int(payload["iteration"]),
            "half": int(payload["half"]),
            "part_id": int(payload["group_id"]),
            "original_index_zero_based": int(payload["original_index"]),
            "current_size": int(payload["current_size"]),
        }
        if "candidate_aa_feature_per_shell" in payload:
            aa_per_shell = np.asarray(payload["scale_aa_per_shell"], dtype=np.float64)
            aa_feature_per_shell = np.asarray(
                payload["candidate_aa_feature_per_shell"],
                dtype=np.float32,
            )
            aa_feature_shell_ids = np.asarray(
                payload["candidate_aa_feature_shell_ids"],
                dtype=np.int32,
            )
        else:
            aa_per_shell = None
            aa_feature_per_shell = None
            aa_feature_shell_ids = None
    _require(
        recovar_probabilities.shape[:1] == recovar_rotations.shape[:1],
        "RECOVAR rotation topology changed",
    )
    _require(
        recovar_probabilities.shape[1] == fine_translations.shape[0],
        "RECOVAR translation topology changed",
    )

    prefix = native_directory / native_prefix
    orientation_num = int(round(_scalar(Path(f"{prefix}orientation_num.bin"))))
    translation_num = int(round(_scalar(Path(f"{prefix}translation_num.bin"))))
    native_sum_weight = np.float32(_scalar(Path(f"{prefix}sum_weight.bin")))
    native_threshold = np.float32(_scalar(Path(f"{prefix}significant_weight.bin")))
    native_raw = _real(Path(f"{prefix}sorted_weights.bin")).astype(np.float32).reshape(
        orientation_num,
        translation_num,
    )
    native_rotations = _real(Path(f"{prefix}eulers.bin")).astype(np.float32).reshape(
        orientation_num,
        3,
        3,
    )
    native_rotations = native_rotations.transpose(0, 2, 1)
    native_trans_xyz = _real(Path(f"{prefix}trans_xyz.bin")).astype(np.float32)
    native_ctf = _real(Path(f"{prefix}ctfs.bin")).astype(np.float32)
    _require(native_trans_xyz.size == 3 * translation_num, "native translation topology changed")
    native_translation_angles = np.stack(
        (
            native_trans_xyz[:translation_num],
            native_trans_xyz[translation_num : 2 * translation_num],
        ),
        axis=1,
    )
    _require(np.isfinite(native_sum_weight) and native_sum_weight > 0.0, "invalid native weight sum")
    native_probabilities = np.where(
        native_raw >= native_threshold,
        native_raw / native_sum_weight,
        np.float32(0.0),
    ).astype(np.float32)

    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _relion_translation_angles_f32,
    )

    recovar_translation_angles = np.asarray(
        _relion_translation_angles_f32(fine_translations, (image_size, image_size)),
        dtype=np.float32,
    )
    _require(
        recovar_translation_angles.shape == native_translation_angles.shape,
        "native and RECOVAR translation counts differ",
    )
    recovar_translation_lookup = {
        _f32_key(x, y): row for row, (x, y) in enumerate(recovar_translation_angles.tolist())
    }
    _require(
        len(recovar_translation_lookup) == translation_num,
        "RECOVAR translation phases are not unique",
    )
    phase_distance = np.max(
        np.abs(
            native_translation_angles[:, None, :].astype(np.float64)
            - recovar_translation_angles[None, :, :].astype(np.float64)
        ),
        axis=2,
    )
    native_to_recovar_translation = np.argmin(phase_distance, axis=1).astype(np.int64)
    translation_nearest = phase_distance[
        np.arange(translation_num),
        native_to_recovar_translation,
    ]
    _require(
        np.unique(native_to_recovar_translation).size == translation_num,
        "native-to-RECOVAR translation match is not bijective",
    )
    _require(
        float(np.max(translation_nearest)) <= 2.0e-8,
        "native translation phase differs by more than the fixed two-ULP-scale bound",
    )
    translation_exact_count = int(
        sum(
            _f32_key(*native_translation_angles[row]) in recovar_translation_lookup
            for row in range(translation_num)
        )
    )

    native_rotation_mass = np.sum(native_probabilities, axis=1, dtype=np.float64)
    recovar_rotation_mass = np.sum(recovar_probabilities, axis=1, dtype=np.float64)
    native_active_rows = np.flatnonzero(native_rotation_mass > 0.0)
    recovar_active_rows = np.flatnonzero(recovar_rotation_mass > 0.0)
    rotation_matches = match_rotations(
        native_rotations[native_active_rows],
        recovar_rotations[recovar_active_rows],
        tolerance=0.0,
    )
    matched_native_rows = native_active_rows[rotation_matches.pairs[:, 0]]
    matched_recovar_rows = recovar_active_rows[rotation_matches.pairs[:, 1]]
    native_unmatched_rows = native_active_rows[rotation_matches.relion_unmatched]
    recovar_unmatched_rows = recovar_active_rows[rotation_matches.recovar_unmatched]

    native_common = native_probabilities[matched_native_rows]
    native_common = native_common[:, np.argsort(native_to_recovar_translation)]
    # The previous line gathers native columns in RECOVAR-column order because
    # ``argsort(native_to_recovar_translation)[recovar_col]`` is the native col.
    recovar_common = recovar_probabilities[matched_recovar_rows]
    common_delta = recovar_common.astype(np.float64) - native_common.astype(np.float64)
    native_unmatched_mass = float(np.sum(native_rotation_mass[native_unmatched_rows], dtype=np.float64))
    recovar_unmatched_mass = float(np.sum(recovar_rotation_mass[recovar_unmatched_rows], dtype=np.float64))
    posterior_tv = 0.5 * (
        float(np.sum(np.abs(common_delta), dtype=np.float64))
        + native_unmatched_mass
        + recovar_unmatched_mass
    )

    native_mass = float(np.sum(native_probabilities, dtype=np.float64))
    recovar_mass = float(np.sum(recovar_probabilities, dtype=np.float64))

    window_indices, _ = make_fourier_window_indices_np(
        (image_size, image_size),
        identity["current_size"],
        square=False,
        include_dc=True,
        exact_radius=True,
    )
    _require(window_indices.size == scale_mask.size == ctf_probs_raw_sum.size, "CTF window topology changed")
    recovar_coordinates = np.rint(
        make_frequency_coords_half_np((image_size, image_size))[window_indices]
    ).astype(np.int32)
    native_xdim = identity["current_size"] // 2 + 1
    _require(native_ctf.size % native_xdim == 0, "native CTF rectangle changed")
    native_ydim = native_ctf.size // native_xdim
    native_ctf_by_coordinate = {}
    for flat_index, value in enumerate(native_ctf.tolist()):
        row = flat_index // native_xdim
        x = flat_index % native_xdim
        y = row if row <= native_ydim // 2 else row - native_ydim
        native_ctf_by_coordinate[(x, y)] = np.float32(value)
    active_pixel_rows = np.flatnonzero(scale_mask)
    native_ctf_active = np.asarray(
        [native_ctf_by_coordinate[tuple(recovar_coordinates[row])] for row in active_pixel_rows],
        dtype=np.float32,
    )
    recovar_ctf2_active = ctf_probs_raw_sum[active_pixel_rows] / recovar_mass
    native_ctf2_active = native_ctf_active.astype(np.float64) ** 2
    positive_ctf = native_ctf2_active > 0.0
    ctf_ratio = recovar_ctf2_active[positive_ctf] / native_ctf2_active[positive_ctf]
    ctf_square = {
        **_metric(recovar_ctf2_active, native_ctf2_active),
        "active_pixel_count": int(active_pixel_rows.size),
        "ratio_median": float(np.median(ctf_ratio)),
        "ratio_p05": float(np.percentile(ctf_ratio, 5)),
        "ratio_p95": float(np.percentile(ctf_ratio, 95)),
    }
    native_rotation_moment = np.sum(
        native_rotations.astype(np.float64) * native_rotation_mass[:, None, None],
        axis=0,
        dtype=np.float64,
    )
    recovar_rotation_moment = np.sum(
        recovar_rotations.astype(np.float64) * recovar_rotation_mass[:, None, None],
        axis=0,
        dtype=np.float64,
    )
    native_translation_mass = np.sum(native_probabilities, axis=0, dtype=np.float64)
    recovar_translation_mass = np.sum(recovar_probabilities, axis=0, dtype=np.float64)
    native_translation_mass_recovar_order = native_translation_mass[
        np.argsort(native_to_recovar_translation)
    ]

    aa_weight_swap = None
    if aa_feature_per_shell is not None and native_components is not None:
        _require(
            aa_feature_per_shell.shape == (recovar_rotations.shape[0], aa_feature_shell_ids.size),
            "candidate AA shell-feature topology changed",
        )
        native_mass_in_recovar_order = np.zeros(recovar_rotations.shape[0], dtype=np.float64)
        native_mass_in_recovar_order[matched_recovar_rows] = native_rotation_mass[matched_native_rows]
        _require(native_unmatched_rows.size == 0, "native AA weight swap has unmatched rotations")
        recovar_feature_replay = np.sum(
            aa_feature_per_shell.astype(np.float64) * recovar_rotation_mass[:, None],
            axis=0,
            dtype=np.float64,
        )
        native_weight_feature_replay = np.sum(
            aa_feature_per_shell.astype(np.float64) * native_mass_in_recovar_order[:, None],
            axis=0,
            dtype=np.float64,
        )
        captured_recovar = aa_per_shell[aa_feature_shell_ids]
        native_shells = _native_aa_shells(
            native_components,
            iteration=identity["iteration"],
            half=identity["half"],
            part_id=identity["part_id"],
        )[aa_feature_shell_ids]
        recovar_native_units = captured_recovar / float(recovar_term_divisor)
        recovar_feature_native_units = recovar_feature_replay / float(recovar_term_divisor)
        native_weight_feature_native_units = native_weight_feature_replay / float(recovar_term_divisor)
        baseline_residual_norm = float(
            np.linalg.norm(recovar_native_units - native_shells)
        )
        swapped_residual_norm = float(
            np.linalg.norm(native_weight_feature_native_units - native_shells)
        )
        closure_fraction = (
            1.0 - swapped_residual_norm / baseline_residual_norm
            if baseline_residual_norm > 0.0
            else 0.0
        )
        aa_weight_swap = {
            "shell_ids": aa_feature_shell_ids.tolist(),
            "recovar_feature_replay_vs_capture": _metric(
                recovar_feature_native_units,
                recovar_native_units,
            ),
            "baseline_recovar_vs_native": _metric(recovar_native_units, native_shells),
            "native_weights_on_recovar_features_vs_native": _metric(
                native_weight_feature_native_units,
                native_shells,
            ),
            "residual_closure_fraction": closure_fraction,
            "classification": (
                "posterior weights are causally sufficient for the AA shell residual"
                if closure_fraction >= 0.8
                else "posterior weights are not the dominant AA shell residual"
                if closure_fraction <= 0.2
                else "posterior weights partially contribute to the AA shell residual"
            ),
        }

    native_files = sorted(native_directory.glob(f"{native_prefix}*.bin"))
    return {
        "schema": "recovar.em.k1_scale_aa_candidates.v1",
        "identity": {
            **identity,
            "native_orientation_count": orientation_num,
            "recovar_padded_rotation_count": int(recovar_rotations.shape[0]),
            "translation_count": translation_num,
        },
        "translation_join": {
            "exact_phase_match_count": translation_exact_count,
            "matched_phase_count": int(native_to_recovar_translation.size),
            "matched_max_abs": float(np.max(translation_nearest)),
            "native_to_recovar": native_to_recovar_translation.tolist(),
        },
        "rotation_join": {
            "native_active_rotation_count": int(native_active_rows.size),
            "recovar_active_rotation_count": int(recovar_active_rows.size),
            "exact_match_count": int(rotation_matches.pairs.shape[0]),
            "native_unmatched_count": int(native_unmatched_rows.size),
            "recovar_unmatched_count": int(recovar_unmatched_rows.size),
            "native_unmatched_mass": native_unmatched_mass,
            "recovar_unmatched_mass": recovar_unmatched_mass,
            "native_ambiguous_count": int(rotation_matches.relion_ambiguous),
            "recovar_ambiguous_count": int(rotation_matches.recovar_ambiguous),
        },
        "posterior": {
            "native_retained_mass": native_mass,
            "recovar_retained_mass": recovar_mass,
            "recovar_over_native_retained_mass": recovar_mass / native_mass,
            "native_positive_candidate_count": int(np.count_nonzero(native_probabilities)),
            "recovar_positive_candidate_count": int(np.count_nonzero(recovar_probabilities)),
            "common_candidate_metric": _metric(recovar_common, native_common),
            "union_total_variation": posterior_tv,
            "native_entropy": _entropy(native_probabilities),
            "recovar_entropy": _entropy(recovar_probabilities),
            "rotation_matrix_first_moment": _metric(
                recovar_rotation_moment,
                native_rotation_moment,
            ),
            "translation_marginal": _metric(
                recovar_translation_mass,
                native_translation_mass_recovar_order,
            ),
        },
        "ctf_square": ctf_square,
        "aa_weight_swap": aa_weight_swap,
        "native_weight_scalars": {
            "sum_weight_float32": float(native_sum_weight),
            "significant_weight_float32": float(native_threshold),
        },
        "artifacts": {
            "recovar_capture": str(recovar_capture.resolve()),
            "recovar_capture_sha256": _sha256(recovar_capture),
            "native_directory": str(native_directory.resolve()),
            "native_file_count": len(native_files),
            "native_files_sha256": {
                path.name: _sha256(path) for path in native_files
            },
            "native_components": (
                None if native_components is None else str(native_components.resolve())
            ),
            "native_components_sha256": (
                None if native_components is None else _sha256(native_components)
            ),
        },
        "classification": (
            aa_weight_swap["classification"]
            if aa_weight_swap is not None
            else "candidate posterior differs; causal sufficiency requires the AA weight-swap replay"
            if posterior_tv > 1e-7
            else "candidate posterior agrees; projected-reference power or CTF is the next boundary"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-capture", type=Path, required=True)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--native-prefix", default="img0_part109_storeWavg_")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--native-components", type=Path)
    parser.add_argument("--recovar-term-divisor", type=float, default=float(128**4))
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.recovar_capture,
        args.native_directory,
        native_prefix=args.native_prefix,
        image_size=args.image_size,
        native_components=args.native_components,
        recovar_term_divisor=args.recovar_term_divisor,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
