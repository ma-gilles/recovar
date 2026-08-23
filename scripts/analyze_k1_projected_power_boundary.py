#!/usr/bin/env python3
"""Compare native and RECOVAR projected-reference power before posterior weighting."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from scripts.analyze_k1_bpref_contributor_membership import match_rotations
from scripts.analyze_k1_scale_aa_boundary import _native_aa_shells
from scripts.analyze_k1_scale_aa_candidates import _metric, _real, _scalar, _sha256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _raw_f32(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    _require(len(payload) >= 8, f"truncated raw float32 array: {path}")
    count = struct.unpack_from("<Q", payload)[0]
    values = np.frombuffer(payload, dtype="<f4", offset=8).copy()
    _require(values.size == count, f"raw float32 size mismatch: {path}")
    return values


def _native_probabilities(prefix: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orientation_num = int(round(_scalar(Path(f"{prefix}orientation_num.bin"))))
    translation_num = int(round(_scalar(Path(f"{prefix}translation_num.bin"))))
    sum_weight = np.float32(_scalar(Path(f"{prefix}sum_weight.bin")))
    threshold = np.float32(_scalar(Path(f"{prefix}significant_weight.bin")))
    raw = _real(Path(f"{prefix}sorted_weights.bin")).astype(np.float32).reshape(
        orientation_num,
        translation_num,
    )
    probabilities = np.where(raw >= threshold, raw / sum_weight, np.float32(0.0)).astype(
        np.float32
    )
    rotations = _real(Path(f"{prefix}eulers.bin")).astype(np.float32).reshape(
        orientation_num,
        3,
        3,
    )
    return probabilities, rotations.transpose(0, 2, 1), np.sum(
        probabilities,
        axis=1,
        dtype=np.float64,
    )


def _shell_features(
    real: np.ndarray,
    imag: np.ndarray,
    ctf: np.ndarray,
    shell_labels: np.ndarray,
    shell_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaled_real = (real * ctf[None, :]).astype(np.float32)
    scaled_imag = (imag * ctf[None, :]).astype(np.float32)
    power = (scaled_real * scaled_real + scaled_imag * scaled_imag).astype(np.float32)
    feature_f32 = np.stack(
        [
            np.sum(power[:, shell_labels == shell], axis=1, dtype=np.float32)
            for shell in shell_ids.tolist()
        ],
        axis=1,
    )
    feature_f64 = np.stack(
        [
            np.sum(power[:, shell_labels == shell], axis=1, dtype=np.float64)
            for shell in shell_ids.tolist()
        ],
        axis=1,
    )
    return feature_f32, feature_f64


def analyze(
    recovar_capture: Path,
    native_directory: Path,
    native_components: Path,
    *,
    native_prefix: str,
    recovar_term_divisor: float,
) -> dict[str, object]:
    with np.load(recovar_capture, allow_pickle=False) as payload:
        recovar_probabilities = np.asarray(payload["candidate_posterior_probs"], dtype=np.float32)
        recovar_rotations = np.asarray(payload["candidate_rotation_matrices"], dtype=np.float32)
        recovar_features = np.asarray(payload["candidate_aa_feature_per_shell"], dtype=np.float32)
        shell_ids = np.asarray(payload["candidate_aa_feature_shell_ids"], dtype=np.int32)
        captured_shells = np.asarray(payload["scale_aa_per_shell"], dtype=np.float64)[shell_ids]
        iteration = int(payload["iteration"])
        half = int(payload["half"])
        part_id = int(payload["group_id"])
        scale = float(np.asarray(payload["scale_for_stats"], dtype=np.float32))
    recovar_rotation_mass = np.sum(recovar_probabilities, axis=1, dtype=np.float64)

    prefix = native_directory / native_prefix
    native_probabilities, native_rotations, native_rotation_mass = _native_probabilities(prefix)
    orientation_num = native_rotations.shape[0]
    panel_size = int(round(_scalar(Path(f"{prefix}project_panel_size.bin"))))
    panel_orientation_num = int(
        round(_scalar(Path(f"{prefix}project_panel_orientation_num.bin")))
    )
    _require(panel_orientation_num == orientation_num, "native panel orientation count changed")
    # The compact integer arrays use the flat-int format, not the flat-real format.
    panel_pixels_payload = Path(f"{prefix}project_panel_pixels.bin").read_bytes()
    panel_shells_payload = Path(f"{prefix}project_panel_shells.bin").read_bytes()
    panel_pixels = np.frombuffer(panel_pixels_payload, dtype="<i4", offset=4).astype(np.int64)
    panel_shells = np.frombuffer(panel_shells_payload, dtype="<i4", offset=4).copy()
    _require(panel_pixels.size == panel_shells.size == panel_size, "native panel topology changed")
    _require(np.array_equal(np.unique(panel_shells), shell_ids), "native and RECOVAR shell sets differ")

    real_path = Path(f"{prefix}project_panel_ref_real.f32")
    imag_path = Path(f"{prefix}project_panel_ref_imag.f32")
    native_real = _raw_f32(real_path).reshape(orientation_num, panel_size)
    native_imag = _raw_f32(imag_path).reshape(orientation_num, panel_size)
    native_ctf_all = _real(Path(f"{prefix}ctfs.bin")).astype(np.float32)
    _require(int(np.max(panel_pixels)) < native_ctf_all.size, "native panel pixel is out of range")
    native_ctf = native_ctf_all[panel_pixels]
    native_feature_f32, native_feature_f64 = _shell_features(
        native_real,
        native_imag,
        native_ctf,
        panel_shells,
        shell_ids,
    )

    native_active = np.flatnonzero(native_rotation_mass > 0.0)
    recovar_active = np.flatnonzero(recovar_rotation_mass > 0.0)
    matches = match_rotations(
        native_rotations[native_active],
        recovar_rotations[recovar_active],
        tolerance=0.0,
    )
    matched_native = native_active[matches.pairs[:, 0]]
    matched_recovar = recovar_active[matches.pairs[:, 1]]
    _require(matches.relion_unmatched.size == 0, "native active rotation is unmatched")
    _require(matches.recovar_unmatched.size == 0, "RECOVAR active rotation is unmatched")

    recovar_feature_native_units = (
        recovar_features[matched_recovar].astype(np.float64) / recovar_term_divisor
    )
    native_feature_f32_matched = native_feature_f32[matched_native].astype(np.float64)
    native_feature_f64_matched = native_feature_f64[matched_native]
    native_shells = _native_aa_shells(
        native_components,
        iteration=iteration,
        half=half,
        part_id=part_id,
    )[shell_ids]
    captured_native_units = captured_shells / recovar_term_divisor

    native_feature_weighted_native = np.sum(
        native_feature_f32.astype(np.float64) * native_rotation_mass[:, None],
        axis=0,
        dtype=np.float64,
    )
    native_feature_weighted_recovar = np.sum(
        native_feature_f32_matched * recovar_rotation_mass[matched_recovar, None],
        axis=0,
        dtype=np.float64,
    )
    baseline_norm = float(np.linalg.norm(captured_native_units - native_shells))
    swapped_norm = float(np.linalg.norm(native_feature_weighted_recovar - native_shells))
    closure = 1.0 - swapped_norm / baseline_norm if baseline_norm > 0.0 else 0.0

    positive = native_feature_f32_matched > 0.0
    feature_ratio = recovar_feature_native_units[positive] / native_feature_f32_matched[positive]
    native_f32_f64_delta = _metric(native_feature_f32_matched, native_feature_f64_matched)
    return {
        "schema": "recovar.em.k1_projected_power_boundary.v1",
        "identity": {
            "iteration": iteration,
            "half": half,
            "part_id": part_id,
            "native_orientation_count": orientation_num,
            "active_rotation_count": int(matched_native.size),
            "panel_pixel_count": panel_size,
            "shell_ids": shell_ids.tolist(),
            "scale_for_stats": scale,
        },
        "rotation_join": {
            "exact_match_count": int(matches.pairs.shape[0]),
            "native_unmatched_count": int(matches.relion_unmatched.size),
            "recovar_unmatched_count": int(matches.recovar_unmatched.size),
        },
        "projected_power_feature": {
            "recovar_vs_native_float32_shell_reduction": _metric(
                recovar_feature_native_units,
                native_feature_f32_matched,
            ),
            "native_float32_vs_float64_shell_reduction": native_f32_f64_delta,
            "recovar_over_native_ratio_median": float(np.median(feature_ratio)),
            "recovar_over_native_ratio_p05": float(np.percentile(feature_ratio, 5)),
            "recovar_over_native_ratio_p95": float(np.percentile(feature_ratio, 95)),
        },
        "aa_replay": {
            "native_feature_native_weights_vs_native": _metric(
                native_feature_weighted_native,
                native_shells,
            ),
            "captured_recovar_vs_native": _metric(captured_native_units, native_shells),
            "native_feature_recovar_weights_vs_native": _metric(
                native_feature_weighted_recovar,
                native_shells,
            ),
            "native_projected_power_residual_closure_fraction": closure,
        },
        "artifacts": {
            "recovar_capture": str(recovar_capture.resolve()),
            "recovar_capture_sha256": _sha256(recovar_capture),
            "native_directory": str(native_directory.resolve()),
            "native_project_panel_real_sha256": _sha256(real_path),
            "native_project_panel_imag_sha256": _sha256(imag_path),
            "native_components": str(native_components.resolve()),
            "native_components_sha256": _sha256(native_components),
        },
        "classification": (
            "projected-reference interpolation/power is causally sufficient for the AA residual"
            if closure >= 0.8
            else "projected-reference power is not the dominant AA residual; Wavg accumulation arithmetic differs"
            if closure <= 0.2
            else "projected-reference interpolation/power partially contributes to the AA residual"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-capture", type=Path, required=True)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--native-components", type=Path, required=True)
    parser.add_argument("--native-prefix", default="img0_part109_storeWavg_")
    parser.add_argument("--recovar-term-divisor", type=float, default=float(128**4))
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.recovar_capture,
        args.native_directory,
        args.native_components,
        native_prefix=args.native_prefix,
        recovar_term_divisor=args.recovar_term_divisor,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
