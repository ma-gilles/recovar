#!/usr/bin/env python3
"""Seal RELION/RECOVAR BPref factor and reduction-precision comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.validate_relion_bpref_factor_capture import load_factor_capture, validate_directory


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, object]:
    left = np.asarray(lhs)
    right = np.asarray(rhs)
    _require(left.shape == right.shape, "comparison shape changed")
    delta = right.astype(np.complex128) - left.astype(np.complex128)
    denominator = max(float(np.linalg.norm(left.astype(np.complex128))), np.finfo(np.float64).tiny)
    return {
        "shape": list(left.shape),
        "lhs_dtype": str(left.dtype),
        "rhs_dtype": str(right.dtype),
        "exact_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "relative_l2_over_lhs": float(np.linalg.norm(delta) / denominator),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
    }


def _distribution(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.size > 0 and np.all(np.isfinite(array)), "invalid per-particle metric")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(np.max(array)),
        "values_in_selection_order": array.tolist(),
    }


def _per_particle_relative_l2(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, object]:
    _require(lhs.shape == rhs.shape and lhs.ndim >= 2, "per-particle shape changed")
    return _distribution([float(_metric(left, right)["relative_l2_over_lhs"]) for left, right in zip(lhs, rhs)])


def _load_sealed_recovar(
    npz_path: Path, json_path: Path, seal_path: Path
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    seal = json.loads(seal_path.read_text())
    _require(seal.get("schema") == "recovar-bpref-factor-authoritative-seal-v1", "RECOVAR seal schema changed")
    authoritative = seal.get("authoritative")
    _require(isinstance(authoritative, dict), "RECOVAR authoritative seal is missing")
    _require(
        int(authoritative.get("minimum_authoritative_version", 0)) >= 5,
        "RECOVAR factor version predates precision controls",
    )
    _require(
        npz_path.name == authoritative.get("npz") and json_path.name == authoritative.get("json"),
        "RECOVAR factor path is not authoritative",
    )
    _require(_sha256(npz_path) == authoritative.get("npz_sha256"), "RECOVAR factor NPZ SHA-256 changed")
    _require(_sha256(json_path) == authoritative.get("json_sha256"), "RECOVAR factor JSON SHA-256 changed")
    report = json.loads(json_path.read_text())
    _require(report.get("factor_ready") is True, "RECOVAR factors are not qualified")
    _require(report.get("output_npz_sha256") == _sha256(npz_path), "RECOVAR JSON/NPZ binding changed")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {
        "numerator_f32",
        "numerator_highest_f32",
        "numerator_sequential_f32",
        "numerator_f64",
        "term_f32",
        "centered_packed_indices",
        "translation_vectors_f32",
    }
    _require(
        required <= arrays.keys(), f"RECOVAR precision controls are incomplete: {sorted(required - arrays.keys())}"
    )
    return report, arrays


def _classify(summary: dict[str, dict[str, object]], factors_close: bool, relion_terms_close: bool) -> str:
    default_error = float(summary["relion_vs_recovar_default_f32"]["global"]["relative_l2_over_lhs"])
    highest_error = float(summary["relion_vs_recovar_highest_f32"]["global"]["relative_l2_over_lhs"])
    sequential_error = float(summary["relion_vs_recovar_sequential_f32"]["global"]["relative_l2_over_lhs"])
    f64_error = float(summary["relion_vs_recovar_genuine_f64"]["global"]["relative_l2_over_lhs"])
    if (
        factors_close
        and relion_terms_close
        and highest_error < 1e-5
        and sequential_error < 1e-5
        and f64_error < 1e-5
        and default_error > 100 * max(highest_error, np.finfo(np.float64).tiny)
        and default_error > 100 * max(f64_error, np.finfo(np.float64).tiny)
    ):
        return "recovar_default_gemm_reduced_precision"
    return "unresolved"


def compare(
    capture_directory: Path,
    selection_json: Path,
    validation_json: Path,
    inertness_json: Path,
    recovar_npz: Path,
    recovar_json: Path,
    recovar_seal: Path,
    *,
    expected_rank: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    validation = json.loads(validation_json.read_text())
    inertness = json.loads(inertness_json.read_text())
    fresh_validation = validate_directory(capture_directory, selection_json, expected_rank=expected_rank)
    _require(
        validation.get("capture_ready") is True and inertness.get("capture_inertness_qualified") is True,
        "RELION capture is not qualified and inert",
    )
    _require(
        validation.get("selection_sha256") == _sha256(selection_json), "RELION validation selection binding changed"
    )
    _require(
        validation.get("artifact_sha256") == fresh_validation.get("artifact_sha256"),
        "RELION capture changed after validation",
    )
    recovar_report, recovar = _load_sealed_recovar(recovar_npz, recovar_json, recovar_seal)
    _require(recovar_report.get("selection_sha256") == _sha256(selection_json), "RECOVAR selection binding changed")

    selection = json.loads(selection_json.read_text())["selected"]
    stacks = np.asarray([int(record["stack_index_1based"]) for record in selection], dtype=np.int64)
    _require(np.array_equal(recovar["stack_indices_1based"], stacks), "RECOVAR stack order changed")
    captures_by_stack = {
        capture.stack_index: capture
        for capture in (load_factor_capture(path) for path in capture_directory.glob("*.bpre-v2.bin"))
    }
    particle_count, window_size = recovar["term_f32"].shape
    relion_arrays = {
        name: np.empty((particle_count, window_size), dtype=dtype)
        for name, dtype in {
            "processed_fft": np.complex64,
            "ctf": np.float32,
            "inverse_noise": np.float32,
            "shifted_image": np.complex64,
            "weighted_ctf": np.float32,
            "term": np.complex64,
            "weight_term": np.float32,
            "summary": np.complex64,
        }.items()
    }
    support = np.zeros((particle_count, window_size), dtype=bool)
    relion_translation = np.empty((particle_count, 2), dtype=np.float32)
    relion_term_to_summary_errors: list[float] = []
    image_box = 256.0
    for row, stack in enumerate(stacks):
        capture = captures_by_stack[int(stack)]
        pixel_lookup = {
            (int(x), int(y)): index for index, (x, y) in enumerate(zip(capture.pixels["x"], capture.pixels["y"]))
        }
        coordinates = [(int(index) % 129, int(index) // 129 - 128) for index in recovar["centered_packed_indices"][row]]
        _require(
            len(set(coordinates)) == window_size and all(value in pixel_lookup for value in coordinates),
            f"stack {stack}: pixel geometry does not close",
        )
        pixel_rows = np.asarray([pixel_lookup[value] for value in coordinates], dtype=np.int64)
        accepted = np.flatnonzero((capture.hypotheses["flags"] & 1) != 0)
        _require(accepted.size == 1, f"stack {stack}: accepted hypothesis count changed")
        hypothesis = capture.hypotheses[int(accepted[0])]
        orientation = int(hypothesis["orientation_local"])
        translation = int(hypothesis["translation"])
        _require(
            orientation == int(recovar["rotation_rows"][row]) and translation == int(recovar["translation_rows"][row]),
            f"stack {stack}: winner identity changed",
        )
        term_panel = capture.terms.reshape(1, capture.pixels.size)[0][pixel_rows]
        scale = float(recovar["scale_f32"][row])
        relion_arrays["processed_fft"][row] = (
            capture.pixels["image_re"][pixel_rows] + 1j * capture.pixels["image_im"][pixel_rows]
        ) * image_box**2
        relion_arrays["ctf"][row] = -capture.pixels["ctf"][pixel_rows] / scale
        relion_arrays["inverse_noise"][row] = capture.pixels["minvsigma2"][pixel_rows] / image_box**4
        relion_arrays["shifted_image"][row] = (
            term_panel["translated_re"] + 1j * term_panel["translated_im"]
        ) * image_box**2
        relion_arrays["weighted_ctf"][row] = -term_panel["weighted_ctf"] / image_box**4
        relion_arrays["term"][row] = -(term_panel["term_re"] + 1j * term_panel["term_im"]) / image_box**2
        relion_arrays["weight_term"][row] = term_panel["weight_term"] / image_box**4
        relion_translation[row] = (
            capture.translations["x"][translation],
            capture.translations["y"][translation],
        )
        summary_lookup = {
            (int(x), int(y)): index for index, (x, y) in enumerate(zip(capture.summaries["x"], capture.summaries["y"]))
        }
        support[row] = np.asarray([value in summary_lookup for value in coordinates])
        _require(
            int(np.count_nonzero(support[row])) == capture.summaries.size, f"stack {stack}: summary support changed"
        )
        summary_rows = np.asarray(
            [summary_lookup[value] for index, value in enumerate(coordinates) if support[row, index]],
            dtype=np.int64,
        )
        summary = capture.summaries["source_re"][summary_rows] + 1j * capture.summaries["source_im"][summary_rows]
        relion_arrays["summary"][row, support[row]] = -summary / image_box**2
        relion_arrays["summary"][row, ~support[row]] = 0
        relion_term_to_summary_errors.append(
            float(_metric(summary, -relion_arrays["term"][row, support[row]] * image_box**2)["relative_l2_over_lhs"])
        )

    factor_pairs = {
        "processed_fft": (relion_arrays["processed_fft"], recovar["processed_fft_f32"]),
        "ctf": (relion_arrays["ctf"], recovar["ctf_f32"]),
        "inverse_noise": (relion_arrays["inverse_noise"], 1.0 / recovar["noise_f32"]),
        "shifted_image": (relion_arrays["shifted_image"], recovar["shifted_image_f32"]),
        "weighted_ctf": (relion_arrays["weighted_ctf"], recovar["weighted_ctf_f32"]),
        "term": (relion_arrays["term"], recovar["term_f32"]),
        "weight_term": (relion_arrays["weight_term"], recovar["weight_term_f32"]),
    }
    factors = {
        name: {"global": _metric(left, right), "per_particle_relative_l2": _per_particle_relative_l2(left, right)}
        for name, (left, right) in factor_pairs.items()
    }
    expected_translation = -2 * np.pi * recovar["translation_vectors_f32"] / image_box
    factors["translation_phase_increment"] = {
        "global": _metric(relion_translation, expected_translation),
        "per_particle_relative_l2": _per_particle_relative_l2(relion_translation, expected_translation),
    }
    summary_pairs = {
        "relion_vs_recovar_default_f32": recovar["numerator_f32"],
        "relion_vs_recovar_highest_f32": recovar["numerator_highest_f32"],
        "relion_vs_recovar_sequential_f32": recovar["numerator_sequential_f32"],
        "relion_vs_recovar_genuine_f64": recovar["numerator_f64"],
        "relion_vs_recovar_direct_f32_term": recovar["term_f32"],
    }
    relion_summary = relion_arrays["summary"][support]
    summaries = {
        name: {
            "global": _metric(relion_summary, candidate[support]),
            "per_particle_relative_l2": _distribution(
                [
                    float(
                        _metric(relion_arrays["summary"][row, support[row]], candidate[row, support[row]])[
                            "relative_l2_over_lhs"
                        ]
                    )
                    for row in range(particle_count)
                ]
            ),
        }
        for name, candidate in summary_pairs.items()
    }
    factors_close = all(float(value["global"]["relative_l2_over_lhs"]) < 1e-5 for value in factors.values())
    relion_terms_close = max(relion_term_to_summary_errors) < 1e-6
    classification = _classify(summaries, factors_close, relion_terms_close)
    _require(classification != "unresolved", "factor boundary remains unresolved")
    output_arrays = {
        "stack_indices_1based": stacks,
        "support": support,
        "relion_summary": relion_arrays["summary"],
        "recovar_default_f32": recovar["numerator_f32"],
        "recovar_highest_f32": recovar["numerator_highest_f32"],
        "recovar_sequential_f32": recovar["numerator_sequential_f32"],
        "recovar_genuine_f64": recovar["numerator_f64"],
    }
    report = {
        "schema": "relion-recovar-bpref-factor-comparison-v1",
        "metric_policy": "exact/array metrics for intermediates; no correlation; FSC/FSC-AUC reserved for maps",
        "selection_json": str(selection_json.resolve()),
        "selection_sha256": _sha256(selection_json),
        "relion_validation_json": str(validation_json.resolve()),
        "relion_validation_sha256": _sha256(validation_json),
        "relion_inertness_json": str(inertness_json.resolve()),
        "relion_inertness_sha256": _sha256(inertness_json),
        "recovar_factor_npz": str(recovar_npz.resolve()),
        "recovar_factor_npz_sha256": _sha256(recovar_npz),
        "recovar_factor_json_sha256": _sha256(recovar_json),
        "recovar_factor_seal_sha256": _sha256(recovar_seal),
        "particle_count": particle_count,
        "support_count_per_particle": np.sum(support, axis=1).astype(int).tolist(),
        "factor_comparisons": factors,
        "relion_term_to_summary_relative_l2": _distribution(relion_term_to_summary_errors),
        "summary_comparisons": summaries,
        "classification": classification,
        "factor_boundary_closed": True,
    }
    return report, output_arrays


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--selection-json", required=True, type=Path)
    parser.add_argument("--validation-json", required=True, type=Path)
    parser.add_argument("--inertness-json", required=True, type=Path)
    parser.add_argument("--recovar-npz", required=True, type=Path)
    parser.add_argument("--recovar-json", required=True, type=Path)
    parser.add_argument("--recovar-seal", required=True, type=Path)
    parser.add_argument("--expected-rank", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists() or args.output_npz.exists():
        raise FileExistsError("refusing to overwrite factor comparison artifact")
    report, arrays = compare(
        args.capture_directory,
        args.selection_json,
        args.validation_json,
        args.inertness_json,
        args.recovar_npz,
        args.recovar_json,
        args.recovar_seal,
        expected_rank=args.expected_rank,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **arrays)
    report["output_npz"] = str(args.output_npz.resolve())
    report["output_npz_sha256"] = _sha256(args.output_npz)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
