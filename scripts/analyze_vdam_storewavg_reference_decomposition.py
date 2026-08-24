#!/usr/bin/env python3
"""Decompose VDAM cutoff XA/AA error into posterior and reference terms."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from scripts.analyze_k1_native_wavg_pixels import _wavg_components
from scripts.analyze_vdam_storewavg_boundary import (
    _flat,
    _load_native,
    _match_rotations,
    _metric,
    _production_score_gradient_rows,
    _require,
)
from scripts.analyze_vdam_storewavg_panel import (
    _part_to_original_indices,
    _score_dumps,
)

SCHEMA = "recovar.vdam_storewavg_reference_decomposition.v1"


def _flat_complex(directory: Path, stem: str) -> np.ndarray:
    return _flat(directory / f"{stem}_real.bin", np.dtype("<f8")) + 1j * _flat(
        directory / f"{stem}_imag.bin", np.dtype("<f8")
    )


def _scalar(path: Path) -> float:
    payload = path.read_bytes()
    _require(len(payload) == 8, f"scalar size mismatch: {path}")
    return float(struct.unpack("<d", payload)[0])


def _current_size_from_rectangle_size(pixel_count: int) -> int:
    matches = [
        size
        for size in range(2, 8193, 2)
        if size * (size // 2 + 1) == int(pixel_count)
    ]
    _require(len(matches) == 1, f"cannot infer current size from {pixel_count} Wavg pixels")
    return matches[0]


def _native_rectangle_coordinates(current_size: int) -> tuple[np.ndarray, np.ndarray]:
    half_width = current_size // 2 + 1
    x = np.tile(np.arange(half_width, dtype=np.float32), current_size)
    row = np.repeat(np.arange(current_size, dtype=np.int32), half_width)
    y = np.where(row <= current_size // 2, row, row - current_size).astype(np.float32)
    return x, y


def _translate_native_rectangle(
    image: np.ndarray,
    translation_angles: np.ndarray,
    current_size: int,
) -> np.ndarray:
    """Translate one native FFTW rectangle in RELION's stored phase frame."""

    image = np.asarray(image, dtype=np.complex64).reshape(-1)
    angles = np.asarray(translation_angles, dtype=np.float32)
    x, y = _native_rectangle_coordinates(current_size)
    _require(image.shape == x.shape, "native masked image is not a complete Wavg rectangle")
    _require(angles.ndim == 2 and angles.shape[1] == 2, "native translation angles changed")
    phase = (
        angles[:, 0, None] * x[None, :] + angles[:, 1, None] * y[None, :]
    ).astype(np.float32)
    cosine = np.cos(phase).astype(np.float32)
    sine = np.sin(phase).astype(np.float32)
    real = cosine * image.real[None, :] - sine * image.imag[None, :]
    imag = cosine * image.imag[None, :] + sine * image.real[None, :]
    return (real + 1j * imag).astype(np.complex64)


def _cutoff_sums(
    projections: np.ndarray,
    translated_images: np.ndarray,
    ctf: np.ndarray,
    posterior: np.ndarray,
    cutoff_mask: np.ndarray,
) -> dict[str, float]:
    components = _wavg_components(projections, translated_images, ctf, posterior)
    return {
        name: float(np.sum(components[name][cutoff_mask], dtype=np.float64))
        for name in ("xa", "aa")
    }


def _decompose_components(
    native_projection: np.ndarray,
    candidate_projection: np.ndarray,
    translated_images: np.ndarray,
    ctf: np.ndarray,
    native_posterior: np.ndarray,
    candidate_posterior: np.ndarray,
    cutoff_mask: np.ndarray,
    captured_reference: dict[str, float],
    captured_candidate: dict[str, float],
) -> dict[str, object]:
    native_native = _cutoff_sums(
        native_projection, translated_images, ctf, native_posterior, cutoff_mask
    )
    native_candidate = _cutoff_sums(
        native_projection, translated_images, ctf, candidate_posterior, cutoff_mask
    )
    candidate_candidate = _cutoff_sums(
        candidate_projection, translated_images, ctf, candidate_posterior, cutoff_mask
    )
    components = {}
    for name in ("xa", "aa"):
        posterior_effect = native_candidate[name] - native_native[name]
        reference_effect = candidate_candidate[name] - native_candidate[name]
        unexplained = captured_candidate[name] - candidate_candidate[name]
        total = captured_candidate[name] - captured_reference[name]
        components[name] = {
            "captured_reference": captured_reference[name],
            "captured_candidate": captured_candidate[name],
            "native_projection_native_posterior_replay": native_native[name],
            "native_projection_candidate_posterior_replay": native_candidate[name],
            "candidate_projection_candidate_posterior_replay": candidate_candidate[name],
            "posterior_effect": posterior_effect,
            "reference_projection_effect": reference_effect,
            "unexplained_after_candidate_projection_replay": unexplained,
            "captured_total_error": total,
            "decomposition_closure_error": posterior_effect + reference_effect + unexplained - total,
        }
    return components


def _fine_reference_rectangle(
    capture_directory: Path,
    orientation_count: int,
    rectangle_size: int,
) -> np.ndarray:
    rotation_rows = _flat(
        capture_directory / "pass1_acc_rot_idx.bin", np.dtype("<i4")
    ).astype(np.int64)
    references = _flat_complex(capture_directory, "pass1_class0_fine_ref").reshape(
        rotation_rows.size, rectangle_size
    )
    result = np.empty((orientation_count, rectangle_size), dtype=np.complex64)
    for orientation in range(orientation_count):
        rows = np.flatnonzero(rotation_rows == orientation)
        _require(rows.size > 0, f"native fine panel omits orientation {orientation}")
        result[orientation] = references[rows[0]]
        _require(
            bool(np.all(references[rows] == result[orientation][None, :])),
            f"native fine projection varies across translations for orientation {orientation}",
        )
    return result


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.size > 0, "cannot summarize an empty decomposition")
    return {
        "signed_sum": float(np.sum(array)),
        "mean_abs": float(np.mean(np.abs(array))),
        "max_abs": float(np.max(np.abs(array))),
    }


def analyze(
    panel_root: Path,
    score_directory: Path,
    relion_data_star: Path,
    cutoff_report: Path,
    *,
    physical_image_size: int,
) -> dict[str, object]:
    cutoff = json.loads(cutoff_report.read_text())
    cutoff_by_part = {int(row["part_id"]): row for row in cutoff["per_particle"]}
    original_by_part = _part_to_original_indices(relion_data_star)
    score_by_original = _score_dumps(score_directory)
    part_directories = {
        int(path.name.removeprefix("part-")): path / "capture"
        for path in panel_root.glob("part-[0-9][0-9][0-9]")
        if (path / "capture").is_dir()
    }
    parts = sorted(set(part_directories).intersection(cutoff_by_part))
    _require(parts, "native capture panel and cutoff report have no common particles")

    records = []
    effects = {
        name: {effect: [] for effect in ("posterior", "reference_projection", "unexplained", "total")}
        for name in ("xa", "aa")
    }
    for part_id in parts:
        _require(part_id in original_by_part, f"part {part_id} is absent from RELION data STAR")
        original_index = original_by_part[part_id]
        _require(original_index in score_by_original, f"part {part_id} has no RECOVAR score dump")
        capture = part_directories[part_id]
        prefix = f"img0_part{part_id}_storeWavg_"
        native = _load_native(capture, prefix, load_projector=False)
        with np.load(score_by_original[original_index], allow_pickle=False) as archive:
            score = {name: archive[name] for name in archive.files}
        _require("debug_proj_for_recon" in score, f"score dump lacks reconstruction projection: {score_by_original[original_index]}")
        _production_data, _production_weight, candidate_posterior = (
            _production_score_gradient_rows(score)
        )
        rotation_map = _match_rotations(
            np.asarray(native["rotations"], dtype=np.float32),
            np.asarray(score["local_rotation_matrices"], dtype=np.float32),
            1.0e-5,
        )
        candidate_posterior = np.asarray(candidate_posterior[rotation_map], dtype=np.float32)
        native_posterior = np.asarray(native["probabilities"], dtype=np.float32)
        _require(candidate_posterior.shape == native_posterior.shape, "posterior topology changed")

        rectangle_size = int(round(_scalar(capture / f"{prefix}image_size.bin")))
        current_size = _current_size_from_rectangle_size(rectangle_size)
        candidate_current_size = int(np.asarray(score["current_size"]).reshape(-1)[0])
        _require(current_size == candidate_current_size, "native/candidate current sizes differ")
        from recovar.em.dense_single_volume.helpers.fourier_window import (
            make_fourier_window_indices_np,
        )
        from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
            _make_relion_wavg_rectangle,
        )

        recon_indices, _ = make_fourier_window_indices_np(
            (physical_image_size, physical_image_size),
            current_size,
            include_dc=True,
            exact_radius=True,
        )
        rectangle = _make_relion_wavg_rectangle(
            (physical_image_size, physical_image_size), current_size, recon_indices
        )
        native_projection = _fine_reference_rectangle(
            capture, int(native["orientation_count"]), rectangle_size
        )
        candidate_exact = np.asarray(score["debug_proj_for_recon"], dtype=np.complex64)[
            rotation_map
        ]
        _require(
            candidate_exact.shape == (native_projection.shape[0], rectangle.exact_positions.size),
            "candidate exact-radius projection topology changed",
        )
        # RECOVAR's debug projection is in its centered DFT/volume frame.  The
        # native accelerator reference carries RELION's 2/N^3 projection scale
        # and opposite Fourier sign at this VDAM boundary.
        candidate_exact = (
            candidate_exact * np.float32(-2.0 / physical_image_size**3)
        ).astype(np.complex64)
        candidate_projection = np.zeros_like(native_projection)
        candidate_projection[:, rectangle.exact_positions] = candidate_exact

        masked_image = _flat_complex(
            capture, "preprocess_img0_masked_fourier_post_optics"
        ).astype(np.complex64)
        translated = _translate_native_rectangle(
            masked_image,
            np.asarray(native["translation_angles"], dtype=np.float32),
            current_size,
        )
        cutoff_mask = np.asarray(rectangle.shell_indices) == current_size // 2
        cutoff_row = cutoff_by_part[part_id]
        captured_reference = {
            name: float(cutoff_row["reference"][name]) for name in ("xa", "aa")
        }
        captured_candidate = {
            name: float(cutoff_row["candidate"][name]) for name in ("xa", "aa")
        }
        decomposition = _decompose_components(
            native_projection,
            candidate_projection,
            translated,
            np.asarray(native["ctf"], dtype=np.float32),
            native_posterior,
            candidate_posterior,
            cutoff_mask,
            captured_reference,
            captured_candidate,
        )
        for name in ("xa", "aa"):
            effects[name]["posterior"].append(decomposition[name]["posterior_effect"])
            effects[name]["reference_projection"].append(
                decomposition[name]["reference_projection_effect"]
            )
            effects[name]["unexplained"].append(
                decomposition[name]["unexplained_after_candidate_projection_replay"]
            )
            effects[name]["total"].append(decomposition[name]["captured_total_error"])
        records.append(
            {
                "part_id": part_id,
                "original_index": original_index,
                "current_size": current_size,
                "cutoff_shell": current_size // 2,
                "projection": _metric(
                    native_projection[:, rectangle.exact_positions], candidate_exact
                ),
                "components": decomposition,
                "artifacts": {
                    "capture_directory": str(capture.resolve()),
                    "score_dump": str(score_by_original[original_index].resolve()),
                },
            }
        )

    return {
        "schema": SCHEMA,
        "identity": {
            "particle_count": len(records),
            "part_ids": parts,
            "physical_image_size": physical_image_size,
        },
        "summary": {
            name: {effect: _summary(values) for effect, values in component.items()}
            for name, component in effects.items()
        },
        "per_particle": records,
        "artifacts": {
            "panel_root": str(panel_root.resolve()),
            "score_directory": str(score_directory.resolve()),
            "relion_data_star": str(relion_data_star.resolve()),
            "cutoff_report": str(cutoff_report.resolve()),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--score-directory", type=Path, required=True)
    parser.add_argument("--relion-data-star", type=Path, required=True)
    parser.add_argument("--cutoff-report", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = analyze(
        args.panel_root,
        args.score_directory,
        args.relion_data_star,
        args.cutoff_report,
        physical_image_size=args.physical_image_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("identity", "summary")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
