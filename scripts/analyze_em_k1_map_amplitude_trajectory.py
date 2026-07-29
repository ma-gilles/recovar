#!/usr/bin/env python3
"""Measure shellwise amplitude differences in paired K=1 reference maps.

RECOVAR intermediate map ``itNNN_halfH_reg.mrc`` is paired with RELION
``run_it(NNN+1)_halfH_class001.mrc``.  This is the reference-map boundary
used by the next scoring iteration in the zero-based RECOVAR loop.

This is a diagnostic, not a parity score.  It deliberately reports
shellwise least-squares amplitude fits and normalized L2 residuals rather
than map correlation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

OUTPUT_SCHEMA = "recovar.em_k1_map_amplitude_trajectory.v1"


@dataclass(frozen=True)
class CaseSpec:
    label: str
    recovar_intermediates: Path
    relion_reference: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_case_spec(value: str) -> CaseSpec:
    fields = value.split("=", 1)
    _require(
        len(fields) == 2 and fields[0].strip() and fields[1].strip(),
        "--case must be LABEL=RECOVAR_INTERMEDIATES,RELION_REFERENCE",
    )
    paths = fields[1].split(",", 1)
    _require(
        len(paths) == 2 and paths[0].strip() and paths[1].strip(),
        "--case must be LABEL=RECOVAR_INTERMEDIATES,RELION_REFERENCE",
    )
    return CaseSpec(
        label=fields[0].strip(),
        recovar_intermediates=Path(paths[0]).expanduser().resolve(),
        relion_reference=Path(paths[1]).expanduser().resolve(),
    )


def parse_reference_iterations(value: str) -> tuple[int, ...]:
    try:
        iterations = tuple(
            int(token)
            for token in value.replace(",", " ").replace(";", " ").split()
        )
    except ValueError as error:
        raise ValueError("--reference-iterations must contain integers") from error
    _require(iterations, "--reference-iterations must not be empty")
    _require(
        all(iteration >= 1 for iteration in iterations),
        "--reference-iterations must be one-based positive integers",
    )
    _require(
        tuple(sorted(set(iterations))) == iterations,
        "--reference-iterations must be unique and increasing",
    )
    return iterations


def centered_fourier(volume: np.ndarray) -> np.ndarray:
    """Match RECOVAR's centered, orthonormal three-dimensional DFT."""

    value = np.asarray(volume)
    _require(
        value.ndim == 3 and len(set(value.shape)) == 1,
        f"expected a cubic three-dimensional volume, got {value.shape}",
    )
    shifted = np.fft.fftshift(value, axes=(-3, -2, -1))
    transformed = np.fft.fftn(
        shifted,
        axes=(-3, -2, -1),
        norm="ortho",
    )
    return np.fft.fftshift(transformed, axes=(-3, -2, -1))


def unshifted_shell_labels(volume_shape: Sequence[int]) -> np.ndarray:
    """Match the state-swap diagnostic's rounded unshifted shell labels."""

    shape = tuple(int(size) for size in volume_shape)
    _require(
        len(shape) == 3 and all(size > 0 for size in shape),
        f"expected a positive three-dimensional shape, got {shape}",
    )
    axes = [np.fft.fftfreq(size) * size for size in shape]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.rint(np.sqrt(sum(grid * grid for grid in grids))).astype(
        np.int32
    )


def summarize_fourier_pair(
    recovar_fourier: np.ndarray,
    relion_fourier: np.ndarray,
) -> dict[str, Any]:
    """Fit positive global and per-shell RECOVAR-to-RELION amplitudes."""

    source = np.asarray(recovar_fourier)
    target = np.asarray(relion_fourier)
    _require(source.shape == target.shape, "paired Fourier maps have different shapes")
    _require(source.ndim == 3, "paired Fourier maps must be three-dimensional")
    _require(
        np.all(np.isfinite(source)) and np.all(np.isfinite(target)),
        "paired Fourier maps must be finite",
    )

    source_flat = source.reshape(-1)
    target_flat = target.reshape(-1)
    target_norm = float(np.linalg.norm(target_flat))
    _require(target_norm > 0.0, "RELION reference map has zero energy")
    source_energy = float(np.vdot(source_flat, source_flat).real)
    _require(source_energy > 0.0, "RECOVAR reference map has zero energy")

    before = float(np.linalg.norm(source_flat - target_flat) / target_norm)
    global_scale = float(
        np.vdot(source_flat, target_flat).real / source_energy
    )
    _require(
        np.isfinite(global_scale) and global_scale > 0.0,
        f"global amplitude scale is not positive: {global_scale}",
    )
    after_global = float(
        np.linalg.norm(source_flat * global_scale - target_flat) / target_norm
    )

    labels = unshifted_shell_labels(source.shape).reshape(-1)
    scaled = source_flat.copy()
    shell_rows: list[dict[str, Any]] = []
    for shell in np.unique(labels):
        mask = labels == shell
        source_shell = source_flat[mask]
        target_shell = target_flat[mask]
        denominator = float(np.vdot(source_shell, source_shell).real)
        target_energy = float(np.vdot(target_shell, target_shell).real)
        if denominator <= 0.0:
            _require(
                target_energy <= 0.0,
                f"shell {int(shell)} has RELION energy but no RECOVAR energy",
            )
            scale = 1.0
        else:
            scale = float(
                np.vdot(source_shell, target_shell).real / denominator
            )
            _require(
                np.isfinite(scale) and scale > 0.0,
                f"shell {int(shell)} amplitude scale is not positive: {scale}",
            )
        scaled[mask] = source_shell * scale
        shell_rows.append(
            {
                "shell": int(shell),
                "scale_recovar_to_relion": scale,
                "recovar_energy": denominator,
                "relion_energy": target_energy,
            }
        )

    after_shell = float(np.linalg.norm(scaled - target_flat) / target_norm)
    scales = np.asarray(
        [row["scale_recovar_to_relion"] for row in shell_rows],
        dtype=np.float64,
    )
    explained_fraction = (
        float((before - after_shell) / before) if before > 0.0 else 0.0
    )
    return {
        "relative_l2_before": before,
        "global_scale_recovar_to_relion": global_scale,
        "relative_l2_after_global_scale": after_global,
        "shell_scale_min": float(np.min(scales)),
        "shell_scale_median": float(np.median(scales)),
        "shell_scale_max": float(np.max(scales)),
        "relative_l2_after_shell_scale": after_shell,
        "shell_scale_explained_fraction": explained_fraction,
        "shells": shell_rows,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_recovar_map(path: Path) -> np.ndarray:
    from recovar.utils.helpers import load_mrc

    return np.asarray(load_mrc(str(path)), dtype=np.float32)


def _load_relion_map(path: Path) -> np.ndarray:
    from recovar.utils.helpers import load_relion_volume

    return np.asarray(load_relion_volume(str(path)), dtype=np.float32)


def analyze_case(
    spec: CaseSpec,
    *,
    reference_iterations: Sequence[int],
) -> dict[str, Any]:
    _require(
        spec.recovar_intermediates.is_dir(),
        f"RECOVAR intermediates directory is missing: {spec.recovar_intermediates}",
    )
    _require(
        spec.relion_reference.is_dir(),
        f"RELION reference directory is missing: {spec.relion_reference}",
    )

    rows = []
    volume_shape: tuple[int, ...] | None = None
    for reference_iteration in reference_iterations:
        recovar_iteration = int(reference_iteration) - 1
        for half in (1, 2):
            recovar_path = (
                spec.recovar_intermediates
                / f"it{recovar_iteration:03d}_half{half}_reg.mrc"
            )
            relion_path = (
                spec.relion_reference
                / f"run_it{int(reference_iteration):03d}_half{half}_class001.mrc"
            )
            _require(recovar_path.is_file(), f"missing RECOVAR map: {recovar_path}")
            _require(relion_path.is_file(), f"missing RELION map: {relion_path}")
            recovar_map = _load_recovar_map(recovar_path)
            relion_map = _load_relion_map(relion_path)
            _require(
                recovar_map.shape == relion_map.shape,
                f"map shape mismatch: {recovar_map.shape} != {relion_map.shape}",
            )
            if volume_shape is None:
                volume_shape = tuple(int(size) for size in recovar_map.shape)
            _require(
                tuple(recovar_map.shape) == volume_shape,
                f"case {spec.label} changes volume shape across iterations",
            )
            metrics = summarize_fourier_pair(
                centered_fourier(recovar_map),
                centered_fourier(relion_map),
            )
            rows.append(
                {
                    "reference_iteration": int(reference_iteration),
                    "recovar_intermediate_iteration": recovar_iteration,
                    "half": half,
                    "recovar_map": str(recovar_path),
                    "recovar_map_sha256": _sha256(recovar_path),
                    "relion_map": str(relion_path),
                    "relion_map_sha256": _sha256(relion_path),
                    **metrics,
                }
            )

    return {
        "label": spec.label,
        "recovar_intermediates": str(spec.recovar_intermediates),
        "relion_reference": str(spec.relion_reference),
        "volume_shape": list(volume_shape or ()),
        "map_pair_count": len(rows),
        "trajectory": rows,
    }


def analyze(
    cases: Sequence[CaseSpec],
    *,
    reference_iterations: Sequence[int],
) -> dict[str, Any]:
    _require(cases, "at least one --case is required")
    labels = [case.label for case in cases]
    _require(len(labels) == len(set(labels)), "case labels must be unique")
    iterations = tuple(int(value) for value in reference_iterations)
    _require(
        iterations and all(value >= 1 for value in iterations),
        "reference iterations must be positive",
    )
    _require(
        tuple(sorted(set(iterations))) == iterations,
        "reference iterations must be unique and increasing",
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "metric_policy": (
            "shellwise least-squares amplitude and normalized L2; "
            "correlation is not used"
        ),
        "parity_score": False,
        "pairing_rule": (
            "RECOVAR it(N-1)_halfH_reg.mrc versus "
            "RELION run_itN_halfH_class001.mrc"
        ),
        "shell_definition": (
            "rounded integer radius from unshifted numpy.fft.fftfreq axes; "
            "matches the state-swap diagnostic"
        ),
        "reference_iterations": list(iterations),
        "cases": [
            analyze_case(case, reference_iterations=iterations)
            for case in cases
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        type=parse_case_spec,
        help="repeatable LABEL=RECOVAR_INTERMEDIATES,RELION_REFERENCE",
    )
    parser.add_argument(
        "--reference-iterations",
        default="1,2,3",
        type=parse_reference_iterations,
        help="one-based RELION reference-map iterations (default: 1,2,3)",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = analyze(
        args.case,
        reference_iterations=args.reference_iterations,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output_json.resolve())


if __name__ == "__main__":
    main()
