#!/usr/bin/env python3
"""Audit the complete VDAM adaptive-sampling trajectory against RELION."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import starfile

from recovar.em import sampling

_ITERATION_RE = re.compile(r"run_it(\d+)_recovar_meta\.json$")


def _native_tables(path: Path) -> dict[str, object]:
    value = starfile.read(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected named STAR blocks in {path}")
    return value


def _native_model_sampling(path: Path) -> dict[str, float | int]:
    tables = _native_tables(path)
    classes = tables["model_classes"]
    if len(classes) != 1:
        raise ValueError(f"K=1 sampling audit expected one model class in {path}")
    general = tables["model_general"]
    if not isinstance(general, dict):
        raise ValueError(f"expected scalar model_general block in {path}")
    return {
        "sampling_acc_rot": float(classes.iloc[0]["rlnAccuracyRotations"]),
        "sampling_acc_trans_angstrom": float(
            classes.iloc[0]["rlnAccuracyTranslationsAngst"]
        ),
        "orientational_prior_mode": int(general["rlnOrientationalPriorMode"]),
        "current_size": int(general["rlnCurrentImageSize"]),
        "current_resolution_angstrom": float(general["rlnCurrentResolution"]),
    }


def _native_sampling(path: Path) -> dict[str, float | int]:
    tables = _native_tables(path)
    general = tables.get("sampling_general", tables)
    if not isinstance(general, dict):
        raise ValueError(f"expected scalar sampling_general block in {path}")
    return {
        "healpix_order": int(general["rlnHealpixOrder"]),
        "offset_range_angstrom": float(general["rlnOffsetRange"]),
        "offset_step_angstrom": float(general["rlnOffsetStep"]),
        "random_perturbation": float(general["rlnSamplingPerturbInstance"]),
    }


def _native_changes(path: Path) -> dict[str, float | int]:
    tables = _native_tables(path)
    general = tables.get("optimiser_general", tables)
    if not isinstance(general, dict):
        raise ValueError(f"expected scalar optimiser_general block in {path}")
    return {
        "current_changes_optimal_offsets_angstrom": float(general["rlnChangesOptimalOffsets"]),
        "nr_iter_without_resolution_gain": int(general["rlnNumberOfIterWithoutResolutionGain"]),
    }


def _native_translation_count(
    native: dict[str, float | int],
    *,
    pixel_size: float,
    oversampling: int,
) -> int:
    coarse = sampling.get_relion_translation_grid(
        float(native["offset_range_angstrom"]) / pixel_size,
        float(native["offset_step_angstrom"]) / pixel_size,
        source_units_per_pixel=pixel_size,
    )
    return int(coarse.shape[0]) * (4 ** int(oversampling))


def _close(candidate: float, native: float, *, atol: float) -> bool:
    return bool(abs(float(candidate) - float(native)) <= float(atol))


def audit_sampling_trajectory(
    candidate_dir: Path,
    relion_dir: Path,
    *,
    pixel_size: float,
    iterations: list[int] | None = None,
) -> dict[str, object]:
    if pixel_size <= 0:
        raise ValueError("pixel_size must be positive")
    if iterations is None:
        iterations = sorted(
            int(match.group(1))
            for path in candidate_dir.glob("run_it*_recovar_meta.json")
            if (match := _ITERATION_RE.search(path.name)) is not None and int(match.group(1)) > 0
        )
    if not iterations:
        raise ValueError("no positive VDAM iterations selected")

    rows: list[dict[str, object]] = []
    for row_index, iteration in enumerate(iterations):
        tag = f"{int(iteration):03d}"
        candidate_path = candidate_dir / f"run_it{tag}_recovar_meta.json"
        sampling_path = relion_dir / f"run_it{tag}_sampling.star"
        model_path = relion_dir / f"run_it{tag}_model.star"
        optimiser_path = relion_dir / f"run_it{tag}_optimiser.star"
        for path in (candidate_path, sampling_path, model_path, optimiser_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        candidate = json.loads(candidate_path.read_text())
        native = _native_sampling(sampling_path)
        native_model = _native_model_sampling(model_path)
        native_changes = _native_changes(optimiser_path)
        candidate_current_resolution = float(candidate["current_resolution"])
        if candidate_current_resolution <= 0.0:
            raise ValueError(
                f"candidate current_resolution must be positive in {candidate_path}"
            )
        candidate_current_resolution_angstrom = 1.0 / candidate_current_resolution
        oversampling = int(candidate["oversampling"])
        native_n_translations = _native_translation_count(
            native,
            pixel_size=pixel_size,
            oversampling=oversampling,
        )
        native_updated = False
        if row_index > 0:
            previous_tag = f"{int(iteration) - 1:03d}"
            previous_sampling_path = relion_dir / f"run_it{previous_tag}_sampling.star"
            if not previous_sampling_path.is_file():
                raise FileNotFoundError(previous_sampling_path)
            previous_native = _native_sampling(previous_sampling_path)
            native_updated = any(
                native[key] != previous_native[key]
                for key in ("healpix_order", "offset_range_angstrom", "offset_step_angstrom")
            )
        candidate_prior_mode_value = candidate.get("orientational_prior_mode")
        candidate_prior_mode = (
            None
            if candidate_prior_mode_value is None
            else int(candidate_prior_mode_value)
        )
        candidate_uniform_prior_value = candidate.get("uniform_local_orientation_prior")
        candidate_uniform_prior = (
            None
            if candidate_uniform_prior_value is None
            else bool(candidate_uniform_prior_value)
        )

        checks = {
            "healpix_order": int(candidate["healpix_order"]) == int(native["healpix_order"]),
            "offset_range": _close(
                candidate["offset_range_angstrom"], native["offset_range_angstrom"], atol=5.1e-7
            ),
            "offset_step": _close(
                candidate["offset_step_angstrom"], native["offset_step_angstrom"], atol=5.1e-7
            ),
            "random_perturbation": _close(
                candidate["random_perturbation"], native["random_perturbation"], atol=5.1e-6
            ),
            "translation_topology": int(candidate["n_translations"]) == native_n_translations,
            "accuracy_rotation": _close(
                candidate["sampling_acc_rot"], native_model["sampling_acc_rot"], atol=5.1e-4
            ),
            "accuracy_translation": _close(
                candidate["sampling_acc_trans_angstrom"],
                native_model["sampling_acc_trans_angstrom"],
                atol=5.1e-7,
            ),
            "optimal_offset_change": _close(
                candidate["current_changes_optimal_offsets_angstrom"],
                native_changes["current_changes_optimal_offsets_angstrom"],
                atol=5.1e-7,
            ),
            "orientational_prior_mode": candidate_prior_mode is not None
            and candidate_prior_mode == int(native_model["orientational_prior_mode"]),
            "current_size": int(candidate["current_size"])
            == int(native_model["current_size"]),
            "current_resolution": _close(
                candidate_current_resolution_angstrom,
                native_model["current_resolution_angstrom"],
                atol=5.1e-7,
            ),
        }
        if row_index > 0:
            checks["sampling_updated"] = bool(candidate["sampling_updated"]) == native_updated

        rows.append(
            {
                "iteration": int(iteration),
                "candidate": {
                    "healpix_order": int(candidate["healpix_order"]),
                    "offset_range_angstrom": float(candidate["offset_range_angstrom"]),
                    "offset_step_angstrom": float(candidate["offset_step_angstrom"]),
                    "random_perturbation": float(candidate["random_perturbation"]),
                    "n_translations": int(candidate["n_translations"]),
                    "sampling_acc_rot": float(candidate["sampling_acc_rot"]),
                    "sampling_acc_trans_angstrom": float(candidate["sampling_acc_trans_angstrom"]),
                    "sampling_updated": bool(candidate["sampling_updated"]),
                    "sampling_accuracy_estimated": bool(
                        candidate.get("sampling_accuracy_estimated", True)
                    ),
                    "current_size": int(candidate["current_size"]),
                    "current_resolution": candidate_current_resolution,
                    "current_resolution_angstrom": candidate_current_resolution_angstrom,
                    "current_resolution_shell": int(candidate["current_resolution_shell"]),
                    "orientational_prior_mode": candidate_prior_mode,
                    "uniform_local_orientation_prior": candidate_uniform_prior,
                    "current_changes_optimal_offsets_angstrom": float(
                        candidate["current_changes_optimal_offsets_angstrom"]
                    ),
                    "nr_iter_without_resolution_gain": int(candidate["sampling_nr_iter_wo_resol_gain"]),
                },
                "native": {
                    **native,
                    "n_translations": native_n_translations,
                    **native_model,
                    "sampling_updated": native_updated,
                    **native_changes,
                },
                "absolute_errors": {
                    "offset_range_angstrom": abs(
                        float(candidate["offset_range_angstrom"])
                        - float(native["offset_range_angstrom"])
                    ),
                    "offset_step_angstrom": abs(
                        float(candidate["offset_step_angstrom"])
                        - float(native["offset_step_angstrom"])
                    ),
                    "sampling_acc_rot": abs(
                        float(candidate["sampling_acc_rot"])
                        - float(native_model["sampling_acc_rot"])
                    ),
                    "sampling_acc_trans_angstrom": abs(
                        float(candidate["sampling_acc_trans_angstrom"])
                        - float(native_model["sampling_acc_trans_angstrom"])
                    ),
                    "current_resolution_angstrom": abs(
                        candidate_current_resolution_angstrom
                        - float(native_model["current_resolution_angstrom"])
                    ),
                    "current_changes_optimal_offsets_angstrom": abs(
                        float(candidate["current_changes_optimal_offsets_angstrom"])
                        - float(native_changes["current_changes_optimal_offsets_angstrom"])
                    ),
                },
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    check_names = sorted({name for row in rows for name in row["checks"]})
    first_mismatch = {
        name: next(
            (int(row["iteration"]) for row in rows if name in row["checks"] and not row["checks"][name]),
            None,
        )
        for name in check_names
    }
    return {
        "schema": "recovar.vdam_sampling_trajectory_audit.v1",
        "candidate_dir": str(candidate_dir.resolve()),
        "relion_dir": str(relion_dir.resolve()),
        "pixel_size": float(pixel_size),
        "iteration_count": len(rows),
        "first_mismatch": first_mismatch,
        "result": "pass" if all(row["pass"] for row in rows) else "fail",
        "iterations": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--relion-dir", type=Path, required=True)
    parser.add_argument("--pixel-size", type=float, required=True)
    parser.add_argument("--iterations", type=int, nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_sampling_trajectory(
        args.candidate_dir,
        args.relion_dir,
        pixel_size=args.pixel_size,
        iterations=args.iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("result", "iteration_count", "first_mismatch")}, indent=2))
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
