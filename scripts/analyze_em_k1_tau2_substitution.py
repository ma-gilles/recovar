#!/usr/bin/env python3
"""Replay a numbered K=1 Wiener solve with RECOVAR and RELION tau2.

The saved RECOVAR numerator and weight are held fixed.  Only the tau2 prior
is substituted, which tests whether a RELION/RECOVAR tau2 difference can
explain an observed numbered-map difference.  Map quality is evaluated with
FSC/FSC-AUC; normalized L2 and shellwise amplitude fits are secondary
diagnostics.  Correlation is not computed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import starfile

if __package__:
    from scripts.analyze_em_k1_map_amplitude_trajectory import (
        centered_fourier,
        summarize_fourier_pair,
    )
    from scripts.summarize_em_completion_bench import (
        normalized_fsc_auc,
        shell_fsc,
    )
else:
    from analyze_em_k1_map_amplitude_trajectory import (
        centered_fourier,
        summarize_fourier_pair,
    )
    from summarize_em_completion_bench import normalized_fsc_auc, shell_fsc

OUTPUT_SCHEMA = "recovar.em_k1_tau2_substitution.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalized_l2(value: np.ndarray, target: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    _require(value.shape == target.shape, "normalized L2 inputs differ in shape")
    target_norm = float(np.linalg.norm(target))
    _require(target_norm > 0.0, "normalized L2 target has zero energy")
    return float(np.linalg.norm(value - target) / target_norm)


def map_metrics(value: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    curve = np.asarray(shell_fsc(value, target), dtype=np.float64)
    _require(
        curve.size > 1 and np.any(np.isfinite(curve[1:])),
        "map pair has no finite non-DC FSC shells",
    )
    amplitude = summarize_fourier_pair(
        centered_fourier(value),
        centered_fourier(target),
    )
    return {
        "fsc_auc": float(normalized_fsc_auc(curve)),
        "n_fsc_shells": int(curve.size),
        "relative_l2": normalized_l2(value, target),
        "amplitude": {
            key: amplitude[key]
            for key in (
                "global_scale_recovar_to_relion",
                "shell_scale_min",
                "shell_scale_median",
                "shell_scale_max",
                "relative_l2_after_shell_scale",
                "shell_scale_explained_fraction",
            )
        },
    }


def classify_substitution(
    baseline_relative_l2: float,
    substituted_relative_l2: float,
    *,
    explanation_fraction_gate: float = 0.5,
) -> dict[str, Any]:
    _require(baseline_relative_l2 > 0.0, "baseline relative L2 must be positive")
    _require(
        substituted_relative_l2 >= 0.0,
        "substituted relative L2 must be nonnegative",
    )
    explained_fraction = float(
        (baseline_relative_l2 - substituted_relative_l2)
        / baseline_relative_l2
    )
    classification = (
        "relion_tau2_explains_majority_of_map_residual"
        if explained_fraction >= explanation_fraction_gate
        else "relion_tau2_does_not_explain_map_residual"
    )
    return {
        "classification": classification,
        "relative_l2_explained_fraction": explained_fraction,
        "explanation_fraction_gate": float(explanation_fraction_gate),
    }


def _model(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing RELION model: {path}")
    value = starfile.read(path)
    _require(isinstance(value, dict), f"RELION model is not multi-block: {path}")
    return value


def _general(model: dict[str, Any]) -> dict[str, Any]:
    block = model["model_general"]
    return dict(block.iloc[0]) if hasattr(block, "iloc") else dict(block)


def _relion_tau2(model: dict[str, Any], n4: float) -> np.ndarray:
    table = model["model_class_1"]
    _require("rlnReferenceTau2" in table, "RELION model has no rlnReferenceTau2")
    value = np.asarray(table["rlnReferenceTau2"], dtype=np.float64) * n4
    _require(
        value.ndim == 1 and value.size > 0 and np.all(np.isfinite(value)),
        "RELION tau2 is empty or non-finite",
    )
    return value


def _infer_cubic_shape(size: int) -> tuple[int, int, int]:
    edge = int(round(int(size) ** (1.0 / 3.0)))
    _require(edge**3 == int(size), f"accumulator size {size} is not cubic")
    return (edge, edge, edge)


def _reconstruct_and_flatten(
    weight: np.ndarray,
    numerator: np.ndarray,
    tau2: np.ndarray,
    *,
    volume_shape: tuple[int, int, int],
    accumulator_shape: tuple[int, int, int],
    padding_factor: int,
    projection_padding_factor: int,
    current_size: int,
    minres_map: int,
    voxel_size: float,
    particle_diameter_angstrom: float,
) -> np.ndarray:
    from recovar.core import fourier_transform_utils, mask
    from recovar.em.dense_single_volume.mean_helpers import (
        _reconstruct_volume_eager,
    )

    reconstructed = _reconstruct_volume_eager(
        weight,
        numerator,
        volume_shape,
        padding_factor,
        tau=tau2,
        tau2_fudge=1.0,
        projection_padding_factor=projection_padding_factor,
        minres_map=minres_map,
        current_size=current_size,
        accumulator_volume_shape=accumulator_shape,
    )
    real = np.asarray(
        fourier_transform_utils.get_idft3(
            np.asarray(reconstructed).reshape(volume_shape)
        )
    ).real
    radius = particle_diameter_angstrom / (2.0 * voxel_size)
    solvent_mask = np.asarray(
        mask.raised_cosine_mask(
            volume_shape,
            radius=radius,
            radius_p=radius + 5.0,
            offset=np.zeros(3),
        )
    )
    return np.asarray(real * solvent_mask, dtype=np.float64)


def analyze(
    *,
    intermediates: Path,
    recovar_results: Path,
    relion_dir: Path,
    reference_iteration: int,
    particle_diameter_angstrom: float,
    projection_padding_factor: int,
    minres_map: int,
    replay_fsc_auc_gate: float,
    replay_relative_l2_gate: float,
    explanation_fraction_gate: float,
) -> dict[str, Any]:
    from recovar.utils.helpers import load_mrc, load_relion_volume

    intermediates = intermediates.resolve()
    recovar_results = recovar_results.resolve()
    relion_dir = relion_dir.resolve()
    _require(reference_iteration >= 1, "reference iteration must be positive")
    recovar_iteration = reference_iteration - 1
    _require(recovar_results.is_file(), f"missing {recovar_results}")
    with np.load(recovar_results, allow_pickle=False) as results:
        volume_shape = tuple(
            int(value) for value in np.asarray(results["volume_shape"]).reshape(-1)
        )
    _require(
        len(volume_shape) == 3 and len(set(volume_shape)) == 1,
        f"unexpected volume shape {volume_shape}",
    )
    n4 = float(volume_shape[0] ** 4)

    meta_path = intermediates / f"it{recovar_iteration:03d}_meta.npy"
    _require(meta_path.is_file(), f"missing {meta_path}")
    meta = np.load(meta_path, allow_pickle=True).item()
    current_size = int(meta["current_size"])
    stored_tau_path = intermediates / f"it{recovar_iteration:03d}_tau2.npy"
    _require(stored_tau_path.is_file(), f"missing {stored_tau_path}")
    stored_tau = np.asarray(np.load(stored_tau_path), dtype=np.float32)

    half_rows = []
    for half in (1, 2):
        model_path = (
            relion_dir
            / f"run_it{reference_iteration:03d}_half{half}_model.star"
        )
        model = _model(model_path)
        general = _general(model)
        padding_factor = int(general["rlnPaddingFactor"])
        voxel_size = float(general["rlnPixelSize"])
        _require(
            int(general["rlnCurrentImageSize"]) == current_size,
            "RELION and RECOVAR current sizes differ",
        )
        relion_tau = _relion_tau2(model, n4)

        numerator_path = (
            intermediates
            / f"it{recovar_iteration:03d}_Ft_y_{half - 1}.npy"
        )
        weight_path = (
            intermediates
            / f"it{recovar_iteration:03d}_Ft_ctf_{half - 1}.npy"
        )
        recovar_map_path = (
            intermediates
            / f"it{recovar_iteration:03d}_half{half}_reg.mrc"
        )
        relion_map_path = (
            relion_dir
            / f"run_it{reference_iteration:03d}_half{half}_class001.mrc"
        )
        for path in (
            numerator_path,
            weight_path,
            recovar_map_path,
            relion_map_path,
        ):
            _require(path.is_file(), f"missing {path}")
        numerator = np.load(numerator_path)
        weight = np.load(weight_path)
        _require(
            numerator.shape == weight.shape,
            f"accumulator shape mismatch for half {half}",
        )
        accumulator_shape = _infer_cubic_shape(numerator.size)
        saved_recovar = np.asarray(load_mrc(str(recovar_map_path)), dtype=np.float64)
        saved_relion = np.asarray(
            load_relion_volume(str(relion_map_path)),
            dtype=np.float64,
        )

        stored_replay = _reconstruct_and_flatten(
            weight,
            numerator,
            stored_tau,
            volume_shape=volume_shape,
            accumulator_shape=accumulator_shape,
            padding_factor=padding_factor,
            projection_padding_factor=projection_padding_factor,
            current_size=current_size,
            minres_map=minres_map,
            voxel_size=voxel_size,
            particle_diameter_angstrom=particle_diameter_angstrom,
        )
        relion_tau_replay = _reconstruct_and_flatten(
            weight,
            numerator,
            relion_tau,
            volume_shape=volume_shape,
            accumulator_shape=accumulator_shape,
            padding_factor=padding_factor,
            projection_padding_factor=projection_padding_factor,
            current_size=current_size,
            minres_map=minres_map,
            voxel_size=voxel_size,
            particle_diameter_angstrom=particle_diameter_angstrom,
        )

        replay_integrity = map_metrics(stored_replay, saved_recovar)
        _require(
            replay_integrity["fsc_auc"] >= replay_fsc_auc_gate,
            f"stored-tau replay FSC-AUC failed for half {half}",
        )
        _require(
            replay_integrity["relative_l2"] <= replay_relative_l2_gate,
            f"stored-tau replay relative L2 failed for half {half}",
        )
        baseline = map_metrics(stored_replay, saved_relion)
        substituted = map_metrics(relion_tau_replay, saved_relion)
        classification = classify_substitution(
            baseline["relative_l2"],
            substituted["relative_l2"],
            explanation_fraction_gate=explanation_fraction_gate,
        )
        half_rows.append(
            {
                "half": half,
                "current_size": current_size,
                "accumulator_shape": list(accumulator_shape),
                "recovar_map": str(recovar_map_path),
                "relion_map": str(relion_map_path),
                "relion_model": str(model_path),
                "stored_tau_replay_vs_saved_recovar": replay_integrity,
                "stored_tau_replay_vs_relion": baseline,
                "relion_tau_replay_vs_relion": substituted,
                **classification,
            }
        )

    all_reject = all(
        row["classification"] == "relion_tau2_does_not_explain_map_residual"
        for row in half_rows
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "metric_policy": (
            "FSC/FSC-AUC primary; normalized L2 and shellwise amplitude "
            "secondary; correlation is not computed"
        ),
        "reference_iteration": reference_iteration,
        "recovar_intermediate_iteration": recovar_iteration,
        "volume_shape": list(volume_shape),
        "particle_diameter_angstrom": float(particle_diameter_angstrom),
        "projection_padding_factor": int(projection_padding_factor),
        "minres_map": int(minres_map),
        "replay_fsc_auc_gate": float(replay_fsc_auc_gate),
        "replay_relative_l2_gate": float(replay_relative_l2_gate),
        "explanation_fraction_gate": float(explanation_fraction_gate),
        "classification": (
            "relion_tau2_rejected_as_map_residual_cause"
            if all_reject
            else "relion_tau2_not_rejected_for_all_halves"
        ),
        "halves": half_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediates", required=True, type=Path)
    parser.add_argument("--recovar-results", required=True, type=Path)
    parser.add_argument("--relion-dir", required=True, type=Path)
    parser.add_argument("--reference-iteration", required=True, type=int)
    parser.add_argument(
        "--particle-diameter-angstrom",
        required=True,
        type=float,
    )
    parser.add_argument("--projection-padding-factor", type=int, default=2)
    parser.add_argument("--minres-map", type=int, default=5)
    parser.add_argument("--replay-fsc-auc-gate", type=float, default=0.99999)
    parser.add_argument("--replay-relative-l2-gate", type=float, default=0.001)
    parser.add_argument(
        "--explanation-fraction-gate",
        type=float,
        default=0.5,
    )
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = analyze(
        intermediates=args.intermediates,
        recovar_results=args.recovar_results,
        relion_dir=args.relion_dir,
        reference_iteration=args.reference_iteration,
        particle_diameter_angstrom=args.particle_diameter_angstrom,
        projection_padding_factor=args.projection_padding_factor,
        minres_map=args.minres_map,
        replay_fsc_auc_gate=args.replay_fsc_auc_gate,
        replay_relative_l2_gate=args.replay_relative_l2_gate,
        explanation_fraction_gate=args.explanation_fraction_gate,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output_json.resolve())


if __name__ == "__main__":
    main()
