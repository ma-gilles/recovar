#!/usr/bin/env python3
"""Qualify dynamic VDAM dense fallback against a static-dense exact oracle."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Sequence

import mrcfile
import numpy as np

SCHEMA = "recovar.vdam_schedule_representation_gate.v1"
REQUIRED_EXACT_META = (
    "selected_particle_ids",
    "best_pose_rotation_ids",
    "best_pose_rotations",
    "pose_assignments",
    "class_assignments",
    "best_pose_translations",
    "max_posterior_per_image",
    "significant_counts",
    "class_direction_posterior_sums",
    "wsum_sigma2_offset",
    "halfset_ids",
    "joint_halfset_particle_stream",
)
REQUIRED_EXACT_SCALARS = (
    "current_resolution",
    "current_resolution_shell",
    "current_size",
    "current_changes_optimal_classes",
    "current_changes_optimal_offsets_angstrom",
    "sigma2_offset_before",
    "sigma_offset_angstrom",
    "sampling_acc_rot",
    "sampling_acc_trans_angstrom",
    "sampling_accuracy_estimated",
)
ATOMIC_META = (
    "ave_Pmax",
    "class_posterior_sums",
    "class_posterior_sums_full",
    "class_reconstruction_support_sums",
    "noise_sumw",
    "sigma2_offset_sumw",
    "wsum_img_power",
    "wsum_sigma2_noise",
    "class_bpref_weight_sums",
    "halfset_0_pmax_mean",
    "halfset_0_class_posterior_sums",
    "halfset_0_class_posterior_sums_full",
    "halfset_0_class_reconstruction_support_sums",
    "halfset_0_noise_sumw",
    "halfset_0_sigma2_offset_sumw",
    "halfset_0_wsum_img_power",
    "halfset_0_wsum_sigma2_noise",
    "halfset_0_class_bpref_weight_sums",
    "halfset_1_class_bpref_weight_sums",
)
PANEL_ARM_ORDER = (
    "dynamic_1",
    "oracle_1",
    "oracle_2",
    "dynamic_2",
    "oracle_3",
    "dynamic_3",
    "dynamic_4",
    "oracle_4",
)
PANEL_SCHEMA = "recovar.vdam_schedule_representation_panel.v1"


class GateSetupError(RuntimeError):
    """Raised when a schedule-comparison artifact is incomplete."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise GateSetupError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise GateSetupError(f"JSON artifact is not an object: {path}")
    return value


def _array_delta(left: Any, right: Any) -> dict[str, Any]:
    lhs = np.asarray(left)
    rhs = np.asarray(right)
    if lhs.shape != rhs.shape:
        return {
            "comparable": False,
            "left_shape": list(lhs.shape),
            "right_shape": list(rhs.shape),
            "exact_equal": False,
        }
    exact = bool(np.array_equal(lhs, rhs, equal_nan=True))
    result: dict[str, Any] = {
        "comparable": True,
        "shape": list(lhs.shape),
        "exact_equal": exact,
    }
    if lhs.dtype.kind not in "biufc" or rhs.dtype.kind not in "biufc":
        result["mismatch_count"] = int(np.count_nonzero(lhs != rhs))
        return result
    left64 = lhs.astype(np.complex128 if np.iscomplexobj(lhs) else np.float64)
    right64 = rhs.astype(np.complex128 if np.iscomplexobj(rhs) else np.float64)
    delta = left64 - right64
    absolute = np.abs(delta)
    scale = max(
        float(np.linalg.norm(left64.reshape(-1))),
        float(np.linalg.norm(right64.reshape(-1))),
        float(np.finfo(np.float64).tiny),
    )
    result.update(
        mismatch_count=int(np.count_nonzero(lhs != rhs)),
        max_abs_delta=float(np.max(absolute)) if absolute.size else 0.0,
        normalized_l2_delta=float(np.linalg.norm(delta.reshape(-1)) / scale),
    )
    return result


def _exact_panel(values: dict[str, Any]) -> dict[str, Any]:
    comparisons = {
        f"{left}__vs__{right}": _array_delta(values[left], values[right])
        for left, right in itertools.combinations(values, 2)
    }
    unequal = [
        pair
        for pair, comparison in comparisons.items()
        if comparison.get("exact_equal") is not True
    ]
    return {
        "pair_count": len(comparisons),
        "all_exact": not unequal,
        "unequal_pairs": unequal,
        "comparisons": comparisons,
    }


def _multi_repeat_envelope(
    values: dict[str, Any],
    *,
    dynamic_labels: Sequence[str],
    oracle_labels: Sequence[str],
) -> dict[str, Any]:
    def distances(pairs):
        return {
            f"{left}__vs__{right}": _array_delta(values[left], values[right])[
                "normalized_l2_delta"
            ]
            for left, right in pairs
        }

    dynamic = distances(itertools.combinations(dynamic_labels, 2))
    oracle = distances(itertools.combinations(oracle_labels, 2))
    cross = distances(itertools.product(dynamic_labels, oracle_labels))
    dynamic_max = max(dynamic.values(), default=0.0)
    oracle_max = max(oracle.values(), default=0.0)
    envelope = max(dynamic_max, oracle_max)
    cross_max = max(cross.values(), default=0.0)
    inside = (
        cross_max <= float(np.nextafter(envelope, math.inf))
        if envelope > 0.0
        else cross_max == 0.0
    )
    return {
        "policy": (
            "every dynamic/oracle cross delta must be no larger than the "
            "maximum within-representation repeat delta"
        ),
        "dynamic_repeat_count": len(dynamic_labels),
        "oracle_repeat_count": len(oracle_labels),
        "dynamic_pair_count": len(dynamic),
        "oracle_pair_count": len(oracle),
        "cross_pair_count": len(cross),
        "dynamic_repeat_max_normalized_l2": dynamic_max,
        "oracle_repeat_max_normalized_l2": oracle_max,
        "repeat_envelope_normalized_l2": envelope,
        "maximum_cross_normalized_l2": cross_max,
        "maximum_cross_over_envelope": (
            cross_max / envelope if envelope > 0.0 else 1.0 if cross_max == 0.0 else math.inf
        ),
        "within_observed_repeat_envelope": inside,
        "dynamic_pairs": dynamic,
        "oracle_pairs": oracle,
        "cross_pairs": cross,
    }


def _numeric_star(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        for token in line.split():
            try:
                values.append(float(token))
            except ValueError:
                continue
    return np.asarray(values, dtype=np.float64)


def _load_output_prefix(prefix: Path) -> dict[str, Any]:
    meta = _load_json(prefix.with_name(prefix.name + "_recovar_meta.json"))
    with mrcfile.open(
        prefix.with_name(prefix.name + "_class001.mrc"), permissive=True
    ) as stream:
        volume = np.array(stream.data, copy=True)
    hybrid = meta.get("halfset_0_profile_summary", {}).get(
        "coarse_gaussian_gemm_hybrid"
    )
    if not isinstance(hybrid, dict):
        raise GateSetupError(f"{prefix} omitted hybrid metadata")
    return {
        "meta": meta,
        "hybrid": hybrid,
        "volume": volume,
        "data_star": _numeric_star(prefix.with_name(prefix.name + "_data.star")),
        "model_star": _numeric_star(prefix.with_name(prefix.name + "_model.star")),
    }


def _load_repeat(root: Path, repeat: str) -> dict[str, Any]:
    summary = _load_json(root / "recovar_profiled" / "profile_summary.json")
    profile = summary.get(repeat)
    if not isinstance(profile, dict):
        raise GateSetupError(f"{root} omitted the {repeat} profile")
    iteration = int(summary.get("profiled_iteration", -1))
    prefix = root / "recovar_profiled" / repeat / f"run_it{iteration:03d}"
    return {
        "summary": summary,
        "profile": profile,
        "wall_s": profile.get("wall_s"),
        **_load_output_prefix(prefix),
    }


def _analyze_loaded_arms(
    arms: dict[str, dict[str, Any]],
    *,
    dynamic_labels: Sequence[str],
    oracle_labels: Sequence[str],
    provenance_exact: bool,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    representation_checks = {}
    for label, arm in arms.items():
        hybrid = arm["hybrid"]
        dynamic = label.startswith("dynamic_")
        expected_representation = (
            "dense_full_direct_dynamic_fallback"
            if dynamic
            else "dense_full_direct_static_capacity"
        )
        expected_fallback = 1 if dynamic else 0
        passed = bool(
            hybrid.get("score_representation_batch_counts")
            == {expected_representation: 1}
            and hybrid.get("fallback_batch_count") == expected_fallback
            and (
                hybrid.get("fallback_reasons") == {"block_capacity_overflow": 1}
                if dynamic
                else hybrid.get("fallback_reasons") == {}
            )
        )
        representation_checks[label] = {
            "pass": passed,
            "score_representation_batch_counts": hybrid.get(
                "score_representation_batch_counts"
            ),
            "fallback_batch_count": hybrid.get("fallback_batch_count"),
            "fallback_reasons": hybrid.get("fallback_reasons"),
        }

    exact = {}
    for field in (*REQUIRED_EXACT_META, *REQUIRED_EXACT_SCALARS):
        try:
            values = {label: arm["meta"][field] for label, arm in arms.items()}
        except KeyError as error:
            raise GateSetupError(f"required exact metadata field is missing: {field}") from error
        exact[field] = _exact_panel(values)
    exact["data_star_numeric_content"] = _exact_panel(
        {label: arm["data_star"] for label, arm in arms.items()}
    )

    atomic = {
        field: _multi_repeat_envelope(
            {
                label: arm["meta"][field]
                for label, arm in arms.items()
                if field in arm["meta"]
            },
            dynamic_labels=dynamic_labels,
            oracle_labels=oracle_labels,
        )
        for field in ATOMIC_META
    }
    incomplete_atomic = {
        field: sorted(set(arms) - set(values))
        for field, values in (
            (
                field,
                {
                    label: arm["meta"][field]
                    for label, arm in arms.items()
                    if field in arm["meta"]
                },
            )
            for field in ATOMIC_META
        )
        if len(values) != len(arms)
    }
    if incomplete_atomic:
        raise GateSetupError(f"required atomic metadata fields are missing: {incomplete_atomic}")
    atomic["class001_volume"] = _multi_repeat_envelope(
        {label: arm["volume"] for label, arm in arms.items()},
        dynamic_labels=dynamic_labels,
        oracle_labels=oracle_labels,
    )
    atomic["model_star_numeric_content"] = _multi_repeat_envelope(
        {label: arm["model_star"] for label, arm in arms.items()},
        dynamic_labels=dynamic_labels,
        oracle_labels=oracle_labels,
    )

    exact_pass = all(row["all_exact"] for row in exact.values())
    atomic_pass = all(
        row["within_observed_repeat_envelope"] for row in atomic.values()
    )
    representation_pass = all(
        row["pass"] for row in representation_checks.values()
    )
    dynamic_walls = np.asarray(
        [arms[label]["wall_s"] for label in dynamic_labels], dtype=np.float64
    )
    oracle_walls = np.asarray(
        [arms[label]["wall_s"] for label in oracle_labels], dtype=np.float64
    )
    dynamic_median = float(np.median(dynamic_walls))
    oracle_median = float(np.median(oracle_walls))
    return {
        "schema": SCHEMA,
        "classification": "diagnostic_schedule_representation_qualification",
        "provenance": provenance,
        "dynamic_repeat_count": len(dynamic_labels),
        "oracle_repeat_count": len(oracle_labels),
        "provenance_exact": provenance_exact,
        "representation_checks": representation_checks,
        "representation_contract_passed": representation_pass,
        "exact_checks": exact,
        "exact_science_contract_passed": exact_pass,
        "atomic_envelopes": atomic,
        "atomic_envelope_contract_passed": atomic_pass,
        "wall_time": {
            "dynamic_values_s": dynamic_walls.tolist(),
            "oracle_values_s": oracle_walls.tolist(),
            "dynamic_median_s": dynamic_median,
            "oracle_median_s": oracle_median,
            "dynamic_speedup_over_oracle": oracle_median / dynamic_median,
            "dynamic_fractional_change_from_oracle": (
                dynamic_median / oracle_median - 1.0
            ),
        },
        "pass": bool(
            provenance_exact and representation_pass and exact_pass and atomic_pass
        ),
    }


def analyze(
    dynamic_roots: Sequence[Path],
    oracle_roots: Sequence[Path],
) -> dict[str, Any]:
    if len(dynamic_roots) != len(oracle_roots) or len(dynamic_roots) < 2:
        raise GateSetupError(
            "the gate requires equal dynamic/oracle root counts and at least two roots each"
        )
    arms: dict[str, dict[str, Any]] = {}
    dynamic_labels = []
    oracle_labels = []
    for backend, roots, labels in (
        ("dynamic", dynamic_roots, dynamic_labels),
        ("oracle", oracle_roots, oracle_labels),
    ):
        for root_index, root in enumerate(roots, 1):
            for repeat in ("cold", "warm"):
                label = f"{backend}_{root_index}_{repeat}"
                arms[label] = _load_repeat(Path(root), repeat)
                labels.append(label)

    first = next(iter(arms.values()))
    checkpoint_sha = first["summary"].get("checkpoint_optimiser_sha256")
    schedule = first["profile"].get("schedule")
    provenance_exact = all(
        arm["summary"].get("checkpoint_optimiser_sha256") == checkpoint_sha
        and arm["profile"].get("schedule") == schedule
        for arm in arms.values()
    )
    return _analyze_loaded_arms(
        arms,
        dynamic_labels=dynamic_labels,
        oracle_labels=oracle_labels,
        provenance_exact=provenance_exact,
        provenance={
            "source": "independent_late_profile_roots",
            "dynamic_roots": [str(Path(root).resolve()) for root in dynamic_roots],
            "oracle_roots": [str(Path(root).resolve()) for root in oracle_roots],
            "checkpoint_optimiser_sha256": checkpoint_sha,
            "schedule": schedule,
        },
    )


def analyze_panel(panel_root: Path) -> dict[str, Any]:
    panel_path = panel_root if panel_root.is_file() else panel_root / "panel.json"
    panel = _load_json(panel_path)
    if panel.get("schema") != PANEL_SCHEMA:
        raise GateSetupError(f"unexpected representation panel schema: {panel.get('schema')!r}")
    if tuple(panel.get("arm_order", ())) != PANEL_ARM_ORDER:
        raise GateSetupError("representation panel does not use the mirrored ABBA+BAAB order")
    raw_arms = panel.get("arms")
    if not isinstance(raw_arms, dict) or set(raw_arms) != set(PANEL_ARM_ORDER):
        raise GateSetupError("representation panel arms are incomplete")

    arms: dict[str, dict[str, Any]] = {}
    schedules = {}
    schedule_keys = (
        "current_size",
        "healpix_order",
        "n_rotations",
        "n_translations",
        "subset_size",
        "random_perturbation",
    )
    capacity_contract = True
    for label in PANEL_ARM_ORDER:
        record = raw_arms[label]
        if not isinstance(record, dict):
            raise GateSetupError(f"representation panel arm {label!r} is invalid")
        output_prefix = Path(record.get("output_prefix", ""))
        if not output_prefix.is_absolute():
            raise GateSetupError(f"representation panel arm {label!r} has no absolute prefix")
        arms[label] = {
            "profile": record,
            "wall_s": record.get("wall_s"),
            **_load_output_prefix(output_prefix),
        }
        try:
            schedules[label] = {
                key: arms[label]["meta"][key] for key in schedule_keys
            }
        except KeyError as error:
            raise GateSetupError(
                f"representation panel arm {label!r} omitted schedule field {error.args[0]}"
            ) from error
        expected_capacity = 64 if label.startswith("dynamic_") else 288
        capacity_contract &= record.get("block_capacity") == expected_capacity

    first_schedule = schedules[PANEL_ARM_ORDER[0]]
    checkpoint_sha = panel.get("checkpoint_optimiser_sha256")
    environment = panel.get("environment_without_block_capacity")
    provenance_exact = bool(
        isinstance(checkpoint_sha, str)
        and len(checkpoint_sha) == 64
        and isinstance(environment, dict)
        and environment
        and "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY"
        not in environment
        and capacity_contract
        and all(schedule == first_schedule for schedule in schedules.values())
    )
    return _analyze_loaded_arms(
        arms,
        dynamic_labels=tuple(
            label for label in PANEL_ARM_ORDER if label.startswith("dynamic_")
        ),
        oracle_labels=tuple(
            label for label in PANEL_ARM_ORDER if label.startswith("oracle_")
        ),
        provenance_exact=provenance_exact,
        provenance={
            "source": "single_process_prewarmed_abba_baab_panel",
            "panel_path": str(panel_path.resolve()),
            "checkpoint_optimiser_sha256": checkpoint_sha,
            "input_star_sha256": panel.get("input_star_sha256"),
            "schedule": first_schedule,
            "capacity_contract_exact": capacity_contract,
            "environment_without_block_capacity": environment,
        },
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-root", action="append", type=Path)
    parser.add_argument("--oracle-root", action="append", type=Path)
    parser.add_argument("--panel-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.panel_root is not None:
        if args.dynamic_root or args.oracle_root:
            raise GateSetupError(
                "--panel-root cannot be combined with independent profile roots"
            )
        report = analyze_panel(args.panel_root)
    else:
        if not args.dynamic_root or not args.oracle_root:
            raise GateSetupError(
                "supply --panel-root or both --dynamic-root and --oracle-root"
            )
        report = analyze(args.dynamic_root, args.oracle_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
