#!/usr/bin/env python3
"""Analyze a same-GPU default-versus-HIGHEST BPref production A/B."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mrcfile
import numpy as np

from scripts.analyze_relion_bpref_factor_inertness import _array_metrics, _map_metrics, _sha256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_mrc(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=False) as handle:
        return np.asarray(handle.data, dtype=np.float32).copy()


def _factor_comparisons(control_path: Path, fixed_path: Path) -> dict[str, object]:
    with np.load(control_path, allow_pickle=False) as archive:
        control = {name: archive[name] for name in archive.files}
    with np.load(fixed_path, allow_pickle=False) as archive:
        fixed = {name: archive[name] for name in archive.files}
    _require(control.keys() == fixed.keys(), "factor key set changed between arms")
    changed = [name for name in control if not np.array_equal(control[name], fixed[name])]
    _require(changed == ["numerator_f32"], f"unexpected factor changes: {changed}")
    result = {
        "changed_arrays": changed,
        "control_default_vs_fixed_production": _array_metrics(control["numerator_f32"], fixed["numerator_f32"]),
        "fixed_production_vs_control_highest": _array_metrics(fixed["numerator_f32"], control["numerator_highest_f32"]),
        "fixed_production_vs_control_sequential": _array_metrics(
            fixed["numerator_f32"], control["numerator_sequential_f32"]
        ),
        "fixed_production_vs_fixed_highest": _array_metrics(fixed["numerator_f32"], fixed["numerator_highest_f32"]),
    }
    result["fixed_matches_highest_exactly"] = bool(
        result["fixed_production_vs_control_highest"]["exact_equal"]
        and result["fixed_production_vs_fixed_highest"]["exact_equal"]
    )
    return result


def _timing(root: Path) -> dict[str, object]:
    wall = json.loads((root / "provenance/walltime.json").read_text())
    factor_wall = json.loads((root / "factors/walltime.json").read_text())
    logs = sorted((root / "logs").glob("recovar_*.log"))
    _require(len(logs) == 1, f"expected one production log under {root}")
    matches = re.findall(r"Sparse pass-2 \(bucketed\): .*?, ([0-9.]+)s E\+M", logs[0].read_text())
    _require(len(matches) == 2, f"expected two half-set pass-2 timings under {root}")
    return {
        "gpu_uuid": wall["gpu_uuid"],
        "production_wall_s": int(wall["wall_s"]),
        "factor_extraction_wall_s": int(factor_wall["wall_s"]),
        "pass2_half_wall_s": [float(value) for value in matches],
        "pass2_median_wall_s": float(np.median([float(value) for value in matches])),
    }


def analyze(control: Path, fixed: Path, relion: Path, *, expected_gpu_uuid: str) -> dict[str, object]:
    timing = {"control": _timing(control), "fixed": _timing(fixed)}
    _require(
        {timing[arm]["gpu_uuid"] for arm in timing} == {expected_gpu_uuid},
        "A/B arms did not run on the required physical GPU",
    )
    factors = _factor_comparisons(control / "factors/operands.npz", fixed / "factors/operands.npz")
    _require(
        bool(factors["fixed_matches_highest_exactly"]), "fixed production numerator does not equal HIGHEST control"
    )

    aggregates = {}
    hashes = {}
    for half in (0, 1):
        for field in ("Ft_y", "Ft_ctf"):
            paths = {
                "control": control / "intermediates" / f"it000_{field}_{half}.npy",
                "fixed": fixed / "intermediates" / f"it000_{field}_{half}.npy",
            }
            aggregates[f"half{half + 1}_{field}"] = _array_metrics(
                np.load(paths["control"], allow_pickle=False),
                np.load(paths["fixed"], allow_pickle=False),
            )
            hashes.update({str(path.resolve()): _sha256(path) for path in paths.values()})

    maps = {
        arm: {
            "half1": _load_mrc(root / "recovar/final_half1.mrc"),
            "half2": _load_mrc(root / "recovar/final_half2.mrc"),
            "merged": _load_mrc(root / "recovar/final_merged.mrc"),
        }
        for arm, root in (("control", control), ("fixed", fixed))
    }
    relion_maps = {
        "half1": _load_mrc(relion / "run_it001_half1_class001.mrc"),
        "half2": _load_mrc(relion / "run_it001_half2_class001.mrc"),
    }
    relion_maps["merged"] = (0.5 * (relion_maps["half1"] + relion_maps["half2"])).astype(np.float32)
    map_fsc = {
        "control_vs_fixed": {
            name: _map_metrics(maps["control"][name], maps["fixed"][name]) for name in ("half1", "half2", "merged")
        },
        "control_vs_relion_signed_telemetry": {
            name: _map_metrics(maps["control"][name], relion_maps[name]) for name in ("half1", "half2", "merged")
        },
        "fixed_vs_relion_signed_telemetry": {
            name: _map_metrics(maps["fixed"][name], relion_maps[name]) for name in ("half1", "half2", "merged")
        },
        "control_vs_relion_sign_aligned": {
            name: _map_metrics(-maps["control"][name], relion_maps[name]) for name in ("half1", "half2", "merged")
        },
        "fixed_vs_relion_sign_aligned": {
            name: _map_metrics(-maps["fixed"][name], relion_maps[name]) for name in ("half1", "half2", "merged")
        },
        "internal_halfmap": {arm: _map_metrics(values["half1"], values["half2"]) for arm, values in maps.items()},
    }
    map_fsc["fixed_minus_control_relion_fsc_auc"] = {
        name: float(
            map_fsc["fixed_vs_relion_sign_aligned"][name]["fsc_auc_non_dc"]
            - map_fsc["control_vs_relion_sign_aligned"][name]["fsc_auc_non_dc"]
        )
        for name in ("half1", "half2", "merged")
    }
    for root in (control, fixed):
        for name in ("final_half1.mrc", "final_half2.mrc", "final_merged.mrc"):
            path = root / "recovar" / name
            hashes[str(path.resolve())] = _sha256(path)
    for name in ("run_it001_half1_class001.mrc", "run_it001_half2_class001.mrc"):
        path = relion / name
        hashes[str(path.resolve())] = _sha256(path)

    timing["fixed_minus_control"] = {
        "production_wall_s": int(timing["fixed"]["production_wall_s"] - timing["control"]["production_wall_s"]),
        "factor_extraction_wall_s": int(
            timing["fixed"]["factor_extraction_wall_s"] - timing["control"]["factor_extraction_wall_s"]
        ),
        "pass2_median_wall_s": float(timing["fixed"]["pass2_median_wall_s"] - timing["control"]["pass2_median_wall_s"]),
    }
    return {
        "schema": "recovar-bpref-reduction-precision-ab-v1",
        "metric_policy": "exact/array metrics for intermediates; FSC/FSC-AUC only for maps; no correlation",
        "relion_map_sign_alignment": {
            "recovar_multiplier": -1,
            "signed_fsc_retained_as_telemetry": True,
            "basis": "explicit RECOVAR-versus-RELION CTF/BPref sign convention established by the sealed factor comparison",
        },
        "control_root": str(control.resolve()),
        "fixed_root": str(fixed.resolve()),
        "relion_root": str(relion.resolve()),
        "same_physical_gpu": True,
        "gpu_uuid": expected_gpu_uuid,
        "factor_array_comparisons": factors,
        "aggregate_array_comparisons": aggregates,
        "map_fsc_comparisons": map_fsc,
        "timing": timing,
        "artifact_sha256": hashes,
        "precision_fix_confirmed": True,
        "absolute_real_data_relion_parity_confirmed": False,
        "absolute_parity_scope_note": "This one-iteration A/B isolates reduction precision; it does not replace the full real-data trajectory gate.",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("fixed", type=Path)
    parser.add_argument("--relion", required=True, type=Path)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite A/B artifact: {args.output_json}")
    report = analyze(args.control, args.fixed, args.relion, expected_gpu_uuid=args.expected_gpu_uuid)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
