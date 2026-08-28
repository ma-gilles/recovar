#!/usr/bin/env python3
"""Audit one VDAM candidate against a frozen native-RELION repeat envelope."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.summarize_em_completion_bench import _load_relion_volume, normalized_fsc_auc, shell_fsc
else:
    from summarize_em_completion_bench import _load_relion_volume, normalized_fsc_auc, shell_fsc


SCHEMA = "recovar.vdam_candidate_native_envelope.v1"
TRAJECTORY_SCHEMA = "recovar.vdam_relion_fsc_trajectory_audit.v1"
SUITE_SCHEMA = "recovar.vdam_relion_parity_suite.v1"
ARTIFACT_SUFFIXES = ("class001.mrc", "model.star", "data.star")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]+")


class CandidateEnvelopeError(RuntimeError):
    """Raised when candidate-envelope evidence is incomplete or mixed."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateEnvelopeError(f"cannot read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CandidateEnvelopeError(f"{label} must contain a JSON object: {path}")
    return value


def classify_candidate_checkpoint(
    *,
    candidate_native_fsc_auc: list[float],
    candidate_gt_fsc_auc: float,
    native_gt_fsc_auc: list[float],
    cross_engine_min: float,
    gt_delta_min: float,
) -> dict[str, Any]:
    """Require a native mode match and quality versus the best native repeat."""

    cross = np.asarray(candidate_native_fsc_auc, dtype=np.float64)
    native_gt = np.asarray(native_gt_fsc_auc, dtype=np.float64)
    if cross.ndim != 1 or cross.size < 2 or not np.all(np.isfinite(cross)):
        raise CandidateEnvelopeError("classification requires at least two finite native mode comparisons")
    if native_gt.shape != cross.shape or not np.all(np.isfinite(native_gt)):
        raise CandidateEnvelopeError("native GT metrics must align with the native mode comparisons")
    if not np.isfinite(candidate_gt_fsc_auc):
        raise CandidateEnvelopeError("candidate GT metric must be finite")
    best_native_match = float(np.max(cross))
    best_native_gt = float(np.max(native_gt))
    conservative_gt_delta = float(candidate_gt_fsc_auc - best_native_gt)
    checks = {
        "candidate_matches_a_native_mode_at_frozen_gate": best_native_match >= float(cross_engine_min),
        "candidate_meets_gt_nondegradation_vs_best_native": conservative_gt_delta
        >= float(gt_delta_min),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "candidate_best_native_fsc_auc": best_native_match,
        "candidate_worst_native_fsc_auc": float(np.min(cross)),
        "candidate_gt_fsc_auc": float(candidate_gt_fsc_auc),
        "best_native_gt_fsc_auc": best_native_gt,
        "candidate_minus_best_native_gt_fsc_auc": conservative_gt_delta,
    }


def _candidate_provenance(candidate_root: Path) -> dict[str, str]:
    run_provenance_path = candidate_root / "run_provenance.json"
    paired_gpu_path = candidate_root / "paired_gpu_uuid.json"
    if run_provenance_path.exists() or paired_gpu_path.exists():
        run_provenance = _load_json(
            run_provenance_path, label="candidate run provenance"
        )
        paired_gpu = _load_json(paired_gpu_path, label="candidate paired GPU report")
        source_head = str(run_provenance.get("git_head", ""))
        cuda_digest = str(
            run_provenance.get("recovar_native_extensions", {})
            .get("cuda_backproject", {})
            .get("sha256", "")
        )
        physical_gpu = str(paired_gpu.get("physical_gpu_uuid", ""))
        if _HEX40.fullmatch(source_head) is None:
            raise CandidateEnvelopeError(f"candidate source head is invalid: {source_head!r}")
        if _HEX64.fullmatch(cuda_digest) is None:
            raise CandidateEnvelopeError("candidate CUDA library digest is invalid")
        if {
            physical_gpu,
            str(paired_gpu.get("relion_gpu_uuid", "")),
            str(paired_gpu.get("recovar_gpu_uuid", "")),
        } != {physical_gpu} or not physical_gpu.startswith("GPU-"):
            raise CandidateEnvelopeError("candidate paired GPU identity is incomplete or mixed")
        return {
            "source_head": source_head,
            "cuda_library_sha256": cuda_digest,
            "physical_gpu_uuid": physical_gpu,
            "source_format": "paired_run_provenance.v1",
        }

    provenance = candidate_root / "provenance"
    try:
        source_head = (provenance / "repo_head.txt").read_text().strip()
        input_lines = (provenance / "input_sha256.txt").read_text().splitlines()
        nvidia_smi = (provenance / "nvidia_smi.txt").read_text()
    except OSError as exc:
        raise CandidateEnvelopeError(f"candidate provenance is incomplete: {exc}") from exc
    if _HEX40.fullmatch(source_head) is None:
        raise CandidateEnvelopeError(f"candidate source head is invalid: {source_head!r}")
    cuda_rows = [line.split()[0] for line in input_lines if "libcuda_backproject.so" in line]
    if len(cuda_rows) != 1 or _HEX64.fullmatch(cuda_rows[0]) is None:
        raise CandidateEnvelopeError("candidate provenance must contain exactly one CUDA library digest")
    gpu_uuids = sorted(set(_GPU_UUID.findall(nvidia_smi)))
    if len(gpu_uuids) != 1:
        raise CandidateEnvelopeError(f"candidate GPU identity is ambiguous: {gpu_uuids}")
    return {
        "source_head": source_head,
        "cuda_library_sha256": cuda_rows[0],
        "physical_gpu_uuid": gpu_uuids[0],
        "source_format": "legacy_provenance_directory.v1",
    }


def _native_panel_provenance(
    native_roots: list[Path], *, suite_id: str, case_id: str, checkpoints: tuple[int, ...]
) -> dict[str, Any]:
    if len(native_roots) < 2:
        raise CandidateEnvelopeError("native envelope requires at least two complete repeats")
    source_heads: set[str] = set()
    executable_hashes: set[str] = set()
    gpu_uuids: set[str] = set()
    for index, root in enumerate(native_roots, start=1):
        trajectory = _load_json(root / "trajectory_audit.json", label=f"native repeat {index} audit")
        provenance = _load_json(root / "run_provenance.json", label=f"native repeat {index} provenance")
        gpu = _load_json(root / "paired_gpu_uuid.json", label=f"native repeat {index} GPU report")
        if trajectory.get("schema") != TRAJECTORY_SCHEMA:
            raise CandidateEnvelopeError(f"native repeat {index} trajectory schema differs")
        if trajectory.get("suite_id") != suite_id or trajectory.get("case_id") != case_id:
            raise CandidateEnvelopeError(f"native repeat {index} suite or case identity differs")
        observed = tuple(int(row["iteration"]) for row in trajectory.get("checkpoints", ()))
        if observed != checkpoints or not bool(trajectory.get("artifact_topology_exact")):
            raise CandidateEnvelopeError(f"native repeat {index} checkpoint topology differs")
        source_heads.add(str(provenance.get("git_head", "")))
        executable_hashes.add(str(provenance.get("relion_reference", {}).get("executable_sha256", "")))
        physical = str(gpu.get("physical_gpu_uuid", ""))
        if {
            physical,
            str(gpu.get("relion_gpu_uuid", "")),
            str(gpu.get("recovar_gpu_uuid", "")),
        } != {physical} or not physical.startswith("GPU-"):
            raise CandidateEnvelopeError(f"native repeat {index} paired GPU identity differs")
        gpu_uuids.add(physical)
    if len(source_heads) != 1 or any(_HEX40.fullmatch(value) is None for value in source_heads):
        raise CandidateEnvelopeError(f"native panel contains mixed or invalid source heads: {source_heads}")
    if len(executable_hashes) != 1 or any(_HEX64.fullmatch(value) is None for value in executable_hashes):
        raise CandidateEnvelopeError("native panel contains mixed or invalid RELION executable hashes")
    if len(gpu_uuids) != 1:
        raise CandidateEnvelopeError(f"native panel spans physical GPUs: {gpu_uuids}")
    return {
        "repeat_count": len(native_roots),
        "source_head": next(iter(source_heads)),
        "relion_executable_sha256": next(iter(executable_hashes)),
        "physical_gpu_uuid": next(iter(gpu_uuids)),
    }


def require_same_physical_gpu(
    candidate_provenance: dict[str, str], native_provenance: dict[str, Any]
) -> None:
    """Reject a candidate/native envelope assembled across physical GPUs."""

    candidate_gpu = str(candidate_provenance.get("physical_gpu_uuid", ""))
    native_gpu = str(native_provenance.get("physical_gpu_uuid", ""))
    if not candidate_gpu or candidate_gpu != native_gpu:
        raise CandidateEnvelopeError(
            f"candidate/native physical GPU differs: {candidate_gpu!r} vs {native_gpu!r}"
        )


def _metric(lhs: np.ndarray, rhs: np.ndarray, *, key: str, shellwise: dict[str, np.ndarray]) -> float:
    curve = np.asarray(shell_fsc(lhs, rhs), dtype=np.float64)
    if curve.size <= 1 or not np.any(np.isfinite(curve[1:])):
        raise CandidateEnvelopeError(f"{key} produced no finite non-DC FSC shells")
    shellwise[key] = curve
    value = float(normalized_fsc_auc(curve))
    if not np.isfinite(value):
        raise CandidateEnvelopeError(f"{key} produced a non-finite FSC-AUC")
    return value


def _map_path(root: Path, engine: str, iteration: int) -> Path:
    return root / engine / f"run_it{iteration:03d}_class001.mrc"


def _require_artifacts(root: Path, engine: str, checkpoints: tuple[int, ...], *, label: str) -> None:
    missing = [
        root / engine / f"run_it{iteration:03d}_{suffix}"
        for iteration in checkpoints
        for suffix in ARTIFACT_SUFFIXES
        if not (root / engine / f"run_it{iteration:03d}_{suffix}").is_file()
    ]
    if missing:
        raise CandidateEnvelopeError(f"{label} is missing {len(missing)} artifacts; first: {missing[0]}")


def audit_candidate_envelope(
    *,
    scorecard_path: Path,
    case_id: str,
    candidate_root: Path,
    native_roots: list[Path],
    fixture_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    scorecard = _load_json(scorecard_path, label="scorecard")
    if scorecard.get("schema") != SUITE_SCHEMA:
        raise CandidateEnvelopeError(f"unsupported scorecard schema: {scorecard.get('schema')!r}")
    matches = [row for row in scorecard.get("cases", ()) if row.get("id") == case_id]
    if len(matches) != 1:
        raise CandidateEnvelopeError(f"expected one scorecard row for {case_id}, found {len(matches)}")
    case = matches[0]
    acceptance = scorecard["acceptance_contract"]
    checkpoints = tuple(int(value) for value in acceptance["required_checkpoints"])
    if checkpoints != tuple(sorted(set(checkpoints))) or checkpoints[0] != 0:
        raise CandidateEnvelopeError("scorecard checkpoints must be sorted, unique, and start at zero")
    if checkpoints[-1] != int(case["definition"]["nr_iter"]):
        raise CandidateEnvelopeError("scorecard terminal checkpoint differs from nr_iter")

    materialization = _load_json(fixture_dir / "fixture_materialization.json", label="fixture materialization")
    source_manifest = scorecard["source_fixture_manifest"]
    if (
        materialization.get("case_id") != case["definition"]["source_em_case_id"]
        or materialization.get("manifest_sha256") != source_manifest["sha256"]
    ):
        raise CandidateEnvelopeError("fixture identity differs from the frozen scorecard")
    gt_path = fixture_dir / "reference_gt_relion.mrc"
    if not gt_path.is_file():
        raise CandidateEnvelopeError(f"frozen GT map is missing: {gt_path}")

    candidate_provenance = _candidate_provenance(candidate_root)
    native_provenance = _native_panel_provenance(
        native_roots, suite_id=str(scorecard["suite_id"]), case_id=case_id, checkpoints=checkpoints
    )
    require_same_physical_gpu(candidate_provenance, native_provenance)
    _require_artifacts(candidate_root, "recovar", checkpoints, label="candidate")
    for index, root in enumerate(native_roots, start=1):
        _require_artifacts(root, "relion", checkpoints, label=f"native repeat {index}")

    gt = _load_relion_volume(gt_path)
    cross_min = float(acceptance["cross_engine_fsc_auc_min"])
    gt_min = float(acceptance["recovar_minus_relion_gt_fsc_auc_min"])
    shellwise: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for iteration in checkpoints:
        candidate = _load_relion_volume(_map_path(candidate_root, "recovar", iteration))
        native = [_load_relion_volume(_map_path(root, "relion", iteration)) for root in native_roots]
        if candidate.shape != gt.shape or any(volume.shape != gt.shape for volume in native):
            raise CandidateEnvelopeError(f"iteration {iteration} map shapes differ")
        cross = [
            _metric(
                candidate,
                volume,
                key=f"it{iteration:03d}_candidate_native{index:02d}",
                shellwise=shellwise,
            )
            for index, volume in enumerate(native, start=1)
        ]
        candidate_gt = _metric(
            candidate, gt, key=f"it{iteration:03d}_candidate_gt", shellwise=shellwise
        )
        native_gt = [
            _metric(
                volume,
                gt,
                key=f"it{iteration:03d}_native{index:02d}_gt",
                shellwise=shellwise,
            )
            for index, volume in enumerate(native, start=1)
        ]
        native_self = [
            _metric(
                native[lhs],
                native[rhs],
                key=f"it{iteration:03d}_native{lhs + 1:02d}_native{rhs + 1:02d}",
                shellwise=shellwise,
            )
            for lhs, rhs in itertools.combinations(range(len(native)), 2)
        ]
        classification = classify_candidate_checkpoint(
            candidate_native_fsc_auc=cross,
            candidate_gt_fsc_auc=candidate_gt,
            native_gt_fsc_auc=native_gt,
            cross_engine_min=cross_min,
            gt_delta_min=gt_min,
        )
        rows.append(
            {
                "iteration": iteration,
                "candidate_native_fsc_auc": cross,
                "native_gt_fsc_auc": native_gt,
                "native_repeat_floor_fsc_auc": float(min(native_self)),
                **classification,
            }
        )

    report = {
        "schema": SCHEMA,
        "suite_id": scorecard["suite_id"],
        "case_id": case_id,
        "case_name": case.get("name"),
        "result": "pass" if all(row["pass"] for row in rows) else "fail",
        "scope": "map native-mode coverage and conservative GT nondegradation; particle and schedule envelopes are separate gates",
        "thresholds": {
            "cross_engine_fsc_auc_min": cross_min,
            "recovar_minus_relion_gt_fsc_auc_min": gt_min,
        },
        "candidate_provenance": candidate_provenance,
        "native_panel_provenance": native_provenance,
        "fixture": {
            "source_em_case_id": case["definition"]["source_em_case_id"],
            "manifest_sha256": source_manifest["sha256"],
        },
        "minimum_candidate_best_native_fsc_auc": min(
            row["candidate_best_native_fsc_auc"] for row in rows
        ),
        "minimum_candidate_minus_best_native_gt_fsc_auc": min(
            row["candidate_minus_best_native_gt_fsc_auc"] for row in rows
        ),
        "checkpoints": rows,
    }
    return report, shellwise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, action="append", required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-shells-npz", type=Path, required=True)
    args = parser.parse_args(argv)
    report, shellwise = audit_candidate_envelope(
        scorecard_path=args.scorecard.resolve(),
        case_id=args.case_id,
        candidate_root=args.candidate_root.resolve(),
        native_roots=[path.resolve() for path in args.native_root],
        fixture_dir=args.fixture_dir.resolve(),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_shells_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_shells_npz, **shellwise)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
