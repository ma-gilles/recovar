#!/usr/bin/env python3
"""Run one native RELION repeat against an already completed VDAM candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.audit_vdam_fsc_trajectory import audit
    from scripts.materialize_em_k1_fixture import sha256_file
    from scripts.run_vdam_relion_parity_case import (
        DEFAULT_RELION_REFINE,
        RunError,
        _assert_git_source_unchanged,
        _definition_with_iteration_override,
        _git_source_state,
        _load_json,
        _physical_gpu_uuid,
        _qualification_cuda_environment,
        _relion_reference_provenance,
        _run_logged,
        _scorecard_case,
        _write_particle_state_audit,
        build_relion_command,
    )
else:
    from audit_vdam_fsc_trajectory import audit
    from materialize_em_k1_fixture import sha256_file
    from run_vdam_relion_parity_case import (
        DEFAULT_RELION_REFINE,
        RunError,
        _assert_git_source_unchanged,
        _definition_with_iteration_override,
        _git_source_state,
        _load_json,
        _physical_gpu_uuid,
        _qualification_cuda_environment,
        _relion_reference_provenance,
        _run_logged,
        _scorecard_case,
        _write_particle_state_audit,
        build_relion_command,
    )


SCHEMA = "recovar.vdam_native_only_repeat.v1"


def _validate_candidate(
    candidate_root: Path,
    *,
    case_id: str,
    scorecard_path: Path,
) -> tuple[Path, Path, dict[str, Any], str]:
    candidate_root = candidate_root.resolve()
    fixture_dir = candidate_root / "data"
    recovar_dir = candidate_root / "recovar"
    provenance = _load_json(candidate_root / "run_provenance.json")
    gpu = _load_json(candidate_root / "paired_gpu_uuid.json")
    if provenance.get("case_id") != case_id:
        raise RunError("completed candidate case identity differs")
    if provenance.get("scorecard_sha256") != sha256_file(scorecard_path):
        raise RunError("completed candidate scorecard identity differs")
    physical = str(gpu.get("physical_gpu_uuid", ""))
    if {
        physical,
        str(gpu.get("relion_gpu_uuid", "")),
        str(gpu.get("recovar_gpu_uuid", "")),
    } != {physical} or not physical.startswith("GPU-"):
        raise RunError("completed candidate did not preserve one physical GPU")
    required = (
        fixture_dir / "fixture_materialization.json",
        recovar_dir / "run_native_options.json",
        candidate_root / "trajectory_audit.json",
        candidate_root / "particle_state_trajectory_audit.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RunError(f"completed candidate evidence is incomplete: {missing}")
    return fixture_dir, recovar_dir, provenance, physical


def run_native_repeat(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    source_head, tracked_dirty = _git_source_state(repo)
    if tracked_dirty:
        raise RunError("native-repeat evidence requires a clean tracked worktree")
    if os.environ.get("VDAM_NR_ITER_OVERRIDE") is not None:
        raise RunError("native-only repeat panels do not permit iteration overrides")
    scorecard_path = args.scorecard.resolve()
    scorecard = _load_json(scorecard_path)
    case = _scorecard_case(scorecard, args.case_id)
    definition = _definition_with_iteration_override(case["definition"])
    fixture_dir, recovar_dir, candidate_provenance, expected_gpu_uuid = (
        _validate_candidate(
            args.candidate_root,
            case_id=args.case_id,
            scorecard_path=scorecard_path,
        )
    )

    case_root = (args.output_root / args.case_id).resolve()
    if case_root.exists() and any(case_root.iterdir()):
        raise RunError(f"refusing to reuse non-empty repeat directory: {case_root}")
    relion_dir = case_root / "relion"
    relion_dir.mkdir(parents=True, exist_ok=True)
    (case_root / "SAFE_TO_DELETE").touch()
    input_name = (
        "particles_relion_identity_ctf.star"
        if (fixture_dir / "particles_relion_identity_ctf.star").is_file()
        else "particles.star"
    )
    relion_argv = build_relion_command(
        input_star=fixture_dir / input_name,
        output_prefix=relion_dir / "run",
        definition=definition,
        relion_refine=args.relion_refine,
        threads=args.threads,
    )
    (relion_dir / "relion_command.json").write_text(
        json.dumps({"argv": relion_argv}, indent=2, sort_keys=True) + "\n"
    )
    provenance = {
        "schema": SCHEMA,
        "run_mode": "native_only_repeat_against_completed_candidate",
        "scorecard": str(scorecard_path),
        "scorecard_sha256": sha256_file(scorecard_path),
        "case_id": args.case_id,
        "source_em_case_id": definition["source_em_case_id"],
        "repo": str(repo),
        "git_head": source_head,
        "tracked_worktree_clean_at_launch": True,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "relion_reference": _relion_reference_provenance(args.relion_refine),
        "candidate": {
            "root": str(args.candidate_root.resolve()),
            "git_head": candidate_provenance.get("git_head"),
            "physical_gpu_uuid": expected_gpu_uuid,
        },
    }
    (case_root / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    current_gpu_uuid = _physical_gpu_uuid()
    if current_gpu_uuid != expected_gpu_uuid:
        raise RunError(
            "native repeat was not allocated on the completed candidate GPU: "
            f"expected {expected_gpu_uuid}, got {current_gpu_uuid}"
        )
    env = _qualification_cuda_environment(
        dict(os.environ),
        deterministic_cuda=False,
    )
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
    timing = _run_logged(
        relion_argv,
        cwd=fixture_dir,
        log_path=relion_dir / "relion.log",
        env=env,
    )
    completed_gpu_uuid = _physical_gpu_uuid()
    if completed_gpu_uuid != expected_gpu_uuid:
        raise RunError("physical GPU changed during native RELION repeat")
    _assert_git_source_unchanged(repo, source_head)
    provenance.update(
        git_head_at_completion=source_head,
        tracked_worktree_clean_at_completion=True,
        relion_wall_s=float(timing["external_wall_s"]),
    )
    (case_root / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    paired_gpu_report = case_root / "paired_gpu_uuid.json"
    paired_gpu_report.write_text(
        json.dumps(
            {
                "physical_gpu_uuid": expected_gpu_uuid,
                "relion_gpu_uuid": completed_gpu_uuid,
                "recovar_gpu_uuid": expected_gpu_uuid,
                "recovar_source": str(args.candidate_root.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    report, shellwise = audit(
        scorecard_path=scorecard_path,
        case_id=args.case_id,
        fixture_dir=fixture_dir,
        recovar_dir=recovar_dir,
        relion_dir=relion_dir,
        paired_gpu_report_path=paired_gpu_report,
    )
    report["particle_state_trajectory"] = _write_particle_state_audit(
        recovar_dir=recovar_dir,
        relion_dir=relion_dir,
        case_root=case_root,
        nr_iter=int(definition["nr_iter"]),
    )
    report_path = case_root / "trajectory_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(case_root / "trajectory_shellwise_fsc.npz", **shellwise)
    (case_root / "NATIVE_SCIENCE_COMPLETED").touch()
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "strict_point_reference_result": report["result"],
        "case_id": args.case_id,
        "candidate_root": str(args.candidate_root.resolve()),
        "output_root": str(case_root),
        "physical_gpu_uuid": expected_gpu_uuid,
        "relion_wall_s": float(timing["external_wall_s"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--relion-refine", type=Path, default=DEFAULT_RELION_REFINE)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.relion_refine.is_file() or not os.access(args.relion_refine, os.X_OK):
        raise RunError(f"RELION executable is unavailable: {args.relion_refine}")
    run_native_repeat(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
