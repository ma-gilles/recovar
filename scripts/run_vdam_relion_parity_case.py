#!/usr/bin/env python3
"""Run one frozen VDAM/RELION InitialModel parity case on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.audit_vdam_fsc_trajectory import audit
    from scripts.materialize_em_k1_fixture import load_case, materialize, sha256_file
else:
    from audit_vdam_fsc_trajectory import audit
    from materialize_em_k1_fixture import load_case, materialize, sha256_file

DEFAULT_FIXTURE_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/em_work/codex")
DEFAULT_RELION_REFINE = Path("/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine")


class RunError(RuntimeError):
    """Raised when case execution or provenance validation fails."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RunError(f"expected JSON object in {path}")
    return value


def _scorecard_case(scorecard: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = [row for row in scorecard.get("cases", []) if row.get("id") == case_id]
    if len(matches) != 1:
        raise RunError(f"expected exactly one scorecard row for {case_id}, found {len(matches)}")
    return matches[0]


def _physical_gpu_uuid() -> str:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(values) != 1 or not values[0].startswith("GPU-"):
        raise RunError(f"expected exactly one visible physical GPU UUID, found {values}")
    slurm_gpu = os.environ.get("SLURM_JOB_GPUS", "").split(",", 1)[0]
    if slurm_gpu.startswith("GPU-") and slurm_gpu != values[0]:
        raise RunError(f"Slurm GPU UUID {slurm_gpu} differs from visible UUID {values[0]}")
    return values[0]


def build_relion_command(
    *, input_star: Path, output_prefix: Path, definition: dict[str, Any], relion_refine: Path, threads: int
) -> list[str]:
    return [
        str(relion_refine),
        "--o",
        str(output_prefix),
        "--iter",
        str(definition["nr_iter"]),
        "--grad",
        "--denovo_3dref",
        "--grad_write_iter",
        "1",
        "--i",
        str(input_star),
        "--ctf",
        "--K",
        str(definition["nr_classes"]),
        "--sym",
        "C1",
        "--flatten_solvent",
        "--zero_mask",
        "--dont_combine_weights_via_disc",
        "--pool",
        "3",
        "--pad",
        str(definition["padding_factor"]),
        "--particle_diameter",
        "200",
        "--oversampling",
        str(definition["oversampling"]),
        "--healpix_order",
        str(definition["healpix_order"]),
        "--offset_range",
        str(definition["offset_range_px"]),
        "--offset_step",
        str(definition["offset_step_px"]),
        "--auto_sampling",
        "--tau2_fudge",
        str(definition["tau2_fudge"]),
        "--random_seed",
        str(definition["random_seed"]),
        "--j",
        str(threads),
        "--gpu",
        "0",
    ]


def build_recovar_command(
    *, input_star: Path, output_prefix: Path, fixture_dir: Path, definition: dict[str, Any]
) -> list[str]:
    return [
        sys.executable,
        "scripts/run_ab_initio.py",
        "--i",
        str(input_star),
        "--o",
        str(output_prefix),
        "--nr_iter",
        str(definition["nr_iter"]),
        "--K",
        str(definition["nr_classes"]),
        "--tau2_fudge",
        str(definition["tau2_fudge"]),
        "--sym",
        "C1",
        "--particle_diameter",
        "200",
        "--random_seed",
        str(definition["random_seed"]),
        "--healpix_order",
        str(definition["healpix_order"]),
        "--oversampling",
        str(definition["oversampling"]),
        "--offset_range",
        str(definition["offset_range_px"]),
        "--offset_step",
        str(definition["offset_step_px"]),
        "--padding_factor",
        str(definition["padding_factor"]),
        "--datadir",
        str(fixture_dir),
        "--gpu",
        "0",
        "--require_custom_cuda",
    ]


def _run_logged(argv: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]) -> dict[str, Any]:
    start = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    end = time.time()
    timing = {
        "argv": argv,
        "cwd": str(cwd),
        "start_epoch": start,
        "end_epoch": end,
        "external_wall_s": end - start,
        "exit_status": proc.returncode,
    }
    log_path.with_suffix(".timing.json").write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n")
    if proc.returncode:
        raise RunError(f"command failed with status {proc.returncode}; see {log_path}")
    return timing


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    scorecard_path = args.scorecard.resolve()
    scorecard = _load_json(scorecard_path)
    case = _scorecard_case(scorecard, args.case_id)
    definition = case["definition"]
    manifest_path = (repo / scorecard["source_fixture_manifest"]["path"]).resolve()
    if sha256_file(manifest_path) != scorecard["source_fixture_manifest"]["sha256"]:
        raise RunError("source fixture manifest digest differs from the frozen scorecard")
    _, source_case = load_case(
        manifest_path,
        case_id=definition["source_em_case_id"],
        case_name=next(
            row["name"]
            for row in _load_json(manifest_path)["cases"]
            if row["id"] == definition["source_em_case_id"]
        ),
    )

    case_root = (args.output_root / args.case_id).resolve()
    if case_root.exists() and any(case_root.iterdir()):
        raise RunError(f"refusing to reuse non-empty case output directory: {case_root}")
    fixture_dir = case_root / "data"
    recovar_dir = case_root / "recovar"
    relion_dir = case_root / "relion"
    for directory in (fixture_dir, recovar_dir, relion_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (case_root / "SAFE_TO_DELETE").touch()
    materialize(
        manifest_path,
        args.fixture_root,
        fixture_dir,
        case_id=definition["source_em_case_id"],
        case_name=source_case["name"],
    )

    input_name = "particles_relion_identity_ctf.star" if (fixture_dir / "particles_relion_identity_ctf.star").is_file() else "particles.star"
    input_star = fixture_dir / input_name
    relion_argv = build_relion_command(
        input_star=input_star,
        output_prefix=relion_dir / "run",
        definition=definition,
        relion_refine=args.relion_refine,
        threads=args.threads,
    )
    recovar_argv = build_recovar_command(
        input_star=input_star,
        output_prefix=recovar_dir / "run",
        fixture_dir=fixture_dir,
        definition=definition,
    )
    (relion_dir / "relion_command.json").write_text(
        json.dumps({"argv": relion_argv}, indent=2, sort_keys=True) + "\n"
    )
    (recovar_dir / "recovar_command.json").write_text(
        json.dumps({"argv": recovar_argv}, indent=2, sort_keys=True) + "\n"
    )
    provenance = {
        "scorecard": str(scorecard_path),
        "scorecard_sha256": sha256_file(scorecard_path),
        "case_id": args.case_id,
        "source_em_case_id": definition["source_em_case_id"],
        "repo": str(repo),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (case_root / "run_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    env = dict(os.environ)
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    gpu_uuid = _physical_gpu_uuid()
    # RELION resolves stack paths recorded in the STAR relative to its process
    # working directory, not relative to the STAR path itself.
    _run_logged(relion_argv, cwd=fixture_dir, log_path=relion_dir / "relion.log", env=env)
    relion_gpu_uuid = _physical_gpu_uuid()
    if relion_gpu_uuid != gpu_uuid:
        raise RunError("physical GPU changed during RELION execution")
    _run_logged(recovar_argv, cwd=repo, log_path=recovar_dir / "recovar.log", env=env)
    recovar_gpu_uuid = _physical_gpu_uuid()
    paired_gpu_report = case_root / "paired_gpu_uuid.json"
    paired_gpu_report.write_text(
        json.dumps(
            {
                "physical_gpu_uuid": gpu_uuid,
                "relion_gpu_uuid": relion_gpu_uuid,
                "recovar_gpu_uuid": recovar_gpu_uuid,
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
    report_path = case_root / "trajectory_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(case_root / "trajectory_shellwise_fsc.npz", **shellwise)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["result"] != "pass":
        raise RunError(f"trajectory completed but failed the frozen gates; see {report_path}")
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--scorecard", type=Path, default=repo / "docs/math/vdam_relion_parity_scorecard_v1.json")
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--relion-refine", type=Path, default=DEFAULT_RELION_REFINE)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.relion_refine.is_file() or not os.access(args.relion_refine, os.X_OK):
        raise RunError(f"RELION executable is unavailable: {args.relion_refine}")
    run_case(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
