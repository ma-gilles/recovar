#!/usr/bin/env python3
"""Run one paired K-class InitialModel comparison on a single Slurm GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.audit_vdam_kclass_trajectory import audit_trajectory
else:
    from audit_vdam_kclass_trajectory import audit_trajectory


DEFAULT_RELION = Path("/scratch/gpfs/GILLES/mg6942/relion_clean_f2c1a384/build_clean_pinned/bin/relion_refine")


class PairRunError(RuntimeError):
    """Raised when paired execution or provenance validation fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gpu_uuid() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(values) != 1 or not values[0].startswith("GPU-"):
        raise PairRunError(f"expected exactly one visible physical GPU, found {values}")
    return values[0]


def build_relion_command(args: argparse.Namespace, output_prefix: Path) -> list[str]:
    return [
        str(args.relion_refine),
        "--o",
        str(output_prefix),
        "--iter",
        str(args.nr_iter),
        "--grad",
        "--denovo_3dref",
        "--grad_write_iter",
        "1",
        "--i",
        "particles.star",
        "--ctf",
        "--K",
        str(args.K),
        "--sym",
        args.symmetry,
        "--flatten_solvent",
        "--zero_mask",
        "--dont_combine_weights_via_disc",
        "--pool",
        "3",
        "--pad",
        str(args.padding_factor),
        "--particle_diameter",
        str(args.particle_diameter),
        "--oversampling",
        str(args.oversampling),
        "--healpix_order",
        str(args.healpix_order),
        "--offset_range",
        str(args.offset_range),
        "--offset_step",
        str(args.offset_step),
        "--auto_sampling",
        "--tau2_fudge",
        str(args.tau2_fudge),
        "--random_seed",
        str(args.random_seed),
        "--j",
        str(args.threads),
        "--gpu",
        "0",
    ]


def build_recovar_command(args: argparse.Namespace, output_prefix: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "recovar.commands.initial_model",
        "--i",
        str(args.fixture_dir / "particles.star"),
        "--datadir",
        str(args.fixture_dir),
        "--o",
        str(output_prefix),
        "--nr-iter",
        str(args.nr_iter),
        "--grad-write-iter",
        "1",
        "--K",
        str(args.K),
        "--tau2-fudge",
        str(args.tau2_fudge),
        "--sym",
        args.symmetry,
        "--particle-diameter",
        str(args.particle_diameter),
        "--random-seed",
        str(args.random_seed),
        "--healpix-order",
        str(args.healpix_order),
        "--oversampling",
        str(args.oversampling),
        "--offset-range",
        str(args.offset_range),
        "--offset-step",
        str(args.offset_step),
        "--padding-factor",
        str(args.padding_factor),
        "--image-batch-size",
        str(args.image_batch_size),
        "--rotation-block-size",
        str(args.rotation_block_size),
        "--gpu",
        "0",
    ]


def _run(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> dict[str, Any]:
    started = time.time()
    with log_path.open("w") as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True)
    timing = {"wall_s": time.time() - started, "exit_code": result.returncode}
    if result.returncode:
        raise PairRunError(f"command failed with exit {result.returncode}; see {log_path}")
    return timing


def run_pair(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        text=True,
    ).strip()
    if git_status and not args.allow_dirty:
        raise PairRunError(
            "paired qualification requires a clean worktree; commit or pass --allow-dirty for diagnostics"
        )
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise PairRunError(f"refusing to reuse non-empty output root: {args.output_root}")
    candidate_dir = args.output_root / "recovar"
    reference_dir = args.output_root / "relion"
    for directory in (args.output_root, candidate_dir, reference_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (args.output_root / "SAFE_TO_DELETE").touch()

    required_fixture = [args.fixture_dir / "particles.star", args.fixture_dir / "particles.128.mrcs"]
    missing = [str(path) for path in required_fixture if not path.is_file()]
    if missing:
        raise PairRunError(f"missing K-class fixture paths: {missing}")
    if not args.relion_refine.is_file() or not os.access(args.relion_refine, os.X_OK):
        raise PairRunError(f"RELION executable is unavailable: {args.relion_refine}")

    relion_command = build_relion_command(args, reference_dir / "run")
    recovar_command = build_recovar_command(args, candidate_dir / "run")
    (reference_dir / "command.json").write_text(json.dumps(relion_command, indent=2) + "\n")
    (candidate_dir / "command.json").write_text(json.dumps(recovar_command, indent=2) + "\n")
    env = dict(os.environ)
    env.pop("JAX_PLATFORM_NAME", None)
    env.update(
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        CUDA_LAUNCH_BLOCKING="0",
        JAX_PLATFORMS="cuda,cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )

    gpu_before = _gpu_uuid()
    relion_timing = _run(
        relion_command,
        cwd=args.fixture_dir,
        env=env,
        log_path=reference_dir / "run.log",
    )
    gpu_between = _gpu_uuid()
    recovar_timing = _run(
        recovar_command,
        cwd=repo_root,
        env=env,
        log_path=candidate_dir / "run.log",
    )
    gpu_after = _gpu_uuid()
    if len({gpu_before, gpu_between, gpu_after}) != 1:
        raise PairRunError("physical GPU identity changed during paired execution")

    report, shellwise = audit_trajectory(
        candidate_dir=candidate_dir,
        reference_dir=reference_dir,
        K=args.K,
        checkpoints=tuple(args.checkpoint),
        minimum_fsc_auc=args.minimum_fsc_auc,
        minimum_assignment_accuracy=args.minimum_assignment_accuracy,
    )
    provenance = {
        "schema": "recovar.vdam_kclass_pair.v1",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip(),
        "git_dirty": bool(git_status),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "physical_gpu_uuid": gpu_before,
        "fixture_dir": str(args.fixture_dir),
        "fixture_sha256": {path.name: _sha256(path) for path in required_fixture},
        "relion_executable": str(args.relion_refine.resolve()),
        "relion_sha256": _sha256(args.relion_refine),
        "relion_timing": relion_timing,
        "recovar_timing": recovar_timing,
        "runtime_ratio_recovar_over_relion": recovar_timing["wall_s"] / relion_timing["wall_s"],
        "audit": report,
    }
    (args.output_root / "pair_report.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    npz_path = args.output_root / "trajectory_shellwise_fsc.npz"
    np.savez_compressed(npz_path, **shellwise)
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return provenance


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--relion-refine", type=Path, default=DEFAULT_RELION)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--nr-iter", type=int, default=8)
    parser.add_argument("--checkpoint", type=int, action="append", default=None)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--tau2-fudge", type=float, default=4.0)
    parser.add_argument("--symmetry", default="C1")
    parser.add_argument("--particle-diameter", type=float, default=200.0)
    parser.add_argument("--healpix-order", type=int, default=1)
    parser.add_argument("--oversampling", type=int, default=1)
    parser.add_argument("--offset-range", type=float, default=6.0)
    parser.add_argument("--offset-step", type=float, default=2.0)
    parser.add_argument("--padding-factor", type=int, default=1)
    parser.add_argument("--image-batch-size", type=int, default=500)
    parser.add_argument("--rotation-block-size", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--minimum-fsc-auc", type=float, default=0.999)
    parser.add_argument("--minimum-assignment-accuracy", type=float, default=0.995)
    parser.add_argument("--allow-dirty", action="store_true", help="Permit non-qualification diagnostic runs")
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        args.checkpoint = sorted({0, 1, 2, 4, args.nr_iter})
    return args


def main(argv: list[str] | None = None) -> int:
    report = run_pair(_parse_args(argv))
    return 0 if report["audit"]["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
