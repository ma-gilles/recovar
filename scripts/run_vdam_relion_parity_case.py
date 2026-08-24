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


def _env_flag(name: str, environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    raw = str(env.get(name, "")).strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _qualification_cuda_environment(
    env: dict[str, str],
    *,
    deterministic_cuda: bool,
) -> dict[str, str]:
    """Return one launch mode shared by the paired reference and candidate."""

    configured = dict(env)
    configured["CUDA_LAUNCH_BLOCKING"] = "1" if deterministic_cuda else "0"
    return configured


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


def _relion_reference_provenance(executable: Path) -> dict[str, Any]:
    """Fingerprint the exact RELION binary and its source checkout when available."""

    resolved = executable.resolve(strict=True)
    report: dict[str, Any] = {
        "executable": str(resolved),
        "executable_sha256": sha256_file(resolved),
        "executable_size_bytes": resolved.stat().st_size,
    }
    source = subprocess.run(
        ["git", "-C", str(resolved.parent), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if source.returncode != 0:
        return report
    source_root = Path(source.stdout.strip()).resolve()
    report.update(
        source_git_root=str(source_root),
        source_git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip(),
        source_git_tracked_dirty=bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=source_root,
                text=True,
            ).strip()
        ),
    )
    return report


def _recovar_native_extension_provenance(
    repo: Path,
    environ: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fingerprint the exact CUDA and RELION-bind binaries used by a run."""
    env = os.environ if environ is None else environ
    cuda_value = env.get("RECOVAR_CUDA_LIB")
    if not cuda_value:
        raise RunError("RECOVAR_CUDA_LIB is required for VDAM parity provenance")
    cuda_path = Path(cuda_value).expanduser().resolve()
    if not cuda_path.is_file():
        raise RunError(f"RECOVAR_CUDA_LIB does not name a file: {cuda_path}")

    bind_matches = sorted((repo / "recovar" / "relion_bind").glob("_relion_bind_core*.so"))
    if len(bind_matches) != 1:
        raise RunError(
            "expected exactly one local _relion_bind_core extension, found "
            f"{[str(path) for path in bind_matches]}"
        )
    bind_path = bind_matches[0].resolve()
    return {
        "cuda_backproject": {
            "path": str(cuda_path),
            "sha256": sha256_file(cuda_path),
            "size_bytes": cuda_path.stat().st_size,
        },
        "relion_bind_core": {
            "path": str(bind_path),
            "sha256": sha256_file(bind_path),
            "size_bytes": bind_path.stat().st_size,
        },
    }


def build_relion_command(
    *, input_star: Path, output_prefix: Path, definition: dict[str, Any], relion_refine: Path, threads: int
) -> list[str]:
    symmetry = str(definition.get("symmetry", "C1"))
    do_run_c1 = bool(definition.get("do_run_C1", True))
    refinement_symmetry = "C1" if do_run_c1 else symmetry
    particle_diameter = float(definition.get("particle_diameter_angstrom", 200.0))
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
        refinement_symmetry,
        "--flatten_solvent",
        "--zero_mask",
        "--dont_combine_weights_via_disc",
        "--pool",
        "3",
        "--pad",
        str(definition["padding_factor"]),
        "--particle_diameter",
        str(particle_diameter),
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
    *,
    input_star: Path,
    output_prefix: Path,
    fixture_dir: Path,
    definition: dict[str, Any],
    image_batch_size: int = 500,
) -> list[str]:
    if int(image_batch_size) <= 0:
        raise ValueError("image_batch_size must be positive")
    symmetry = str(definition.get("symmetry", "C1"))
    do_run_c1 = bool(definition.get("do_run_C1", True))
    particle_diameter = float(definition.get("particle_diameter_angstrom", 200.0))
    return [
        sys.executable,
        "-m",
        "scripts.run_ab_initio",
        "--i",
        str(input_star),
        "--o",
        str(output_prefix),
        "--nr_iter",
        str(definition["nr_iter"]),
        "--grad_write_iter",
        "1",
        "--K",
        str(definition["nr_classes"]),
        "--tau2_fudge",
        str(definition["tau2_fudge"]),
        "--sym",
        symmetry,
        "--do_run_C1",
        "1" if do_run_c1 else "0",
        "--particle_diameter",
        str(particle_diameter),
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
        "--image_batch_size",
        str(int(image_batch_size)),
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


def _recovar_gpu_env(base: dict[str, str], *, repo: Path) -> dict[str, str]:
    env = dict(base)
    env.pop("JAX_PLATFORM_NAME", None)
    # RECOVAR's JAX configuration uses a CPU device for explicit host-side
    # work, so keep CPU available while making CUDA the first/default backend.
    env["JAX_PLATFORMS"] = "cuda,cpu"
    # Bind the actual InitialModel subprocess to this checkout.  Pixi may keep
    # an editable-install link to a sibling worktree, and direct script
    # execution puts only ``scripts/`` ahead of that link on sys.path.
    env["PYTHONPATH"] = str(repo.resolve())
    env["RECOVAR_EXPECTED_REPO_ROOT"] = str(repo.resolve())
    return env


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
    image_batch_size = int(os.environ.get("RECOVAR_VDAM_IMAGE_BATCH_SIZE", "500"))
    recovar_argv = build_recovar_command(
        input_star=input_star,
        output_prefix=recovar_dir / "run",
        fixture_dir=fixture_dir,
        definition=definition,
        image_batch_size=image_batch_size,
    )
    allow_async_cuda = _env_flag("VDAM_ALLOW_ASYNC_CUDA")
    deterministic_cuda = _env_flag("VDAM_DETERMINISTIC_CUDA")
    if allow_async_cuda and deterministic_cuda:
        raise RunError("VDAM_ALLOW_ASYNC_CUDA and VDAM_DETERMINISTIC_CUDA are mutually exclusive")
    if deterministic_cuda:
        recovar_argv.append("--deterministic_cuda")
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
        "relion_reference": _relion_reference_provenance(args.relion_refine),
        "recovar_native_extensions": _recovar_native_extension_provenance(repo),
        "recovar_image_batch_size": image_batch_size,
        "recovar_sparse_big_jit_mstep_max_gb": os.environ.get(
            "RECOVAR_EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB"
        ),
        "cuda_launch_blocking": deterministic_cuda,
    }
    (case_root / "run_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    env = _qualification_cuda_environment(
        dict(os.environ),
        deterministic_cuda=deterministic_cuda,
    )
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
    recovar_env = _recovar_gpu_env(env, repo=repo)
    recorded_env_keys = (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_LAUNCH_BLOCKING",
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
        "PYTHONPATH",
        "RECOVAR_CUDA_LIB",
        "RECOVAR_EXPECTED_REPO_ROOT",
        "RECOVAR_DISABLE_CUDA",
        "RECOVAR_EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB",
        "RECOVAR_INITIAL_MODEL_EXACT_FINE_DIFF2",
        "RECOVAR_INITIAL_MODEL_UNIFY_LOCAL_BUCKET_SIZES",
        "RECOVAR_INITIAL_MODEL_COMPACT_SPARSE_PASS2",
        "RECOVAR_INITIAL_MODEL_PROJECTOR_DUMP_DIR",
        "RECOVAR_DISABLE_LOCAL_BIG_JIT",
        "RECOVAR_PASS2_DUMP_DIR",
        "RECOVAR_PASS2_DUMP_ORIGINAL_INDICES",
        "RECOVAR_PASS2_DUMP_CURRENT_SIZE",
        "RECOVAR_PASS2_DUMP_ITERATION",
        "RECOVAR_PASS2_DUMP_RAW_OPERANDS",
        "RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET",
        "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR",
        "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES",
        "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_CURRENT_SIZE",
        "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_ITERATION",
        "RECOVAR_LOCAL_SCORE_DUMP_DIR",
        "RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES",
        "RECOVAR_LOCAL_SCORE_DUMP_CURRENT_SIZE",
        "RECOVAR_LOCAL_SCORE_DUMP_ITERATION",
        "RECOVAR_LOCAL_SCORE_DUMP_OPERANDS",
        "RELION_DUMP_SIGMA2_NOISE_DIR",
        "RELION_DUMP_SIGMA2_AA_PART_ID",
        "RELION_DUMP_SIGMA2_AA_ITER",
        "SLURM_JOB_GPUS",
    )
    (recovar_dir / "runtime_environment.json").write_text(
        json.dumps({key: recovar_env.get(key) for key in recorded_env_keys}, indent=2, sort_keys=True) + "\n"
    )
    _run_logged(recovar_argv, cwd=repo, log_path=recovar_dir / "recovar.log", env=recovar_env)
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
