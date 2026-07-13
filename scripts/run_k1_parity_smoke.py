#!/usr/bin/env python3
"""Portable one-iteration K=1 supplied-map RELION parity smoke.

This launcher validates a complete RELION iter-N -> iter-(N+1) replay,
runs it locally or through Slurm, and gates the result on FSC/FSC-AUC.
Correlation is recorded only as an auxiliary diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_multi_iter_parity.py"
QUALITY_LABELS = (
    "recovar_half1",
    "recovar_half2",
    "recovar_merged",
    "relion_half1",
    "relion_half2",
    "relion_merged",
)


@dataclass(frozen=True)
class SmokeInputs:
    data_star: Path
    gt_volume: Path
    relion_dir: Path
    output_dir: Path
    particle_root: Path
    start_iter: int
    run_prefix: str


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def resolve_inputs(args: argparse.Namespace) -> SmokeInputs:
    fixture = _resolve_path(args.fixture_dir) if args.fixture_dir else None
    data_star = _resolve_path(args.data_star or (fixture / "particles.star" if fixture else ""))
    gt_volume = _resolve_path(args.gt_volume or (fixture / "reference_gt.mrc" if fixture else ""))
    relion_dir = _resolve_path(args.relion_dir or (fixture / "relion" if fixture else ""))
    output_dir = _resolve_path(args.output_dir)
    particle_root = _resolve_path(args.particle_root) if args.particle_root else data_star.parent
    return SmokeInputs(
        data_star=data_star,
        gt_volume=gt_volume,
        relion_dir=relion_dir,
        output_dir=output_dir,
        particle_root=particle_root,
        start_iter=int(args.start_iter),
        run_prefix=args.relion_run_prefix,
    )


def required_relion_paths(inputs: SmokeInputs) -> list[Path]:
    """Return the strict split-half iter-N -> iter-(N+1) file contract."""

    paths: list[Path] = []
    for iteration in (inputs.start_iter, inputs.start_iter + 1):
        stem = inputs.relion_dir / f"{inputs.run_prefix}_it{iteration:03d}"
        paths.extend(
            [
                Path(f"{stem}_half1_model.star"),
                Path(f"{stem}_half2_model.star"),
                Path(f"{stem}_data.star"),
                Path(f"{stem}_sampling.star"),
                Path(f"{stem}_optimiser.star"),
                Path(f"{stem}_half1_class001.mrc"),
                Path(f"{stem}_half2_class001.mrc"),
            ]
        )
    return paths


def referenced_particle_stacks(data_star: Path, particle_root: Path) -> list[Path]:
    """Find stack paths in RELION image-name tokens without importing RECOVAR."""

    stacks: set[Path] = set()
    for raw_line in data_star.read_text(errors="replace").splitlines():
        if "@" not in raw_line:
            continue
        try:
            tokens = shlex.split(raw_line, comments=False, posix=True)
        except ValueError:
            tokens = raw_line.split()
        for token in tokens:
            left, separator, right = token.partition("@")
            if separator and left.isdigit() and right:
                stack = Path(right)
                stacks.add(stack if stack.is_absolute() else particle_root / stack)
    return sorted(path.resolve() for path in stacks)


def validate_inputs(inputs: SmokeInputs, *, relion_src_dir: str | None = None) -> list[Path]:
    missing = [path for path in (inputs.data_star, inputs.gt_volume) if not path.is_file()]
    if not inputs.relion_dir.is_dir():
        missing.append(inputs.relion_dir)
    missing.extend(path for path in required_relion_paths(inputs) if not path.is_file())
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required K=1 smoke inputs:\n{rendered}")

    stacks = referenced_particle_stacks(inputs.data_star, inputs.particle_root)
    if not stacks:
        raise ValueError(f"No RELION image-name stack references found in {inputs.data_star}")
    missing_stacks = [path for path in stacks if not path.is_file()]
    if missing_stacks:
        rendered = "\n".join(f"  - {path}" for path in missing_stacks)
        raise FileNotFoundError(
            f"Particle stacks referenced by {inputs.data_star} were not found. "
            f"Use --particle-root when paths are relative to another directory:\n{rendered}"
        )
    if relion_src_dir:
        projector = _resolve_path(relion_src_dir) / "projector.h"
        if not projector.is_file():
            raise FileNotFoundError(f"--relion-src-dir must be RELION's src directory: missing {projector}")
    return stacks


def build_runner_command(args: argparse.Namespace, inputs: SmokeInputs, python: Path) -> list[str]:
    command = [
        str(python),
        str(RUNNER),
        "--relion_dir",
        str(inputs.relion_dir),
        "--relion_run_prefix",
        inputs.run_prefix,
        "--data_star",
        str(inputs.data_star),
        "--gt_volume",
        str(inputs.gt_volume),
        "--iter",
        str(inputs.start_iter),
        "--max_iter",
        "1",
        "--skip_final_iteration",
        "--force_max_iter_after_convergence",
        "--output_dir",
        str(inputs.output_dir),
        "--image_batch_size",
        str(args.image_batch_size),
        "--rotation_block_size",
        str(args.rotation_block_size),
    ]
    if args.max_particles is not None:
        command.extend(["--max_particles", str(args.max_particles)])
    return command


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_provenance(command: list[str], inputs: SmokeInputs, stacks: list[Path]) -> dict:
    def git(*git_args: str) -> str:
        result = subprocess.run(
            ["git", *git_args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
        )
        return result.stdout.strip()

    tracked_inputs = [inputs.data_star, inputs.gt_volume, *required_relion_paths(inputs), *stacks]
    return {
        "repo_root": str(REPO_ROOT),
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("symbolic-ref", "--short", "HEAD") or "<detached>",
        "git_status_porcelain": git("status", "--porcelain=v1"),
        "git_diff_sha256": hashlib.sha256(
            subprocess.run(
                ["git", "diff", "--binary", "HEAD", "--"], cwd=REPO_ROOT, capture_output=True, check=False
            ).stdout
        ).hexdigest(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": command[0],
        "command": command,
        "cwd": str(REPO_ROOT),
        "inputs": [
            {"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)}
            for path in tracked_inputs
        ],
        "quality_policy": {
            "primary": "FSC curves and normalized FSC-AUC",
            "correlation": "auxiliary only; never a pass/fail gate",
        },
    }


def ensure_relion_binding(
    python: Path,
    env: dict[str, str],
    *,
    relion_src_dir: str | None,
    output_dir: Path,
) -> None:
    """Use an existing RELION binding or build one into output-local scratch."""

    import_command = [
        str(python),
        "-c",
        "from recovar.relion_bind import _relion_bind_core; print(_relion_bind_core.__file__)",
    ]
    probe = subprocess.run(
        import_command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0:
        print(f"RELION binding: {probe.stdout.strip()}")
        return
    if relion_src_dir is None:
        raise RuntimeError(
            "RECOVAR's RELION binding is unavailable. Pass --relion-src-dir /path/to/relion/src "
            "or build it before running. Import error:\n" + probe.stderr.strip()
        )

    build_dir = output_dir / "runtime" / "relion_bind"
    build_dir.mkdir(parents=True, exist_ok=True)
    env["RELION_SRC_DIR"] = str(_resolve_path(relion_src_dir))
    env["RECOVAR_RELION_BIND_BUILD_DIR"] = str(build_dir)
    build_log = output_dir / "relion_bind_build.log"
    with build_log.open("w") as log:
        result = subprocess.run(
            [str(python), str(REPO_ROOT / "recovar" / "relion_bind" / "build.py")],
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"RELION binding build failed; see {build_log}")
    subprocess.run(import_command, cwd=REPO_ROOT, env=env, check=True)
    print(f"RELION binding built under {build_dir}")


def _finite_curve(values) -> tuple[list[float | None], int]:
    import numpy as np

    curve = np.asarray(values, dtype=np.float64).reshape(-1)
    finite_non_dc = int(np.isfinite(curve[1:]).sum())
    return [float(value) if np.isfinite(value) else None for value in curve], finite_non_dc


def quality_summary(npz_path: Path, *, auc_tolerance: float, min_relion_fsc_auc: float) -> dict:
    import numpy as np

    with np.load(npz_path, allow_pickle=False) as values:
        primary: dict[str, dict] = {}
        failures: list[str] = []
        for label in QUALITY_LABELS:
            curve_key = f"{label}_fsc_vs_gt"
            auc_key = f"{label}_fsc_auc_vs_gt"
            if curve_key not in values or auc_key not in values:
                failures.append(f"missing {curve_key}/{auc_key}")
                continue
            curve, finite_non_dc = _finite_curve(values[curve_key])
            auc = float(values[auc_key])
            if finite_non_dc < 2:
                failures.append(f"{curve_key} has fewer than two finite non-DC shells")
            if not np.isfinite(auc):
                failures.append(f"{auc_key} is not finite")
            primary[label] = {
                "fsc_curve": curve,
                "finite_non_dc_shells": finite_non_dc,
                "fsc_auc_vs_gt": auc,
                "shell_05": int(values[f"{label}_shell_05"]),
                "shell_0143": int(values[f"{label}_shell_0143"]),
            }

        for key in ("final_merged_fsc_vs_relion", "final_merged_fsc_auc_vs_relion"):
            if key not in values:
                failures.append(f"missing {key}")
        direct_auc = float(values["final_merged_fsc_auc_vs_relion"]) if "final_merged_fsc_auc_vs_relion" in values else float("nan")
        if not np.isfinite(direct_auc) or direct_auc < min_relion_fsc_auc:
            failures.append(
                f"final merged FSC-AUC vs RELION {direct_auc:.6g} < {min_relion_fsc_auc:.6g}"
            )
        if "final_merged_fsc_vs_relion" in values:
            _, finite_non_dc = _finite_curve(values["final_merged_fsc_vs_relion"])
            if finite_non_dc < 2:
                failures.append("final_merged_fsc_vs_relion has fewer than two finite non-DC shells")

        if "recovar_merged" in primary and "relion_merged" in primary:
            deficit = primary["relion_merged"]["fsc_auc_vs_gt"] - primary["recovar_merged"]["fsc_auc_vs_gt"]
            if deficit > auc_tolerance:
                failures.append(
                    f"RECOVAR merged FSC-AUC trails RELION by {deficit:.6g}, tolerance {auc_tolerance:.6g}"
                )
        else:
            deficit = float("nan")

        auxiliary = {
            key: float(values[key])
            for key in values.files
            if "corr" in key and np.asarray(values[key]).ndim == 0
        }
    return {
        "passed": not failures,
        "failures": failures,
        "primary_fsc_quality": primary,
        "merged_fsc_auc_deficit_vs_relion": float(deficit) if np.isfinite(deficit) else None,
        "final_merged_fsc_auc_vs_relion": float(direct_auc) if np.isfinite(direct_auc) else None,
        "thresholds": {
            "max_merged_fsc_auc_deficit": auc_tolerance,
            "min_final_merged_fsc_auc_vs_relion": min_relion_fsc_auc,
        },
        "auxiliary_correlations_not_gates": auxiliary,
        "scope_warning": "A one-iteration smoke does not establish full-trajectory or convergence parity.",
    }


def _idle_local_gpu() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    gpu_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    process_result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    if gpu_result.returncode or process_result.returncode:
        return None
    busy = {line.strip() for line in process_result.stdout.splitlines() if line.strip()}
    allowed = os.environ.get("CUDA_VISIBLE_DEVICES")
    allowed_tokens = set(allowed.split(",")) if allowed else None
    for line in gpu_result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        index, uuid, used_text = fields
        if allowed_tokens is not None and index not in allowed_tokens and uuid not in allowed_tokens:
            continue
        try:
            used_mib = int(used_text)
        except ValueError:
            continue
        if uuid not in busy and used_mib < 1024:
            return index
    return None


def choose_mode(requested: str) -> tuple[str, str | None]:
    if os.environ.get("SLURM_JOB_ID"):
        return "worker", None
    if requested == "slurm":
        return "slurm", None
    gpu = _idle_local_gpu()
    if requested == "local":
        if gpu is None:
            raise RuntimeError("No conservatively idle local GPU was found; use --mode slurm")
        return "local", gpu
    if gpu is not None:
        return "local", gpu
    if shutil.which("sbatch"):
        return "slurm", None
    raise RuntimeError("No idle local GPU and no sbatch executable were found")


def write_slurm_script(args: argparse.Namespace, worker_command: list[str], output_dir: Path) -> Path:
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    script = logs / "k1_parity_smoke.sbatch"
    directives = [
        "#!/bin/bash",
        f"#SBATCH --job-name={args.slurm_job_name}",
        f"#SBATCH --gres={args.slurm_gres}",
        f"#SBATCH --cpus-per-task={args.slurm_cpus}",
        f"#SBATCH --mem={args.slurm_mem}",
        f"#SBATCH --time={args.slurm_time}",
        f"#SBATCH --output={logs}/slurm-%j.out",
    ]
    if args.slurm_partition:
        directives.append(f"#SBATCH --partition={args.slurm_partition}")
    if args.slurm_account:
        directives.append(f"#SBATCH --account={args.slurm_account}")
    runtime = output_dir / "runtime"
    body = [
        "set -euo pipefail",
        "unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV",
        "export PYTHONNOUSERSITE=1",
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
        f"export TMPDIR={shlex.quote(str(runtime / 'tmp'))}",
        f"export PIXI_HOME={shlex.quote(str(runtime / 'pixi_home'))}",
        f"export RATTLER_CACHE_DIR={shlex.quote(str(runtime / 'rattler_cache'))}",
        'mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"',
        f"cd {shlex.quote(str(REPO_ROOT))}",
        shlex.join(worker_command),
    ]
    script.write_text("\n".join([*directives, *body, ""]))
    return script


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--fixture-dir", help="Convention root: particles.star, reference_gt.mrc, relion/")
    result.add_argument("--data-star")
    result.add_argument("--gt-volume")
    result.add_argument("--relion-dir")
    result.add_argument("--particle-root", help="Base for relative stack paths; default: STAR directory")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--relion-run-prefix", default="run")
    result.add_argument("--start-iter", type=int, default=3)
    result.add_argument("--mode", choices=("auto", "local", "slurm"), default="auto")
    result.add_argument("--validate-only", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--python", help="Python executable; default: repo pixi environment")
    result.add_argument("--relion-src-dir", help="RELION src/ used to build bindings when absent")
    result.add_argument("--max-particles", type=int)
    result.add_argument("--image-batch-size", type=int, default=64)
    result.add_argument("--rotation-block-size", type=int, default=2048)
    result.add_argument("--fsc-auc-tolerance", type=float, default=1e-4)
    result.add_argument("--min-relion-fsc-auc", type=float, default=0.995)
    result.add_argument("--slurm-job-name", default="recovar-k1-smoke")
    result.add_argument("--slurm-gres", default="gpu:1")
    result.add_argument("--slurm-cpus", type=int, default=4)
    result.add_argument("--slurm-mem", default="64G")
    result.add_argument("--slurm-time", default="01:00:00")
    result.add_argument("--slurm-partition")
    result.add_argument("--slurm-account")
    result.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    inputs = resolve_inputs(args)
    stacks = validate_inputs(inputs, relion_src_dir=args.relion_src_dir)
    default_python = REPO_ROOT / ".pixi" / "envs" / "default" / "bin" / "python"
    python = _resolve_path(args.python) if args.python else default_python
    if not python.is_file():
        raise FileNotFoundError(f"Python not found: {python}. Run `pixi install` first or pass --python.")
    command = build_runner_command(args, inputs, python)
    print(json.dumps({"inputs": {key: str(value) for key, value in inputs.__dict__.items()}, "command": command}, indent=2))
    if args.validate_only:
        print(f"VALID: {len(stacks)} referenced particle stack(s); no GPU was used")
        return 0

    mode, gpu = ("worker", None) if args.internal_worker else choose_mode(args.mode)
    if mode == "slurm":
        worker_argv = [str(python), str(Path(__file__).resolve()), *sys.argv[1:], "--internal-worker"]
        script = write_slurm_script(args, worker_argv, inputs.output_dir)
        print(f"Slurm script: {script}")
        if args.dry_run:
            return 0
        result = subprocess.run(["sbatch", "--parsable", str(script)], text=True, capture_output=True, check=True)
        print(f"Submitted Slurm job: {result.stdout.strip()}")
        return 0
    if args.dry_run:
        print(f"DRY RUN: mode={mode}, gpu={gpu}")
        return 0

    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    (inputs.output_dir / "SAFE_TO_DELETE").touch()
    env = os.environ.copy()
    env.update({"PYTHONNOUSERSITE": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
    for name in ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "VIRTUAL_ENV"):
        env.pop(name, None)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    if args.relion_src_dir:
        env["RELION_SRC_DIR"] = str(_resolve_path(args.relion_src_dir))
    ensure_relion_binding(
        python,
        env,
        relion_src_dir=args.relion_src_dir,
        output_dir=inputs.output_dir,
    )
    provenance = collect_provenance(command, inputs, stacks)
    (inputs.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    log_path = inputs.output_dir / "k1_parity_smoke.log"
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    summary = quality_summary(
        inputs.output_dir / "refinement_results.npz",
        auc_tolerance=args.fsc_auc_tolerance,
        min_relion_fsc_auc=args.min_relion_fsc_auc,
    )
    summary_path = inputs.output_dir / "quality_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print("Correlations above are AUXILIARY ONLY and were not used as quality gates.")
    if not summary["passed"]:
        print(f"FAILED FSC quality gates; see {summary_path}", file=sys.stderr)
        return 1
    print(f"PASSED one-iteration FSC/FSC-AUC smoke; see {summary_path}")
    print("This smoke does not prove full-trajectory or convergence parity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
