#!/usr/bin/env python3
"""Generate or submit a sealed real-data K-class InitialModel Slurm pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime")
DEFAULT_RELION = Path(
    "/scratch/gpfs/GILLES/mg6942/relion_clean_f2c1a384/build_clean_pinned/bin/relion_refine"
)


@dataclass(frozen=True)
class FixtureSpec:
    dataset: str
    fixture_dir: Path
    particles_star_sha256: str
    source_indices_sha256: str
    particle_diameter: float = 200.0


FIXTURES = {
    "10345": FixtureSpec(
        dataset="EMPIAR-10345",
        fixture_dir=Path(
            "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
            "vdam_real10345_10k_fixture_v1_20260823/data"
        ),
        particles_star_sha256="e5d9f77ff38d0e5137412892e7cc7591ba09265fb928b649cdeab58208a540f5",
        source_indices_sha256="9f812a7bfd6bb9dd071786143a501c6803f6c05541faee36c7d6e07f0aa787a3",
    ),
    "10076": FixtureSpec(
        dataset="EMPIAR-10076",
        fixture_dir=Path(
            "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
            "em_k1_real10076_10k_fixture_20260712/data"
        ),
        particles_star_sha256="2560afeea6839dddbb38b47d26cdf8944535a799d1e6d3e1441535c96043998f",
        source_indices_sha256="b58a6d11fb292a9ed9573ac75c6a0673f4a0e8c216f4dadf11a7f0537b0e2c9d",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quote(value: object) -> str:
    return shlex.quote(str(value))


def _git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def relion_source_provenance(source_dir: Path) -> dict[str, str | bool]:
    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=source_dir, text=True
        ).strip()
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_dir, text=True
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=source_dir, text=True
        ).strip()
        tracked_status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=source_dir,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"cannot resolve RELION source provenance from {source_dir}") from exc
    if tracked_status:
        raise SystemExit(f"RELION source has tracked changes:\n{tracked_status}")
    return {
        "repo_root": repo_root,
        "source_dir": str(source_dir),
        "git_head": head,
        "git_tree": tree,
        "tracked_dirty": False,
    }


def validate_fixture(spec: FixtureSpec) -> None:
    required = [
        spec.fixture_dir / "particles.star",
        spec.fixture_dir / "source_indices.npy",
        spec.fixture_dir / "fixture_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing frozen {spec.dataset} fixture files: {missing}")
    actual_star = _sha256(required[0])
    actual_indices = _sha256(required[1])
    if actual_star != spec.particles_star_sha256:
        raise SystemExit(
            f"{spec.dataset} particles.star drift: {actual_star} != {spec.particles_star_sha256}"
        )
    if actual_indices != spec.source_indices_sha256:
        raise SystemExit(
            f"{spec.dataset} source_indices.npy drift: {actual_indices} != {spec.source_indices_sha256}"
        )


def build_pair_command(args: argparse.Namespace, spec: FixtureSpec, pair_root: Path) -> list[str]:
    command = [
        str(args.pixi_python),
        "-m",
        "scripts.run_em_real_kclass_initialmodel_pair",
        "--dataset",
        spec.dataset,
        "--fixture-dir",
        str(spec.fixture_dir),
        "--output-root",
        str(pair_root),
        "--relion-refine",
        str(args.relion_refine),
        "--relion-source-dir",
        str(args.relion_source_dir),
        "--K",
        str(args.K),
        "--nr-iter",
        str(args.nr_iter),
        "--random-seed",
        str(args.random_seed),
        "--tau2-fudge",
        str(args.tau2_fudge),
        "--symmetry",
        args.symmetry,
        "--particle-diameter",
        str(args.particle_diameter or spec.particle_diameter),
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
        "--image-fourier-backend",
        args.image_fourier_backend,
        "--rotation-block-size",
        str(args.rotation_block_size),
        "--threads",
        str(args.cpus_per_task),
        "--minimum-fsc-auc",
        str(args.minimum_fsc_auc),
        "--minimum-assignment-accuracy",
        str(args.minimum_assignment_accuracy),
        "--minimum-class-fraction",
        str(args.minimum_class_fraction),
    ]
    for checkpoint in args.checkpoint:
        command.extend(("--checkpoint", str(checkpoint)))
    if args.reference_pair_report is not None:
        command.extend(("--reference-pair-report", str(args.reference_pair_report)))
    return command


def render_sbatch(
    args: argparse.Namespace,
    *,
    expected_head: str,
    pair_command: list[str],
) -> str:
    run_root = args.output_root
    runtime_prefix = DEFAULT_RUNTIME_ROOT / f"real_kclass_{args.dataset}"
    venv = run_root / "build" / "venv"
    run_python = venv / "bin" / "python"
    binding_dir = run_root / "build" / "relion_bind"
    cuda_lib = run_root / "build" / "cuda" / "libcuda_backproject.so"
    pixi_root = args.pixi_python.parent.parent
    pair_command = [str(run_python), *pair_command[1:]]
    command_text = " \\\n  ".join(_quote(value) for value in pair_command)
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=real_k4_{args.dataset}
#SBATCH --output={_quote(run_root / 'logs' / 'pair-%j.out')}
#SBATCH --error={_quote(run_root / 'logs' / 'pair-%j.err')}
#SBATCH --partition={args.partition}
#SBATCH --account={args.account}
#SBATCH --constraint={args.constraint}
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={args.mem}
#SBATCH --time={args.time_limit}

set -euo pipefail
cd {_quote(REPO_ROOT)}
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_PROMPT_MODIFIER CONDA_SHLVL
while IFS='=' read -r env_name _; do
  case "${{env_name}}" in
    RECOVAR_*|RELION_*|JAX_*|XLA_*) unset "${{env_name}}" ;;
  esac
done < <(env)
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}"
export RELION_SRC_DIR={_quote(args.relion_source_dir)}
export RECOVAR_RELION_BIND_BUILD_DIR={_quote(binding_dir)}
export RECOVAR_RELION_BIND_JOBS="${{SLURM_CPUS_PER_TASK}}"
export RECOVAR_CUDA_LIB={_quote(cuda_lib)}
RUNTIME_ROOT={_quote(runtime_prefix)}_${{SLURM_JOB_ID}}
export TMPDIR="${{RUNTIME_ROOT}}/tmp"
export PIXI_HOME="${{RUNTIME_ROOT}}/pixi_home"
export RATTLER_CACHE_DIR="${{RUNTIME_ROOT}}/rattler_cache"
mkdir -p "${{TMPDIR}}" "${{PIXI_HOME}}" "${{RATTLER_CACHE_DIR}}" {_quote(run_root / 'build')} {_quote(run_root / 'logs')}
touch "${{RUNTIME_ROOT}}/SAFE_TO_DELETE"

if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
module load {_quote(args.cuda_module)}
export CUDA_HOME="${{CUDA_HOME:-/usr/local/cuda-12.8}}"
export PATH="${{CUDA_HOME}}/bin:${{PATH}}"
CUDA_TARGET_LIB_DIR="${{CUDA_HOME}}/targets/x86_64-linux/lib"
PIXI_NVIDIA_ROOT={_quote(pixi_root / 'lib' / 'python3.11' / 'site-packages' / 'nvidia')}
PIXI_NVIDIA_LIB_DIRS="$(find "${{PIXI_NVIDIA_ROOT}}" -type d -name lib 2>/dev/null | paste -sd: - || true)"
export LD_LIBRARY_PATH="${{PIXI_NVIDIA_LIB_DIRS:+${{PIXI_NVIDIA_LIB_DIRS}}:}}${{CUDA_TARGET_LIB_DIR}}:${{LD_LIBRARY_PATH:-}}"
export CMAKE_INCLUDE_PATH={_quote(pixi_root / 'include' / 'fftw')}:{_quote(pixi_root / 'include')}:${{CMAKE_INCLUDE_PATH:-}}
export CMAKE_LIBRARY_PATH={_quote(pixi_root / 'lib')}:${{CMAKE_LIBRARY_PATH:-}}

EXPECTED_HEAD={_quote(expected_head)}
ACTUAL_HEAD="$(git rev-parse HEAD)"
if [[ "${{ACTUAL_HEAD}}" != "${{EXPECTED_HEAD}}" ]]; then
  echo "ERROR: queued source drift: ${{ACTUAL_HEAD}} != ${{EXPECTED_HEAD}}" >&2
  exit 2
fi
if [[ -n "$(git status --short --untracked-files=all)" ]]; then
  echo "ERROR: source tree changed after submission" >&2
  git status --short --untracked-files=all >&2
  exit 2
fi

{_quote(args.pixi_python)} -m venv --system-site-packages {_quote(venv)}
if ! grep -qx 'include-system-site-packages = true' {_quote(venv / 'pyvenv.cfg')}; then
  echo "ERROR: sealed run venv did not retain the pixi environment" >&2
  exit 2
fi
PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 {_quote(venv / 'bin' / 'pip')} install \
  -e {_quote(REPO_ROOT)} --no-deps --no-build-isolation --ignore-installed

{_quote(run_python)} recovar/relion_bind/build.py
env PYTHON={_quote(run_python)} make -C recovar/cuda \
  LIB={_quote(cuda_lib)} \
  CUDA_ARCH='-gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90' all

{_quote(run_python)} -c "import pathlib,recovar,jax; repo=pathlib.Path.cwd().resolve(); rf=pathlib.Path(recovar.__file__).resolve(); jf=pathlib.Path(jax.__file__).resolve(); assert str(rf).startswith(str(repo) + '/'), rf; assert '.pixi/envs/default/' in str(jf), jf; assert len(jax.devices('gpu')) == 1, jax.devices(); print(rf); print(jf); print(jax.devices())"

{command_text}
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(FIXTURES), default="10345")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-pair-report", type=Path, default=None)
    parser.add_argument("--relion-refine", type=Path, default=DEFAULT_RELION)
    parser.add_argument("--relion-source-dir", type=Path, required=True)
    parser.add_argument("--pixi-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--nr-iter", type=int, default=8)
    parser.add_argument("--checkpoint", type=int, action="append", default=None)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--tau2-fudge", type=float, default=4.0)
    parser.add_argument("--symmetry", default="C1")
    parser.add_argument("--particle-diameter", type=float, default=None)
    parser.add_argument("--healpix-order", type=int, default=1)
    parser.add_argument("--oversampling", type=int, default=1)
    parser.add_argument("--offset-range", type=float, default=6.0)
    parser.add_argument("--offset-step", type=float, default=2.0)
    parser.add_argument("--padding-factor", type=int, default=1)
    parser.add_argument("--image-batch-size", type=int, default=500)
    parser.add_argument(
        "--image-fourier-backend",
        choices=("host_numpy", "jax_gpu", "relion_cuda"),
        default="relion_cuda",
    )
    parser.add_argument("--rotation-block-size", type=int, default=5000)
    parser.add_argument("--minimum-fsc-auc", type=float, default=0.999)
    parser.add_argument("--minimum-assignment-accuracy", type=float, default=0.995)
    parser.add_argument("--minimum-class-fraction", type=float, default=0.01)
    parser.add_argument("--partition", default=os.environ.get("SBATCH_PARTITION", "cryoem"))
    parser.add_argument("--account", default=os.environ.get("SBATCH_ACCOUNT", "gilles"))
    parser.add_argument("--constraint", default=os.environ.get("SBATCH_CONSTRAINT", "h100"))
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem", default="192G")
    parser.add_argument("--time-limit", default="04:00:00")
    parser.add_argument("--cuda-module", default=os.environ.get("CUDA_MODULE", "cudatoolkit/12.8"))
    parser.add_argument("--submit", action="store_true", help="Submit after writing the sealed sbatch script")
    args = parser.parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    args.relion_refine = args.relion_refine.expanduser().resolve()
    args.relion_source_dir = args.relion_source_dir.expanduser().resolve()
    args.pixi_python = args.pixi_python.expanduser().resolve()
    if args.reference_pair_report is not None:
        args.reference_pair_report = args.reference_pair_report.expanduser().resolve()
    if args.checkpoint is None:
        # Native InitialModel writes numbered artifacts after each completed
        # update, so a fresh run has iterations 1..nr_iter and no iteration 0.
        args.checkpoint = list(range(1, args.nr_iter + 1))
    if args.K < 2 or args.nr_iter < 1:
        parser.error("K must be >=2 and nr-iter must be positive")
    if args.cpus_per_task < 1:
        parser.error("cpus-per-task must be positive")
    if not 0.0 <= args.minimum_class_fraction < 1.0 / args.K:
        parser.error("minimum-class-fraction must be in [0, 1/K)")
    if sorted(set(args.checkpoint)) != args.checkpoint or not all(
        0 <= value <= args.nr_iter for value in args.checkpoint
    ):
        parser.error("checkpoints must be sorted, unique, and within 0..nr-iter")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = FIXTURES[args.dataset]
    validate_fixture(spec)
    for path, label in (
        (args.relion_refine, "RELION executable"),
        (args.relion_source_dir / "projector.h", "RELION source projector.h"),
        (args.pixi_python, "pixi Python"),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    if args.reference_pair_report is not None and not args.reference_pair_report.is_file():
        raise SystemExit(f"missing frozen reference report: {args.reference_pair_report}")
    relion_source = relion_source_provenance(args.relion_source_dir)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to reuse non-empty output root: {args.output_root}")

    expected_head = _git_text("rev-parse", "HEAD")
    tracked_status = _git_text("status", "--porcelain", "--untracked-files=all")
    if tracked_status:
        raise SystemExit(f"launcher requires a clean source tree:\n{tracked_status}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "SAFE_TO_DELETE").touch()
    (args.output_root / "logs").mkdir()
    scripts_dir = args.output_root / "scripts"
    scripts_dir.mkdir()
    pair_root = args.output_root / "pair"
    placeholder_command = build_pair_command(args, spec, pair_root)
    script_path = scripts_dir / "run_pair.sbatch"
    script_path.write_text(
        render_sbatch(
            args,
            expected_head=expected_head,
            pair_command=placeholder_command,
        )
    )
    script_path.chmod(0o755)
    manifest = {
        "schema": "recovar.em_real_kclass_initialmodel_submission.v1",
        "source_root": str(REPO_ROOT),
        "git_head": expected_head,
        "git_tree": _git_text("rev-parse", "HEAD^{tree}"),
        "relion_executable": str(args.relion_refine),
        "relion_executable_sha256": _sha256(args.relion_refine),
        "relion_source": relion_source,
        "dataset": spec.dataset,
        "fixture_dir": str(spec.fixture_dir),
        "particles_star_sha256": spec.particles_star_sha256,
        "source_indices_sha256": spec.source_indices_sha256,
        "reference_mode": "frozen_pair_report" if args.reference_pair_report else "fresh_same_gpu",
        "reference_pair_report": str(args.reference_pair_report) if args.reference_pair_report else None,
        "sbatch_script": str(script_path),
        "sbatch_script_sha256": _sha256(script_path),
        "requested_resources": {
            "partition": args.partition,
            "account": args.account,
            "constraint": args.constraint,
            "gpus": 1,
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": args.cpus_per_task,
            "mem": args.mem,
            "time_limit": args.time_limit,
            "exclusive": False,
        },
        "pair_command": placeholder_command,
        "safe_to_delete_marker": str(args.output_root / "SAFE_TO_DELETE"),
    }
    manifest_path = args.output_root / "submission_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    submit_command = ["sbatch", "--parsable", str(script_path)]
    print(f"Run root: {args.output_root}")
    print(f"Sbatch script: {script_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Submit: {shlex.join(submit_command)}")
    if not args.submit:
        print("Dry run only; no Slurm job submitted.")
        return 0
    result = subprocess.run(submit_command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";", 1)[0]
    print(f"Submitted Slurm job {job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
