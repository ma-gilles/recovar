#!/usr/bin/env python3
"""Submit a PDB-backed K-class EM robustness matrix to Slurm.

The K=1 robustness launcher covers image-count, SNR, angle-distribution, and
outlier stress for AutoRefine. This launcher covers the orthogonal K-class axis:
number of volumes/classes, class balance, PDB family, noise model, and pose
distribution. Each case generates a target-grid PDB synthetic dataset, runs a
RELION Class3D baseline, runs RECOVAR's K-class full refinement with matching
GUI-style Class3D defaults, evaluates both against GT, and writes artifacts that
``scripts/summarize_em_robustness_matrix.py`` can aggregate.
"""

from __future__ import annotations

import argparse
import os
import shlex
import time
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RIBO_PDB_DIR = Path("/home/mg6942/mytigress/cryobench2/Ribosembly/pdbs")
DEFAULT_IGG_PDB_DIR = Path("/home/mg6942/mytigress/cryobench2/IgG-1D/pdbs")
DEFAULT_TOMOTWIN_PDB_DIR = Path("/home/mg6942/mytigress/cryobench2/Tomotwin-100/pdbs")
DEFAULT_IGG_RL_PDB_DIR = Path("/home/mg6942/mytigress/cryobench2/IgG-RL/pdbs")


@dataclass(frozen=True)
class Case:
    index: int
    name: str
    pdb_dir: Path
    n_classes: int
    n_images: int
    grid_size: int
    noise_level: float
    noise_model: str
    dataset_params_option: str
    class_distribution: str
    seed: int
    pdb_bfactor: float
    init_radius: int
    noise_scale_std: float
    contrast_std: float
    volume_radius: float
    image_offset_n_std: float
    percent_outliers: float
    max_iter: int
    time_limit: str
    mem: str
    streaming_chunk_size: int
    streaming_mmap: bool

    @property
    def row_fields(self) -> list[str]:
        return [
            str(self.index),
            self.name,
            str(self.n_classes),
            str(self.n_images),
            str(self.grid_size),
            f"{self.noise_level:g}",
            self.noise_model,
            self.dataset_params_option,
            str(self.seed),
            f"{self.pdb_bfactor:g}",
            str(self.init_radius),
            f"{self.noise_scale_std:g}",
            f"{self.contrast_std:g}",
            f"{self.volume_radius:g}",
            f"{self.image_offset_n_std:g}",
            f"{self.percent_outliers:g}",
            str(self.max_iter),
            self.class_distribution,
            self.time_limit,
            self.mem,
        ]


DEFAULT_CASES: tuple[Case, ...] = (
    Case(1, "ribo_k2_10k_g128_white_noise1_uniform", DEFAULT_RIBO_PDB_DIR, 2, 10_000, 128, 1.0, "white", "uniform", "uniform", 2801, 80.0, 10, 0.0, 0.0, 0.7, 0.0, 0.0, 5, "05:00:00", "192G", 500, False),
    Case(2, "ribo_k4_10k_g128_white_noise1_uniform", DEFAULT_RIBO_PDB_DIR, 4, 10_000, 128, 1.0, "white", "uniform", "uniform", 2802, 80.0, 10, 0.0, 0.0, 0.7, 0.0, 0.0, 5, "06:00:00", "256G", 500, False),
    Case(3, "ribo_k4_10k_g128_radial_noise3_nonuniform_linear", DEFAULT_RIBO_PDB_DIR, 4, 10_000, 128, 3.0, "radial1", "nonuniform", "linear", 2803, 80.0, 10, 0.2, 0.2, 0.7, 0.0, 0.0, 5, "06:00:00", "256G", 500, False),
    Case(4, "ribo_k8_10k_g128_white_noise3_kent_headheavy", DEFAULT_RIBO_PDB_DIR, 8, 10_000, 128, 3.0, "white", "kent", "head-heavy", 2804, 80.0, 10, 0.0, 0.0, 0.7, 0.0, 0.0, 5, "08:00:00", "320G", 500, False),
    Case(5, "ribo_k4_50k_g256_white_noise1_uniform", DEFAULT_RIBO_PDB_DIR, 4, 50_000, 256, 1.0, "white", "uniform", "uniform", 2805, 80.0, 10, 0.0, 0.0, 0.7, 0.0, 0.0, 8, "18:00:00", "500G", 1000, True),
    Case(6, "ribo_k4_50k_g256_radial_noise3_nonuniform_linear", DEFAULT_RIBO_PDB_DIR, 4, 50_000, 256, 3.0, "radial1", "nonuniform", "linear", 2806, 80.0, 10, 0.2, 0.2, 0.7, 0.0, 0.0, 8, "18:00:00", "500G", 1000, True),
    Case(7, "ribo_k16_20k_g128_white_noise3_uniform", DEFAULT_RIBO_PDB_DIR, 16, 20_000, 128, 3.0, "white", "uniform", "uniform", 2807, 80.0, 10, 0.0, 0.0, 0.7, 0.0, 0.0, 5, "12:00:00", "500G", 500, False),
    Case(8, "igg_k4_10k_g128_white_noise1_uniform", DEFAULT_IGG_PDB_DIR, 4, 10_000, 128, 1.0, "white", "uniform", "uniform", 2808, 80.0, 10, 0.0, 0.0, 0.7, 0.0, 0.0, 5, "06:00:00", "256G", 500, False),
    Case(9, "igg_k8_10k_g128_radial_noise3_nonuniform", DEFAULT_IGG_PDB_DIR, 8, 10_000, 128, 3.0, "radial1", "nonuniform", "linear", 2809, 80.0, 10, 0.2, 0.2, 0.7, 0.0, 0.0, 5, "08:00:00", "320G", 500, False),
    Case(10, "ribo_k4_10k_g128_radial_noise3_nonuniform_outliers_pct20", DEFAULT_RIBO_PDB_DIR, 4, 10_000, 128, 3.0, "radial1", "nonuniform", "linear", 2810, 80.0, 10, 0.2, 0.2, 0.7, 0.5, 0.20, 5, "06:00:00", "256G", 500, False),
    Case(11, "igg_k4_10k_g128_white_noise1_uniform_outliers_pct20", DEFAULT_IGG_PDB_DIR, 4, 10_000, 128, 1.0, "white", "uniform", "uniform", 2811, 80.0, 10, 0.0, 0.0, 0.7, 0.0, 0.20, 5, "06:00:00", "256G", 500, False),
    Case(12, "tomotwin_k4_10k_g128_white_noise1_uniform", DEFAULT_TOMOTWIN_PDB_DIR, 4, 10_000, 128, 1.0, "white", "uniform", "uniform", 2812, 80.0, 10, 0.0, 0.0, 0.7, 0.0, 0.0, 5, "06:00:00", "256G", 500, False),
    Case(13, "tomotwin_k8_10k_g128_radial_noise3_kent_headheavy", DEFAULT_TOMOTWIN_PDB_DIR, 8, 10_000, 128, 3.0, "radial1", "kent", "head-heavy", 2813, 80.0, 10, 0.2, 0.2, 0.7, 0.0, 0.0, 5, "08:00:00", "320G", 500, False),
    Case(14, "igg_rl_k4_10k_g128_white_noise1_uniform", DEFAULT_IGG_RL_PDB_DIR, 4, 10_000, 128, 1.0, "white", "uniform", "uniform", 2814, 80.0, 10, 0.0, 0.0, 0.7, 0.0, 0.0, 5, "06:00:00", "256G", 500, False),
    Case(15, "igg_rl_k4_10k_g128_radial_noise3_nonuniform_outliers_pct20", DEFAULT_IGG_RL_PDB_DIR, 4, 10_000, 128, 3.0, "radial1", "nonuniform", "linear", 2815, 80.0, 10, 0.2, 0.2, 0.7, 0.5, 0.20, 5, "06:00:00", "256G", 500, False),
)

RECOVAR_OPTIONAL_ENV_PASSTHROUGH = (
    "TF_GPU_ALLOCATOR",
    "RECOVAR_SPARSE_KCLASS_GROUP_TIMING",
    "RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE",
    "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT",
    "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER",
    "RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT",
    "RECOVAR_RELION_FIRSTITER_RECON_COMPLEX_BUDGET",
    "RECOVAR_K_CLASS_DENSE_PASS2",
    "RECOVAR_K_CLASS_DENSE_PASS2_SUPPORT_FRACTION",
    "RECOVAR_K_CLASS_DENSE_PASS2_MEAN_SUPPORT_FRACTION",
    "RECOVAR_K_CLASS_DENSE_PASS2_SMALL_DATASET_IMAGES",
    "RECOVAR_K_CLASS_DENSE_PASS2_SMALL_DATASET_MEAN_SUPPORT_FRACTION",
    "RECOVAR_K_CLASS_RELION_X_HALF_MSTEP",
    "RECOVAR_K_CLASS_FULL_VOLUME_MSTEP",
    "RECOVAR_K_CLASS_HALF_VOLUME_MSTEP",
    "RECOVAR_KCLASS_REPLAY_TAU2",
    "RECOVAR_KCLASS_REPLAY_TAU2_SAME_ITER",
    "RECOVAR_SPARSE_KCLASS_REUSE_COMPACT_NOISE_SUMS",
    "RECOVAR_SPARSE_KCLASS_FUSE_COMPACT_IMAGE_SUMS",
    "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS",
    "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_CHECK",
    "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_STATS",
    "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP",
    "RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS",
    "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE",
    "RECOVAR_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE",
    "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL",
    "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_PREMATMUL_MAX_GROUPED_DENSE_RATIO",
    "RECOVAR_SPARSE_PASS2_MAX_NOISE_BLOCK_BYTES",
    "RECOVAR_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES",
    "RECOVAR_SPARSE_PASS2_MAX_HYPOTHESES",
    "RECOVAR_KCLASS_DUMP_DIR",
)


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Poll squeue until the summary job leaves the queue.")
    parser.add_argument("--dry-run", action="store_true", help="Write scripts and tables but do not submit sbatch jobs.")
    parser.add_argument("--case", action="append", default=[], help="Case index or name. Repeatable.")
    parser.add_argument("--scratch-dir", type=Path, default=None)
    parser.add_argument("--summary-partition", default=os.environ.get("EM_KCLASS_MATRIX_SUMMARY_PARTITION"))
    parser.add_argument(
        "--max-iter-override",
        type=int,
        default=None,
        help="Override max_iter for all selected cases. Also available as EM_KCLASS_MATRIX_MAX_ITER.",
    )
    parser.add_argument(
        "--time-limit-override",
        default=None,
        help="Override Slurm time limit for all selected cases. Also available as EM_KCLASS_MATRIX_TIME_LIMIT.",
    )
    parser.add_argument(
        "--seed-override",
        type=int,
        default=None,
        help=(
            "Override the simulator/RELION/RECOVAR seed for all selected cases. "
            "Also available as EM_KCLASS_MATRIX_SEED."
        ),
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=None,
        help=(
            "Add an offset to each selected case seed. "
            "Also available as EM_KCLASS_MATRIX_SEED_OFFSET."
        ),
    )
    return parser.parse_args()


def selected_cases(args: argparse.Namespace) -> list[Case]:
    requested = list(args.case)
    env_cases = os.environ.get("EM_KCLASS_MATRIX_CASES", "")
    requested.extend(part.strip() for part in env_cases.split(",") if part.strip())
    if not requested:
        return list(DEFAULT_CASES)
    out: list[Case] = []
    for case in DEFAULT_CASES:
        if str(case.index) in requested or case.name in requested:
            out.append(case)
    missing = sorted(set(requested) - {str(case.index) for case in out} - {case.name for case in out})
    if missing:
        raise SystemExit(f"Unknown case(s): {', '.join(missing)}")
    out = apply_case_overrides(out, args)
    return out


def apply_case_overrides(cases: list[Case], args: argparse.Namespace) -> list[Case]:
    max_iter_override = getattr(args, "max_iter_override", None)
    if max_iter_override is None:
        raw_max_iter = os.environ.get("EM_KCLASS_MATRIX_MAX_ITER")
        max_iter_override = int(raw_max_iter) if raw_max_iter else None
    if max_iter_override is not None and max_iter_override <= 0:
        raise SystemExit("EM_KCLASS_MATRIX_MAX_ITER / --max-iter-override must be positive")

    time_limit_override = getattr(args, "time_limit_override", None) or os.environ.get("EM_KCLASS_MATRIX_TIME_LIMIT")
    seed_override = getattr(args, "seed_override", None)
    if seed_override is None:
        raw_seed_override = os.environ.get("EM_KCLASS_MATRIX_SEED")
        seed_override = int(raw_seed_override) if raw_seed_override else None
    seed_offset = getattr(args, "seed_offset", None)
    if seed_offset is None:
        raw_seed_offset = os.environ.get("EM_KCLASS_MATRIX_SEED_OFFSET")
        seed_offset = int(raw_seed_offset) if raw_seed_offset else None
    if seed_override is not None and seed_offset is not None:
        raise SystemExit("Use either EM_KCLASS_MATRIX_SEED / --seed-override or EM_KCLASS_MATRIX_SEED_OFFSET / --seed-offset, not both")

    if not max_iter_override and not time_limit_override and seed_override is None and seed_offset is None:
        return cases

    out = []
    for case in cases:
        updated = case
        if max_iter_override:
            updated = replace(updated, max_iter=max_iter_override)
        if time_limit_override:
            updated = replace(updated, time_limit=time_limit_override)
        if seed_override is not None or seed_offset is not None:
            new_seed = seed_override if seed_override is not None else case.seed + int(seed_offset or 0)
            updated = replace(updated, seed=new_seed, name=f"{updated.name}_seed{new_seed}")
        out.append(updated)
    return out


def sbatch_directive(flag: str, value: str | None) -> str:
    return f"#SBATCH {flag}={value}" if value else ""


def optional_exports(names: tuple[str, ...]) -> str:
    lines = []
    for name in names:
        value = os.environ.get(name)
        if value:
            lines.append(f"export {name}={q(value)}")
    return "\n".join(lines)


def build_cuda_lib_command(*, scratch_dir: Path) -> str:
    return f"""mkdir -p "$(dirname "${{RECOVAR_CUDA_LIB}}")"
CUDA_LIB_TMP="${{RECOVAR_CUDA_LIB}}.${{SLURM_JOB_ID:-$$}}.tmp"
export CUDA_LIB_TMP PIXI_PY
flock {q(scratch_dir / "cuda" / "build.lock")} bash -lc '
  set -euo pipefail
  rm -f "${{CUDA_LIB_TMP}}"
  env PYTHON="${{PIXI_PY}}" make -C recovar/cuda LIB="${{CUDA_LIB_TMP}}" all
  mv -f "${{CUDA_LIB_TMP}}" "${{RECOVAR_CUDA_LIB}}"
'
"""


def job_preamble(*, scratch_dir: Path, cuda_lib: Path, cuda_module: str, job_name: str) -> str:
    return f"""set -euo pipefail
cd {q(REPO_ROOT)}
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_PROMPT_MODIFIER CONDA_SHLVL
# Submit shells often run CPU-only local tests. GPU Slurm jobs must not inherit
# those overrides or the CUDA provenance gate will correctly fail.
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
export PYTHONNOUSERSITE=1
export RECOVAR_EXPECTED_REPO_ROOT={q(REPO_ROOT)}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PIXI_FROZEN=true
export TMPDIR={q(scratch_dir)}/tmp/{job_name}_${{SLURM_JOB_ID}}
export PIXI_HOME={q(scratch_dir)}/pixi_home/{job_name}_${{SLURM_JOB_ID}}
export RATTLER_CACHE_DIR={q(scratch_dir)}/rattler_cache/{job_name}_${{SLURM_JOB_ID}}
export RECOVAR_JAX_CACHE_DIR={q(scratch_dir)}/jax_cache
export JAX_COMPILATION_CACHE_DIR="${{RECOVAR_JAX_CACHE_DIR}}"
export RECOVAR_CUDA_LIB={q(cuda_lib)}
export RECOVAR_CUDA_CACHE_DIR={q(scratch_dir)}/cuda_cache/{job_name}_${{SLURM_JOB_ID}}
export RECOVAR_RELION_BIND_BUILD_DIR={q(scratch_dir)}/relion_bind_build/{job_name}_${{SLURM_JOB_ID}}
mkdir -p "${{TMPDIR}}" "${{PIXI_HOME}}" "${{RATTLER_CACHE_DIR}}" "${{RECOVAR_JAX_CACHE_DIR}}" "${{RECOVAR_CUDA_CACHE_DIR}}" "${{RECOVAR_RELION_BIND_BUILD_DIR}}" "$(dirname "${{RECOVAR_CUDA_LIB}}")" {q(REPO_ROOT / ".pixi")}

if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
if ! module load {q(cuda_module)}; then
  echo "WARNING: failed to load CUDA module {cuda_module}; falling back to CUDA_HOME if available" >&2
fi
CUDA_HOME="${{CUDA_HOME:-/usr/local/cuda-12.8}}"
export CUDA_HOME
if [[ -d "${{CUDA_HOME}}/bin" ]]; then
  export PATH="${{CUDA_HOME}}/bin:${{PATH}}"
fi
CUDA_TARGET_LIB_DIR="${{CUDA_HOME}}/targets/x86_64-linux/lib"
PIXI_NVIDIA_ROOT={q(REPO_ROOT)}/.pixi/envs/default/lib/python3.11/site-packages/nvidia
if [[ -d "${{PIXI_NVIDIA_ROOT}}" ]]; then
  PIXI_NVIDIA_LIB_DIRS="$(find "${{PIXI_NVIDIA_ROOT}}" -type d -name lib 2>/dev/null | paste -sd: -)"
else
  PIXI_NVIDIA_LIB_DIRS=""
fi
if [[ -n "${{PIXI_NVIDIA_LIB_DIRS}}" ]]; then
  export LD_LIBRARY_PATH="${{PIXI_NVIDIA_LIB_DIRS}}:${{CUDA_TARGET_LIB_DIR}}:${{LD_LIBRARY_PATH:-}}"
else
  export LD_LIBRARY_PATH="${{CUDA_TARGET_LIB_DIR}}:${{LD_LIBRARY_PATH:-}}"
fi
if [[ -z "${{CUDA_VISIBLE_DEVICES:-}}" ]]; then
  SLURM_VISIBLE_GPUS="${{SLURM_STEP_GPUS:-${{SLURM_JOB_GPUS:-}}}}"
  CUDA_FIRST_GPU="${{SLURM_VISIBLE_GPUS%%,*}}"
  if [[ -n "${{CUDA_FIRST_GPU}}" ]]; then
    export CUDA_VISIBLE_DEVICES="${{CUDA_FIRST_GPU}}"
  fi
fi

echo "=== {job_name} ==="
echo "Repo: {REPO_ROOT}"
echo "HEAD: $(git rev-parse HEAD)"
echo "Branch: $(git symbolic-ref --short HEAD || echo '<detached>')"
echo "Dirty status:"
git status --short
echo "Slurm job: ${{SLURM_JOB_ID}}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-}}"
echo "TMPDIR=${{TMPDIR}}"
echo "RECOVAR_CUDA_LIB=${{RECOVAR_CUDA_LIB}}"
echo "RECOVAR_RELION_BIND_BUILD_DIR=${{RECOVAR_RELION_BIND_BUILD_DIR}}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
{optional_exports(RECOVAR_OPTIONAL_ENV_PASSTHROUGH)}
"""


def write_setup_script(
    *,
    scratch_dir: Path,
    jobs_dir: Path,
    cuda_lib: Path,
    account: str,
    partition: str,
    constraint: str,
    setup_gres: str,
    cuda_module: str,
) -> Path:
    script = jobs_dir / "em_kclass_matrix_setup.sh"
    text = f"""#!/usr/bin/env bash
#SBATCH --job-name=em_kclass_setup
#SBATCH --output={q(scratch_dir / "em_kclass_matrix_setup.out")}
#SBATCH --error={q(scratch_dir / "em_kclass_matrix_setup.err")}
#SBATCH --partition={partition}
#SBATCH --account={account}
{sbatch_directive("--constraint", constraint)}
{sbatch_directive("--gres", setup_gres)}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

{job_preamble(scratch_dir=scratch_dir, cuda_lib=cuda_lib, cuda_module=cuda_module, job_name="em_kclass_matrix_setup")}

flock {q(REPO_ROOT / ".pixi" / "install-recovar.lock")} bash -lc '
set -euo pipefail
pixi run --frozen install-recovar
pixi run --frozen python recovar/relion_bind/build.py
'
PIXI_PY="$(pixi run --frozen which python)"
"${{PIXI_PY}}" - <<'PY'
import os
import pathlib
import jax
import recovar
from recovar.relion_bind import _relion_bind_core as relion_bind

repo = pathlib.Path.cwd().resolve()
relion_bind_file = pathlib.Path(relion_bind.__file__).resolve()
external_bind_dir = os.environ.get("RECOVAR_RELION_BIND_BUILD_DIR")
external_bind_root = pathlib.Path(external_bind_dir).resolve() if external_bind_dir else None
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert str(relion_bind_file).startswith(str(repo) + "/") or (
    external_bind_root is not None
    and str(relion_bind_file).startswith(str(external_bind_root) + "/")
)
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
print("setup provenance gate ok")
PY
"""
    script.write_text(text)
    script.chmod(0o755)
    return script


def write_case_script(
    *,
    case: Case,
    scratch_dir: Path,
    jobs_dir: Path,
    cuda_lib: Path,
    account: str,
    partition: str,
    constraint: str,
    exclusive: bool,
    cuda_module: str,
    relion_module: str,
    relion_refine_mpi: str,
    relion_mpi_ranks: int,
    relion_pool: int,
    particle_diameter: float,
    image_batch_size: int,
    rotation_block_size: int,
    gt_align_refine_orders: str,
    noise_rng_batch_size: str,
) -> Path:
    case_root = scratch_dir / "cases" / f"{case.index}_{case.name}"
    script = jobs_dir / f"em_kclass_matrix_{case.index}_{case.name}.sh"
    exclusive_directive = "#SBATCH --exclusive" if exclusive else ""
    streaming_flag = "--streaming-mmap" if case.streaming_mmap else "--no-streaming-mmap"
    noise_rng_lines = ""
    if noise_rng_batch_size:
        noise_rng_lines = f"  --noise-rng-batch-size {q(noise_rng_batch_size)} \\\n"
    refine_orders_args = ""
    if gt_align_refine_orders.strip():
        quoted = " ".join(q(part) for part in gt_align_refine_orders.split(",") if part.strip())
        refine_orders_args = f"--gt_align_refine_orders {quoted}"
    required_pdb_count = case.n_classes + (1 if case.percent_outliers > 0.0 else 0)
    outlier_pdb_setup = ""
    outlier_prepare_arg = ""
    if case.percent_outliers > 0.0:
        outlier_pdb_setup = f"""
OUTLIER_PDB="${{PDBS[{case.n_classes}]}}"
echo "Using holdout outlier PDB: ${{OUTLIER_PDB}}"
"""
        outlier_prepare_arg = '  --outlier-pdb-path "${OUTLIER_PDB}" \\\n'
    text = f"""#!/usr/bin/env bash
#SBATCH --job-name=em_kcls_{case.index}_{case.name[:16]}
#SBATCH --output={q(scratch_dir / f"em_kclass_matrix_{case.index}_{case.name}.out")}
#SBATCH --error={q(scratch_dir / f"em_kclass_matrix_{case.index}_{case.name}.err")}
#SBATCH --partition={partition}
#SBATCH --account={account}
{sbatch_directive("--constraint", constraint)}
#SBATCH --gres=gpu:1
{exclusive_directive}
#SBATCH --nodes=1
#SBATCH --ntasks={relion_mpi_ranks}
#SBATCH --cpus-per-task=8
#SBATCH --mem={case.mem}
#SBATCH --time={case.time_limit}

{job_preamble(scratch_dir=scratch_dir, cuda_lib=cuda_lib, cuda_module=cuda_module, job_name=f"em_kclass_matrix_{case.index}_{case.name}")}

CASE_ROOT={q(case_root)}
DATA_DIR="${{CASE_ROOT}}/data"
RECOVAR_DIR="${{CASE_ROOT}}/recovar"
RELION_DIR="${{CASE_ROOT}}/relion_ref"
RELION_DISPATCH_LOG="${{RELION_DIR}}/dispatch.tsv"
RELION_DISPATCH_SCHEDULE="${{RELION_DIR}}/dispatch_schedule.npz"
SUB_PDB_DIR="${{CASE_ROOT}}/pdbs_k{case.n_classes}"
mkdir -p "${{CASE_ROOT}}" "${{DATA_DIR}}" "${{RECOVAR_DIR}}" "${{RELION_DIR}}" "${{SUB_PDB_DIR}}"

cat > "${{CASE_ROOT}}/case_config.json" <<JSON
{{
  "index": {case.index},
  "name": "{case.name}",
  "pdb_dir": "{case.pdb_dir}",
  "n_classes": {case.n_classes},
  "n_images": {case.n_images},
  "grid_size": {case.grid_size},
  "noise_level": {case.noise_level},
  "noise_model": "{case.noise_model}",
  "dataset_params_option": "{case.dataset_params_option}",
  "class_distribution": "{case.class_distribution}",
  "seed": {case.seed},
  "pdb_bfactor": {case.pdb_bfactor},
  "init_radius": {case.init_radius},
  "noise_scale_std": {case.noise_scale_std},
  "contrast_std": {case.contrast_std},
  "volume_radius": {case.volume_radius},
  "image_offset_n_std": {case.image_offset_n_std},
  "percent_outliers": {case.percent_outliers},
  "max_iter": {case.max_iter},
  "particle_diameter_ang": {particle_diameter}
}}
JSON

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv -l 60 > "${{CASE_ROOT}}/gpu_monitor.csv" &
MONITOR_PID="$!"
trap 'kill "${{MONITOR_PID}}" 2>/dev/null || true' EXIT

mapfile -t PDBS < <(find {q(case.pdb_dir)} -maxdepth 1 -type f -name '*.pdb' | sort)
if [[ "${{#PDBS[@]}}" -lt {required_pdb_count} ]]; then
  echo "Need {required_pdb_count} PDB files under {case.pdb_dir}, found ${{#PDBS[@]}}" >&2
  exit 2
fi
rm -f "${{SUB_PDB_DIR}}"/*.pdb
for ((i=0; i<{case.n_classes}; i++)); do
  src="${{PDBS[$i]}}"
  ln -sf "${{src}}" "${{SUB_PDB_DIR}}/$(printf '%03d_%s' "$i" "$(basename "${{src}}")")"
done
{outlier_pdb_setup}

flock {q(REPO_ROOT / ".pixi" / "install-recovar.lock")} bash -lc 'pixi run --frozen install-recovar'
PIXI_PY="$(pixi run --frozen which python)"
"${{PIXI_PY}}" recovar/relion_bind/build.py
{build_cuda_lib_command(scratch_dir=scratch_dir)}
"${{PIXI_PY}}" - <<'PY'
import os
import pathlib
import jax
import recovar
import recovar.cuda_backproject as cb
from recovar.relion_bind import _relion_bind_core as relion_bind

repo = pathlib.Path.cwd().resolve()
relion_bind_file = pathlib.Path(relion_bind.__file__).resolve()
external_bind_dir = os.environ.get("RECOVAR_RELION_BIND_BUILD_DIR")
external_bind_root = pathlib.Path(external_bind_dir).resolve() if external_bind_dir else None
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert str(relion_bind_file).startswith(str(repo) + "/") or (
    external_bind_root is not None
    and str(relion_bind_file).startswith(str(external_bind_root) + "/")
)
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
print("jax.devices() =", jax.devices())
assert any(getattr(d, "platform", "") in {"gpu", "cuda"} for d in jax.devices())
assert cb.cuda_available(), cb.cuda_unavailable_error()
print("case provenance/cuda gate ok")
PY

echo "=== Prepare K-class dataset: {case.name} ==="
"${{PIXI_PY}}" -m scripts.prepare_cryobench_pdb_multiclass_relion_parity_benchmark \\
  --pdb-dir "${{SUB_PDB_DIR}}" \\
  --output-dir "${{DATA_DIR}}" \\
  --n-images {case.n_images} \\
  --grid-size {case.grid_size} \\
  --noise-level {case.noise_level} \\
  --noise-model {q(case.noise_model)} \\
  --dataset-params-option {q(case.dataset_params_option)} \\
  --class-distribution {q(case.class_distribution)} \\
  --pdb-bfactor {case.pdb_bfactor} \\
  --init-radius {case.init_radius} \\
  --noise-scale-std {case.noise_scale_std} \\
  --contrast-std {case.contrast_std} \\
  --volume-radius {case.volume_radius} \\
  --image-offset-n-std {case.image_offset_n_std} \\
  --percent-outliers {case.percent_outliers} \\
{outlier_prepare_arg}{noise_rng_lines}  --relion-normalize \\
  {streaming_flag} \\
  --streaming-chunk-size {case.streaming_chunk_size} \\
  --disc-type cubic \\
  --seed {case.seed} \\
  2>&1 | tee "${{CASE_ROOT}}/prepare.log"

echo "=== Run RELION Class3D: {case.name} ==="
RELION_START="$(date +%s)"
set +e
(
  unset LD_LIBRARY_PATH
  if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
  fi
  export PS1="${{PS1:-}}"
  set +u
  module load {q(relion_module)}
  set -u
  export CUDA_VISIBLE_DEVICES=0
  RELION_TMPDIR="${{SLURM_TMPDIR:-/tmp/${{USER:-mg6942}}/relion_${{SLURM_JOB_ID:-manual}}_{case.index}_{case.name}}}"
  mkdir -p "${{RELION_TMPDIR}}"
  export TMPDIR="${{RELION_TMPDIR}}"
  export TMP="${{RELION_TMPDIR}}"
  export TEMP="${{RELION_TMPDIR}}"
  export OMPI_MCA_orte_tmpdir_base="${{RELION_TMPDIR}}"
  export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
  cd "${{DATA_DIR}}"
  RELION_CTF_ARGS=(--ctf)
  if [[ {q(case.dataset_params_option)} == "noctf" ]]; then
    RELION_CTF_ARGS=()
  fi
  ITER_PADDED="$(printf "%03d" {case.max_iter})"
  if [[ ! -s "${{RELION_DIR}}/run_it${{ITER_PADDED}}_model.star" ]]; then
    rm -f "${{RELION_DISPATCH_LOG}}" "${{RELION_DISPATCH_SCHEDULE}}"
    export RELION_DISPATCH_LOG
    mpirun -n {relion_mpi_ranks} {q(relion_refine_mpi)} \\
      --i particles.star \\
      --ref reference_init_classes_relion.star \\
      --o "${{RELION_DIR}}/run" \\
      --iter {case.max_iter} \\
      --tau2_fudge 4 \\
      --particle_diameter {particle_diameter:g} \\
	      --K {case.n_classes} \\
	      --flatten_solvent \\
	      --zero_mask \\
	      --firstiter_cc \\
	      "${{RELION_CTF_ARGS[@]}}" \\
	      --norm \\
	      --scale \\
      --sym C1 \\
      --oversampling 1 \\
      --healpix_order 1 \\
      --offset_range 6 \\
      --offset_step 2 \\
      --pad 2 \\
      --pool {relion_pool} \\
      --dont_combine_weights_via_disc \\
      --random_seed {case.seed} \\
      --gpu 0 \\
      --j 4
  else
    echo "Reusing RELION output in ${{RELION_DIR}}"
    if [[ ! -s "${{RELION_DISPATCH_SCHEDULE}}" ]]; then
      echo "Refusing to reconstruct strict ownership from a loose/stale dispatch log." >&2
      echo "A reused RELION run must already have its content-bound dispatch_schedule.npz." >&2
      exit 2
    fi
  fi
) 2>&1 | tee "${{CASE_ROOT}}/relion_class3d.log"
RELION_STATUS="${{PIPESTATUS[0]}}"
set -e
RELION_END="$(date +%s)"
cat > "${{RELION_DIR}}/slurm_walltime.json" <<JSON
{{"slurm_job_id":"${{SLURM_JOB_ID}}","start_epoch":${{RELION_START}},"end_epoch":${{RELION_END}},"external_wall_s":$((RELION_END - RELION_START)),"exit_status":${{RELION_STATUS}}}}
JSON
if [[ "${{RELION_STATUS}}" -ne 0 ]]; then
  exit "${{RELION_STATUS}}"
fi
if [[ ! -s "${{RELION_DISPATCH_SCHEDULE}}" ]]; then
  if [[ ! -s "${{RELION_DISPATCH_LOG}}" ]]; then
    echo "Strict K>1 parity requires a same-run dynamic dispatch capture." >&2
    echo "The selected RELION executable did not write ${{RELION_DISPATCH_LOG}} via RELION_DISPATCH_LOG." >&2
    exit 2
  fi
  "${{PIXI_PY}}" -m scripts.build_relion_dispatch_schedule \\
    --dispatch-log "${{RELION_DISPATCH_LOG}}" \\
    --output "${{RELION_DISPATCH_SCHEDULE}}" \\
    --n-particles {case.n_images} \\
    --n-followers {relion_mpi_ranks - 1} \\
    --pool-size {relion_pool * 4} \\
    --random-seed {case.seed} \\
    --oracle-dir "${{RELION_DIR}}"
fi

echo "=== Run RECOVAR K-class refinement: {case.name} ==="
START_EPOCH="$(date +%s)"
set +e
"${{PIXI_PY}}" -m scripts.run_full_refinement \\
  --data_dir "${{DATA_DIR}}" \\
  --output "${{RECOVAR_DIR}}" \\
  --max_iter {case.max_iter} \\
  --n_classes {case.n_classes} \\
  --healpix_order 1 \\
  --offset_range 6 \\
	  --offset_step 2 \\
	  --adaptive_oversampling 1 \\
	  --init_resolution 30.0 \\
	  --firstiter_cc \\
	  --image_batch_size {image_batch_size} \\
	  --rotation_block_size {rotation_block_size} \\
  --seed {case.seed} \\
  --relion_optimiser "${{RELION_DIR}}/run_it000_optimiser.star" \\
  --relion_init_dir "${{RELION_DIR}}" \\
  --perturb_replay_relion_dir "${{RELION_DIR}}" \\
  --relion-dispatch-schedule "${{RELION_DISPATCH_SCHEDULE}}" \\
  --particle_diameter_ang {particle_diameter:g} \\
  --tau2_fudge 4.0 \\
  --benchmark_ledger_json "${{RECOVAR_DIR}}/benchmark_ledger.json" \\
  --timing_dir "${{RECOVAR_DIR}}/timing" \\
  2>&1 | tee "${{RECOVAR_DIR}}/run_full_refinement.log"
STATUS="${{PIPESTATUS[0]}}"
set -e
END_EPOCH="$(date +%s)"
cat > "${{RECOVAR_DIR}}/slurm_walltime.json" <<JSON
{{"slurm_job_id":"${{SLURM_JOB_ID}}","start_epoch":${{START_EPOCH}},"end_epoch":${{END_EPOCH}},"external_wall_s":$((END_EPOCH - START_EPOCH)),"exit_status":${{STATUS}}}}
JSON
if [[ "${{STATUS}}" -ne 0 ]]; then
  exit "${{STATUS}}"
fi

echo "=== Evaluate K-class GT metrics: {case.name} ==="
ITER_PADDED="$(printf "%03d" {case.max_iter})"
REC_ARGS=()
REL_ARGS=()
GT_ARGS=()
for class_no in $(seq -f "%03g" 1 {case.n_classes}); do
  REC_ARGS+=(--volume "${{RECOVAR_DIR}}/final_class${{class_no}}.mrc")
  REL_ARGS+=(--volume "${{RELION_DIR}}/run_it${{ITER_PADDED}}_class${{class_no}}.mrc")
  GT_ARGS+=(--gt_volume "${{DATA_DIR}}/reference_gt_class${{class_no}}.mrc")
done
"${{PIXI_PY}}" -m scripts.evaluate_kclass_gt \\
  "${{REC_ARGS[@]}}" \\
  "${{GT_ARGS[@]}}" \\
  --label RECOVAR \\
  --volume_frame recovar \\
  --gt_frame recovar \\
  --gt_align_healpix_order 2 \\
  {refine_orders_args} \\
  --output_json "${{CASE_ROOT}}/kclass_gt_fsc.json" \\
  2>&1 | tee "${{CASE_ROOT}}/evaluate_kclass_gt.log"
"${{PIXI_PY}}" -m scripts.evaluate_kclass_gt \\
  "${{REL_ARGS[@]}}" \\
  "${{GT_ARGS[@]}}" \\
  --label RELION \\
  --volume_frame relion \\
  --gt_frame recovar \\
  --gt_align_healpix_order 2 \\
  {refine_orders_args} \\
  --output_json "${{CASE_ROOT}}/relion_kclass_gt_fsc.json" \\
  2>&1 | tee "${{CASE_ROOT}}/relion_evaluate_kclass_gt.log"
"""
    script.write_text(text)
    script.chmod(0o755)
    return script


def write_summary_script(
    *,
    scratch_dir: Path,
    jobs_dir: Path,
    account: str,
    partition: str,
    constraint: str,
    dependency: str,
    tracked_jobs: list[str],
) -> Path:
    script = jobs_dir / "em_kclass_matrix_summary.sh"
    text = f"""#!/usr/bin/env bash
#SBATCH --job-name=em_kclass_summary
#SBATCH --output={q(scratch_dir / "em_kclass_matrix_summary.out")}
#SBATCH --error={q(scratch_dir / "em_kclass_matrix_summary.err")}
#SBATCH --partition={partition}
#SBATCH --account={account}
{sbatch_directive("--constraint", constraint)}
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --dependency={dependency}

set -euo pipefail
cd {q(REPO_ROOT)}
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export RECOVAR_DISABLE_CUDA=1
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu
export PIXI_FROZEN=true
export TMPDIR={q(scratch_dir)}/tmp/em_kclass_matrix_summary_${{SLURM_JOB_ID}}
export PIXI_HOME={q(scratch_dir)}/pixi_home/em_kclass_matrix_summary_${{SLURM_JOB_ID}}
export RATTLER_CACHE_DIR={q(scratch_dir)}/rattler_cache/em_kclass_matrix_summary_${{SLURM_JOB_ID}}
export RECOVAR_JAX_CACHE_DIR={q(scratch_dir)}/jax_cache
export JAX_COMPILATION_CACHE_DIR="${{RECOVAR_JAX_CACHE_DIR}}"
mkdir -p "${{TMPDIR}}" "${{PIXI_HOME}}" "${{RATTLER_CACHE_DIR}}" "${{RECOVAR_JAX_CACHE_DIR}}"

echo "=== EM K-class robustness matrix summary ==="
echo "Repo: {REPO_ROOT}"
echo "HEAD: $(git rev-parse HEAD)"
echo "Branch: $(git symbolic-ref --short HEAD || echo '<detached>')"
echo "Scratch: {scratch_dir}"
echo
for job_id in {' '.join(tracked_jobs)}; do
  sacct -j "${{job_id}}" -X -o JobID,JobName%40,State,Elapsed,MaxRSS,ReqMem,AllocTRES || true
done
echo
pixi run --frozen python -m scripts.summarize_em_robustness_matrix \\
  {q(scratch_dir)} \\
  --output-markdown {q(scratch_dir / "em_kclass_robustness_summary.md")} \\
  --output-json {q(scratch_dir / "em_kclass_robustness_summary.json")} \\
  --dedupe-case-reruns
tail -200 {q(scratch_dir / "em_kclass_robustness_summary.md")} || true
"""
    script.write_text(text)
    script.chmod(0o755)
    return script


def submit(script: Path, *, dry_run: bool, extra_args: list[str] | None = None) -> str:
    if dry_run:
        print("DRY-RUN sbatch", *(extra_args or []), script)
        return "DRYRUN"
    cmd = ["sbatch", "--parsable", *(extra_args or []), str(script)]
    env = os.environ.copy()
    for name in (
        "SBATCH_ACCOUNT",
        "SBATCH_PARTITION",
        "SBATCH_CONSTRAINT",
        "SBATCH_GRES",
        "SBATCH_GPUS",
        "SBATCH_GPUS_PER_NODE",
    ):
        env.pop(name, None)
    return subprocess.check_output(cmd, text=True, env=env).strip()


def git_text(*args: str, default: str = "<unknown>") -> str:
    proc = subprocess.run(["git", "-C", str(REPO_ROOT), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    text = proc.stdout.strip()
    if proc.returncode != 0 or not text:
        return default
    return text


def main() -> int:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"em_kclass_robustness_{timestamp}_{os.getpid()}"
    scratch_dir = args.scratch_dir or Path(
        os.environ.get(
            "EM_KCLASS_MATRIX_SCRATCH_DIR",
            f"/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/{run_id}",
        )
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    (scratch_dir / "SAFE_TO_DELETE").touch()
    jobs_dir = scratch_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (scratch_dir / "tmp").mkdir(exist_ok=True)

    cases = selected_cases(args)
    account = os.environ.get("SBATCH_ACCOUNT", "gilles")
    partition = os.environ.get("SBATCH_PARTITION", "cryoem")
    summary_partition = args.summary_partition or "cpu"
    constraint = os.environ.get("SBATCH_CONSTRAINT", "")
    summary_constraint = os.environ.get("EM_KCLASS_MATRIX_SUMMARY_CONSTRAINT", "")
    setup_partition = os.environ.get("EM_KCLASS_MATRIX_SETUP_PARTITION", "cpu")
    setup_constraint = os.environ.get("EM_KCLASS_MATRIX_SETUP_CONSTRAINT", "")
    setup_gres = os.environ.get("EM_KCLASS_MATRIX_SETUP_GRES", "")
    exclusive = os.environ.get("EM_KCLASS_MATRIX_EXCLUSIVE", "0") != "0"
    cuda_module = os.environ.get("CUDA_MODULE", "cudatoolkit/12.8")
    relion_module = os.environ.get("RELION_MODULE", "relion/5.0.1/gcc-11.5.0-gpu")
    relion_refine_mpi = os.environ.get("EM_KCLASS_MATRIX_RELION_REFINE_MPI", "").strip()
    if not relion_refine_mpi:
        raise SystemExit(
            "EM_KCLASS_MATRIX_RELION_REFINE_MPI must name an absolute, executable "
            "RELION build instrumented to honor RELION_DISPATCH_LOG; the stock binary "
            "cannot supply strict dynamic-dispatch parity."
        )
    relion_refine_path = Path(relion_refine_mpi).expanduser().resolve()
    if not relion_refine_path.is_file() or not os.access(relion_refine_path, os.X_OK):
        raise SystemExit(
            "EM_KCLASS_MATRIX_RELION_REFINE_MPI is not an executable file: "
            f"{relion_refine_path}"
        )
    relion_refine_mpi = str(relion_refine_path)
    relion_mpi_ranks = int(os.environ.get("RELION_MPI_RANKS", "3"))
    relion_pool = int(os.environ.get("EM_KCLASS_MATRIX_RELION_POOL", "3"))
    particle_diameter = float(os.environ.get("EM_KCLASS_MATRIX_PARTICLE_DIAMETER", "380"))
    image_batch_size = int(os.environ.get("KCLASS_IMAGE_BATCH_SIZE", "50"))
    rotation_block_size = int(os.environ.get("KCLASS_ROTATION_BLOCK_SIZE", "2000"))
    gt_align_refine_orders = os.environ.get("EM_KCLASS_MATRIX_GT_ALIGN_REFINE_ORDERS", "3")
    noise_rng_batch_size = os.environ.get("EM_KCLASS_MATRIX_NOISE_RNG_BATCH_SIZE", "")
    max_iter_override_for_env = getattr(args, "max_iter_override", None)
    if max_iter_override_for_env is None:
        max_iter_override_for_env = os.environ.get("EM_KCLASS_MATRIX_MAX_ITER", "")
    time_limit_override_for_env = getattr(args, "time_limit_override", None) or os.environ.get("EM_KCLASS_MATRIX_TIME_LIMIT", "")
    seed_override_for_env = getattr(args, "seed_override", None)
    if seed_override_for_env is None:
        seed_override_for_env = os.environ.get("EM_KCLASS_MATRIX_SEED", "")
    seed_offset_for_env = getattr(args, "seed_offset", None)
    if seed_offset_for_env is None:
        seed_offset_for_env = os.environ.get("EM_KCLASS_MATRIX_SEED_OFFSET", "")
    cuda_lib = scratch_dir / "cuda" / "libcuda_backproject.so"

    print("EM K-class robustness matrix launcher")
    print(f"Repo: {REPO_ROOT}")
    print(f"HEAD: {git_text('rev-parse', 'HEAD')}")
    print(f"Branch: {git_text('symbolic-ref', '--short', 'HEAD', default='<detached>')}")
    print(f"Scratch: {scratch_dir}")
    print(f"Cases: {', '.join(str(case.index) for case in cases)}")
    print(f"Partition/account: {partition}/{account}")
    print(f"Setup partition: {setup_partition}")
    print(f"Setup constraint: {setup_constraint or '<none>'}")
    print(f"Summary partition: {summary_partition}")
    print(f"Summary constraint: {summary_constraint or '<none>'}")
    print(f"Constraint: {constraint or '<none>'}")
    print(f"RELION module: {relion_module}")
    print(f"RELION dispatch-capture executable: {relion_refine_mpi}")
    if max_iter_override_for_env:
        print(f"Max iter override: {max_iter_override_for_env}")
    if time_limit_override_for_env:
        print(f"Time limit override: {time_limit_override_for_env}")
    if seed_override_for_env:
        print(f"Seed override: {seed_override_for_env}")
    if seed_offset_for_env:
        print(f"Seed offset: {seed_offset_for_env}")

    case_table = scratch_dir / "case_table.tsv"
    header = [
        "index",
        "name",
        "n_classes",
        "n_images",
        "grid",
        "noise_level",
        "noise_model",
        "dataset_params_option",
        "seed",
        "pdb_bfactor",
        "init_radius",
        "noise_scale_std",
        "contrast_std",
        "volume_radius",
        "image_offset_n_std",
        "percent_outliers",
        "max_iter",
        "class_distribution",
        "time_limit",
        "mem",
        "case_root",
        "script",
        "job_id",
    ]
    case_table.write_text("|".join(header) + "\n")

    setup_script = write_setup_script(
        scratch_dir=scratch_dir,
        jobs_dir=jobs_dir,
        cuda_lib=cuda_lib,
        account=account,
        partition=setup_partition,
        constraint=setup_constraint,
        setup_gres=setup_gres,
        cuda_module=cuda_module,
    )
    setup_job = submit(setup_script, dry_run=args.dry_run)
    tracked_jobs = [setup_job]
    case_jobs: list[str] = []

    for case in cases:
        if not case.pdb_dir.exists():
            raise SystemExit(f"PDB directory missing for case {case.index}: {case.pdb_dir}")
        script = write_case_script(
            case=case,
            scratch_dir=scratch_dir,
            jobs_dir=jobs_dir,
            cuda_lib=cuda_lib,
            account=account,
            partition=partition,
            constraint=constraint,
            exclusive=exclusive,
            cuda_module=cuda_module,
            relion_module=relion_module,
            relion_refine_mpi=relion_refine_mpi,
            relion_mpi_ranks=relion_mpi_ranks,
            relion_pool=relion_pool,
            particle_diameter=particle_diameter,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
            gt_align_refine_orders=gt_align_refine_orders,
            noise_rng_batch_size=noise_rng_batch_size,
        )
        job_id = submit(script, dry_run=args.dry_run, extra_args=[f"--dependency=afterok:{setup_job}"] if not args.dry_run else None)
        tracked_jobs.append(job_id)
        case_jobs.append(job_id)
        case_root = scratch_dir / "cases" / f"{case.index}_{case.name}"
        with case_table.open("a", encoding="utf-8") as f:
            f.write("|".join([*case.row_fields, str(case_root), str(script), job_id]) + "\n")
        print(f"Case {case.index} {case.name}: {job_id}")

    dependency = "afterany:" + ":".join(tracked_jobs)
    summary_script = write_summary_script(
        scratch_dir=scratch_dir,
        jobs_dir=jobs_dir,
        account=account,
        partition=summary_partition,
        constraint=summary_constraint,
        dependency=dependency,
        tracked_jobs=tracked_jobs,
    )
    summary_job = submit(summary_script, dry_run=args.dry_run)
    print(f"Setup job: {setup_job}")
    print(f"Case jobs: {' '.join(case_jobs)}")
    print(f"Summary job: {summary_job}")
    print(f"Scratch: {scratch_dir}")

    (scratch_dir / "submission.env").write_text(
        "\n".join(
            [
                f"REPO_ROOT={REPO_ROOT}",
                f"SCRATCH_DIR={scratch_dir}",
                f"EM_KCLASS_MATRIX_SETUP_PARTITION={setup_partition}",
                f"EM_KCLASS_MATRIX_SETUP_CONSTRAINT={setup_constraint}",
                f"EM_KCLASS_MATRIX_SUMMARY_PARTITION={summary_partition}",
                f"EM_KCLASS_MATRIX_SUMMARY_CONSTRAINT={summary_constraint}",
                f"SETUP_JOB_ID={setup_job}",
                f"CASE_JOB_IDS={' '.join(case_jobs)}",
                f"SUMMARY_JOB_ID={summary_job}",
                f"CASE_TABLE={case_table}",
                f"SBATCH_PARTITION={partition}",
                f"SBATCH_ACCOUNT={account}",
                f"SBATCH_CONSTRAINT={constraint}",
                f"RELION_MODULE={relion_module}",
                f"EM_KCLASS_MATRIX_RELION_REFINE_MPI={relion_refine_mpi}",
                f"RELION_MPI_RANKS={relion_mpi_ranks}",
                f"KCLASS_IMAGE_BATCH_SIZE={image_batch_size}",
                f"KCLASS_ROTATION_BLOCK_SIZE={rotation_block_size}",
                f"EM_KCLASS_MATRIX_GT_ALIGN_REFINE_ORDERS={gt_align_refine_orders}",
                f"EM_KCLASS_MATRIX_MAX_ITER={max_iter_override_for_env}",
                f"EM_KCLASS_MATRIX_TIME_LIMIT={time_limit_override_for_env}",
                f"EM_KCLASS_MATRIX_SEED={seed_override_for_env}",
                f"EM_KCLASS_MATRIX_SEED_OFFSET={seed_offset_for_env}",
                *[f"{name}={os.environ.get(name, '')}" for name in RECOVAR_OPTIONAL_ENV_PASSTHROUGH],
            ]
        )
        + "\n",
    )

    if args.watch and not args.dry_run:
        job_list = ",".join([*tracked_jobs, summary_job])
        while subprocess.run(["squeue", "-h", "-j", summary_job], text=True, stdout=subprocess.PIPE).stdout.strip():
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            subprocess.run(["squeue", "-j", job_list], check=False)
            time.sleep(60)
        print((scratch_dir / "em_kclass_matrix_summary.out").read_text(errors="replace")[-12000:])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
