#!/usr/bin/env python3
"""Plan or submit matched EMPIAR-10202 set-6 I1 refinements.

The default action is non-mutating outside the newly created run/runtime roots:
it validates every sealed input and native artifact, writes one shared balanced
64-particle STAR, and emits four audited Slurm scripts.  ``--submit`` launches
the two smoke jobs and makes both full jobs depend on both smoke jobs succeeding.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import mrcfile
import numpy as np

# Launch planning and integrity checks are CPU-only.  Importing the RECOVAR
# STAR reader otherwise initializes every visible login-node GPU through the
# package import, even though this process never performs a JAX computation.
# The generated Slurm scripts explicitly clear all JAX_* variables before
# starting either refinement, so this cannot change the submitted engines.
os.environ["JAX_PLATFORMS"] = "cpu"

from recovar.data_io.starfile import read_star

SCHEMA = "recovar.empiar10202_set6_i1_matched_launch.v1"
PREPARATION_SCHEMA = "recovar.empiar10202_set6_i1_preparation.v1"
DEFAULT_PREPARATION_MANIFEST = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
    "empiar10202_set6_i1_20260830/preparation_manifest.json"
)
ALLOWED_RUN_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/em_work/codex")
RUNTIME_PARENT = ALLOWED_RUN_ROOT / "runtime"

PARTICLE_COUNT = 30_515
HALF_COUNTS = {1: 15_258, 2: 15_257}
HALF_ASSIGNMENT_SHA256 = "1a06a8e0fccc553a99ecbdd24b21115b2b20b14cd1ae267310c6d4fe449d054f"
PREPARED_STAR_SHA256 = "d66afb3001e6e43463fb699804fe8b50f8cef1f2ccd9730f5275955b6be7b512"
PARTICLE_STACK_SHA256 = "8eecf0fbf8e645ac51feff278a86e43e7e4be117921333dc6d3e22e52a628453"
PARTICLE_STACK_SIZE_BYTES = 78_118_401_024
BOX_SIZE = 800
SYMMETRY = "I1"
SYMMETRY_OPERATOR_COUNT = 60
SYMMETRY_OPERATORS_SHA256 = "093a0876b93610ec141c87840ae3ff4dc4491b27dec87143558358ef556557b8"
SYMMETRY_OPERATOR_ORDER = (
    "identity at index 0, then RELION SymList::get_matrices(isym) for isym=0..58"
)
SYMMETRY_LEFT_OPERATOR_POLICY = (
    "all left operators are identity and are not applied in BPref reconstruction"
)
SYMMETRY_HASH_ENCODING = (
    "float64 rounded to 12 decimals, stacked [left,right], little-endian C-order bytes"
)
SEED = 10_202

RELION_MPI = Path(
    "/projects/MOLBIO/local/relion-5.0.1-gcc-11.5.0-cuda-12.6-rhel9-arch80/"
    "bin/relion_refine_mpi"
)
RELION_MPI_SHA256 = "92cf3ba54038d5e162e238b952fe88f1414f440d4e6cba23bc4b097428087b4a"
RELION_SOURCE = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/"
    "relion_d476e6f_clean_binding_20260713"
)
RELION_SOURCE_COMMIT = "d476e6f6a4f1f37627c06ace5227fc374c0c2b05"
RELION_BIND_DIR = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/"
    "pr158_symmetry_build_20260830/relion_bind"
)
RELION_BIND_SHA256 = "82b0a8cf2c189463f9cf0181099f4e92ce4365cce3819463c27af44b3c1014a2"
CUDA_LIBRARY = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/"
    "pr158_symmetry_h100_gate_r3_20260830/libcuda_backproject_sm90.so"
)
CUDA_LIBRARY_SHA256 = "e637d87a8ed9fff1c5b7007747e3a6781db021966e099e9a18023b99e7e67fa4"
CUDA_SOURCE_SHA256 = "da15a71d23a35e541cc95fd25279d6b1f80c1e7cf5be451497c6756d1b621a4b"

RELION_CUDA_LIB_DIR = Path("/usr/local/cuda-12/targets/x86_64-linux/lib")
RELION_MPI_LIB_DIR = Path("/usr/local/openmpi/cuda-12.6/4.1.6/gcc/lib64")


@dataclass(frozen=True)
class PreparedInputs:
    preparation_manifest: Path
    preparation_manifest_sha256: str
    particle_star: Path
    particle_star_sha256: str
    particle_stack: Path
    particle_stack_sha256: str
    particle_stack_size_bytes: int
    recovar_reference: Path
    recovar_reference_sha256: str
    relion_reference: Path
    relion_reference_sha256: str
    canonical_reference_sha256: str


@dataclass(frozen=True)
class Resources:
    nodes: int
    ntasks: int
    cpus_per_task: int
    memory: str
    h100_gpus: int
    walltime: str


@dataclass(frozen=True)
class RunSpec:
    key: str
    engine: str
    phase: str
    resources: Resources
    data_dir: Path
    output_dir: Path
    command: tuple[str, ...]
    expected_outputs: tuple[Path, ...]
    depends_on: tuple[str, ...]


def sha256_file(path: Path, *, limit_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = limit_bytes
    with path.open("rb") as stream:
        while remaining is None or remaining > 0:
            size = 8 * 1024 * 1024 if remaining is None else min(8 * 1024 * 1024, remaining)
            chunk = stream.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    if remaining:
        raise ValueError(f"{path} is shorter than the requested {limit_bytes} bytes")
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON value using the launch manifest's canonical encoding."""

    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def require_sha256(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected}")
    return observed


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_subject(repo: Path, expected_commit: str) -> dict[str, str]:
    repo = repo.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("--subject-commit must be a complete lowercase 40-character Git SHA")
    observed = _git(repo, "rev-parse", "HEAD")
    if observed != expected_commit:
        raise ValueError(f"subject HEAD mismatch: {observed} != {expected_commit}")
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError(f"subject worktree is not clean:\n{status}")
    python = repo / ".pixi/envs/default/bin/python"
    driver = repo / "scripts/run_full_refinement.py"
    cuda_source = repo / "recovar/cuda/cuda_backproject.cu"
    for path in (python, driver, cuda_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    require_sha256(cuda_source, CUDA_SOURCE_SHA256)
    return {
        "repo": str(repo),
        "commit": observed,
        "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "tree_clean": True,
        "diff_sha256": hashlib.sha256(b"").hexdigest(),
        "python": str(python),
        "driver": str(driver),
        "driver_sha256": sha256_file(driver),
        "cuda_source": str(cuda_source),
        "cuda_source_sha256": CUDA_SOURCE_SHA256,
    }


def _half_assignment_sha256(table: Any) -> str:
    values = table["_rlnRandomSubset"].to_numpy(dtype=np.uint8)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _canonical_relion_array_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with mrcfile.mmap(path, mode="r", permissive=False) as mrc:
        raw = mrc.data
        for canonical_z in range(raw.shape[2]):
            plane = np.ascontiguousarray(-raw[:, :, canonical_z].T, dtype="<f4")
            digest.update(memoryview(plane).cast("B"))
    return digest.hexdigest()


def _canonical_recovar_array_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with mrcfile.mmap(path, mode="r", permissive=False) as mrc:
        raw = mrc.data
        for canonical_z in range(raw.shape[2]):
            canonical = np.ascontiguousarray(raw[:, :, canonical_z].T, dtype="<f4")
            digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def validate_preparation_manifest(path: Path) -> PreparedInputs:
    path = path.resolve()
    manifest_sha = sha256_file(path)
    payload = json.loads(path.read_text())
    if payload.get("schema") != PREPARATION_SCHEMA:
        raise ValueError(f"unexpected preparation schema: {payload.get('schema')!r}")
    if payload.get("dataset") != "EMPIAR-10202" or payload.get("image_set") != 6:
        raise ValueError("preparation manifest is not EMPIAR-10202 set 6")
    symmetry = payload.get("symmetry", {})
    if symmetry.get("requested_label") != SYMMETRY or symmetry.get("relion_label") != SYMMETRY:
        raise ValueError("preparation manifest must use explicit I1, never bare I/I2")
    if symmetry.get("recovar_label") != SYMMETRY or symmetry.get("operator_count") != 60:
        raise ValueError("preparation manifest has the wrong RECOVAR I1 operator contract")
    contract = payload.get("refinement_contract", {})
    expected_contract = {
        "k": 1,
        "autonomous_refinement": True,
        "fixed_poses": False,
        "local_only": False,
        "replay": False,
        "oracle": False,
        "same_particles_and_order": True,
        "same_metadata": True,
        "same_half_assignment": True,
        "same_initial_reference_canonical_array": True,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"preparation refinement contract {key}={contract.get(key)!r} != {expected!r}")

    inputs = payload["input_contract"]
    prepared = inputs["prepared_star"]
    star = Path(prepared["path"]).resolve()
    if prepared.get("sha256") != PREPARED_STAR_SHA256:
        raise ValueError("preparation manifest does not pin the authoritative STAR digest")
    require_sha256(star, PREPARED_STAR_SHA256)
    if Path(inputs["authoritative_particle_star"]).resolve() != star:
        raise ValueError("authoritative_particle_star does not name prepared_star.path")
    particles, optics = read_star(str(star))
    if optics is None or len(optics) != 1 or len(particles) != PARTICLE_COUNT:
        raise ValueError("authoritative STAR must have one optics row and 30,515 particle rows")
    halves = particles["_rlnRandomSubset"].to_numpy(dtype=np.int64)
    observed_counts = {half: int(np.count_nonzero(halves == half)) for half in (1, 2)}
    if observed_counts != HALF_COUNTS or not np.isin(halves, (1, 2)).all():
        raise ValueError(f"authoritative STAR half counts changed: {observed_counts}")
    if _half_assignment_sha256(particles) != HALF_ASSIGNMENT_SHA256:
        raise ValueError("authoritative STAR half assignment changed")

    stack_info = inputs["particle_stack"]
    stack = Path(stack_info["path"]).resolve()
    if stack_info.get("sha256") != PARTICLE_STACK_SHA256:
        raise ValueError("preparation manifest has the wrong particle-stack SHA-256")
    if stack_info.get("size_bytes") != PARTICLE_STACK_SIZE_BYTES:
        raise ValueError("preparation manifest has the wrong particle-stack size")
    if not stack.is_file() or stack.stat().st_size != PARTICLE_STACK_SIZE_BYTES:
        raise ValueError(f"particle-stack file is absent or has changed size: {stack}")
    require_sha256(stack, PARTICLE_STACK_SHA256)

    reference = payload["initial_reference"]
    relion_info = reference["relion_file"]
    recovar_info = reference["recovar_frame_file"]
    relion_reference = Path(relion_info["path"]).resolve()
    recovar_reference = Path(recovar_info["path"]).resolve()
    relion_sha = require_sha256(relion_reference, relion_info["sha256"])
    recovar_sha = require_sha256(recovar_reference, recovar_info["sha256"])
    canonical_sha = reference["canonical_array_sha256"]
    if not reference.get("canonical_exact"):
        raise ValueError("preparation manifest does not assert exact canonical references")
    if _canonical_relion_array_sha256(relion_reference) != canonical_sha:
        raise ValueError("RELION reference canonical array no longer matches the manifest")
    if _canonical_recovar_array_sha256(recovar_reference) != canonical_sha:
        raise ValueError("RECOVAR reference canonical array no longer matches the RELION reference")
    return PreparedInputs(
        preparation_manifest=path,
        preparation_manifest_sha256=manifest_sha,
        particle_star=star,
        particle_star_sha256=PREPARED_STAR_SHA256,
        particle_stack=stack,
        particle_stack_sha256=PARTICLE_STACK_SHA256,
        particle_stack_size_bytes=PARTICLE_STACK_SIZE_BYTES,
        recovar_reference=recovar_reference,
        recovar_reference_sha256=recovar_sha,
        relion_reference=relion_reference,
        relion_reference_sha256=relion_sha,
        canonical_reference_sha256=canonical_sha,
    )


def validate_native_artifacts() -> dict[str, Any]:
    require_sha256(RELION_MPI, RELION_MPI_SHA256)
    require_sha256(CUDA_LIBRARY, CUDA_LIBRARY_SHA256)
    bind_libraries = sorted(RELION_BIND_DIR.glob("_relion_bind_core*.so"))
    if len(bind_libraries) != 1:
        raise ValueError(f"expected one RELION binding library, found {bind_libraries}")
    bind_library = bind_libraries[0]
    require_sha256(bind_library, RELION_BIND_SHA256)
    source_head = _git(RELION_SOURCE, "rev-parse", "HEAD")
    source_status = _git(RELION_SOURCE, "status", "--porcelain=v1", "--untracked-files=all")
    if source_head != RELION_SOURCE_COMMIT or source_status:
        raise ValueError(
            f"RELION binding source is not sealed: head={source_head} status={source_status!r}"
        )
    for path in (RELION_CUDA_LIB_DIR, RELION_MPI_LIB_DIR):
        if not path.is_dir():
            raise FileNotFoundError(path)
    return {
        "relion_refine_mpi": {"path": str(RELION_MPI), "sha256": RELION_MPI_SHA256},
        "relion_binding": {"path": str(bind_library), "sha256": RELION_BIND_SHA256},
        "relion_binding_source": {
            "path": str(RELION_SOURCE),
            "commit": RELION_SOURCE_COMMIT,
            "tree": _git(RELION_SOURCE, "rev-parse", "HEAD^{tree}"),
        },
        "recovar_cuda_sm90": {"path": str(CUDA_LIBRARY), "sha256": CUDA_LIBRARY_SHA256},
        "relion_runtime_library_dirs": [str(RELION_CUDA_LIB_DIR), str(RELION_MPI_LIB_DIR)],
    }


def _write_star_block(stream: Any, table: Any, name: str) -> None:
    stream.write(f"{name}\n\nloop_\n")
    for index, column in enumerate(table.columns, start=1):
        stream.write(f"{column} #{index}\n")
    for row in table.itertuples(index=False, name=None):
        stream.write(" ".join(str(value) for value in row) + "\n")


def write_balanced_smoke_star(source: Path, destination: Path) -> dict[str, Any]:
    particles, optics = read_star(str(source))
    if optics is None:
        raise ValueError("smoke source STAR must contain an optics table")
    halves = particles["_rlnRandomSubset"].to_numpy(dtype=np.int64)
    selected = np.sort(
        np.concatenate(
            [
                np.flatnonzero(halves == 1)[:32],
                np.flatnonzero(halves == 2)[:32],
            ]
        )
    )
    if selected.size != 64:
        raise ValueError("source STAR does not contain at least 32 particles in each half")
    subset = particles.iloc[selected].reset_index(drop=True)
    destination.parent.mkdir(parents=True, exist_ok=False)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write("# Deterministic balanced EMPIAR-10202 set-6 smoke subset\n\n")
        _write_star_block(stream, optics, "data_optics")
        stream.write("\n")
        _write_star_block(stream, subset, "data_particles")
    reread, reread_optics = read_star(str(destination))
    observed = reread["_rlnRandomSubset"].to_numpy(dtype=np.int64)
    if reread_optics is None or len(reread) != 64:
        raise ValueError("balanced smoke STAR did not round-trip")
    counts = {half: int(np.count_nonzero(observed == half)) for half in (1, 2)}
    if counts != {1: 32, 2: 32}:
        raise ValueError(f"balanced smoke STAR half counts changed: {counts}")
    if not np.array_equal(
        reread["_rlnImageName"].astype(str).to_numpy(),
        particles.iloc[selected]["_rlnImageName"].astype(str).to_numpy(),
    ):
        raise ValueError("balanced smoke STAR changed particle identity/order")
    return {
        "path": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "particle_count": 64,
        "half_counts": {"1": 32, "2": 32},
        "selection_policy": "first 32 source rows in each deposited half, restored to source-row order",
        "source_row_indices_zero_based": selected.tolist(),
        "source_image_names_sha256": hashlib.sha256(
            b"\0".join(value.encode() for value in reread["_rlnImageName"].astype(str))
        ).hexdigest(),
        "half_assignment_sha256": _half_assignment_sha256(reread),
    }


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _recovar_command(
    subject_repo: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    smoke: bool,
) -> tuple[str, ...]:
    command = [
        "srun",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--cpu-bind=none",
        str(subject_repo / ".pixi/envs/default/bin/python"),
        "-m",
        "scripts.run_full_refinement",
        "--data_dir",
        str(data_dir),
        "--output",
        str(output_dir),
        "--init_volume",
        str(data_dir / "reference_init.mrc"),
        "--relion_half_sets",
        str(data_dir / "particles.star"),
    ]
    if smoke:
        command.extend(["--max_iter", "1", "--skip_final_iteration", "--relion_current_sizes", "800"])
    else:
        command.extend(["--max_iter", "50"])
    command.extend(
        [
            "--n_classes",
            "1",
            "--sym",
            "I1",
            "--initial-pose-source",
            "input-star",
            "--healpix_order",
            "3",
            "--max_healpix_order",
            "7",
            "--auto_local_healpix_order",
            "4",
            "--offset_range",
            "5",
            "--offset_step",
            "2",
            "--offset_sigma_angstrom",
            "10",
            "--adaptive_oversampling",
            "1",
            "--max_significants",
            "-1",
            "--tau2_fudge",
            "1",
            "--perturb_factor",
            ".5",
            "--seed",
            str(SEED),
            "--perturb_seed",
            str(SEED),
            "--init_resolution",
            "30",
            "--firstiter_cc",
            "--apply-initial-lowpass",
            "--particle_diameter_ang",
            "300",
            "--width_mask_edge_px",
            "5",
            "--image-fourier-backend",
            "relion_cuda",
            "--image_batch_size",
            "64",
            "--rotation_block_size",
            "8192",
        ]
    )
    return tuple(command)


def _relion_command(data_dir: Path, output_dir: Path, *, smoke: bool) -> tuple[str, ...]:
    command = [
        "srun",
        "--ntasks=3",
        "--ntasks-per-node=3",
        "--cpus-per-task=4",
        "--cpu-bind=none",
        str(RELION_MPI),
        "--i",
        str(data_dir / "particles.star"),
        "--ref",
        str(data_dir / "reference_init_relion.mrc"),
        "--o",
        str(output_dir / "run"),
        "--auto_refine",
        "--split_random_halves",
        "--K",
        "1",
        "--sym",
        "I1",
        "--particle_diameter",
        "300",
        "--maskedge",
        "5",
        "--ini_high",
        "30",
        "--firstiter_cc",
        "--healpix_order",
        "3",
        "--auto_local_healpix_order",
        "4",
        "--offset_range",
        "5",
        "--offset_step",
        "2",
        "--offset",
        "10",
        "--oversampling",
        "1",
        "--perturb",
        ".5",
        "--ctf",
        "--norm",
        "--scale",
        "--zero_mask",
        "--flatten_solvent",
        "--low_resol_join_halves",
        "40",
        "--tau2_fudge",
        "1",
        "--maxsig",
        "-1",
        "--pad",
        "2",
        "--random_seed",
        str(SEED),
    ]
    if smoke:
        command.extend(["--auto_iter_max", "1", "--incr_size", "800"])
    else:
        command.extend(["--auto_iter_max", "50"])
    command.extend(
        [
            "--pool",
            "3",
            "--j",
            "4",
            "--gpu",
            "0:1",
            "--free_gpu_memory",
            "2000",
            "--preread_images",
            "--no_parallel_disc_io",
            "--dont_combine_weights_via_disc",
        ]
    )
    return tuple(command)


def build_run_specs(subject_repo: Path, run_root: Path) -> tuple[RunSpec, ...]:
    smoke_data = run_root / "inputs/smoke"
    full_data = run_root / "inputs/full"
    recovar_smoke = run_root / "outputs/recovar_smoke"
    relion_smoke = run_root / "outputs/relion_smoke"
    recovar_full = run_root / "outputs/recovar_full"
    relion_full = run_root / "outputs/relion_full"
    recovar_smoke_resources = Resources(1, 1, 4, "500G", 1, "12:00:00")
    relion_smoke_resources = Resources(1, 3, 4, "500G", 2, "12:00:00")
    recovar_full_resources = Resources(1, 1, 4, "500G", 1, "5-00:00:00")
    relion_full_resources = Resources(1, 3, 4, "500G", 2, "5-00:00:00")
    return (
        RunSpec(
            "recovar_smoke",
            "recovar",
            "smoke",
            recovar_smoke_resources,
            smoke_data,
            recovar_smoke,
            _recovar_command(subject_repo, smoke_data, recovar_smoke, smoke=True),
            (recovar_smoke / "refinement_results.npz",),
            (),
        ),
        RunSpec(
            "relion_smoke",
            "relion",
            "smoke",
            relion_smoke_resources,
            smoke_data,
            relion_smoke,
            _relion_command(smoke_data, relion_smoke, smoke=True),
            (
                relion_smoke / "run_it001_data.star",
                relion_smoke / "run_it001_half1_class001.mrc",
                relion_smoke / "run_it001_half2_class001.mrc",
            ),
            (),
        ),
        RunSpec(
            "recovar_full",
            "recovar",
            "full",
            recovar_full_resources,
            full_data,
            recovar_full,
            _recovar_command(subject_repo, full_data, recovar_full, smoke=False),
            (
                recovar_full / "refinement_results.npz",
                recovar_full / "final_merged.mrc",
                recovar_full / "final_half1_unfil.mrc",
                recovar_full / "final_half2_unfil.mrc",
            ),
            ("recovar_smoke", "relion_smoke"),
        ),
        RunSpec(
            "relion_full",
            "relion",
            "full",
            relion_full_resources,
            full_data,
            relion_full,
            _relion_command(full_data, relion_full, smoke=False),
            (
                relion_full / "run_class001.mrc",
                relion_full / "run_half1_class001_unfil.mrc",
                relion_full / "run_half2_class001_unfil.mrc",
                relion_full / "run_data.star",
                relion_full / "run_model.star",
            ),
            ("recovar_smoke", "relion_smoke"),
        ),
    )


def _sbatch_header(spec: RunSpec, run_root: Path) -> list[str]:
    resources = spec.resources
    return [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name=10202-s6-{spec.key.replace('_', '-')}",
        "#SBATCH --partition=cryoem",
        "#SBATCH --account=gilles",
        "#SBATCH --qos=della-cryoem",
        "#SBATCH --constraint=h100",
        f"#SBATCH --nodes={resources.nodes}",
        f"#SBATCH --ntasks={resources.ntasks}",
        *(["#SBATCH --ntasks-per-node=3"] if spec.engine == "relion" else []),
        f"#SBATCH --cpus-per-task={resources.cpus_per_task}",
        f"#SBATCH --mem={resources.memory}",
        f"#SBATCH --gres=gpu:h100:{resources.h100_gpus}",
        f"#SBATCH --time={resources.walltime}",
        f"#SBATCH --output={run_root}/logs/{spec.key}-%j.out",
        f"#SBATCH --error={run_root}/logs/{spec.key}-%j.err",
        "",
    ]


def _common_preflight(
    spec: RunSpec,
    *,
    run_root: Path,
    runtime_root: Path,
    subject: dict[str, str],
    inputs: PreparedInputs,
    star_sha256: str,
    bind_library: Path,
) -> list[str]:
    stack_header_sha = sha256_file(inputs.particle_stack, limit_bytes=1024 * 1024)
    lines = [
        "set -euo pipefail",
        f"ROOT={_q(run_root)}",
        f"RUNTIME_BASE={_q(runtime_root)}",
        f"SUBJECT_REPO={_q(subject['repo'])}",
        f"SUBJECT_COMMIT={_q(subject['commit'])}",
        f"DATA_DIR={_q(spec.data_dir)}",
        f"OUTPUT_DIR={_q(spec.output_dir)}",
        f"PYTHON={_q(subject['python'])}",
        f"PIXI_ENV={_q(Path(subject['python']).parent.parent)}",
        f"RELION_BINARY={_q(RELION_MPI)}",
        f"RELION_BIND_DIR={_q(RELION_BIND_DIR)}",
        f"RELION_BIND_LIB={_q(bind_library)}",
        f"CUDA_LIB={_q(CUDA_LIBRARY)}",
        f"RELION_SOURCE={_q(RELION_SOURCE)}",
        f"PARTICLE_STACK={_q(inputs.particle_stack)}",
        "",
        "require_sha() {",
        "    local expected=$1",
        "    local path=$2",
        "    local observed",
        "    observed=$(sha256sum \"${path}\" | awk '{print $1}')",
        "    test \"${observed}\" = \"${expected}\" || {",
        "        echo \"SHA-256 mismatch: ${path}: ${observed} != ${expected}\" >&2",
        "        exit 2",
        "    }",
        "}",
        "",
        "test -f \"${ROOT}/SAFE_TO_DELETE\" -a -f \"${RUNTIME_BASE}/SAFE_TO_DELETE\"",
        "test \"$(git -C \"${SUBJECT_REPO}\" rev-parse HEAD)\" = \"${SUBJECT_COMMIT}\"",
        "test -z \"$(git -C \"${SUBJECT_REPO}\" status --porcelain=v1 --untracked-files=all)\"",
        f"test \"$(git -C \"${{RELION_SOURCE}}\" rev-parse HEAD)\" = {_q(RELION_SOURCE_COMMIT)}",
        "test -z \"$(git -C \"${RELION_SOURCE}\" status --porcelain=v1 --untracked-files=all)\"",
        f"require_sha {_q(RELION_MPI_SHA256)} \"${{RELION_BINARY}}\"",
        f"require_sha {_q(RELION_BIND_SHA256)} \"${{RELION_BIND_LIB}}\"",
        f"require_sha {_q(CUDA_LIBRARY_SHA256)} \"${{CUDA_LIB}}\"",
        f"require_sha {_q(inputs.preparation_manifest_sha256)} {_q(inputs.preparation_manifest)}",
        f"require_sha {_q(star_sha256)} \"${{DATA_DIR}}/particles.star\"",
        f"require_sha {_q(inputs.recovar_reference_sha256)} \"${{DATA_DIR}}/reference_init.mrc\"",
        f"require_sha {_q(inputs.relion_reference_sha256)} \"${{DATA_DIR}}/reference_init_relion.mrc\"",
        f"test \"$(stat -Lc %s \"${{PARTICLE_STACK}}\")\" = {_q(str(inputs.particle_stack_size_bytes))}",
        f"test \"$(head -c 1048576 \"${{PARTICLE_STACK}}\" | sha256sum | awk '{{print $1}}')\" = {_q(stack_header_sha)}",
        "test -z \"$(find \"${OUTPUT_DIR}\" -mindepth 1 -print -quit)\"",
        "",
        "unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE",
        "while IFS='=' read -r variable_name _; do",
        "    case \"${variable_name}\" in",
        "        RECOVAR_*|RELION_*|JAX_*|XLA_*) unset \"${variable_name}\" ;;",
        "    esac",
        "done < <(env)",
        "export PYTHONNOUSERSITE=1",
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "RUNTIME=${RUNTIME_BASE}/${SLURM_JOB_ID}",
        "export TMPDIR=${RUNTIME}/tmp",
        "export PIXI_HOME=${RUNTIME}/pixi_home",
        "export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache",
        "export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/jax_cache",
        "export PYTHONPYCACHEPREFIX=${RUNTIME}/pycache",
        "mkdir -p \"${TMPDIR}\" \"${PIXI_HOME}\" \"${RATTLER_CACHE_DIR}\" \"${JAX_COMPILATION_CACHE_DIR}\" \"${PYTHONPYCACHEPREFIX}\"",
        "touch \"${RUNTIME}/SAFE_TO_DELETE\"",
        "scontrol show job -o \"${SLURM_JOB_ID}\" > \"${ROOT}/provenance/scontrol_${SLURM_JOB_ID}.txt\"",
        f"test \"${{SLURM_NTASKS}}\" = {_q(str(spec.resources.ntasks))}",
        f"test \"${{SLURM_CPUS_PER_TASK}}\" = {_q(str(spec.resources.cpus_per_task))}",
        "mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)",
        f"test \"${{#GPU_NAMES[@]}}\" = {_q(str(spec.resources.h100_gpus))}",
        "for gpu_name in \"${GPU_NAMES[@]}\"; do",
        "    case \"${gpu_name}\" in *H100*) ;; *) echo \"Expected H100, got ${gpu_name}\" >&2; exit 2 ;; esac",
        "done",
        "nvidia-smi -q > \"${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt\"",
        "printf 'RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=unset\\nRECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=unset\\n' > \"${ROOT}/provenance/final_env_${SLURM_JOB_ID}.txt\"",
        "",
    ]
    return lines


def render_sbatch(
    spec: RunSpec,
    *,
    run_root: Path,
    runtime_root: Path,
    subject: dict[str, str],
    inputs: PreparedInputs,
    star_sha256: str,
    bind_library: Path,
) -> str:
    lines = _sbatch_header(spec, run_root)
    lines.extend(
        _common_preflight(
            spec,
            run_root=run_root,
            runtime_root=runtime_root,
            subject=subject,
            inputs=inputs,
            star_sha256=star_sha256,
            bind_library=bind_library,
        )
    )
    if spec.engine == "recovar":
        lines.extend(
            [
                "source /etc/profile.d/modules.sh",
                "set +u",
                "module purge",
                "module load cudatoolkit/12.8",
                "set -u",
                "CUDA_HOME=/usr/local/cuda-12.8",
                "NVIDIA_ROOT=$(find \"${PIXI_ENV}/lib\" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)",
                "test -n \"${NVIDIA_ROOT}\"",
                "NVIDIA_LIBS=$(find \"${NVIDIA_ROOT}\" -type d -name lib | paste -sd: -)",
                "export PATH=${CUDA_HOME}/bin:${PATH}",
                "export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib",
                "export RECOVAR_CUDA_LIB=${CUDA_LIB}",
                "export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND_DIR}",
                "export RELION_SRC_DIR=${RELION_SOURCE}/src",
                "export RECOVAR_EXPECTED_REPO_ROOT=${SUBJECT_REPO}",
                "unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER",
                "ldd \"${RELION_BIND_LIB}\" | tee \"${ROOT}/provenance/ldd_relion_bind_${SLURM_JOB_ID}.txt\"",
                "! grep -q 'not found' \"${ROOT}/provenance/ldd_relion_bind_${SLURM_JOB_ID}.txt\"",
                "cd \"${SUBJECT_REPO}\"",
                "\"${PYTHON}\" - <<'PY'",
                "from pathlib import Path",
                "import jax",
                "import recovar",
                "root = Path.cwd().resolve()",
                "assert Path(recovar.__file__).resolve().is_relative_to(root)",
                "assert '.pixi/envs/default/' in str(Path(jax.__file__).resolve())",
                "assert len(jax.devices('gpu')) == 1",
                "print(f'recovar={Path(recovar.__file__).resolve()}')",
                "print(f'jax={Path(jax.__file__).resolve()}')",
                "print(f'devices={jax.devices()}')",
                "PY",
            ]
        )
    else:
        lines.extend(
            [
                "source /etc/profile.d/modules.sh",
                "set +u",
                "module purge",
                "module load relion/5.0.0/gcc-11.5.0",
                "set -u",
                f"export LD_LIBRARY_PATH={RELION_CUDA_LIB_DIR}:{RELION_MPI_LIB_DIR}:${{LD_LIBRARY_PATH:-}}",
                "export OMPI_MCA_orte_tmpdir_base=${TMPDIR}",
                "export OMPI_MCA_shmem_mmap_enable_nfs_warning=0",
                "ldd \"${RELION_BINARY}\" | tee \"${ROOT}/provenance/ldd_relion_${SLURM_JOB_ID}.txt\"",
                "! grep -q 'not found' \"${ROOT}/provenance/ldd_relion_${SLURM_JOB_ID}.txt\"",
            ]
        )
    lines.extend(
        [
            "env | sort > \"${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt\"",
            "command=(",
            *[f"    {_q(token)}" for token in spec.command],
            ")",
            "printf '%q ' \"${command[@]}\" > \"${ROOT}/provenance/command_${SLURM_JOB_ID}.sh\"",
            "printf '\\n' >> \"${ROOT}/provenance/command_${SLURM_JOB_ID}.sh\"",
            "start_epoch=$(date +%s)",
            '"${command[@]}"',
            "end_epoch=$(date +%s)",
            f"printf '{{\"schema\":\"recovar.em.walltime.v1\",\"job_id\":\"%s\",\"run_key\":\"{spec.key}\",\"start_epoch\":%d,\"end_epoch\":%d,\"wall_s\":%d}}\\n' \\",
            "    \"${SLURM_JOB_ID}\" \"${start_epoch}\" \"${end_epoch}\" \"$((end_epoch - start_epoch))\" > \"${OUTPUT_DIR}/slurm_walltime.json\"",
        ]
    )
    for output in spec.expected_outputs:
        lines.append(f"test -s {_q(output)}")
    lines.extend(
        [
            "nvidia-smi --query-gpu=uuid,name --format=csv,noheader > \"${OUTPUT_DIR}/gpu_identity.txt\"",
            "touch \"${OUTPUT_DIR}/COMPLETED\"",
            "",
        ]
    )
    rendered = "\n".join(lines)
    if "exclusive" in rendered:
        raise AssertionError("generated Slurm scripts must never request exclusive nodes")
    return rendered


def _safe_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.parent != ALLOWED_RUN_ROOT:
        raise ValueError(f"--run-root must be one new direct child of {ALLOWED_RUN_ROOT}")
    if resolved.name in {"", ".", "..", "runtime"}:
        raise ValueError(f"unsafe --run-root: {resolved}")
    if resolved.exists():
        raise FileExistsError(f"refusing to reuse run root: {resolved}")
    runtime = RUNTIME_PARENT / resolved.name
    if runtime.exists():
        raise FileExistsError(f"refusing to reuse runtime root: {runtime}")
    return resolved


def _atomic_json(path: Path, payload: Any) -> None:
    def encode(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        return value

    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(encode(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _relative_or_absolute_link(target: Path, link: Path) -> None:
    link.symlink_to(target.resolve())


def prepare_launch(
    *,
    subject_repo: Path,
    subject_commit: str,
    preparation_manifest: Path,
    run_root: Path,
) -> tuple[Path, tuple[RunSpec, ...]]:
    subject = validate_subject(subject_repo, subject_commit)
    inputs = validate_preparation_manifest(preparation_manifest)
    native = validate_native_artifacts()
    run_root = _safe_root(run_root)
    runtime_root = RUNTIME_PARENT / run_root.name
    run_root.mkdir(parents=False)
    (run_root / "SAFE_TO_DELETE").touch()
    runtime_root.mkdir(parents=False)
    (runtime_root / "SAFE_TO_DELETE").touch()
    for path in (
        run_root / "logs",
        run_root / "evidence",
        run_root / "provenance",
        run_root / "scripts",
        run_root / "outputs/recovar_smoke",
        run_root / "outputs/relion_smoke",
        run_root / "outputs/recovar_full",
        run_root / "outputs/relion_full",
    ):
        path.mkdir(parents=True, exist_ok=False)

    smoke_star = run_root / "inputs/smoke/particles.star"
    smoke = write_balanced_smoke_star(inputs.particle_star, smoke_star)
    full_data = run_root / "inputs/full"
    full_data.mkdir(parents=True, exist_ok=False)
    _relative_or_absolute_link(inputs.particle_star, full_data / "particles.star")
    for data_dir in (smoke_star.parent, full_data):
        _relative_or_absolute_link(inputs.recovar_reference, data_dir / "reference_init.mrc")
        _relative_or_absolute_link(inputs.relion_reference, data_dir / "reference_init_relion.mrc")

    specs = build_run_specs(Path(subject["repo"]), run_root)
    bind_library = Path(native["relion_binding"]["path"])
    script_rows: dict[str, Any] = {}
    for spec in specs:
        star_sha = smoke["sha256"] if spec.phase == "smoke" else inputs.particle_star_sha256
        script_path = run_root / "scripts" / f"{spec.key}.sbatch"
        script_path.write_text(
            render_sbatch(
                spec,
                run_root=run_root,
                runtime_root=runtime_root,
                subject=subject,
                inputs=inputs,
                star_sha256=star_sha,
                bind_library=bind_library,
            )
        )
        script_path.chmod(0o750)
        script_rows[spec.key] = {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
            "engine": spec.engine,
            "phase": spec.phase,
            "resources": asdict(spec.resources),
            "data_dir": str(spec.data_dir),
            "particle_star": str(spec.data_dir / "particles.star"),
            "particle_star_sha256": star_sha,
            "command": list(spec.command),
            "command_sha256": sha256_json(list(spec.command)),
            "expected_outputs": [str(path) for path in spec.expected_outputs],
            "depends_on": list(spec.depends_on),
            "slurm_logs": {
                "stdout_template": str(run_root / "logs" / f"{spec.key}-%j.out"),
                "stderr_template": str(run_root / "logs" / f"{spec.key}-%j.err"),
            },
        }
    launch_manifest = run_root / "launch_manifest.json"
    payload = {
        "schema": SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "planned",
        "subject": subject,
        "preparation": asdict(inputs),
        "native_artifacts": native,
        "symmetry": {
            "family": "icosahedral",
            "requested_label": SYMMETRY,
            "relion_label": SYMMETRY,
            "recovar_label": SYMMETRY,
            "operator_count": SYMMETRY_OPERATOR_COUNT,
            "operator_order": SYMMETRY_OPERATOR_ORDER,
            "left_operator_policy": SYMMETRY_LEFT_OPERATOR_POLICY,
            "hash_encoding": SYMMETRY_HASH_ENCODING,
            "operators_sha256": SYMMETRY_OPERATORS_SHA256,
        },
        "scientific_contract": {
            "dataset": "EMPIAR-10202",
            "image_set": 6,
            "k": 1,
            "autonomous": True,
            "fixed_poses": False,
            "local_only": False,
            "replay": False,
            "same_particle_star_per_phase": True,
            "same_deposited_half_assignments": True,
            "same_initial_reference_canonical_array": True,
            "final_all_data_grid_correct_environment": "unset",
            "final_all_data_after_max_iter_environment": "unset",
            "quality_metric": "signed canonical-frame shellwise FSC/FSC-AUC",
        },
        "smoke_subset": smoke,
        "run_root": str(run_root),
        "runtime_root": str(runtime_root),
        "safe_to_delete_markers": [
            str(run_root / "SAFE_TO_DELETE"),
            str(runtime_root / "SAFE_TO_DELETE"),
        ],
        "runs": script_rows,
        "submission": None,
    }
    _atomic_json(launch_manifest, payload)
    return launch_manifest, specs


def _job_id(stdout: str) -> str:
    value = stdout.strip().split(";", maxsplit=1)[0]
    if not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"sbatch did not return a numeric job ID: {stdout!r}")
    return value


def _validate_requested_resources(scontrol_text: str, spec: RunSpec) -> None:
    expected_cpu = spec.resources.ntasks * spec.resources.cpus_per_task
    checks = (
        f"NumNodes={spec.resources.nodes}",
        f"NumCPUs={expected_cpu}",
        f"gres/gpu={spec.resources.h100_gpus}",
    )
    missing = [token for token in checks if token not in scontrol_text]
    if missing:
        raise ValueError(f"Slurm resource audit for {spec.key} is missing {missing}: {scontrol_text}")


def _resolved_job_artifacts(run_root: Path, spec: RunSpec, job_id: str) -> dict[str, str]:
    """Return the canonical logs and provenance artifacts written by one job."""

    ldd_name = "ldd_relion_bind" if spec.engine == "recovar" else "ldd_relion"
    return {
        "stdout_log": str(run_root / "logs" / f"{spec.key}-{job_id}.out"),
        "stderr_log": str(run_root / "logs" / f"{spec.key}-{job_id}.err"),
        "scontrol_submit": str(run_root / "provenance" / f"scontrol_submit_{job_id}.txt"),
        "scontrol_runtime": str(run_root / "provenance" / f"scontrol_{job_id}.txt"),
        "executed_command": str(run_root / "provenance" / f"command_{job_id}.sh"),
        "environment": str(run_root / "provenance" / f"environment_{job_id}.txt"),
        "final_environment": str(run_root / "provenance" / f"final_env_{job_id}.txt"),
        "gpu_inventory": str(run_root / "provenance" / f"nvidia_smi_{job_id}.txt"),
        "dynamic_libraries": str(run_root / "provenance" / f"{ldd_name}_{job_id}.txt"),
        "walltime": str(spec.output_dir / "slurm_walltime.json"),
        "gpu_identity": str(spec.output_dir / "gpu_identity.txt"),
        "completed_marker": str(spec.output_dir / "COMPLETED"),
    }


def submit_launch(
    launch_manifest: Path,
    specs: Sequence[RunSpec],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    payload = json.loads(launch_manifest.read_text())
    if payload.get("status") != "planned" or payload.get("submission") is not None:
        raise ValueError("launch manifest was already submitted or is not planned")
    by_key = {spec.key: spec for spec in specs}
    job_ids: dict[str, str] = {}
    jobs: dict[str, Any] = {}
    submitted: list[str] = []
    try:
        for key in ("recovar_smoke", "relion_smoke", "recovar_full", "relion_full"):
            spec = by_key[key]
            command = ["sbatch", "--parsable"]
            if spec.depends_on:
                dependencies = ":".join(job_ids[item] for item in spec.depends_on)
                command.append(f"--dependency=afterok:{dependencies}")
            command.append(payload["runs"][key]["path"])
            result = runner(command, check=True, capture_output=True, text=True)
            job_id = _job_id(result.stdout)
            job_ids[key] = job_id
            submitted.append(job_id)
            audit = runner(
                ["scontrol", "show", "job", "-o", job_id],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            _validate_requested_resources(audit, spec)
            submit_audit = launch_manifest.parent / "provenance" / f"scontrol_submit_{job_id}.txt"
            submit_audit.write_text(audit)
            artifacts = _resolved_job_artifacts(launch_manifest.parent, spec, job_id)
            jobs[key] = {
                "job_id": job_id,
                "sbatch_command": command,
                "artifacts": artifacts,
                "scontrol_submit_sha256": sha256_file(submit_audit),
            }
    except Exception:
        for job_id in submitted:
            runner(["scancel", job_id], check=False, capture_output=True, text=True)
        raise
    submission = {
        "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "job_ids": job_ids,
        "jobs": jobs,
        "dependency": {
            "recovar_full": [job_ids["recovar_smoke"], job_ids["relion_smoke"]],
            "relion_full": [job_ids["recovar_smoke"], job_ids["relion_smoke"]],
            "type": "afterok",
        },
    }
    payload["status"] = "submitted"
    payload["submission"] = submission
    _atomic_json(launch_manifest, payload)
    _atomic_json(launch_manifest.parent / "submission.json", submission)
    return submission


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-repo", type=Path, required=True)
    parser.add_argument("--subject-commit", required=True)
    parser.add_argument(
        "--preparation-manifest",
        type=Path,
        default=DEFAULT_PREPARATION_MANIFEST,
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit smoke jobs and dependency-gated full jobs after writing the plan.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    launch_manifest, specs = prepare_launch(
        subject_repo=args.subject_repo,
        subject_commit=args.subject_commit,
        preparation_manifest=args.preparation_manifest,
        run_root=args.run_root,
    )
    result: dict[str, Any] = {"launch_manifest": str(launch_manifest), "submitted": False}
    if args.submit:
        result["submission"] = submit_launch(launch_manifest, specs)
        result["submitted"] = True
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
