#!/usr/bin/env python3
"""Prepare and optionally submit genuine real-data K=4 half-map runs.

RELION forbids ``--split_random_halves`` together with multiple classes.  This
launcher therefore creates two disjoint frozen particle STARs and runs one
independent K=4 Class3D process per STAR for each engine.  The four processes
run serially on one physical GPU.  Dry-run is the default; ``--submit`` is an
explicit state-changing action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import starfile

from recovar.utils import helpers
from scripts.audit_em_real_kclass_halfmaps import (
    EXPECTED_THRESHOLDS,
    MANIFEST_SCHEMA,
    expected_analysis_policy,
    sha256_ints,
    sha256_strings,
)
from scripts.run_em_kclass_robustness_matrix_slurm import (
    RELION_DISPATCH_LOG_SCHEMA_MARKER,
    base_pixi_python,
    job_preamble,
    q,
    write_setup_script,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/em_work/codex")
DEFAULT_RUNTIME_ROOT = DEFAULT_RUN_ROOT / "runtime"
SOURCE_FIXTURE = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
    "em_k1_real10076_10k_fixture_20260712/data"
)
SHARED200_SELECTION = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
    "real_k4_10076_it1_shared200_3942224f5_20260901/outputs/"
    "shared_visited_particles_it001.json"
)
INITIAL_MAP_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
    "real_k4_10076_initialmodel_realgate_3942224f5_20260901/pair/relion"
)
STACKS = {
    128: Path("/scratch/gpfs/AMITS/mg6942/cryodrgn_empiar/empiar10076/inputs/particles.128.mrcs"),
    256: Path("/tigress/CRYOEM/singerlab/mg6942/10076/test_new/downsampled/particles.256.mrcs"),
}
CANONICAL_HASHES = {
    str(SOURCE_FIXTURE / "particles.star"): "2560afeea6839dddbb38b47d26cdf8944535a799d1e6d3e1441535c96043998f",
    str(SOURCE_FIXTURE / "source_indices.npy"): "b58a6d11fb292a9ed9573ac75c6a0673f4a0e8c216f4dadf11a7f0537b0e2c9d",
    str(SOURCE_FIXTURE / "fixture_manifest.json"): "762645e0d77c53bb9ce61701e1fd0d03868021c494f84b73a281b39d753d9750",
    str(SHARED200_SELECTION): "581157ff693aac6f5853d335d9cd0c59aa3fc11e60f54b325feb692ff05a9bd7",
    str(STACKS[128]): "24c52006eeb6f778a2b1a447a4ff790af0d2c78b7b20366281f54ec82c8f9382",
    str(STACKS[256]): "70d0c19995221491d27c9323f21c40df27e153bdf1e783bc78c4d38fe41a9c09",
    str(INITIAL_MAP_ROOT / "run_it000_class001.mrc"): "36c6b856c4a7718a52d7fbec1bc6d58619c7355ab4d1f1589e0ec352fec7303d",
    str(INITIAL_MAP_ROOT / "run_it000_class002.mrc"): "5601d3bbc4e1e12aa62b24365cc9d2493b76cf3b69c50dc5a3f08834969eb26f",
    str(INITIAL_MAP_ROOT / "run_it000_class003.mrc"): "dd405a82bac91e9361e62129a47daf2e49e07ea5a92ab73a19ede7735d377201",
    str(INITIAL_MAP_ROOT / "run_it000_class004.mrc"): "c80b1339f82f1bd155d4b4a28e61502ee535abdad4351d6479af05bd8c9f64d5",
}
SOURCE_FIXTURE_10345 = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
    "vdam_real10345_10k_fixture_v1_20260823/data"
)
INITIAL_MAP_ROOT_10345 = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
    "real_k4_10345_offset_prior_fullpair_92438c285_20260901/pair/relion"
)
STACKS_10345 = {
    256: Path("/projects/CRYOEM/singerlab/mg6942/10345/recovar_data/particles.256.mrcs"),
}
CANONICAL_HASHES_10345 = {
    str(SOURCE_FIXTURE_10345 / "particles.star"): "e5d9f77ff38d0e5137412892e7cc7591ba09265fb928b649cdeab58208a540f5",
    str(SOURCE_FIXTURE_10345 / "source_indices.npy"): "9f812a7bfd6bb9dd071786143a501c6803f6c05541faee36c7d6e07f0aa787a3",
    str(SOURCE_FIXTURE_10345 / "fixture_manifest.json"): "175972d7911dc512ceca2668c0f9b9372d76a3828ee4e9217c9870f55cec4683",
    "/projects/CRYOEM/singerlab/mg6942/10345/recovar_data/filt_particles.star": "8ab202046b07914c45636df73f1e6551a20c1f4476cb72797b1df9e9cd107b12",
    str(STACKS_10345[256]): "7909a695db68b65bfe6d0391054a1b19ae37fc4cd8da5cc4eb9595d76e4116e4",
    str(INITIAL_MAP_ROOT_10345 / "run_it000_class001.mrc"): "976d13ba09a2385266a3913558ad3db0b98002ae286d919177f35803a84f0d9b",
    str(INITIAL_MAP_ROOT_10345 / "run_it000_class002.mrc"): "2d07cec3661b1f7a07ba91ea38a3b0628bbcddcf08f6559a1417a7fb8643732d",
    str(INITIAL_MAP_ROOT_10345 / "run_it000_class003.mrc"): "cd9e289c638b6f23301151d5372ba50178ba29a25ee1162443dda09b0e5e07d3",
    str(INITIAL_MAP_ROOT_10345 / "run_it000_class004.mrc"): "1c4cbed1c79a0c6f5f1be78c0117bfc0fb825821110fe0f0caab7b7ad0b4f484",
}
DEFAULT_RELION_REFINE_MPI = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/"
    "relion_k4_100k_dispatchv2_20260717/build/bin/relion_refine_mpi"
)
DEFAULT_RELION_SOURCE = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/"
    "relion_k4_100k_dispatchv2_20260717/source/src"
)
DEFAULT_RELION_SHA256 = "01fa9cc870fdce6c19d981d6e917765753406972abcddb0311a40ef88e69a782"
DEFAULT_RELION_BASE_COMMIT = "d476e6f6a4f1f37627c06ace5227fc374c0c2b05"
DEFAULT_RELION_BASE_TREE = "1633d228e89d91ede8ad0996e727ec6ab1bc96ee"
DEFAULT_RELION_TRACKED_DIFF_SHA256 = "6987c5ce397cbdd98835682cf1481a150c38c48cda621e006341d01a77e11c11"
class LaunchError(RuntimeError):
    """Raised when a run cannot be prepared without weakening provenance."""


@dataclass(frozen=True)
class Profile:
    name: str
    grid_size: int
    selection: str
    expected_count: int
    time_limit: str
    memory: str
    image_batch_size: int


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    source_fixture: Path
    source_index_semantics: str
    source_particles_star: Path | None
    shared200_selection: Path | None
    initial_map_root: Path
    stacks: Mapping[int, Path]
    canonical_hashes: Mapping[str, str]
    supported_profiles: frozenset[str]


PROFILES = {
    # Sealed 128-grid runs used 6.3 GiB RSS for shared-200 and at most
    # 17.7 GiB for the larger 10k initial-model pair.  These requests retain
    # substantial headroom without reserving hundreds of unused host GiB.
    "shared200-128": Profile("shared200-128", 128, "shared200", 200, "04:00:00", "32G", 100),
    "pilot10k-128": Profile("pilot10k-128", 128, "full10k", 10_000, "12:00:00", "64G", 500),
    # Native-grid host residency is not yet qualified; keep the larger guard
    # until its first sealed peak-RSS measurement is available.
    "native10k-256": Profile("native10k-256", 256, "full10k", 10_000, "24:00:00", "256G", 250),
}


def _dataset_spec(key: str) -> DatasetSpec:
    if key == "10076":
        # Resolve the legacy module constants here rather than capturing them
        # at import time. Existing callers and tests may still override those
        # constants while the default dataset remains backward compatible.
        return DatasetSpec(
            key="10076",
            label="EMPIAR-10076",
            source_fixture=SOURCE_FIXTURE,
            source_index_semantics="particle_stack_index",
            source_particles_star=None,
            shared200_selection=SHARED200_SELECTION,
            initial_map_root=INITIAL_MAP_ROOT,
            stacks=STACKS,
            canonical_hashes=CANONICAL_HASHES,
            supported_profiles=frozenset(PROFILES),
        )
    if key == "10345":
        return DatasetSpec(
            key="10345",
            label="EMPIAR-10345",
            source_fixture=SOURCE_FIXTURE_10345,
            source_index_semantics="source_star_row_index",
            source_particles_star=Path(
                "/projects/CRYOEM/singerlab/mg6942/10345/recovar_data/filt_particles.star"
            ),
            shared200_selection=None,
            initial_map_root=INITIAL_MAP_ROOT_10345,
            stacks=STACKS_10345,
            canonical_hashes=CANONICAL_HASHES_10345,
            supported_profiles=frozenset({"native10k-256"}),
        )
    raise LaunchError(f"unsupported dataset: {key}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchError(message)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _git_text(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _source_provenance() -> dict[str, Any]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    _require(not status, "qualification launcher requires a completely clean worktree")
    return {
        "repo_root": str(REPO_ROOT),
        "commit": _git_text("rev-parse", "HEAD"),
        "tree": _git_text("rev-parse", "HEAD^{tree}"),
        "branch": _git_text("symbolic-ref", "--short", "HEAD"),
        "clean": True,
    }


def _relion_source_provenance(source_dir: Path) -> dict[str, Any]:
    repo = source_dir.parent
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=repo, text=True
    ).splitlines()
    _require(not untracked, f"RELION source contains untracked files: {untracked}")
    commit = _git_text("rev-parse", "HEAD", cwd=repo)
    tree = _git_text("rev-parse", "HEAD^{tree}", cwd=repo)
    tracked_diff = subprocess.check_output(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD"], cwd=repo
    )
    tracked_diff_sha256 = hashlib.sha256(tracked_diff).hexdigest()
    _require(commit == DEFAULT_RELION_BASE_COMMIT, "RELION base commit changed")
    _require(tree == DEFAULT_RELION_BASE_TREE, "RELION base tree changed")
    _require(
        tracked_diff_sha256 == DEFAULT_RELION_TRACKED_DIFF_SHA256,
        "RELION dispatch-instrumentation source diff changed",
    )
    return {
        "source_dir": str(source_dir.resolve()),
        "repo_root": str(repo.resolve()),
        "base_commit": commit,
        "base_tree": tree,
        "tracked_diff_sha256": tracked_diff_sha256,
        "tracked_dirty": bool(tracked_diff),
        "untracked_files": [],
    }


def _verify_canonical(
    path: Path,
    canonical_hashes: Mapping[str, str] | None = None,
) -> str:
    resolved = path.resolve()
    hashes = CANONICAL_HASHES if canonical_hashes is None else canonical_hashes
    expected = hashes.get(str(resolved)) or hashes.get(str(path))
    _require(expected is not None, f"no frozen checksum is declared for {resolved}")
    _require(resolved.is_file(), f"missing canonical artifact: {resolved}")
    # Preparation is the provenance boundary.  Deferring a large-stack hash
    # until the allocated job starts can waste a GPU allocation on an input
    # that was never eligible to run.  The 34.6 GB native 10076 stack hashes
    # in well under a minute on Della, so verify every canonical artifact here.
    observed = sha256_file(resolved)
    _require(
        observed == expected,
        f"canonical checksum changed: {resolved}; expected={expected} observed={observed}",
    )
    return expected


def _input_record(path: Path, *, role: str, expected_hash: str | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    _require(resolved.is_file(), f"missing input artifact: {resolved}")
    digest = expected_hash or sha256_file(resolved)
    _require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"invalid checksum for {resolved}")
    return {
        "role": role,
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest,
    }


def _particle_tables(source_fixture: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixture = SOURCE_FIXTURE if source_fixture is None else source_fixture
    payload = starfile.read(fixture / "particles.star")
    _require(isinstance(payload, dict) and set(payload) >= {"optics", "particles"}, "fixture STAR topology changed")
    return payload["optics"].copy(), payload["particles"].copy()


def _particle_table(path: Path) -> pd.DataFrame:
    payload = starfile.read(path)
    if isinstance(payload, pd.DataFrame):
        table = payload
    else:
        candidates = [table for table in payload.values() if "rlnImageName" in table]
        _require(len(candidates) == 1, f"cannot identify one particle table in {path}")
        table = candidates[0]
    _require("rlnImageName" in table, f"particle table has no rlnImageName: {path}")
    return table


def _image_stack_index(value: str) -> int:
    fields = str(value).split("@", 1)
    _require(len(fields) == 2 and fields[0].isdigit(), f"invalid RELION image identity: {value}")
    one_based = int(fields[0])
    _require(one_based >= 1, f"RELION image identity is not one-based: {value}")
    return one_based - 1


def _selected_particles(
    profile: Profile,
    particles: pd.DataFrame,
    shared200_selection: Path | None = None,
) -> pd.DataFrame:
    names = particles["rlnImageName"].astype(str)
    _require(names.is_unique, "source rlnImageName identities are not unique")
    if profile.selection == "full10k":
        selected = particles.copy()
    else:
        selection_path = SHARED200_SELECTION if shared200_selection is None else shared200_selection
        payload = json.loads(selection_path.read_text())
        _require(payload.get("same_visited_particle_ids") is True, "shared200 source was not admitted")
        requested = {str(value) for value in payload["visited_particle_ids"]}
        _require(len(requested) == profile.expected_count, "shared200 identity count changed")
        selected = particles.loc[names.isin(requested)].copy()
        _require(set(selected["rlnImageName"].astype(str)) == requested, "shared200 identities are absent")
    _require(len(selected) == profile.expected_count, "selected particle count changed")
    return selected.reset_index(drop=True)


def _prepare_references(
    root: Path,
    profile: Profile,
    dataset: DatasetSpec | None = None,
) -> tuple[list[Path], list[Path], Path]:
    initial_map_root = INITIAL_MAP_ROOT if dataset is None else dataset.initial_map_root
    canonical_hashes = CANONICAL_HASHES if dataset is None else dataset.canonical_hashes
    reference_dir = root / "data" / "references"
    reference_dir.mkdir(parents=True)
    recovar_paths: list[Path] = []
    relion_paths: list[Path] = []
    class_rows: list[dict[str, Any]] = []
    for class_id in range(1, 5):
        source = initial_map_root / f"run_it000_class{class_id:03d}.mrc"
        if dataset is None:
            _verify_canonical(source)
        else:
            _verify_canonical(source, canonical_hashes)
        volume, voxel = helpers.load_relion_volume(str(source), return_voxel_size=True)
        volume = np.asarray(volume, dtype=np.float32)
        source_grid = int(volume.shape[0])
        _require(source_grid == 256, f"canonical class {class_id} grid changed")
        if profile.grid_size != source_grid:
            volume = np.real(helpers.downsample_vol_by_fourier_truncation(volume, profile.grid_size)).astype(np.float32)
        voxel_array = np.asarray(voxel)
        if voxel_array.dtype.names:
            voxel_size = float(voxel_array["x"].item())
        else:
            voxel_size = float(voxel_array.reshape(-1)[0])
        voxel_size *= source_grid / profile.grid_size
        recovar_output = reference_dir / f"reference_init_class{class_id:03d}.mrc"
        relion_output = reference_dir / f"reference_init_class{class_id:03d}_relion.mrc"
        # RECOVAR and RELION use different real-space coordinate frames.  Keep
        # one explicitly encoded file for each consumer; passing the RELION
        # file through ``run_full_refinement --init_class_volumes`` silently
        # rotates/inverts every class reference before the first E-step.
        helpers.write_mrc(recovar_output, volume, voxel_size=voxel_size)
        helpers.write_relion_mrc(relion_output, volume, voxel_size=voxel_size)
        recovar_paths.append(recovar_output)
        relion_paths.append(relion_output)
        class_rows.append(
            {
                "rlnReferenceImage": str(relion_output.resolve()),
                "rlnClassDistribution": 0.25,
            }
        )
    star_path = reference_dir / "reference_init_classes_relion.star"
    starfile.write({"model_classes": pd.DataFrame(class_rows)}, star_path, overwrite=True)
    return recovar_paths, relion_paths, star_path


def _write_particle_inputs(
    root: Path,
    profile: Profile,
    dataset: DatasetSpec | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[int]]:
    if dataset is None:
        source_fixture = SOURCE_FIXTURE
        stacks = STACKS
        optics, particles = _particle_tables()
        selected = _selected_particles(profile, particles)
    else:
        source_fixture = dataset.source_fixture
        stacks = dataset.stacks
        optics, particles = _particle_tables(source_fixture)
        selected = _selected_particles(profile, particles, dataset.shared200_selection)
    source_indices_raw = np.load(source_fixture / "source_indices.npy", allow_pickle=False)
    _require(
        source_indices_raw.ndim == 1
        and np.issubdtype(source_indices_raw.dtype, np.integer)
        and source_indices_raw.size == len(particles),
        "immutable source indices do not match the source particle table",
    )
    source_indices = np.asarray(source_indices_raw, dtype=np.int64)
    origin_image_indices = [_image_stack_index(str(value)) for value in particles["rlnImageName"]]
    _require(
        all(value >= 0 for value in origin_image_indices)
        and len(origin_image_indices) == len(set(origin_image_indices)),
        "source STAR has invalid or duplicate image-stack indices",
    )
    source_index_semantics = "particle_stack_index" if dataset is None else dataset.source_index_semantics
    if source_index_semantics == "particle_stack_index":
        _require(
            origin_image_indices == source_indices.tolist(),
            "source STAR image-stack indices differ from immutable source-index order",
        )
    elif source_index_semantics == "source_star_row_index":
        _require(
            dataset is not None and dataset.source_particles_star is not None,
            "source-star row indices require a frozen source particle STAR",
        )
        source_table = _particle_table(dataset.source_particles_star)
        _require(
            np.all(source_indices < len(source_table)),
            "immutable source indices exceed the frozen source particle STAR",
        )
        selected_source_names = source_table.iloc[source_indices]["rlnImageName"].astype(str).tolist()
        _require(
            selected_source_names == particles["rlnImageName"].astype(str).tolist(),
            "fixture particle identities differ from the frozen source STAR row selection",
        )
    else:
        raise LaunchError(f"unknown source-index semantics: {source_index_semantics}")
    source_position = {
        image_index: position for position, image_index in enumerate(origin_image_indices)
    }
    source_apix = float(optics["rlnImagePixelSize"].iloc[0])
    source_grid = int(optics["rlnImageSize"].iloc[0])
    _require(source_grid == 256, "source fixture image grid changed")
    optics.loc[:, "rlnImageSize"] = profile.grid_size
    optics.loc[:, "rlnImagePixelSize"] = source_apix * source_grid / profile.grid_size
    _require(profile.grid_size in stacks, f"dataset has no qualified grid-{profile.grid_size} stack")
    stack_path = stacks[profile.grid_size].resolve()
    _require(stack_path.is_file(), f"runtime particle stack is unavailable: {stack_path}")
    selected.loc[:, "rlnImageName"] = [
        f"{str(value).split('@', 1)[0]}@{stack_path}" for value in selected["rlnImageName"]
    ]
    selected_names = selected["rlnImageName"].astype(str).tolist()
    selected_source_indices = [
        int(source_indices[source_position[_image_stack_index(value)]])
        for value in selected_names
    ]
    selected_star = root / "data" / "selected_particles.star"
    selected_star.parent.mkdir(parents=True, exist_ok=True)
    starfile.write({"optics": optics, "particles": selected}, selected_star, overwrite=True)
    halves: list[dict[str, Any]] = []
    for half_id in (1, 2):
        half_root = root / f"half{half_id}"
        data_dir = half_root / "data"
        relion_dir = half_root / "relion"
        recovar_dir = half_root / "recovar"
        for path in (data_dir, relion_dir, recovar_dir / "intermediates"):
            path.mkdir(parents=True)
        half_particles = selected.loc[selected["rlnRandomSubset"].astype(int) == half_id].copy()
        _require(len(half_particles) > 0, f"frozen half {half_id} is empty")
        half_star = data_dir / "particles.star"
        starfile.write({"optics": optics, "particles": half_particles}, half_star, overwrite=True)
        names = half_particles["rlnImageName"].astype(str).tolist()
        halves.append(
            {
                "half": half_id,
                "particle_count": len(names),
                "particles_star": str(half_star.resolve()),
                "ordered_image_names_sha256": sha256_strings(names),
                "data_dir": str(data_dir.resolve()),
                "relion_dir": str(relion_dir.resolve()),
                "recovar_dir": str(recovar_dir.resolve()),
                "recovar_intermediates_dir": str((recovar_dir / "intermediates").resolve()),
            }
        )
    return halves, selected_names, selected_source_indices


def build_relion_command(
    *,
    executable: Path,
    row: Mapping[str, Any],
    reference_star: Path,
    max_iter: int,
    seed: int,
    particle_diameter: float,
    mpi_ranks: int,
    pool: int,
) -> list[str]:
    return [
        "mpirun",
        "-n",
        str(mpi_ranks),
        str(executable.resolve()),
        "--i",
        "particles.star",
        "--ref",
        str(reference_star.resolve()),
        "--o",
        str(Path(row["relion_dir"]) / "run"),
        "--iter",
        str(max_iter),
        "--tau2_fudge",
        "4",
        "--particle_diameter",
        f"{particle_diameter:g}",
        "--K",
        "4",
        "--flatten_solvent",
        "--zero_mask",
        "--firstiter_cc",
        "--ini_high",
        "30",
        "--ctf",
        "--norm",
        "--scale",
        "--sym",
        "C1",
        "--oversampling",
        "1",
        "--healpix_order",
        "1",
        "--offset_range",
        "6",
        "--offset_step",
        "2",
        "--pad",
        "2",
        "--pool",
        str(pool),
        "--dont_combine_weights_via_disc",
        "--random_seed",
        str(seed),
        "--gpu",
        "0",
        "--j",
        "4",
    ]


def build_recovar_command(
    *,
    python: Path,
    row: Mapping[str, Any],
    initial_class_volumes: Sequence[Path],
    max_iter: int,
    seed: int,
    particle_diameter: float,
    image_batch_size: int,
    mpi_ranks: int,
) -> list[str]:
    _require(len(initial_class_volumes) == 4, "K=4 requires exactly four initial class volumes")
    initial_class_volume_arg = ",".join(str(path.resolve()) for path in initial_class_volumes)
    relion_dir = Path(row["relion_dir"])
    recovar_dir = Path(row["recovar_dir"])
    return [
        str(python),
        "-m",
        "scripts.run_full_refinement",
        "--data_dir",
        str(Path(row["data_dir"])),
        "--output",
        str(recovar_dir),
        "--max_iter",
        str(max_iter),
        "--healpix_order",
        "1",
        "--max_healpix_order",
        "1",
        "--sym",
        "C1",
        "--offset_range",
        "6",
        "--offset_step",
        "2",
        "--adaptive_oversampling",
        "1",
        "--tau2_fudge",
        "4",
        "--perturb_factor",
        "0.5",
        "--perturb_seed",
        str(seed),
        "--seed",
        str(seed),
        "--init_resolution",
        "30",
        "--image-fourier-backend",
        "relion_cuda",
        "--image_batch_size",
        str(image_batch_size),
        "--rotation_block_size",
        "8192",
        "--relion-scale-followers",
        str(mpi_ranks - 1),
        "--relion-dispatch-schedule",
        str(relion_dir / "dispatch_schedule.npz"),
        "--relion_optimiser",
        str(relion_dir / "run_it000_optimiser.star"),
        "--relion_init_dir",
        str(relion_dir),
        "--particle_diameter_ang",
        f"{particle_diameter:g}",
        "--firstiter_cc",
        "--apply-initial-lowpass",
        "--n_classes",
        "4",
        "--init_class_volumes",
        initial_class_volume_arg,
        "--initial-pose-source",
        "none",
        "--timing_dir",
        str(recovar_dir / "timing"),
        "--save_intermediates_dir",
        str(recovar_dir / "intermediates"),
        "--save_intermediates_skip_unregularized",
    ]


def _write_command(path: Path, command: Sequence[str]) -> None:
    path.write_text(json.dumps(list(command), indent=2) + "\n")


def _shell_array(name: str, values: Sequence[str]) -> str:
    return f"{name}=({ ' '.join(shlex.quote(value) for value in values) })"


def _parse_scontrol_fields(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/:.-]*)=", text))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields[match.group(1)] = text[start:end].strip()
    return fields


def _gpu_count_from_tres(value: str) -> int | None:
    generic = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", value)
    if generic is not None:
        return int(generic.group(1))
    typed = re.findall(r"(?:^|,)gres/gpu:[^=,]+=(\d+)(?:,|$)", value)
    return sum(int(item) for item in typed) if typed else None


def validate_submitted_job(job_id: str, script: Path) -> dict[str, Any]:
    """Immediately reject an exclusive or overallocated newly submitted job."""

    output = subprocess.check_output(["scontrol", "show", "job", "-o", job_id], text=True).strip()
    fields = _parse_scontrol_fields(output)
    requested = fields.get("ReqTRES", "")
    allocated = fields.get("AllocTRES", "")
    _require(_gpu_count_from_tres(requested) == 1, f"job {job_id} did not request exactly one GPU")
    if allocated not in {"", "(null)", "N/A"}:
        _require(requested == allocated, f"job {job_id} ReqTRES != AllocTRES")
        _require(_gpu_count_from_tres(allocated) == 1, f"job {job_id} did not allocate exactly one GPU")
    _require(fields.get("OverSubscribe") == "OK", f"job {job_id} is exclusive")
    _require("#SBATCH --exclusive" not in script.read_text(), f"job script requests --exclusive: {script}")
    _require(fields.get("JobState") not in {"CANCELLED", "FAILED", "REJECTED"}, f"job {job_id} was rejected")
    return {
        "job_id": job_id,
        "job_state": fields.get("JobState"),
        "ReqTRES": requested,
        "AllocTRES": None if allocated in {"", "(null)", "N/A"} else allocated,
        "OverSubscribe": fields.get("OverSubscribe"),
        "requested_gpus": 1,
        "allocated_gpus": None if allocated in {"", "(null)", "N/A"} else 1,
        "script": str(script.resolve()),
        "raw_scontrol": output,
        "valid_at_submission": True,
    }


def submit_scripts(setup_script: Path, run_script: Path) -> dict[str, Any]:
    """Submit setup/run jobs and cancel all newly submitted jobs on validation failure."""

    submitted: list[str] = []
    try:
        setup_output = subprocess.check_output(["sbatch", "--parsable", str(setup_script)], text=True).strip()
        setup_job = setup_output.split(";", 1)[0]
        _require(bool(setup_job), "sbatch returned an empty setup job ID")
        submitted.append(setup_job)
        setup_audit = validate_submitted_job(setup_job, setup_script)

        run_output = subprocess.check_output(
            ["sbatch", "--parsable", f"--dependency=afterok:{setup_job}", str(run_script)], text=True
        ).strip()
        run_job = run_output.split(";", 1)[0]
        _require(bool(run_job), "sbatch returned an empty qualification job ID")
        submitted.append(run_job)
        run_audit = validate_submitted_job(run_job, run_script)
    except (LaunchError, OSError, subprocess.CalledProcessError) as exc:
        for job_id in reversed(submitted):
            subprocess.run(["scancel", job_id], check=False, capture_output=True, text=True)
        cancelled = ", ".join(submitted) if submitted else "none"
        raise LaunchError(f"submission validation failed; cancelled newly submitted jobs: {cancelled}: {exc}") from exc
    return {
        "setup_job_id": setup_job,
        "run_job_id": run_job,
        "submission_audit": {"setup": setup_audit, "run": run_audit},
    }


def render_run_script(
    *,
    root: Path,
    profile: Profile,
    source: Mapping[str, Any],
    relion_source: Path,
    relion_module: str,
    relion_refine_mpi: Path,
    cuda_module: str,
    halves: Sequence[Mapping[str, Any]],
    max_iter: int,
    seed: int,
    mpi_ranks: int,
    pool: int,
    analysis_policy: Mapping[str, Any],
    dataset_key: str = "10076",
) -> str:
    _require(
        analysis_policy == expected_analysis_policy(profile.grid_size),
        "run script analysis policy differs from the frozen profile policy",
    )
    alignment_policy = analysis_policy["alignment"]
    fsc_policy = analysis_policy["fsc"]
    mask_policy = analysis_policy["common_mask"]
    refine_order_flags = "".join(
        f"  --refine-healpix-order {int(value)} \\\n"
        for value in alignment_policy["refine_healpix_orders"]
    )
    cuda_lib = root / "build" / "cuda" / "libcuda_backproject.so"
    preamble = job_preamble(
        scratch_dir=root,
        cuda_lib=cuda_lib,
        cuda_module=cuda_module,
        relion_src_dir=relion_source,
        job_name=f"real_k4_halfmap_{dataset_key}_{profile.name}_seed{seed}",
        expected_commit=str(source["commit"]),
    )
    command_arrays = []
    for row in halves:
        half_id = int(row["half"])
        command_arrays.extend(
            [
                _shell_array(f"RELION_COMMAND_{half_id}", row["relion_command"]),
                _shell_array(f"RECOVAR_COMMAND_{half_id}", row["recovar_command"]),
            ]
        )
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=k4h_{dataset_key}_{profile.name[:10]}_{seed}
#SBATCH --output={q(root / 'logs' / 'qualification-%j.out')}
#SBATCH --error={q(root / 'logs' / 'qualification-%j.err')}
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks={mpi_ranks}
#SBATCH --cpus-per-task=8
#SBATCH --mem={profile.memory}
#SBATCH --time={profile.time_limit}

{preamble}

ROOT={q(root)}
MANIFEST="${{ROOT}}/submission_manifest.json"
mkdir -p "${{ROOT}}/logs" "${{ROOT}}/provenance"
sha256sum --check "${{ROOT}}/submission_manifest.sha256"
sha256sum --check "${{ROOT}}/inputs.sha256" | tee "${{ROOT}}/provenance/input_sha256_check.txt"
sha256sum --check "${{ROOT}}/relion_bind_build/shared.sha256"
sha256sum --check "${{RECOVAR_CUDA_LIB}}.sha256"

"${{PIXI_PY}}" - "${{ROOT}}/provenance/slurm_allocation.json" <<'PY'
import json
import pathlib
import sys
from scripts.run_em_real_kclass_initialmodel_pair import _gpu_count_from_tres, _slurm_allocation

row = _slurm_allocation()
row["requested_gpus"] = _gpu_count_from_tres(row["ReqTRES"])
row["allocated_gpus"] = _gpu_count_from_tres(row["AllocTRES"])
pathlib.Path(sys.argv[1]).write_text(json.dumps(row, indent=2, sort_keys=True) + "\\n")
PY

"${{PIXI_PY}}" - <<'PY'
import pathlib
import jax
import recovar
from recovar.relion_bind import _relion_bind_core as relion_bind

repo = pathlib.Path.cwd().resolve()
assert pathlib.Path(recovar.__file__).resolve().is_relative_to(repo)
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
bind_root = pathlib.Path(__import__("os").environ["RECOVAR_RELION_BIND_BUILD_DIR"]).resolve()
assert pathlib.Path(relion_bind.__file__).resolve().is_relative_to(bind_root)
assert any(getattr(device, "platform", "") in {{"cuda", "gpu"}} for device in jax.devices())
print("qualification import/GPU provenance gate ok", jax.devices())
PY

capture_gpu_uuid() {{
  mapfile -t uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed 's/[[:space:]]//g' | sed '/^$/d')
  if [[ "${{#uuids[@]}}" -ne 1 || "${{uuids[0]}}" != GPU-* ]]; then
    echo "ERROR: expected exactly one visible physical GPU" >&2
    return 2
  fi
  printf '%s\\n' "${{uuids[0]}}"
}}

PHYSICAL_GPU_UUID="$(capture_gpu_uuid)"
printf '%s\\n' "${{PHYSICAL_GPU_UUID}}" > "${{ROOT}}/provenance/physical_gpu_uuid.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv > "${{ROOT}}/provenance/gpu_inventory.csv"

MONITOR_PID=""
start_monitor() {{
  local output="$1"
  printf 'epoch,gpu_uuid,memory_used_mib\\n' > "${{output}}"
  (
    while true; do
      local_epoch="$(date +%s)"
      nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits \
        | awk -F',' -v epoch="${{local_epoch}}" '{{gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); print epoch "," $1 "," $2}}'
      sleep 1
    done
  ) >> "${{output}}" &
  MONITOR_PID="$!"
}}

stop_monitor() {{
  if [[ -n "${{MONITOR_PID}}" ]]; then
    kill "${{MONITOR_PID}}" 2>/dev/null || true
    wait "${{MONITOR_PID}}" 2>/dev/null || true
    MONITOR_PID=""
  fi
}}
trap stop_monitor EXIT

record_wall() {{
  local output="$1" start="$2" end="$3" status="$4"
  "${{PIXI_PY}}" - "${{output}}" "${{SLURM_JOB_ID}}" "${{start}}" "${{end}}" "${{status}}" <<'PY'
import json
import pathlib
import sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({{
    "slurm_job_id": sys.argv[2],
    "start_epoch": float(sys.argv[3]),
    "end_epoch": float(sys.argv[4]),
    "external_wall_s": float(sys.argv[4]) - float(sys.argv[3]),
    "exit_status": int(sys.argv[5]),
}}, sort_keys=True) + "\\n")
PY
}}

{os.linesep.join(command_arrays)}

run_relion_half() {{
  local half="$1"
  local data_dir="${{ROOT}}/half${{half}}/data"
  local output_dir="${{ROOT}}/half${{half}}/relion"
  local -n command_ref="RELION_COMMAND_${{half}}"
  local start end status
  rm -f "${{output_dir}}/dispatch.tsv" "${{output_dir}}/dispatch_schedule.npz"
  start="$(date +%s.%N)"
  start_monitor "${{output_dir}}/gpu_monitor.csv"
  set +e
  (
    unset LD_LIBRARY_PATH
    source /etc/profile.d/modules.sh
    export PS1="${{PS1:-}}"
    set +u
    module load {q(relion_module)}
    set -u
    export RELION_DISPATCH_LOG="${{output_dir}}/dispatch.tsv"
    export TMPDIR="${{RUNTIME_ROOT}}/relion_half${{half}}"
    mkdir -p "${{TMPDIR}}"
    cd "${{data_dir}}"
    /usr/bin/time -v -o "${{output_dir}}/time.txt" "${{command_ref[@]}}"
  ) > "${{output_dir}}/run.log" 2>&1
  status="$?"
  set -e
  stop_monitor
  end="$(date +%s.%N)"
  record_wall "${{output_dir}}/slurm_walltime.json" "${{start}}" "${{end}}" "${{status}}"
  if [[ "${{status}}" -ne 0 ]]; then return "${{status}}"; fi
  test -s "${{output_dir}}/dispatch.tsv"
  "${{PIXI_PY}}" -m scripts.build_relion_dispatch_schedule \
    --dispatch-log "${{output_dir}}/dispatch.tsv" \
    --output "${{output_dir}}/dispatch_schedule.npz" \
    --n-particles "$("${{PIXI_PY}}" -c 'import starfile,sys; print(len(starfile.read(sys.argv[1])["particles"]))' "${{data_dir}}/particles.star")" \
    --n-followers {mpi_ranks - 1} \
    --pool-size {pool * 4} \
    --random-seed {seed} \
    --oracle-dir "${{output_dir}}"
}}

run_recovar_half() {{
  local half="$1"
  local output_dir="${{ROOT}}/half${{half}}/recovar"
  local -n command_ref="RECOVAR_COMMAND_${{half}}"
  local start end status
  start="$(date +%s.%N)"
  start_monitor "${{output_dir}}/gpu_monitor.csv"
  set +e
  /usr/bin/time -v -o "${{output_dir}}/time.txt" "${{command_ref[@]}}" \
    > "${{output_dir}}/run.log" 2>&1
  status="$?"
  set -e
  stop_monitor
  end="$(date +%s.%N)"
  record_wall "${{output_dir}}/slurm_walltime.json" "${{start}}" "${{end}}" "${{status}}"
  if [[ "${{status}}" -ne 0 ]]; then return "${{status}}"; fi
}}

run_particle_state_audit_half() {{
  local half="$1"
  local iteration padded
  local -a audit_command=(
    "${{PIXI_PY}}" -m scripts.audit_em_particle_state_distribution
    --recovar-results "${{ROOT}}/half${{half}}/recovar/refinement_results.npz"
    --recovar-particles-star "${{ROOT}}/half${{half}}/data/particles.star"
    --output-json "${{ROOT}}/audit/half${{half}}_particle_state.json"
    --output-npz "${{ROOT}}/audit/half${{half}}_particle_state_arrays.npz"
    --output-hash-manifest "${{ROOT}}/audit/half${{half}}_particle_state.sha256"
  )
  mkdir -p "${{ROOT}}/audit"
  for ((iteration = 1; iteration <= {max_iter}; iteration++)); do
    printf -v padded '%03d' "${{iteration}}"
    audit_command+=(
      --relion-star
      "${{ROOT}}/half${{half}}/relion/run_it${{padded}}_data.star"
    )
  done
  "${{audit_command[@]}}"
}}

for half in 1 2; do
  run_relion_half "${{half}}"
  test "$(capture_gpu_uuid)" = "${{PHYSICAL_GPU_UUID}}"
  run_recovar_half "${{half}}"
  test "$(capture_gpu_uuid)" = "${{PHYSICAL_GPU_UUID}}"
  run_particle_state_audit_half "${{half}}"
done

"${{PIXI_PY}}" -m scripts.audit_em_real_kclass_halfmaps \
  --manifest "${{MANIFEST}}" \
  --output-dir "${{ROOT}}/audit" \
  --fit-max-shell {int(alignment_policy["fit_max_shell"])} \
  --crossing-consecutive-shells {int(fsc_policy["crossing_consecutive_shells"])} \
  --phase-randomization-corrected {str(bool(fsc_policy["phase_randomization_corrected"])).lower()} \
  --absolute-resolution-claim {str(bool(fsc_policy["absolute_resolution_claim"])).lower()} \
  --coarse-healpix-order {int(alignment_policy["coarse_healpix_order"])} \
{refine_order_flags}  --interpolation-order {int(alignment_policy["interpolation_order"])} \
  --mask-threshold {q(str(mask_policy["threshold"]))} \
  --mask-lowpass-sigma {int(mask_policy["lowpass_sigma"])} \
  --mask-extend {int(mask_policy["extend"])} \
  --mask-soft-edge {int(mask_policy["soft_edge"])} \
  --mask-cleanup {str(bool(mask_policy["cleanup"])).lower()}
"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("10076", "10345"), default="10076")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="shared200-128")
    parser.add_argument("--seed", type=int, choices=(42001, 42002, 42003), default=42001)
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--particle-diameter", type=float, default=200.0)
    parser.add_argument("--relion-refine-mpi", type=Path, default=DEFAULT_RELION_REFINE_MPI)
    parser.add_argument("--relion-source-dir", type=Path, default=DEFAULT_RELION_SOURCE)
    parser.add_argument("--relion-module", default="relion/5.0.0/gcc-11.5.0")
    parser.add_argument("--cuda-module", default="cudatoolkit/12.8")
    parser.add_argument("--mpi-ranks", type=int, default=3)
    parser.add_argument("--pool", type=int, default=3)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args(argv)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root.expanduser().resolve()
    _require(root.is_relative_to(DEFAULT_RUN_ROOT), f"output root must be under {DEFAULT_RUN_ROOT}")
    _require(not root.exists() and not root.is_symlink(), f"refusing to reuse output root: {root}")
    _require(args.max_iter >= 1, "max_iter must be positive")
    _require(args.mpi_ranks >= 2 and args.pool >= 1, "invalid RELION MPI/pool topology")
    profile = PROFILES[args.profile]
    dataset = _dataset_spec(args.dataset)
    _require(
        profile.name in dataset.supported_profiles,
        f"{dataset.label} does not have qualified inputs for profile {profile.name}; "
        f"supported profiles: {', '.join(sorted(dataset.supported_profiles))}",
    )
    source = _source_provenance()
    _require(base_pixi_python().is_file(), f"base pixi Python is unavailable: {base_pixi_python()}")
    relion_source = _relion_source_provenance(args.relion_source_dir.resolve())
    executable = args.relion_refine_mpi.resolve()
    _require(executable.is_file() and os.access(executable, os.X_OK), f"RELION executable unavailable: {executable}")
    _require(sha256_file(executable) == DEFAULT_RELION_SHA256, "RELION executable checksum changed")
    _require(RELION_DISPATCH_LOG_SCHEMA_MARKER in executable.read_bytes(), "RELION dispatch schema-v2 marker missing")

    root.mkdir(parents=True)
    (root / "SAFE_TO_DELETE").touch()
    (root / "logs").mkdir()
    (root / "jobs").mkdir()
    (root / "provenance").mkdir()
    (root / "build" / "cuda").mkdir(parents=True)
    relion_diff_artifact = root / "provenance" / "relion_source_tracked.diff"
    relion_diff_artifact.write_bytes(
        subprocess.check_output(
            ["git", "diff", "--binary", "--no-ext-diff", "HEAD"],
            cwd=args.relion_source_dir.resolve().parent,
        )
    )
    _require(
        sha256_file(relion_diff_artifact) == relion_source["tracked_diff_sha256"],
        "captured RELION source diff changed after provenance validation",
    )
    relion_source["tracked_diff_artifact"] = {
        "path": str(relion_diff_artifact.resolve()),
        "sha256": sha256_file(relion_diff_artifact),
        "size_bytes": relion_diff_artifact.stat().st_size,
    }

    source_fixture = dataset.source_fixture
    initial_map_root = dataset.initial_map_root
    canonical_hashes = dataset.canonical_hashes
    stacks = dataset.stacks
    for path in (
        source_fixture / "particles.star",
        source_fixture / "source_indices.npy",
        source_fixture / "fixture_manifest.json",
    ):
        _verify_canonical(path, canonical_hashes)
    if dataset.source_particles_star is not None:
        _verify_canonical(dataset.source_particles_star, canonical_hashes)
    if profile.selection == "shared200":
        _require(dataset.shared200_selection is not None, "dataset has no qualified shared200 selection")
        _verify_canonical(dataset.shared200_selection, canonical_hashes)
    _require(profile.grid_size in stacks, f"dataset has no qualified grid-{profile.grid_size} stack")
    stack_path = stacks[profile.grid_size]
    stack_hash = _verify_canonical(stack_path, canonical_hashes)
    analysis_policy = expected_analysis_policy(profile.grid_size)
    recovar_reference_paths, relion_reference_paths, reference_star = _prepare_references(
        root,
        profile,
        dataset,
    )
    halves, selected_names, selected_source_indices = _write_particle_inputs(
        root,
        profile,
        dataset,
    )

    run_python = root / "venv" / "bin" / "python"
    for row in halves:
        relion_command = build_relion_command(
            executable=executable,
            row=row,
            reference_star=reference_star,
            max_iter=args.max_iter,
            seed=args.seed,
            particle_diameter=args.particle_diameter,
            mpi_ranks=args.mpi_ranks,
            pool=args.pool,
        )
        recovar_command = build_recovar_command(
            python=run_python,
            row=row,
            initial_class_volumes=recovar_reference_paths,
            max_iter=args.max_iter,
            seed=args.seed,
            particle_diameter=args.particle_diameter,
            image_batch_size=profile.image_batch_size,
            mpi_ranks=args.mpi_ranks,
        )
        relion_command_path = Path(row["relion_dir"]) / "command.json"
        recovar_command_path = Path(row["recovar_dir"]) / "command.json"
        _write_command(relion_command_path, relion_command)
        _write_command(recovar_command_path, recovar_command)
        row["relion_command"] = relion_command
        row["recovar_command"] = recovar_command
        row["relion_command_path"] = str(relion_command_path.resolve())
        row["recovar_command_path"] = str(recovar_command_path.resolve())

    setup_script = write_setup_script(
        scratch_dir=root,
        jobs_dir=root / "jobs",
        cuda_lib=root / "build" / "cuda" / "libcuda_backproject.so",
        account="gilles",
        partition="cryoem",
        constraint="h100",
        setup_gres="gpu:1",
        cuda_module=args.cuda_module,
        relion_src_dir=args.relion_source_dir.resolve(),
        expected_commit=str(source["commit"]),
        setup_allocation_record=root / "provenance" / "setup_slurm_allocation.json",
    )
    run_script = root / "jobs" / "run_k4_independent_halfmaps.sh"
    run_script.write_text(
        render_run_script(
            root=root,
            profile=profile,
            source=source,
            relion_source=args.relion_source_dir.resolve(),
            relion_module=args.relion_module,
            relion_refine_mpi=executable,
            cuda_module=args.cuda_module,
            halves=halves,
            max_iter=args.max_iter,
            seed=args.seed,
            mpi_ranks=args.mpi_ranks,
            pool=args.pool,
            analysis_policy=analysis_policy,
            dataset_key=dataset.key,
        )
    )
    run_script.chmod(0o755)

    input_artifacts = [
        _input_record(source_fixture / "particles.star", role="frozen_source_particles_star", expected_hash=canonical_hashes[str(source_fixture / 'particles.star')]),
        _input_record(source_fixture / "source_indices.npy", role="frozen_source_indices", expected_hash=canonical_hashes[str(source_fixture / 'source_indices.npy')]),
        _input_record(source_fixture / "fixture_manifest.json", role="frozen_fixture_manifest", expected_hash=canonical_hashes[str(source_fixture / 'fixture_manifest.json')]),
        _input_record(stack_path, role=f"particle_stack_grid{profile.grid_size}", expected_hash=stack_hash),
        _input_record(executable, role="instrumented_relion_refine_mpi", expected_hash=DEFAULT_RELION_SHA256),
        _input_record(
            relion_diff_artifact,
            role="instrumented_relion_source_tracked_diff",
            expected_hash=DEFAULT_RELION_TRACKED_DIFF_SHA256,
        ),
        *[
            _input_record(
                path,
                role=f"shared_initial_class{index:03d}_source",
                expected_hash=canonical_hashes[
                    str(initial_map_root / f"run_it000_class{index:03d}.mrc")
                ],
            )
            for index, path in enumerate(
                [initial_map_root / f"run_it000_class{i:03d}.mrc" for i in range(1, 5)],
                start=1,
            )
        ],
        *[
            _input_record(path, role=f"prepared_recovar_initial_class{index:03d}")
            for index, path in enumerate(recovar_reference_paths, start=1)
        ],
        *[
            _input_record(path, role=f"prepared_relion_initial_class{index:03d}")
            for index, path in enumerate(relion_reference_paths, start=1)
        ],
        _input_record(reference_star, role="prepared_relion_reference_star"),
        _input_record(root / "data" / "selected_particles.star", role="frozen_selected_particles_star"),
    ]
    if dataset.source_particles_star is not None:
        input_artifacts.append(
            _input_record(
                dataset.source_particles_star,
                role="source_index_origin_particles_star",
                expected_hash=canonical_hashes[str(dataset.source_particles_star)],
            )
        )
    if profile.selection == "shared200":
        assert dataset.shared200_selection is not None
        input_artifacts.append(
            _input_record(
                dataset.shared200_selection,
                role="shared200_selection",
                expected_hash=canonical_hashes[str(dataset.shared200_selection)],
            )
        )
    for row in halves:
        input_artifacts.extend(
            [
                _input_record(Path(row["particles_star"]), role=f"half{row['half']}_particles_star"),
                _input_record(Path(row["relion_command_path"]), role=f"half{row['half']}_relion_command"),
                _input_record(Path(row["recovar_command_path"]), role=f"half{row['half']}_recovar_command"),
            ]
        )
    inputs_sha = root / "inputs.sha256"
    inputs_sha.write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in input_artifacts))

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "dataset": dataset.label,
        "profile": profile.name,
        "source": source,
        "relion_source": relion_source,
        "relion_executable": {
            "path": str(executable),
            "sha256": DEFAULT_RELION_SHA256,
            "dispatch_schema_marker": RELION_DISPATCH_LOG_SCHEMA_MARKER.decode(),
            "source_binding": {
                "cryptographically_attested": False,
                "reason": (
                    "the executable SHA-256, base source tree, and exact instrumentation-diff SHA-256 "
                    "are sealed independently; no build-system attestation binds that binary to that source"
                ),
            },
        },
        "config": {
            "K": 4,
            "symmetry": "C1",
            "grid_size": profile.grid_size,
            "max_iter": args.max_iter,
            "seed": args.seed,
            "particle_diameter_angstrom": args.particle_diameter,
            "initial_lowpass_angstrom": 30.0,
            "mpi_ranks": args.mpi_ranks,
            "followers": args.mpi_ranks - 1,
            "pool": args.pool,
            "dispatch_pool_size": args.pool * 4,
            "image_batch_size": profile.image_batch_size,
            "rotation_block_size": 8192,
            "fourier_backend": "relion_cuda",
            "final_all_data_after_max_iter": False,
        },
        "thresholds": EXPECTED_THRESHOLDS,
        "analysis_policy": analysis_policy,
        "particle_selection": {
            "mode": profile.selection,
            "source_particles_star": str((root / "data" / "selected_particles.star").resolve()),
            "origin_particles_star": str((source_fixture / "particles.star").resolve()),
            "origin_particles_star_sha256": canonical_hashes[
                str(source_fixture / "particles.star")
            ],
            "source_indices_npy": str((source_fixture / "source_indices.npy").resolve()),
            "source_indices_sha256": canonical_hashes[
                str(source_fixture / "source_indices.npy")
            ],
            "source_index_semantics": dataset.source_index_semantics,
            "source_index_origin_particles_star": (
                str(dataset.source_particles_star.resolve())
                if dataset.source_particles_star is not None
                else None
            ),
            "source_index_origin_particles_star_sha256": (
                canonical_hashes[str(dataset.source_particles_star)]
                if dataset.source_particles_star is not None
                else None
            ),
            "selection_source_json": (
                str(dataset.shared200_selection.resolve())
                if dataset.shared200_selection is not None and profile.selection == "shared200"
                else None
            ),
            "selection_source_sha256": (
                canonical_hashes[str(dataset.shared200_selection)]
                if dataset.shared200_selection is not None and profile.selection == "shared200"
                else None
            ),
            "selected_particles_star": str((root / "data" / "selected_particles.star").resolve()),
            "selected_image_names": selected_names,
            "ordered_image_names_sha256": sha256_strings(selected_names),
            "selected_source_indices": selected_source_indices,
            "ordered_source_indices_sha256": sha256_ints(selected_source_indices),
            "particle_stack_path": str(stack_path.resolve()),
            "particle_stack_sha256": stack_hash,
        },
        "halves": halves,
        "input_artifacts": input_artifacts,
        "environment": {
            "relion_module": args.relion_module,
            "cuda_module": args.cuda_module,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONNOUSERSITE": "1",
            "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT": "unset",
            "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER": "unset",
        },
        "provenance": {
            "direct_rehash_max_bytes": 1_000_000_000,
            "input_sha256_manifest": str(inputs_sha.resolve()),
            "input_sha256_check_log": str((root / "provenance" / "input_sha256_check.txt").resolve()),
            "setup_slurm_allocation_json": str(
                (root / "provenance" / "setup_slurm_allocation.json").resolve()
            ),
            "slurm_allocation_json": str((root / "provenance" / "slurm_allocation.json").resolve()),
            "setup_script": str(setup_script.resolve()),
            "setup_script_sha256": sha256_file(setup_script),
            "run_script": str(run_script.resolve()),
            "run_script_sha256": sha256_file(run_script),
            "setup_base_pixi_python": str(base_pixi_python()),
            "runtime_root_policy": str(
                DEFAULT_RUNTIME_ROOT
                / f"real_k4_halfmap_{dataset.key}_{profile.name}_seed{args.seed}_<job_id>"
            ),
            "relion_tmpdir_policy": "<runtime_root>/relion_half<half>",
        },
        "claim_boundary": {
            "independent_halfmaps": True,
            "construction": "two independent K=4 processes per engine, one process per frozen particle half",
            "recovar_internal_half_labels_are_combined_replicas": True,
            "final_all_data_maps_are_not_halfmaps": True,
            "initial_maps_use_both_halves_but_are_shared_and_lowpassed_to_30_angstrom": True,
        },
    }
    manifest_path = root / "submission_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (root / "submission_manifest.sha256").write_text(f"{sha256_file(manifest_path)}  {manifest_path.resolve()}\n")

    result: dict[str, Any] = {
        "output_root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "setup_script": str(setup_script),
        "run_script": str(run_script),
        "profile": profile.name,
        "dataset": dataset.label,
        "particle_count": len(selected_names),
        "half_counts": [row["particle_count"] for row in halves],
        "submitted": False,
    }
    if args.submit:
        submission = submit_scripts(setup_script, run_script)
        result.update({"submitted": True, **submission})
        (root / "submitted_jobs.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = prepare(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LaunchError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2) from exc
