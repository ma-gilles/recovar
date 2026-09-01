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
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RIBO_PDB_DIR = Path("/home/mg6942/mytigress/cryobench2/Ribosembly/pdbs")
DEFAULT_IGG_PDB_DIR = Path("/home/mg6942/mytigress/cryobench2/IgG-1D/pdbs")
DEFAULT_TOMOTWIN_PDB_DIR = Path("/home/mg6942/mytigress/cryobench2/Tomotwin-100/pdbs")
DEFAULT_IGG_RL_PDB_DIR = Path("/home/mg6942/mytigress/cryobench2/IgG-RL/pdbs")
DEFAULT_RUNTIME_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime")
RELION_DISPATCH_LOG_SCHEMA_MARKER = b"RELION_DISPATCH_LOG_SCHEMA_V2"
KCLASS_INITIAL_RESOLUTION_ANG = 60.0
THREE_SEED_VALUES = (41001, 41002, 41003)


def base_pixi_python() -> Path:
    """Return the installed pixi Python used to seed the run-local venv."""

    configured = os.environ.get(
        "EM_KCLASS_MATRIX_PIXI_PY",
        str(REPO_ROOT / ".pixi" / "envs" / "default" / "bin" / "python"),
    )
    return Path(configured).expanduser().resolve()


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
    image_batch_size: int | None = None
    rotation_block_size: int | None = None
    symmetry: str = "C1"
    base_name: str | None = None
    base_seed: int | None = None
    seed_replicate: int | None = None
    shared_input_group: str | None = None
    shared_input_producer: bool = False

    @property
    def seed_suite_base_name(self) -> str:
        return self.base_name or self.name

    @property
    def seed_suite_base_seed(self) -> int:
        return self.seed if self.base_seed is None else self.base_seed

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
            "" if self.image_batch_size is None else str(self.image_batch_size),
            "" if self.rotation_block_size is None else str(self.rotation_block_size),
            self.symmetry,
            self.seed_suite_base_name,
            str(self.seed_suite_base_seed),
            "" if self.seed_replicate is None else str(self.seed_replicate),
            "" if self.shared_input_group is None else self.shared_input_group,
            "producer" if self.shared_input_producer else ("consumer" if self.shared_input_group else ""),
            str(self.pdb_dir),
        ]


DEFAULT_CASES: tuple[Case, ...] = (
    Case(
        1,
        "ribo_k2_10k_g128_white_noise1_uniform",
        DEFAULT_RIBO_PDB_DIR,
        2,
        10_000,
        128,
        1.0,
        "white",
        "uniform",
        "uniform",
        2801,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "05:00:00",
        "192G",
        500,
        False,
    ),
    Case(
        2,
        "ribo_k4_10k_g128_white_noise1_uniform",
        DEFAULT_RIBO_PDB_DIR,
        4,
        10_000,
        128,
        1.0,
        "white",
        "uniform",
        "uniform",
        2802,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "06:00:00",
        "256G",
        500,
        False,
    ),
    Case(
        3,
        "ribo_k4_10k_g128_radial_noise3_nonuniform_linear",
        DEFAULT_RIBO_PDB_DIR,
        4,
        10_000,
        128,
        3.0,
        "radial1",
        "nonuniform",
        "linear",
        2803,
        80.0,
        10,
        0.2,
        0.2,
        0.7,
        0.0,
        0.0,
        5,
        "06:00:00",
        "256G",
        500,
        False,
    ),
    Case(
        4,
        "ribo_k8_10k_g128_white_noise3_kent_headheavy",
        DEFAULT_RIBO_PDB_DIR,
        8,
        10_000,
        128,
        3.0,
        "white",
        "kent",
        "head-heavy",
        2804,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "08:00:00",
        "320G",
        500,
        False,
    ),
    Case(
        5,
        "ribo_k4_50k_g256_white_noise1_uniform",
        DEFAULT_RIBO_PDB_DIR,
        4,
        50_000,
        256,
        1.0,
        "white",
        "uniform",
        "uniform",
        2805,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        8,
        "18:00:00",
        "500G",
        1000,
        True,
    ),
    Case(
        6,
        "ribo_k4_50k_g256_radial_noise3_nonuniform_linear",
        DEFAULT_RIBO_PDB_DIR,
        4,
        50_000,
        256,
        3.0,
        "radial1",
        "nonuniform",
        "linear",
        2806,
        80.0,
        10,
        0.2,
        0.2,
        0.7,
        0.0,
        0.0,
        8,
        "18:00:00",
        "500G",
        1000,
        True,
    ),
    Case(
        7,
        "ribo_k16_20k_g128_white_noise3_uniform",
        DEFAULT_RIBO_PDB_DIR,
        16,
        20_000,
        128,
        3.0,
        "white",
        "uniform",
        "uniform",
        2807,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "12:00:00",
        "500G",
        500,
        False,
    ),
    Case(
        8,
        "igg_k4_10k_g128_white_noise1_uniform",
        DEFAULT_IGG_PDB_DIR,
        4,
        10_000,
        128,
        1.0,
        "white",
        "uniform",
        "uniform",
        2808,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "06:00:00",
        "256G",
        500,
        False,
    ),
    Case(
        9,
        "igg_k8_10k_g128_radial_noise3_nonuniform",
        DEFAULT_IGG_PDB_DIR,
        8,
        10_000,
        128,
        3.0,
        "radial1",
        "nonuniform",
        "linear",
        2809,
        80.0,
        10,
        0.2,
        0.2,
        0.7,
        0.0,
        0.0,
        5,
        "08:00:00",
        "320G",
        500,
        False,
    ),
    Case(
        10,
        "ribo_k4_10k_g128_radial_noise3_nonuniform_outliers_pct20",
        DEFAULT_RIBO_PDB_DIR,
        4,
        10_000,
        128,
        3.0,
        "radial1",
        "nonuniform",
        "linear",
        2810,
        80.0,
        10,
        0.2,
        0.2,
        0.7,
        0.5,
        0.20,
        5,
        "06:00:00",
        "256G",
        500,
        False,
    ),
    Case(
        11,
        "igg_k4_10k_g128_white_noise1_uniform_outliers_pct20",
        DEFAULT_IGG_PDB_DIR,
        4,
        10_000,
        128,
        1.0,
        "white",
        "uniform",
        "uniform",
        2811,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.20,
        5,
        "06:00:00",
        "256G",
        500,
        False,
    ),
    Case(
        12,
        "tomotwin_k4_10k_g128_white_noise1_uniform",
        DEFAULT_TOMOTWIN_PDB_DIR,
        4,
        10_000,
        128,
        1.0,
        "white",
        "uniform",
        "uniform",
        2812,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "06:00:00",
        "256G",
        500,
        False,
    ),
    Case(
        13,
        "tomotwin_k8_10k_g128_radial_noise3_kent_headheavy",
        DEFAULT_TOMOTWIN_PDB_DIR,
        8,
        10_000,
        128,
        3.0,
        "radial1",
        "kent",
        "head-heavy",
        2813,
        80.0,
        10,
        0.2,
        0.2,
        0.7,
        0.0,
        0.0,
        5,
        "08:00:00",
        "320G",
        500,
        False,
    ),
    Case(
        14,
        "igg_rl_k4_10k_g128_white_noise1_uniform",
        DEFAULT_IGG_RL_PDB_DIR,
        4,
        10_000,
        128,
        1.0,
        "white",
        "uniform",
        "uniform",
        2814,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "06:00:00",
        "256G",
        500,
        False,
    ),
    Case(
        15,
        "igg_rl_k4_10k_g128_radial_noise3_nonuniform_outliers_pct20",
        DEFAULT_IGG_RL_PDB_DIR,
        4,
        10_000,
        128,
        3.0,
        "radial1",
        "nonuniform",
        "linear",
        2815,
        80.0,
        10,
        0.2,
        0.2,
        0.7,
        0.5,
        0.20,
        5,
        "06:00:00",
        "256G",
        500,
        False,
    ),
    Case(
        16,
        "ribo_k4_3k_g128_white_noise10_uniform",
        DEFAULT_RIBO_PDB_DIR,
        4,
        3_000,
        128,
        10.0,
        "white",
        "uniform",
        "uniform",
        2816,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "04:00:00",
        "192G",
        500,
        False,
    ),
    Case(
        17,
        "ribo_k4_3k_g128_radial_noise3_noctf_uniform",
        DEFAULT_RIBO_PDB_DIR,
        4,
        3_000,
        128,
        3.0,
        "radial1",
        "noctf",
        "uniform",
        2817,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "04:00:00",
        "192G",
        500,
        False,
    ),
    Case(
        18,
        "ribo_k4_3k_g128_white_noise1_contrast_noise_scale",
        DEFAULT_RIBO_PDB_DIR,
        4,
        3_000,
        128,
        1.0,
        "white",
        "uniform",
        "uniform",
        2818,
        80.0,
        10,
        0.5,
        0.5,
        0.7,
        0.0,
        0.0,
        5,
        "04:00:00",
        "192G",
        500,
        False,
    ),
    Case(
        19,
        "ribo_k4_3k_g128_white_noise1_image_offset",
        DEFAULT_RIBO_PDB_DIR,
        4,
        3_000,
        128,
        1.0,
        "white",
        "uniform",
        "uniform",
        2819,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        1.0,
        0.0,
        5,
        "04:00:00",
        "192G",
        500,
        False,
    ),
    Case(
        20,
        "ribo_k4_3k_g128_radial_noise5_severe_outliers_pct50",
        DEFAULT_RIBO_PDB_DIR,
        4,
        3_000,
        128,
        5.0,
        "radial1",
        "nonuniform",
        "linear",
        2820,
        80.0,
        10,
        0.7,
        0.7,
        0.7,
        1.5,
        0.50,
        5,
        "06:00:00",
        "256G",
        500,
        False,
    ),
    Case(
        21,
        "ribo_k4_3k_g128_white_noise0p2_uniform",
        DEFAULT_RIBO_PDB_DIR,
        4,
        3_000,
        128,
        0.2,
        "white",
        "uniform",
        "uniform",
        2821,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "04:00:00",
        "192G",
        500,
        False,
    ),
    Case(
        22,
        "ribo_k4_3k_g128_white_noise0p2_kent_headheavy",
        DEFAULT_RIBO_PDB_DIR,
        4,
        3_000,
        128,
        0.2,
        "white",
        "kent",
        "head-heavy",
        2822,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "04:00:00",
        "192G",
        500,
        False,
    ),
    Case(
        23,
        "ribo_k4_3k_g128_white_noise1_extreme_class_imbalance",
        DEFAULT_RIBO_PDB_DIR,
        4,
        3_000,
        128,
        1.0,
        "white",
        "uniform",
        "custom:0.80,0.10,0.07,0.03",
        2823,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "04:00:00",
        "192G",
        500,
        False,
    ),
    Case(
        24,
        "ribo_k4_3k_g256_radial_noise3_highres",
        DEFAULT_RIBO_PDB_DIR,
        4,
        3_000,
        256,
        3.0,
        "radial1",
        "uniform",
        "uniform",
        2824,
        0.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "08:00:00",
        "500G",
        500,
        True,
    ),
    Case(
        25,
        "ribo_k4_3k_g128_white_noise1_batch50",
        DEFAULT_RIBO_PDB_DIR,
        4,
        3_000,
        128,
        1.0,
        "white",
        "uniform",
        "uniform",
        2825,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "04:00:00",
        "192G",
        500,
        False,
        50,
        8192,
        shared_input_group="ribo_k4_3k_g128_white_noise1_invariance",
        shared_input_producer=True,
    ),
    Case(
        26,
        "ribo_k4_3k_g128_white_noise1_batch17",
        DEFAULT_RIBO_PDB_DIR,
        4,
        3_000,
        128,
        1.0,
        "white",
        "uniform",
        "uniform",
        2825,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "04:00:00",
        "192G",
        500,
        False,
        17,
        8192,
        shared_input_group="ribo_k4_3k_g128_white_noise1_invariance",
    ),
    Case(
        27,
        "ribo_k4_3k_g128_white_noise1_rotation_block257",
        DEFAULT_RIBO_PDB_DIR,
        4,
        3_000,
        128,
        1.0,
        "white",
        "uniform",
        "uniform",
        2825,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "04:00:00",
        "192G",
        500,
        False,
        50,
        257,
        shared_input_group="ribo_k4_3k_g128_white_noise1_invariance",
    ),
    Case(
        28,
        "ribo_k4_3k_g128_white_noise1_seed3802",
        DEFAULT_RIBO_PDB_DIR,
        4,
        3_000,
        128,
        1.0,
        "white",
        "uniform",
        "uniform",
        3802,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "04:00:00",
        "192G",
        500,
        False,
    ),
    Case(
        29,
        "ribo_k4_3k_g128_white_noise1_seed4802",
        DEFAULT_RIBO_PDB_DIR,
        4,
        3_000,
        128,
        1.0,
        "white",
        "uniform",
        "uniform",
        4802,
        80.0,
        10,
        0.0,
        0.0,
        0.7,
        0.0,
        0.0,
        5,
        "04:00:00",
        "192G",
        500,
        False,
    ),
    Case(
        index=30,
        name="ribo_k4_3k_g128_white_noise1_noctf_positive_control",
        pdb_dir=DEFAULT_RIBO_PDB_DIR,
        n_classes=4,
        n_images=3_000,
        grid_size=128,
        noise_level=1.0,
        noise_model="white",
        dataset_params_option="noctf",
        class_distribution="uniform",
        seed=2830,
        pdb_bfactor=80.0,
        init_radius=10,
        noise_scale_std=0.0,
        contrast_std=0.0,
        volume_radius=0.7,
        image_offset_n_std=0.0,
        percent_outliers=0.0,
        max_iter=5,
        time_limit="04:00:00",
        mem="192G",
        streaming_chunk_size=500,
        streaming_mmap=False,
    ),
    Case(
        index=31,
        name="ribo_k4_5k_g128_white_noise1_c4_uniform",
        pdb_dir=DEFAULT_RIBO_PDB_DIR,
        n_classes=4,
        n_images=5_000,
        grid_size=128,
        noise_level=1.0,
        noise_model="white",
        dataset_params_option="uniform",
        class_distribution="uniform",
        seed=41001,
        pdb_bfactor=80.0,
        init_radius=10,
        noise_scale_std=0.0,
        contrast_std=0.0,
        volume_radius=0.7,
        image_offset_n_std=0.0,
        percent_outliers=0.0,
        max_iter=5,
        time_limit="06:00:00",
        mem="256G",
        streaming_chunk_size=500,
        streaming_mmap=False,
        symmetry="C4",
    ),
    Case(
        index=32,
        name="ribo_k4_5k_g128_white_noise1_d4_uniform",
        pdb_dir=DEFAULT_RIBO_PDB_DIR,
        n_classes=4,
        n_images=5_000,
        grid_size=128,
        noise_level=1.0,
        noise_model="white",
        dataset_params_option="uniform",
        class_distribution="uniform",
        seed=41001,
        pdb_bfactor=80.0,
        init_radius=10,
        noise_scale_std=0.0,
        contrast_std=0.0,
        volume_radius=0.7,
        image_offset_n_std=0.0,
        percent_outliers=0.0,
        max_iter=5,
        time_limit="06:00:00",
        mem="256G",
        streaming_chunk_size=500,
        streaming_mmap=False,
        symmetry="D4",
    ),
    Case(
        index=33,
        name="ribo_k4_5k_g128_white_noise1_o_uniform",
        pdb_dir=DEFAULT_RIBO_PDB_DIR,
        n_classes=4,
        n_images=5_000,
        grid_size=128,
        noise_level=1.0,
        noise_model="white",
        dataset_params_option="uniform",
        class_distribution="uniform",
        seed=41001,
        pdb_bfactor=80.0,
        init_radius=10,
        noise_scale_std=0.0,
        contrast_std=0.0,
        volume_radius=0.7,
        image_offset_n_std=0.0,
        percent_outliers=0.0,
        max_iter=5,
        time_limit="06:00:00",
        mem="256G",
        streaming_chunk_size=500,
        streaming_mmap=False,
        symmetry="O",
    ),
    Case(
        index=34,
        name="ribo_k4_5k_g128_white_noise1_i1_uniform",
        pdb_dir=DEFAULT_RIBO_PDB_DIR,
        n_classes=4,
        n_images=5_000,
        grid_size=128,
        noise_level=1.0,
        noise_model="white",
        dataset_params_option="uniform",
        class_distribution="uniform",
        seed=41001,
        pdb_bfactor=80.0,
        init_radius=10,
        noise_scale_std=0.0,
        contrast_std=0.0,
        volume_radius=0.7,
        image_offset_n_std=0.0,
        percent_outliers=0.0,
        max_iter=5,
        time_limit="08:00:00",
        mem="320G",
        streaming_chunk_size=500,
        streaming_mmap=False,
        symmetry="I1",
    ),
)


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_contains_marker(path: Path, marker: bytes) -> bool:
    """Return whether *marker* occurs in *path* without loading the whole file."""

    overlap = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            candidate = overlap + chunk
            if marker in candidate:
                return True
            overlap = candidate[-(len(marker) - 1) :] if len(marker) > 1 else b""
    return False


def validate_relion_dispatch_executable(value: str) -> Path:
    """Validate that RELION can emit the v2 five-column dispatch capture."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SystemExit(f"EM_KCLASS_MATRIX_RELION_REFINE_MPI must be absolute: {path}")
    path = path.resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"EM_KCLASS_MATRIX_RELION_REFINE_MPI is not an executable file: {path}")
    if not _file_contains_marker(path, RELION_DISPATCH_LOG_SCHEMA_MARKER):
        marker = RELION_DISPATCH_LOG_SCHEMA_MARKER.decode("ascii")
        raise SystemExit(
            "EM_KCLASS_MATRIX_RELION_REFINE_MPI lacks required instrumentation marker "
            f"{marker}: {path}. Strict K>1 parity requires RELION dispatch-log schema v2 "
            "five-column identity records; legacy four-column range capture is rejected."
        )
    return path


def audit_numbered_class_maps(
    *,
    recovar_intermediates_dir: Path,
    relion_dir: Path,
    n_classes: int,
) -> dict[str, object]:
    """Validate RECOVAR numbered class maps against actual RELION iterations."""

    if n_classes <= 0:
        raise ValueError("n_classes must be positive")

    relion_pattern = re.compile(r"run_it(\d{3})_model\.star")
    relion_iterations = sorted(
        int(match.group(1))
        for path in relion_dir.glob("run_it*_model.star")
        if (match := relion_pattern.fullmatch(path.name)) is not None and int(match.group(1)) > 0
    )
    if not relion_iterations:
        raise ValueError(f"No numbered RELION model STARs found in {relion_dir}")
    expected_relion_iterations = list(range(1, relion_iterations[-1] + 1))
    if relion_iterations != expected_relion_iterations:
        raise ValueError(
            "RELION numbered iterations are not contiguous: "
            f"expected {expected_relion_iterations}, found {relion_iterations}"
        )

    map_pattern = re.compile(r"it(\d{3})_half([12])_class(\d+)_reg\.mrc")
    map_paths = sorted(recovar_intermediates_dir.glob("it*_half*_class*_reg.mrc"))
    actual_maps: dict[tuple[int, int, int], Path] = {}
    for path in map_paths:
        match = map_pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Malformed numbered RECOVAR class-map name: {path}")
        key = tuple(int(value) for value in match.groups())
        if key in actual_maps:
            raise ValueError(f"Duplicate numbered RECOVAR class-map identity {key}: {path}")
        if path.stat().st_size <= 0:
            raise ValueError(f"Empty numbered RECOVAR class map: {path}")
        actual_maps[key] = path

    expected_maps = {
        (relion_iteration - 1, half, class_number)
        for relion_iteration in relion_iterations
        for half in (1, 2)
        for class_number in range(1, n_classes + 1)
    }
    actual_keys = set(actual_maps)
    missing = sorted(expected_maps - actual_keys)
    unexpected = sorted(actual_keys - expected_maps)
    if missing or unexpected:
        raise ValueError(
            "RECOVAR numbered class maps do not match the actual RELION trajectory: "
            f"missing={missing}, unexpected={unexpected}"
        )

    checksums = []
    for key in sorted(actual_maps):
        path = actual_maps[key]
        checksums.append(
            {
                "recovar_iteration": key[0],
                "relion_iteration": key[0] + 1,
                "half": key[1],
                "class": key[2],
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
            }
        )
    report: dict[str, object] = {
        "schema_version": 1,
        "n_classes": n_classes,
        "relion_numbered_iterations": relion_iterations,
        "recovar_numbered_iterations": [iteration - 1 for iteration in relion_iterations],
        "maps_per_iteration": 2 * n_classes,
        "map_count": len(checksums),
        "maps": checksums,
    }
    recovar_intermediates_dir.mkdir(parents=True, exist_ok=True)
    (recovar_intermediates_dir / "numbered_class_map_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (recovar_intermediates_dir / "numbered_class_maps.sha256").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in checksums)
    )
    return report


def audit_relion_class_populations(*, relion_dir: Path, n_classes: int) -> dict[str, object]:
    """Fail closed if any numbered RELION iteration contains a collapsed class."""

    import starfile

    if n_classes <= 0:
        raise ValueError("n_classes must be positive")
    model_pattern = re.compile(r"run_it(\d{3})_model\.star")
    model_paths = sorted(
        path
        for path in relion_dir.glob("run_it*_model.star")
        if (match := model_pattern.fullmatch(path.name)) is not None and int(match.group(1)) > 0
    )
    if not model_paths:
        raise ValueError(f"No numbered RELION model STARs found in {relion_dir}")

    rows: list[dict[str, object]] = []
    collapsed: list[dict[str, object]] = []
    for path in model_paths:
        iteration = int(model_pattern.fullmatch(path.name).group(1))  # type: ignore[union-attr]
        data = starfile.read(path, always_dict=True)
        classes = data.get("model_classes")
        if classes is None or not hasattr(classes, "columns"):
            raise ValueError(f"RELION model STAR lacks model_classes table: {path}")
        if len(classes) != n_classes or "rlnClassDistribution" not in classes.columns:
            raise ValueError(
                f"RELION model STAR has invalid class table at iteration {iteration}: "
                f"rows={len(classes)}, expected={n_classes}, columns={list(classes.columns)}"
            )
        for class_number in range(1, n_classes + 1):
            distribution = float(classes.iloc[class_number - 1]["rlnClassDistribution"])
            orient_key = f"model_pdf_orient_class_{class_number}"
            orientations = data.get(orient_key)
            if (
                orientations is None
                or not hasattr(orientations, "columns")
                or "rlnOrientationDistribution" not in orientations.columns
            ):
                raise ValueError(f"RELION model STAR lacks {orient_key} distribution: {path}")
            orientation_mass = float(orientations["rlnOrientationDistribution"].sum())
            row = {
                "iteration": iteration,
                "class": class_number,
                "class_distribution": distribution,
                "orientation_mass": orientation_mass,
            }
            rows.append(row)
            if (
                not math.isfinite(distribution)
                or distribution <= 0.0
                or not math.isfinite(orientation_mass)
                or orientation_mass <= 0.0
            ):
                collapsed.append(row)

    report: dict[str, object] = {
        "schema": "recovar.em.relion_class_population_audit.v1",
        "n_classes": n_classes,
        "numbered_iterations": sorted({int(row["iteration"]) for row in rows}),
        "rows": rows,
        "collapsed": collapsed,
        "passed": not collapsed,
    }
    output = relion_dir / "class_population_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if collapsed:
        identities = ", ".join(f"it{int(row['iteration']):03d}/class{int(row['class']):03d}" for row in collapsed)
        raise ValueError(
            "RELION class-collapse gate failed: every numbered class must retain "
            f"positive class and orientation mass; collapsed={identities}; audit={output}"
        )
    return report


SHARED_INPUT_SIGNATURE_FIELDS = (
    "pdb_dir",
    "n_classes",
    "n_images",
    "grid_size",
    "noise_level",
    "noise_model",
    "dataset_params_option",
    "class_distribution",
    "seed",
    "pdb_bfactor",
    "init_radius",
    "noise_scale_std",
    "contrast_std",
    "volume_radius",
    "image_offset_n_std",
    "percent_outliers",
    "streaming_chunk_size",
    "streaming_mmap",
    "symmetry",
)


def shared_input_key(case: Case) -> str | None:
    """Return the seed-qualified shared-input identity for one case."""

    if case.shared_input_group is None:
        return None
    return f"{case.shared_input_group}_seed{case.seed}"


def validate_shared_input_groups(cases: list[Case]) -> dict[str, Case]:
    """Validate one producer and one immutable generator contract per group."""

    grouped: dict[str, list[Case]] = {}
    for case in cases:
        key = shared_input_key(case)
        if key is not None:
            grouped.setdefault(key, []).append(case)

    producers: dict[str, Case] = {}
    for key, members in grouped.items():
        group_producers = [case for case in members if case.shared_input_producer]
        if len(group_producers) != 1:
            raise SystemExit(
                f"shared input group {key} requires exactly one selected producer, "
                f"found {[case.name for case in group_producers]}; include the producer case"
            )
        producer = group_producers[0]
        expected = tuple(getattr(producer, field) for field in SHARED_INPUT_SIGNATURE_FIELDS)
        for case in members:
            observed = tuple(getattr(case, field) for field in SHARED_INPUT_SIGNATURE_FIELDS)
            if observed != expected:
                changed = [
                    field
                    for field, left, right in zip(
                        SHARED_INPUT_SIGNATURE_FIELDS,
                        expected,
                        observed,
                        strict=True,
                    )
                    if left != right
                ]
                raise SystemExit(f"shared input group {key} changes generator fields for {case.name}: {changed}")
        producers[key] = producer
    return producers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Poll squeue until the summary job leaves the queue.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Write scripts and tables but do not submit sbatch jobs."
    )
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
        help=("Add an offset to each selected case seed. Also available as EM_KCLASS_MATRIX_SEED_OFFSET."),
    )
    parser.add_argument(
        "--three-seed-suite",
        action="store_true",
        help=(
            "Expand every selected scientific case over frozen seeds "
            f"{','.join(str(seed) for seed in THREE_SEED_VALUES)} and emit a multi-seed aggregate. "
            "Also available as EM_KCLASS_MATRIX_THREE_SEED_SUITE=1."
        ),
    )
    return parser.parse_args()


def selected_cases(args: argparse.Namespace) -> list[Case]:
    requested = list(args.case)
    env_cases = os.environ.get("EM_KCLASS_MATRIX_CASES", "")
    requested.extend(part.strip() for part in env_cases.split(",") if part.strip())
    if not requested:
        out = list(DEFAULT_CASES)
    else:
        out = []
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
    three_seed_raw = os.environ.get("EM_KCLASS_MATRIX_THREE_SEED_SUITE", "")
    three_seed_suite = bool(getattr(args, "three_seed_suite", False)) or three_seed_raw.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if seed_override is not None and seed_offset is not None:
        raise SystemExit(
            "Use either EM_KCLASS_MATRIX_SEED / --seed-override or EM_KCLASS_MATRIX_SEED_OFFSET / --seed-offset, not both"
        )
    if three_seed_suite and (seed_override is not None or seed_offset is not None):
        raise SystemExit(
            "EM_KCLASS_MATRIX_THREE_SEED_SUITE / --three-seed-suite cannot be combined "
            "with a seed override or seed offset"
        )

    if (
        not max_iter_override
        and not time_limit_override
        and seed_override is None
        and seed_offset is None
        and not three_seed_suite
    ):
        return cases

    configured_cases = []
    for case in cases:
        updated = case
        if max_iter_override:
            updated = replace(updated, max_iter=max_iter_override)
        if time_limit_override:
            updated = replace(updated, time_limit=time_limit_override)
        if seed_override is not None or seed_offset is not None:
            new_seed = seed_override if seed_override is not None else case.seed + int(seed_offset or 0)
            updated = replace(
                updated,
                seed=new_seed,
                name=f"{updated.name}_seed{new_seed}",
                base_name=case.seed_suite_base_name,
                base_seed=case.seed_suite_base_seed,
            )
        configured_cases.append(updated)

    if not three_seed_suite:
        return configured_cases

    expanded = []
    for case in configured_cases:
        for replicate, seed in enumerate(THREE_SEED_VALUES, start=1):
            expanded.append(
                replace(
                    case,
                    name=f"{case.name}_seed{seed}",
                    seed=seed,
                    base_name=case.seed_suite_base_name,
                    base_seed=case.seed_suite_base_seed,
                    seed_replicate=replicate,
                )
            )
    return expanded


def sbatch_directive(flag: str, value: str | None) -> str:
    return f"#SBATCH {flag}={value}" if value else ""


def build_cuda_lib_command() -> str:
    return """mkdir -p "$(dirname "${RECOVAR_CUDA_LIB}")"
CUDA_LIB_TMP="${RECOVAR_CUDA_LIB}.${SLURM_JOB_ID:-$$}.tmp"
export CUDA_LIB_TMP PIXI_PY
flock "$(dirname "${RECOVAR_CUDA_LIB}")/build.lock" bash -lc '
  set -euo pipefail
  if [[ -s "${RECOVAR_CUDA_LIB}" && -s "${RECOVAR_CUDA_LIB}.sha256" ]]; then
    sha256sum --check "${RECOVAR_CUDA_LIB}.sha256"
    echo "Reusing sealed CUDA library ${RECOVAR_CUDA_LIB}"
    exit 0
  fi
  rm -f "${CUDA_LIB_TMP}"
  env PYTHON="${PIXI_PY}" make -C recovar/cuda LIB="${CUDA_LIB_TMP}" all
  mv -f "${CUDA_LIB_TMP}" "${RECOVAR_CUDA_LIB}"
  sha256sum "${RECOVAR_CUDA_LIB}" > "${RECOVAR_CUDA_LIB}.sha256"
'
sha256sum --check "${RECOVAR_CUDA_LIB}.sha256"
"""


def verify_cuda_lib_command() -> str:
    return """if [[ ! -s "${RECOVAR_CUDA_LIB}" || ! -s "${RECOVAR_CUDA_LIB}.sha256" ]]; then
  echo "ERROR: setup did not seal the shared CUDA library: ${RECOVAR_CUDA_LIB}" >&2
  exit 2
fi
sha256sum --check "${RECOVAR_CUDA_LIB}.sha256"
"""


def git_provenance_gate(*, expected_commit: str) -> str:
    return f"""EXPECTED_GIT_HEAD={q(expected_commit)}
ACTUAL_GIT_HEAD="$(git rev-parse HEAD)"
if [[ "${{ACTUAL_GIT_HEAD}}" != "${{EXPECTED_GIT_HEAD}}" ]]; then
  echo "ERROR: queued-job Git HEAD drift: expected ${{EXPECTED_GIT_HEAD}}, got ${{ACTUAL_GIT_HEAD}}" >&2
  exit 2
fi
TRACKED_GIT_STATUS="$(git status --short --untracked-files=no)"
if [[ -n "${{TRACKED_GIT_STATUS}}" ]]; then
  echo "ERROR: queued-job worktree has tracked changes:" >&2
  printf '%s\\n' "${{TRACKED_GIT_STATUS}}" >&2
  exit 2
fi
echo "Queued-job Git provenance gate ok: ${{ACTUAL_GIT_HEAD}}"
"""


def job_preamble(
    *,
    scratch_dir: Path,
    cuda_lib: Path,
    cuda_module: str,
    relion_src_dir: Path,
    job_name: str,
    expected_commit: str,
) -> str:
    matrix_venv = scratch_dir / "venv"
    matrix_python = matrix_venv / "bin" / "python"
    pixi_env_root = base_pixi_python().parent.parent
    shared_relion_bind_dir = scratch_dir / "relion_bind_build" / "shared"
    return f"""set -euo pipefail
cd {q(REPO_ROOT)}
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_PROMPT_MODIFIER CONDA_SHLVL
# Strict matrix jobs do not inherit experiment/debug toggles from the submitter.
while IFS='=' read -r ENV_NAME _; do
  case "${{ENV_NAME}}" in
    RECOVAR_*|RELION_*|JAX_*|XLA_*) unset "${{ENV_NAME}}" ;;
  esac
done < <(env)
unset TF_GPU_ALLOCATOR
export PYTHONNOUSERSITE=1
export RECOVAR_EXPECTED_REPO_ROOT={q(REPO_ROOT)}
export RELION_SRC_DIR={q(relion_src_dir)}
if [[ ! -f "${{RELION_SRC_DIR}}/projector.h" ]]; then
  echo "ERROR: RELION_SRC_DIR does not contain projector.h: ${{RELION_SRC_DIR}}" >&2
  exit 2
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PIXI_FROZEN=true
export EM_KCLASS_MATRIX_VENV={q(matrix_venv)}
export PIXI_PY={q(matrix_python)}
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export CMAKE_INCLUDE_PATH={q(pixi_env_root / "include" / "fftw")}:{q(pixi_env_root / "include")}:${{CMAKE_INCLUDE_PATH:-}}
export CMAKE_LIBRARY_PATH={q(pixi_env_root / "lib")}:${{CMAKE_LIBRARY_PATH:-}}
export RECOVAR_RELION_BIND_JOBS="${{SLURM_CPUS_ON_NODE:-${{SLURM_CPUS_PER_TASK:-1}}}}"
RUNTIME_ROOT={q(DEFAULT_RUNTIME_ROOT / job_name)}_${{SLURM_JOB_ID}}
export TMPDIR="${{RUNTIME_ROOT}}/tmp"
export PIXI_HOME="${{RUNTIME_ROOT}}/pixi_home"
export RATTLER_CACHE_DIR="${{RUNTIME_ROOT}}/rattler_cache"
export RECOVAR_JAX_CACHE_DIR={q(scratch_dir)}/jax_cache
export JAX_COMPILATION_CACHE_DIR="${{RECOVAR_JAX_CACHE_DIR}}"
export RECOVAR_CUDA_LIB={q(cuda_lib)}
export RECOVAR_CUDA_CACHE_DIR={q(scratch_dir)}/cuda_cache/{job_name}_${{SLURM_JOB_ID}}
export RECOVAR_RELION_BIND_BUILD_DIR={q(shared_relion_bind_dir)}
mkdir -p "${{TMPDIR}}" "${{PIXI_HOME}}" "${{RATTLER_CACHE_DIR}}" "${{RECOVAR_JAX_CACHE_DIR}}" "${{RECOVAR_CUDA_CACHE_DIR}}" "$(dirname "${{RECOVAR_CUDA_LIB}}")"
touch "${{RUNTIME_ROOT}}/SAFE_TO_DELETE"

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
PIXI_NVIDIA_ROOT={q(pixi_env_root)}/lib/python3.11/site-packages/nvidia
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

{git_provenance_gate(expected_commit=expected_commit)}

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
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader || true
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
    relion_src_dir: Path,
    expected_commit: str | None = None,
) -> Path:
    expected_commit = expected_commit or git_text("rev-parse", "HEAD")
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

{job_preamble(scratch_dir=scratch_dir, cuda_lib=cuda_lib, cuda_module=cuda_module, relion_src_dir=relion_src_dir, job_name="em_kclass_matrix_setup", expected_commit=expected_commit)}

BASE_PIXI_PY={q(base_pixi_python())}
if [[ ! -x "${{BASE_PIXI_PY}}" ]]; then
  echo "ERROR: EM_KCLASS_MATRIX_PIXI_PY must name an installed pixi Python: ${{BASE_PIXI_PY}}" >&2
  exit 2
fi
export BASE_PIXI_PY
flock {q(scratch_dir / "install-recovar.lock")} bash -lc '
set -euo pipefail
rm -rf "${{RECOVAR_RELION_BIND_BUILD_DIR:?}}"
rm -rf "${{EM_KCLASS_MATRIX_VENV:?}}"
mkdir -p "${{RECOVAR_RELION_BIND_BUILD_DIR}}"
"${{BASE_PIXI_PY}}" -m venv --system-site-packages "${{EM_KCLASS_MATRIX_VENV}}"
"${{PIXI_PY}}" -m pip install -e . --no-deps --no-build-isolation --ignore-installed
"${{PIXI_PY}}" recovar/relion_bind/build.py
'
mapfile -t RELION_BIND_LIBS < <(find "${{RECOVAR_RELION_BIND_BUILD_DIR}}" -maxdepth 1 -type f -name '_relion_bind_core*.so' -print)
if [[ "${{#RELION_BIND_LIBS[@]}}" -ne 1 ]]; then
  echo "ERROR: expected one sealed RELION binding, found ${{#RELION_BIND_LIBS[@]}}" >&2
  exit 2
fi
sha256sum "${{RELION_BIND_LIBS[0]}}" > {q(scratch_dir / "relion_bind_build" / "shared.sha256")}
sha256sum --check {q(scratch_dir / "relion_bind_build" / "shared.sha256")}
{build_cuda_lib_command()}
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export RECOVAR_DISABLE_CUDA=1
export CUDA_VISIBLE_DEVICES=""
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
    relion_src_dir: Path,
    relion_module: str,
    relion_refine_mpi: str,
    relion_mpi_ranks: int,
    relion_pool: int,
    particle_diameter: float,
    image_batch_size: int,
    rotation_block_size: int,
    gt_align_refine_orders: str,
    noise_rng_batch_size: str,
    expected_commit: str | None = None,
) -> Path:
    expected_commit = expected_commit or git_text("rev-parse", "HEAD")
    effective_image_batch_size = case.image_batch_size if case.image_batch_size is not None else image_batch_size
    effective_rotation_block_size = (
        case.rotation_block_size if case.rotation_block_size is not None else rotation_block_size
    )
    if effective_image_batch_size <= 0 or effective_rotation_block_size <= 0:
        raise ValueError("K-class image and rotation batch sizes must be positive")
    case_root = scratch_dir / "cases" / f"{case.index}_{case.name}"
    input_group_key = shared_input_key(case)
    if input_group_key is None:
        data_dir = case_root / "data"
        sub_pdb_dir = case_root / f"pdbs_k{case.n_classes}"
        shared_input_manifest = None
        relion_dir = case_root / "relion_ref"
        shared_relion_manifest = None
        generate_input = True
        run_relion = True
    else:
        shared_input_root = scratch_dir / "shared_inputs" / input_group_key
        data_dir = shared_input_root / "data"
        sub_pdb_dir = shared_input_root / f"pdbs_k{case.n_classes}"
        shared_input_manifest = shared_input_root / "sealed_inputs.sha256"
        relion_dir = shared_input_root / "relion_ref"
        shared_relion_manifest = shared_input_root / "sealed_relion_oracle.sha256"
        generate_input = case.shared_input_producer
        run_relion = case.shared_input_producer
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

{job_preamble(scratch_dir=scratch_dir, cuda_lib=cuda_lib, cuda_module=cuda_module, relion_src_dir=relion_src_dir, job_name=f"em_kclass_matrix_{case.index}_{case.name}", expected_commit=expected_commit)}

CASE_ROOT={q(case_root)}
DATA_DIR={q(data_dir)}
RECOVAR_DIR="${{CASE_ROOT}}/recovar"
RECOVAR_INTERMEDIATES_DIR="${{RECOVAR_DIR}}/intermediates"
RELION_DIR={q(relion_dir)}
RELION_DISPATCH_LOG="${{RELION_DIR}}/dispatch.tsv"
RELION_DISPATCH_SCHEDULE="${{RELION_DIR}}/dispatch_schedule.npz"
SUB_PDB_DIR={q(sub_pdb_dir)}
SHARED_INPUT_GROUP={q(input_group_key or "")}
SHARED_INPUT_MANIFEST={q(shared_input_manifest or "")}
SHARED_RELION_MANIFEST={q(shared_relion_manifest or "")}
GENERATE_INPUT={1 if generate_input else 0}
RUN_RELION={1 if run_relion else 0}
if [[ -e "${{CASE_ROOT}}" || -L "${{CASE_ROOT}}" ]]; then
  echo "ERROR: refusing to reuse an existing case root: ${{CASE_ROOT}}" >&2
  echo "Use a fresh immutable scratch root; a retry must not mix old and new evidence." >&2
  exit 2
fi
if [[ "${{GENERATE_INPUT}}" == "1" ]]; then
  if [[ -e "${{DATA_DIR}}" || -L "${{DATA_DIR}}" || ( -n "${{SHARED_INPUT_MANIFEST}}" && -e "${{SHARED_INPUT_MANIFEST}}" ) ]]; then
    echo "ERROR: refusing to regenerate or reseal an existing dataset: ${{DATA_DIR}}" >&2
    echo "Use a fresh immutable scratch root; stale partial outputs are not reusable evidence." >&2
    exit 2
  fi
fi
if [[ "${{RUN_RELION}}" == "1" && -n "${{SHARED_RELION_MANIFEST}}" ]]; then
  if [[ -e "${{RELION_DIR}}" || -L "${{RELION_DIR}}" || -e "${{SHARED_RELION_MANIFEST}}" ]]; then
    echo "ERROR: refusing to regenerate or reseal an existing shared RELION oracle: ${{RELION_DIR}}" >&2
    echo "Use a fresh immutable scratch root." >&2
    exit 2
  fi
fi
mkdir -p "${{CASE_ROOT}}" "${{RECOVAR_DIR}}" "${{RECOVAR_INTERMEDIATES_DIR}}"
if [[ "${{GENERATE_INPUT}}" == "1" ]]; then
  mkdir -p "${{DATA_DIR}}" "${{SUB_PDB_DIR}}"
fi
if [[ "${{RUN_RELION}}" == "1" ]]; then
  mkdir -p "${{RELION_DIR}}"
fi

RELION_DISPATCH_SCHEMA_MARKER={q(RELION_DISPATCH_LOG_SCHEMA_MARKER.decode("ascii"))}
if ! LC_ALL=C grep -aFq -- "${{RELION_DISPATCH_SCHEMA_MARKER}}" {q(relion_refine_mpi)}; then
  echo "ERROR: RELION dispatch-capture executable lacks required instrumentation marker ${{RELION_DISPATCH_SCHEMA_MARKER}}: {q(relion_refine_mpi)}" >&2
  echo "Strict K>1 parity requires dispatch-log schema v2 five-column identity records; legacy four-column range capture is rejected." >&2
  exit 2
fi

capture_physical_gpu_uuid() {{
  local slurm_gpu_token="${{SLURM_JOB_GPUS:-}}"
  slurm_gpu_token="${{slurm_gpu_token%%,*}}"
  local gpu_uuid=""
  mapfile -t visible_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed 's/[[:space:]]//g' | sed '/^$/d')
  if [[ "${{#visible_uuids[@]}}" -ne 1 || "${{visible_uuids[0]}}" != GPU-* ]]; then
    echo "ERROR: cannot identify exactly one visible physical GPU UUID (found ${{#visible_uuids[@]}}: ${{visible_uuids[*]:-<none>}})" >&2
    return 2
  fi
  gpu_uuid="${{visible_uuids[0]}}"
  # Numeric SLURM_JOB_GPUS values are host-physical indices, while nvidia-smi
  # is commonly cgroup-remapped to visible index 0 inside the allocation.
  if [[ "${{slurm_gpu_token}}" == GPU-* && "${{slurm_gpu_token}}" != "${{gpu_uuid}}" ]]; then
    echo "ERROR: Slurm GPU UUID ${{slurm_gpu_token}} does not match visible GPU UUID ${{gpu_uuid}}" >&2
    return 2
  fi
  printf '%s\\n' "${{gpu_uuid}}"
}}

CASE_GPU_UUID="$(capture_physical_gpu_uuid)"
printf '%s\\n' "${{CASE_GPU_UUID}}" > "${{CASE_ROOT}}/physical_gpu_uuid.txt"
nvidia-smi --query-gpu=timestamp,index,name,uuid,memory.total,driver_version --format=csv > "${{CASE_ROOT}}/physical_gpu_inventory.csv"

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
  "gpu_monitor_interval_s": 5,
  "legacy_combined_gpu_monitor_interval_s": 60,
  "initial_resolution_ang": {KCLASS_INITIAL_RESOLUTION_ANG},
  "particle_diameter_ang": {particle_diameter},
  "image_batch_size": {effective_image_batch_size},
  "rotation_block_size": {effective_rotation_block_size},
  "symmetry": "{case.symmetry}",
  "base_name": "{case.seed_suite_base_name}",
  "base_seed": {case.seed_suite_base_seed},
  "seed_replicate": {"null" if case.seed_replicate is None else case.seed_replicate},
  "case_root": "{case_root}",
  "slurm_job_id": "${{SLURM_JOB_ID}}",
  "data_dir": "{data_dir}",
  "shared_input_group": {json.dumps(case.shared_input_group)},
  "shared_input_key": {json.dumps(input_group_key)},
  "shared_input_role": {json.dumps("producer" if case.shared_input_producer else ("consumer" if input_group_key else None))},
  "shared_input_manifest": {json.dumps(str(shared_input_manifest) if shared_input_manifest else None)},
  "shared_relion_dir": {json.dumps(str(relion_dir) if input_group_key else None)},
  "shared_relion_manifest": {json.dumps(str(shared_relion_manifest) if shared_relion_manifest else None)}
}}
JSON

GPU_MONITOR_QUERY="timestamp,index,name,memory.used,memory.total,utilization.gpu"
COMBINED_MONITOR_PID=""
ENGINE_MONITOR_PID=""

stop_engine_gpu_monitor() {{
  if [[ -n "${{ENGINE_MONITOR_PID}}" ]]; then
    kill "${{ENGINE_MONITOR_PID}}" 2>/dev/null || true
    wait "${{ENGINE_MONITOR_PID}}" 2>/dev/null || true
    ENGINE_MONITOR_PID=""
  fi
}}

start_engine_gpu_monitor() {{
  local output_path="$1"
  stop_engine_gpu_monitor
  nvidia-smi --query-gpu="${{GPU_MONITOR_QUERY}}" --format=csv -l 5 > "${{output_path}}" &
  ENGINE_MONITOR_PID="$!"
}}

cleanup_gpu_monitors() {{
  stop_engine_gpu_monitor
  if [[ -n "${{COMBINED_MONITOR_PID}}" ]]; then
    kill "${{COMBINED_MONITOR_PID}}" 2>/dev/null || true
    wait "${{COMBINED_MONITOR_PID}}" 2>/dev/null || true
  fi
}}

# Keep the historical whole-case monitor for continuity, but never attribute
# its peak to either engine. Dedicated monitors below delimit RELION and
# RECOVAR independently.
nvidia-smi --query-gpu="${{GPU_MONITOR_QUERY}}" --format=csv -l 60 > "${{CASE_ROOT}}/gpu_monitor.csv" &
COMBINED_MONITOR_PID="$!"
trap cleanup_gpu_monitors EXIT

if [[ "${{GENERATE_INPUT}}" == "1" ]]; then
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
fi

sha256sum --check {q(scratch_dir / "relion_bind_build" / "shared.sha256")}
{verify_cuda_lib_command()}
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

if [[ "${{GENERATE_INPUT}}" == "1" ]]; then
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
  --symmetry {q(case.symmetry)} \\
  --seed {case.seed} \\
  2>&1 | tee "${{CASE_ROOT}}/prepare.log"
fi

if [[ -n "${{SHARED_INPUT_GROUP}}" ]]; then
  if [[ "${{GENERATE_INPUT}}" == "1" ]]; then
    MANIFEST_TMP="${{SHARED_INPUT_MANIFEST}}.${{SLURM_JOB_ID:-$$}}.tmp"
    find "${{DATA_DIR}}" -type f -print0 | sort -z | xargs -0 sha256sum > "${{MANIFEST_TMP}}"
    mv -f "${{MANIFEST_TMP}}" "${{SHARED_INPUT_MANIFEST}}"
  fi
  if [[ ! -s "${{SHARED_INPUT_MANIFEST}}" ]]; then
    echo "ERROR: shared input manifest is missing: ${{SHARED_INPUT_MANIFEST}}" >&2
    exit 2
  fi
  sha256sum --check "${{SHARED_INPUT_MANIFEST}}" \
    | tee -a "${{CASE_ROOT}}/prepare.log"
  ln -sfn "${{DATA_DIR}}" "${{CASE_ROOT}}/shared_data"
fi

if [[ "${{RUN_RELION}}" == "1" ]]; then
echo "=== Run RELION Class3D: {case.name} ==="
RELION_GPU_UUID="$(capture_physical_gpu_uuid)"
if [[ "${{RELION_GPU_UUID}}" != "${{CASE_GPU_UUID}}" ]]; then
  echo "ERROR: RELION physical GPU changed: expected ${{CASE_GPU_UUID}}, got ${{RELION_GPU_UUID}}" >&2
  exit 2
fi
printf '%s\\n' "${{RELION_GPU_UUID}}" > "${{RELION_DIR}}/physical_gpu_uuid.txt"
RELION_START="$(date +%s)"
start_engine_gpu_monitor "${{CASE_ROOT}}/relion_gpu_monitor.csv"
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
  RELION_RUNTIME_GPU_UUID="$(capture_physical_gpu_uuid)"
  if [[ "${{RELION_RUNTIME_GPU_UUID}}" != "${{CASE_GPU_UUID}}" ]]; then
    echo "ERROR: RELION runtime physical GPU changed: expected ${{CASE_GPU_UUID}}, got ${{RELION_RUNTIME_GPU_UUID}}" >&2
    exit 2
  fi
  printf '%s\\n' "${{RELION_RUNTIME_GPU_UUID}}" > "${{RELION_DIR}}/runtime_physical_gpu_uuid.txt"
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
      --ini_high {KCLASS_INITIAL_RESOLUTION_ANG:g} \\
      "${{RELION_CTF_ARGS[@]}}" \\
      --norm \\
      --scale \\
      --sym {q(case.symmetry)} \\
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
stop_engine_gpu_monitor
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

"${{PIXI_PY}}" - "${{RELION_DIR}}" {case.n_classes} <<'PY'
import pathlib
import sys

from scripts.run_em_kclass_robustness_matrix_slurm import audit_relion_class_populations

report = audit_relion_class_populations(
    relion_dir=pathlib.Path(sys.argv[1]),
    n_classes=int(sys.argv[2]),
)
print(
    "RELION class-population audit ok: "
    f"iterations={{report['numbered_iterations']}} rows={{len(report['rows'])}}"
)
PY
if [[ -n "${{SHARED_RELION_MANIFEST}}" ]]; then
  RELION_MANIFEST_TMP="${{SHARED_RELION_MANIFEST}}.${{SLURM_JOB_ID:-$$}}.tmp"
  find "${{RELION_DIR}}" -type f -print0 | sort -z | xargs -0 sha256sum > "${{RELION_MANIFEST_TMP}}"
  mv -f "${{RELION_MANIFEST_TMP}}" "${{SHARED_RELION_MANIFEST}}"
  sha256sum --check "${{SHARED_RELION_MANIFEST}}"
fi
else
  echo "=== Reuse sealed RELION oracle: {case.name} ==="
  if [[ ! -s "${{SHARED_RELION_MANIFEST}}" ]]; then
    echo "ERROR: shared RELION oracle manifest is missing: ${{SHARED_RELION_MANIFEST}}" >&2
    exit 2
  fi
  sha256sum --check "${{SHARED_RELION_MANIFEST}}" \
    | tee "${{CASE_ROOT}}/relion_oracle_verify.log"
  for REQUIRED_RELION_PATH in \
    "${{RELION_DISPATCH_SCHEDULE}}" \
    "${{RELION_DIR}}/run_it000_optimiser.star" \
    "${{RELION_DIR}}/class_population_audit.json" \
    "${{RELION_DIR}}/physical_gpu_uuid.txt"; do
    if [[ ! -s "${{REQUIRED_RELION_PATH}}" ]]; then
      echo "ERROR: sealed shared RELION oracle lacks ${{REQUIRED_RELION_PATH}}" >&2
      exit 2
    fi
  done
  RELION_GPU_UUID="$(<"${{RELION_DIR}}/physical_gpu_uuid.txt")"
fi
if [[ -n "${{SHARED_INPUT_GROUP}}" ]]; then
  # Keep one stable per-case oracle path for producers and consumers alike.
  # The summary and multiseed aggregation tools deliberately resolve the
  # class-population audit through this path, while RELION_DIR may point at a
  # producer-owned shared oracle outside CASE_ROOT.
  ln -sfn "${{RELION_DIR}}" "${{CASE_ROOT}}/relion_ref"
fi

echo "=== Run RECOVAR K-class refinement: {case.name} ==="
RECOVAR_GPU_UUID="$(capture_physical_gpu_uuid)"
if [[ "${{RECOVAR_GPU_UUID}}" != "${{CASE_GPU_UUID}}" ]]; then
  echo "ERROR: RECOVAR physical GPU changed: expected ${{CASE_GPU_UUID}}, got ${{RECOVAR_GPU_UUID}}" >&2
  exit 2
fi
if [[ "${{RUN_RELION}}" == "1" && "${{RECOVAR_GPU_UUID}}" != "${{RELION_GPU_UUID}}" ]]; then
  echo "ERROR: RECOVAR and RELION did not use the same physical GPU: RELION=${{RELION_GPU_UUID}} RECOVAR=${{RECOVAR_GPU_UUID}}" >&2
  exit 2
fi
printf '%s\\n' "${{RECOVAR_GPU_UUID}}" > "${{RECOVAR_DIR}}/physical_gpu_uuid.txt"
cat > "${{CASE_ROOT}}/paired_gpu_uuid.json" <<JSON
{{"physical_gpu_uuid":"${{CASE_GPU_UUID}}","relion_gpu_uuid":"${{RELION_GPU_UUID}}","recovar_gpu_uuid":"${{RECOVAR_GPU_UUID}}","relion_ran_in_case":$([[ "${{RUN_RELION}}" == "1" ]] && echo true || echo false),"hardware_comparable":$([[ "${{RUN_RELION}}" == "1" ]] && echo true || echo false)}}
JSON
rm -rf "${{RECOVAR_INTERMEDIATES_DIR}}"
mkdir -p "${{RECOVAR_INTERMEDIATES_DIR}}"
START_EPOCH="$(date +%s)"
start_engine_gpu_monitor "${{CASE_ROOT}}/recovar_gpu_monitor.csv"
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
  --sym {q(case.symmetry)} \\
  --init_resolution {KCLASS_INITIAL_RESOLUTION_ANG:g} \\
  --apply-initial-lowpass \\
  --firstiter_cc \\
  --image-fourier-backend relion_cuda \\
  --image_batch_size {effective_image_batch_size} \\
  --rotation_block_size {effective_rotation_block_size} \\
  --seed {case.seed} \\
  --relion_optimiser "${{RELION_DIR}}/run_it000_optimiser.star" \\
  --relion_init_dir "${{RELION_DIR}}" \\
  --perturb_replay_relion_dir "${{RELION_DIR}}" \\
  --relion-dispatch-schedule "${{RELION_DISPATCH_SCHEDULE}}" \\
  --particle_diameter_ang {particle_diameter:g} \\
  --tau2_fudge 4.0 \\
  --benchmark_ledger_json "${{RECOVAR_DIR}}/benchmark_ledger.json" \\
  --timing_dir "${{RECOVAR_DIR}}/timing" \\
  --save_intermediates_dir "${{RECOVAR_INTERMEDIATES_DIR}}" \\
  2>&1 | tee "${{RECOVAR_DIR}}/run_full_refinement.log"
STATUS="${{PIPESTATUS[0]}}"
set -e
END_EPOCH="$(date +%s)"
stop_engine_gpu_monitor
cat > "${{RECOVAR_DIR}}/slurm_walltime.json" <<JSON
{{"slurm_job_id":"${{SLURM_JOB_ID}}","start_epoch":${{START_EPOCH}},"end_epoch":${{END_EPOCH}},"external_wall_s":$((END_EPOCH - START_EPOCH)),"exit_status":${{STATUS}}}}
JSON
if [[ "${{STATUS}}" -ne 0 ]]; then
  exit "${{STATUS}}"
fi
RECOVAR_POST_GPU_UUID="$(capture_physical_gpu_uuid)"
if [[ "${{RECOVAR_POST_GPU_UUID}}" != "${{CASE_GPU_UUID}}" ]]; then
  echo "ERROR: RECOVAR runtime physical GPU changed: expected ${{CASE_GPU_UUID}}, got ${{RECOVAR_POST_GPU_UUID}}" >&2
  exit 2
fi
printf '%s\\n' "${{RECOVAR_POST_GPU_UUID}}" > "${{RECOVAR_DIR}}/runtime_physical_gpu_uuid.txt"

"${{PIXI_PY}}" - "${{RECOVAR_INTERMEDIATES_DIR}}" "${{RELION_DIR}}" {case.n_classes} <<'PY'
import pathlib
import sys

from scripts.run_em_kclass_robustness_matrix_slurm import audit_numbered_class_maps

report = audit_numbered_class_maps(
    recovar_intermediates_dir=pathlib.Path(sys.argv[1]),
    relion_dir=pathlib.Path(sys.argv[2]),
    n_classes=int(sys.argv[3]),
)
print(
    "Numbered class-map audit ok: "
    f"RELION iterations={{report['relion_numbered_iterations']}} "
    f"RECOVAR iterations={{report['recovar_numbered_iterations']}} "
    f"maps={{report['map_count']}}"
)
PY

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
    three_seed_suite: bool = False,
    expected_commit: str | None = None,
) -> Path:
    expected_commit = expected_commit or git_text("rev-parse", "HEAD")
    matrix_python = scratch_dir / "venv" / "bin" / "python"
    fallback_python = base_pixi_python()
    script = jobs_dir / "em_kclass_matrix_summary.sh"
    multiseed_command = ""
    if three_seed_suite:
        expected_seeds = ",".join(str(seed) for seed in THREE_SEED_VALUES)
        multiseed_command = f"""
"${{PIXI_PY}}" -m scripts.aggregate_em_kclass_multiseed \\
  {q(scratch_dir)} \\
  --matrix-summary {q(scratch_dir / "em_kclass_robustness_summary.json")} \\
  --case-table {q(scratch_dir / "case_table.tsv")} \\
  --expected-seeds {q(expected_seeds)} \\
  --output-markdown {q(scratch_dir / "em_kclass_multiseed_summary.md")} \\
  --output-json {q(scratch_dir / "em_kclass_multiseed_summary.json")}
tail -200 {q(scratch_dir / "em_kclass_multiseed_summary.md")} || true
"""
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
while IFS='=' read -r ENV_NAME _; do
  case "${{ENV_NAME}}" in
    RECOVAR_*|RELION_*|JAX_*|XLA_*) unset "${{ENV_NAME}}" ;;
  esac
done < <(env)
unset TF_GPU_ALLOCATOR
export PYTHONNOUSERSITE=1
export RECOVAR_DISABLE_CUDA=1
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu
export PIXI_FROZEN=true
MATRIX_PY={q(matrix_python)}
BASE_PIXI_PY={q(fallback_python)}
if [[ -x "${{MATRIX_PY}}" ]]; then
  export PIXI_PY="${{MATRIX_PY}}"
elif [[ -x "${{BASE_PIXI_PY}}" ]]; then
  export PIXI_PY="${{BASE_PIXI_PY}}"
else
  echo "ERROR: neither run-local nor base pixi Python is executable" >&2
  exit 2
fi
RUNTIME_ROOT={q(DEFAULT_RUNTIME_ROOT / "em_kclass_matrix_summary")}_${{SLURM_JOB_ID}}
export TMPDIR="${{RUNTIME_ROOT}}/tmp"
export PIXI_HOME="${{RUNTIME_ROOT}}/pixi_home"
export RATTLER_CACHE_DIR="${{RUNTIME_ROOT}}/rattler_cache"
export RECOVAR_JAX_CACHE_DIR={q(scratch_dir)}/jax_cache
export JAX_COMPILATION_CACHE_DIR="${{RECOVAR_JAX_CACHE_DIR}}"
mkdir -p "${{TMPDIR}}" "${{PIXI_HOME}}" "${{RATTLER_CACHE_DIR}}" "${{RECOVAR_JAX_CACHE_DIR}}"
touch "${{RUNTIME_ROOT}}/SAFE_TO_DELETE"

{git_provenance_gate(expected_commit=expected_commit)}

echo "=== EM K-class robustness matrix summary ==="
echo "Repo: {REPO_ROOT}"
echo "HEAD: $(git rev-parse HEAD)"
echo "Branch: $(git symbolic-ref --short HEAD || echo '<detached>')"
echo "Scratch: {scratch_dir}"
echo
for job_id in {" ".join(tracked_jobs)}; do
  sacct -j "${{job_id}}" -o JobID,JobName%40,State,Elapsed,MaxRSS,ReqMem,AllocTRES || true
done
echo
"${{PIXI_PY}}" -m scripts.summarize_em_robustness_matrix \\
  {q(scratch_dir)} \\
  --output-markdown {q(scratch_dir / "em_kclass_robustness_summary.md")} \\
  --output-json {q(scratch_dir / "em_kclass_robustness_summary.json")} \\
  --slurm-accounting-json-out {q(scratch_dir / "slurm_case_accounting.json")} \\
  --dedupe-case-reruns
tail -200 {q(scratch_dir / "em_kclass_robustness_summary.md")} || true
{multiseed_command}
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
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    text = proc.stdout.strip()
    if proc.returncode != 0 or not text:
        return default
    return text


def main() -> int:
    args = parse_args()
    expected_commit = git_text("rev-parse", "HEAD")
    if expected_commit == "<unknown>":
        raise SystemExit("Cannot resolve the repository commit for queued-job provenance")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"em_kclass_robustness_{timestamp}_{os.getpid()}"
    scratch_dir = (
        (
            args.scratch_dir
            or Path(
                os.environ.get(
                    "EM_KCLASS_MATRIX_SCRATCH_DIR",
                    f"/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/{run_id}",
                )
            )
        )
        .expanduser()
        .resolve()
    )
    if scratch_dir.exists() and (not scratch_dir.is_dir() or any(scratch_dir.iterdir())):
        raise SystemExit(
            f"K-class matrix scratch root must be new or empty; refusing to mix or reseal evidence: {scratch_dir}"
        )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    (scratch_dir / "SAFE_TO_DELETE").touch()
    jobs_dir = scratch_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (scratch_dir / "tmp").mkdir(exist_ok=True)

    cases = selected_cases(args)
    three_seed_suite = bool(cases) and all(case.seed_replicate is not None for case in cases)
    if any(case.seed_replicate is not None for case in cases) and not three_seed_suite:
        raise SystemExit("Three-seed suite expansion produced a mixture of replicated and base cases")
    shared_input_producers = validate_shared_input_groups(cases)
    account = os.environ.get("SBATCH_ACCOUNT", "gilles")
    partition = os.environ.get("SBATCH_PARTITION", "cryoem")
    summary_partition = args.summary_partition or "cpu"
    constraint = os.environ.get("SBATCH_CONSTRAINT", "")
    summary_constraint = os.environ.get("EM_KCLASS_MATRIX_SUMMARY_CONSTRAINT", "")
    # The setup job seals the custom CUDA library once for all dependent case
    # jobs.  It therefore needs the same GPU/toolkit contract as the cases;
    # CPU-only setup nodes on Della do not expose nvcc or nvidia-smi.
    setup_partition = os.environ.get("EM_KCLASS_MATRIX_SETUP_PARTITION", partition)
    setup_constraint = os.environ.get("EM_KCLASS_MATRIX_SETUP_CONSTRAINT", constraint)
    setup_gres = os.environ.get("EM_KCLASS_MATRIX_SETUP_GRES", "gpu:1")
    if "gpu" not in setup_gres.lower():
        raise SystemExit(
            "EM_KCLASS_MATRIX_SETUP_GRES must request a GPU because the setup job "
            "builds and seals the shared CUDA library"
        )
    if os.environ.get("EM_KCLASS_MATRIX_EXCLUSIVE", "0") != "0":
        raise SystemExit("EM_KCLASS_MATRIX_EXCLUSIVE is unsupported: K-class matrix jobs must be non-exclusive")
    exclusive = False
    cuda_module = os.environ.get("CUDA_MODULE", "cudatoolkit/12.8")
    relion_src_value = os.environ.get("RELION_SRC_DIR", "").strip()
    if not relion_src_value:
        raise SystemExit("RELION_SRC_DIR must name an absolute RELION source directory containing projector.h")
    relion_src_path = Path(relion_src_value).expanduser()
    if not relion_src_path.is_absolute():
        raise SystemExit(f"RELION_SRC_DIR must be absolute: {relion_src_path}")
    relion_src_dir = relion_src_path.resolve()
    if not (relion_src_dir / "projector.h").is_file():
        raise SystemExit(
            f"RELION_SRC_DIR must name an absolute RELION source directory containing projector.h: {relion_src_dir}"
        )
    relion_module = os.environ.get("RELION_MODULE", "relion/5.0.1/gcc-11.5.0-gpu")
    relion_refine_mpi = os.environ.get("EM_KCLASS_MATRIX_RELION_REFINE_MPI", "").strip()
    if not relion_refine_mpi:
        raise SystemExit(
            "EM_KCLASS_MATRIX_RELION_REFINE_MPI must name an absolute, executable "
            "RELION build instrumented to honor RELION_DISPATCH_LOG; the stock binary "
            "cannot supply strict dynamic-dispatch parity."
        )
    relion_refine_path = validate_relion_dispatch_executable(relion_refine_mpi)
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
    time_limit_override_for_env = getattr(args, "time_limit_override", None) or os.environ.get(
        "EM_KCLASS_MATRIX_TIME_LIMIT", ""
    )
    seed_override_for_env = getattr(args, "seed_override", None)
    if seed_override_for_env is None:
        seed_override_for_env = os.environ.get("EM_KCLASS_MATRIX_SEED", "")
    seed_offset_for_env = getattr(args, "seed_offset", None)
    if seed_offset_for_env is None:
        seed_offset_for_env = os.environ.get("EM_KCLASS_MATRIX_SEED_OFFSET", "")
    base_python = base_pixi_python()
    if not base_python.is_file() or not os.access(base_python, os.X_OK):
        raise SystemExit(f"EM_KCLASS_MATRIX_PIXI_PY must name an installed pixi Python: {base_python}")
    shared_cuda_lib = scratch_dir / "cuda" / "shared" / "libcuda_backproject.so"

    print("EM K-class robustness matrix launcher")
    print(f"Repo: {REPO_ROOT}")
    print(f"HEAD: {expected_commit}")
    print(f"Branch: {git_text('symbolic-ref', '--short', 'HEAD', default='<detached>')}")
    print(f"Scratch: {scratch_dir}")
    print(f"Cases: {', '.join(str(case.index) for case in cases)}")
    if three_seed_suite:
        print(f"Three-seed suite: {','.join(str(seed) for seed in THREE_SEED_VALUES)}")
    if shared_input_producers:
        print(
            "Shared input producers: "
            + ", ".join(f"{key}={producer.name}" for key, producer in sorted(shared_input_producers.items()))
        )
    print(f"Partition/account: {partition}/{account}")
    print(f"Setup partition: {setup_partition}")
    print(f"Setup constraint: {setup_constraint or '<none>'}")
    print(f"Setup gres: {setup_gres}")
    print(f"Summary partition: {summary_partition}")
    print(f"Summary constraint: {summary_constraint or '<none>'}")
    print(f"Constraint: {constraint or '<none>'}")
    print(f"RELION source: {relion_src_dir}")
    print(f"RELION module: {relion_module}")
    print(f"RELION dispatch-capture executable: {relion_refine_mpi}")
    print(f"Base pixi Python: {base_python}")
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
        "image_batch_size_override",
        "rotation_block_size_override",
        "symmetry",
        "base_name",
        "base_seed",
        "seed_replicate",
        "shared_input_group",
        "shared_input_role",
        "pdb_dir",
        "case_root",
        "script",
        "job_id",
    ]
    case_table.write_text("|".join(header) + "\n")

    setup_script = write_setup_script(
        scratch_dir=scratch_dir,
        jobs_dir=jobs_dir,
        cuda_lib=shared_cuda_lib,
        account=account,
        partition=setup_partition,
        constraint=setup_constraint,
        setup_gres=setup_gres,
        cuda_module=cuda_module,
        relion_src_dir=relion_src_dir,
        expected_commit=expected_commit,
    )
    setup_job = submit(setup_script, dry_run=args.dry_run)
    tracked_jobs = [setup_job]
    case_jobs: list[str] = []
    shared_input_producer_jobs: dict[str, str] = {}

    for case in cases:
        if not case.pdb_dir.exists():
            raise SystemExit(f"PDB directory missing for case {case.index}: {case.pdb_dir}")
        script = write_case_script(
            case=case,
            scratch_dir=scratch_dir,
            jobs_dir=jobs_dir,
            cuda_lib=shared_cuda_lib,
            account=account,
            partition=partition,
            constraint=constraint,
            exclusive=exclusive,
            cuda_module=cuda_module,
            relion_src_dir=relion_src_dir,
            relion_module=relion_module,
            relion_refine_mpi=relion_refine_mpi,
            relion_mpi_ranks=relion_mpi_ranks,
            relion_pool=relion_pool,
            particle_diameter=particle_diameter,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
            gt_align_refine_orders=gt_align_refine_orders,
            noise_rng_batch_size=noise_rng_batch_size,
            expected_commit=expected_commit,
        )
        dependencies = [setup_job]
        input_group_key = shared_input_key(case)
        if input_group_key is not None and not case.shared_input_producer:
            producer_job = shared_input_producer_jobs.get(input_group_key)
            if producer_job is None:
                raise SystemExit(
                    f"shared input consumer {case.name} was reached before its producer job for group {input_group_key}"
                )
            dependencies.append(producer_job)
        dependency_arg = f"--dependency=afterok:{':'.join(dependencies)}"
        job_id = submit(
            script,
            dry_run=args.dry_run,
            extra_args=[dependency_arg] if not args.dry_run else None,
        )
        if input_group_key is not None and case.shared_input_producer:
            shared_input_producer_jobs[input_group_key] = job_id
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
        three_seed_suite=three_seed_suite,
        expected_commit=expected_commit,
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
                f"EXPECTED_GIT_HEAD={expected_commit}",
                f"RUNTIME_ROOT={DEFAULT_RUNTIME_ROOT}",
                f"SCRATCH_DIR={scratch_dir}",
                f"EM_KCLASS_MATRIX_VENV={scratch_dir / 'venv'}",
                f"PIXI_PY={scratch_dir / 'venv' / 'bin' / 'python'}",
                f"EM_KCLASS_MATRIX_PIXI_PY={base_python}",
                f"RECOVAR_CUDA_LIB={shared_cuda_lib}",
                f"RECOVAR_RELION_BIND_BUILD_DIR={scratch_dir / 'relion_bind_build' / 'shared'}",
                f"EM_KCLASS_MATRIX_SETUP_PARTITION={setup_partition}",
                f"EM_KCLASS_MATRIX_SETUP_CONSTRAINT={setup_constraint}",
                f"EM_KCLASS_MATRIX_SETUP_GRES={setup_gres}",
                f"EM_KCLASS_MATRIX_SUMMARY_PARTITION={summary_partition}",
                f"EM_KCLASS_MATRIX_SUMMARY_CONSTRAINT={summary_constraint}",
                f"SETUP_JOB_ID={setup_job}",
                f"CASE_JOB_IDS={q(' '.join(case_jobs))}",
                f"SUMMARY_JOB_ID={summary_job}",
                f"CASE_TABLE={case_table}",
                f"SBATCH_PARTITION={partition}",
                f"SBATCH_ACCOUNT={account}",
                f"SBATCH_CONSTRAINT={constraint}",
                f"RELION_MODULE={relion_module}",
                f"RELION_SRC_DIR={relion_src_dir}",
                f"EM_KCLASS_MATRIX_RELION_REFINE_MPI={relion_refine_mpi}",
                f"RELION_MPI_RANKS={relion_mpi_ranks}",
                f"KCLASS_IMAGE_BATCH_SIZE={image_batch_size}",
                f"KCLASS_ROTATION_BLOCK_SIZE={rotation_block_size}",
                f"EM_KCLASS_MATRIX_GT_ALIGN_REFINE_ORDERS={gt_align_refine_orders}",
                f"EM_KCLASS_MATRIX_MAX_ITER={max_iter_override_for_env}",
                f"EM_KCLASS_MATRIX_TIME_LIMIT={time_limit_override_for_env}",
                f"EM_KCLASS_MATRIX_SEED={seed_override_for_env}",
                f"EM_KCLASS_MATRIX_SEED_OFFSET={seed_offset_for_env}",
                f"EM_KCLASS_MATRIX_THREE_SEED_SUITE={int(three_seed_suite)}",
                f"EM_KCLASS_MATRIX_THREE_SEED_VALUES={','.join(str(seed) for seed in THREE_SEED_VALUES)}",
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
