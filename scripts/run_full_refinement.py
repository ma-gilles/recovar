#!/usr/bin/env python
"""Run a full multi-iteration EM refinement and save per-iteration results.

This script loads the synthetic benchmark dataset (5000 images, 128px),
initializes from the low-pass filtered reference volume, and calls
refine_single_volume() with parameters matching the RELION auto-refine run.

Results are saved as a single .npz file with per-iteration arrays for
downstream comparison via compare_vs_relion.py.

Usage:
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        pixi run python scripts/run_full_refinement.py [--output DIR] [--max_iter N]

Environment variables:
    CUDA_VISIBLE_DEVICES: GPU to use
    XLA_PYTHON_CLIENT_PREALLOCATE: set to false for dynamic allocation
"""

import argparse
import hashlib
import importlib
import json
import logging
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.em.dense_single_volume.helpers.relion_projector_capture import (
    build_relion_projector_replay_state,
)
from recovar.em.dense_single_volume.relion_replay import (
    read_relion_single_optics_sigma2_noise as _read_relion_single_optics_sigma2_noise,
)
from recovar.em.dense_single_volume.relion_replay import (
    relion_mpi_process_start_scoring_noise_pair as _relion_mpi_process_start_scoring_noise_pair,
)
from recovar.em.dense_single_volume.relion_worker_scale import (
    load_relion_dispatch_schedule,
    load_relion_follower_scale_replay,
    relion_class3d_follower_owners_from_schedule,
    relion_ordered_particle_sha256,
    validate_relion_follower_scale_replay,
    verify_relion_dispatch_schedule_oracle,
)
from recovar.utils.parity_provenance import _safe_git_commit, git_worktree_provenance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


_CONCRETE_RECOVAR_PROVENANCE_MODULES = (
    "recovar",
    "recovar.em.dense_single_volume.iteration_loop",
    "recovar.em.dense_single_volume.k_class",
    "recovar.em.dense_single_volume.helpers.significance",
)


def _assert_expected_repo_imports() -> None:
    """Fail fast if EM modules were imported from another editable checkout."""
    expected_root_value = os.environ.get("RECOVAR_EXPECTED_REPO_ROOT")
    if not expected_root_value:
        return

    expected_root = Path(expected_root_value).expanduser().resolve()
    failures = []
    for module_name in _CONCRETE_RECOVAR_PROVENANCE_MODULES:
        module = importlib.import_module(module_name)
        module_file_value = getattr(module, "__file__", None)
        module_file = Path(module_file_value).resolve() if module_file_value else None
        logger.info("Import provenance: %s=%s", module_name, module_file)
        if module_file is None or not module_file.is_relative_to(expected_root):
            failures.append(f"{module_name}={module_file}")

    if failures:
        raise RuntimeError(
            "RECOVAR import provenance failure: expected every concrete EM module under "
            f"{expected_root}, found " + ", ".join(failures)
        )


def _shell_index_to_resolution_angstrom(shell_index, grid_size, voxel_size):
    if voxel_size <= 0:
        return float(shell_index)
    shell_index = float(shell_index)
    if shell_index <= 0:
        return float("inf")
    return float(grid_size) * float(voxel_size) / shell_index


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_relion_sampling_orders(healpix_order: int, adaptive_oversampling: int) -> tuple[int, int]:
    """Return RELION coarse pass-1 and fine pass-2 HEALPix orders.

    RELION's ``--healpix_order`` is the coarse pass-1 order printed as
    ``OrientationalSampling`` under ``Oversampling=0``. Adaptive oversampling
    refines pass 2 to ``healpix_order + adaptive_oversampling``.
    """
    coarse_order = int(healpix_order)
    oversampling = int(adaptive_oversampling)
    if coarse_order < 0:
        raise ValueError(f"healpix_order must be non-negative, got {healpix_order}")
    if oversampling < 0:
        raise ValueError(f"adaptive_oversampling must be non-negative, got {adaptive_oversampling}")
    return coarse_order, coarse_order + oversampling


def _resolve_effective_max_healpix_order(
    *,
    n_classes: int,
    healpix_order: int,
    max_healpix_order: int | None,
) -> tuple[int, str]:
    """Resolve the backend HEALPix refinement cap.

    RELION Class3D runs launched with ``--healpix_order`` use fixed coarse
    sampling (``_rlnDoAutoSampling 0`` in the optimiser STAR).  In that mode,
    adaptive oversampling only controls the pass-2 child grid; it does not let
    later iterations increase the coarse HEALPix order.  K=1 auto-refine keeps
    RECOVAR's historical broad cap unless an explicit cap is provided.
    """
    init_order = int(healpix_order)
    if init_order < 0:
        raise ValueError(f"healpix_order must be non-negative, got {healpix_order}")

    if max_healpix_order is None:
        if int(n_classes) > 1:
            return init_order, "RELION Class3D fixed --healpix_order"
        return 7, "K=1 auto-refine default"

    cap = int(max_healpix_order)
    if cap < init_order:
        raise ValueError(
            "max_healpix_order must be >= healpix_order "
            f"({cap} < {init_order})",
        )
    return cap, "explicit CLI"


def _npz_scalar_to_float(npz, key):
    if key not in npz.files:
        return None
    return float(np.asarray(npz[key]))


def _pose_history_half_arrays(iter_entry, *, dtype=np.float32):
    if iter_entry is None:
        return None
    if not isinstance(iter_entry, (list, tuple)):
        return [np.asarray(iter_entry, dtype=dtype)]
    return [None if arr is None else np.asarray(arr, dtype=dtype) for arr in iter_entry]


def _pose_history_by_image(iter_entry, half_indices, n_images, trailing_shape, *, dtype=np.float32):
    half_arrays = _pose_history_half_arrays(iter_entry, dtype=dtype)
    if half_arrays is None or all(arr is None for arr in half_arrays):
        return None
    out = np.full((int(n_images), *trailing_shape), np.nan, dtype=dtype)
    for half_idx, arr in zip(half_indices, half_arrays):
        if arr is None:
            continue
        half_idx = np.asarray(half_idx, dtype=np.int64)
        if arr.shape[0] != half_idx.shape[0]:
            raise ValueError(
                f"Pose history length {arr.shape[0]} does not match half-set index length {half_idx.shape[0]}"
            )
        out[half_idx] = arr
    return out


def _add_significant_count_artifacts(save_dict, significant_counts, half_indices, n_images):
    """Save significant-count history in both half and original image order."""
    half_order_indices = np.concatenate(
        [np.asarray(indices, dtype=np.int64) for indices in half_indices],
    )
    for iteration, counts in enumerate(significant_counts):
        if counts is None:
            continue
        counts_half_order = np.asarray(counts)
        # Keep the legacy key value/shape/dtype-compatible: it has always
        # stored the concatenated half-1, half-2 refinement-loop order.
        save_dict[f"sig_counts_iter_{iteration:03d}"] = counts_half_order
        save_dict[f"sig_counts_half_order_iter_{iteration:03d}"] = counts_half_order
        flat_counts = counts_half_order.reshape(-1)
        if flat_counts.shape[0] != half_order_indices.shape[0]:
            continue
        counts_by_image = np.full(int(n_images), -1, dtype=flat_counts.dtype)
        counts_by_image[half_order_indices] = flat_counts
        save_dict[f"sig_counts_by_image_iter_{iteration:03d}"] = counts_by_image


def _jsonable_profile_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable_profile_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_profile_value(item) for item in value]
    try:
        arr = np.asarray(value)
    except Exception:
        return str(value)
    if arr.shape == ():
        return _jsonable_profile_value(arr.item())
    return _jsonable_profile_value(arr.tolist())


def _jsonable_profile_rows(rows):
    return [
        {str(key): _jsonable_profile_value(value) for key, value in row.items()}
        for row in rows
        if isinstance(row, dict)
    ]


def _read_timing_npz(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as npz:
        row = {
            "path": str(npz_path),
            "iteration": int(np.asarray(npz["iteration"])) if "iteration" in npz.files else None,
            "relion_iteration": int(np.asarray(npz["relion_iteration"])) if "relion_iteration" in npz.files else None,
            "wall_time_s": _npz_scalar_to_float(npz, "wall_time_s"),
            "stages": {},
        }
        for name in npz.files:
            if name.startswith("stage_seconds_"):
                row["stages"][name[len("stage_seconds_") :]] = float(np.asarray(npz[name]))
    return row


def _collect_timing_rows(timing_dir):
    if timing_dir is None:
        return []
    timing_path = Path(timing_dir)
    if not timing_path.exists():
        return []
    return [_read_timing_npz(path) for path in sorted(timing_path.glob("iter_*.npz"))]


def _stage_deltas_from_cumulative(stages: dict[str, float]) -> dict[str, float]:
    if not stages:
        return {}
    ordered_names = ["e_step", "recon", "fsc", "noise_update", "convergence"]
    deltas: dict[str, float] = {}
    prev = 0.0
    for name in ordered_names:
        value = stages.get(name)
        if value is None:
            continue
        deltas[name] = max(0.0, float(value) - prev)
        prev = float(value)
    for name, value in sorted(stages.items()):
        if name in deltas:
            continue
        deltas[name] = float(value)
    return deltas


def _summarize_timing_rows(rows):
    summary = {
        "n_rows": len(rows),
        "sum_wall_time_s": float(
            np.sum([row["wall_time_s"] for row in rows if row.get("wall_time_s") is not None], dtype=np.float64)
        )
        if rows
        else 0.0,
        "stage_cumulative_by_relion_iter": {},
        "stage_delta_by_relion_iter": {},
        "sum_stage_delta_s": {},
    }
    for row in rows:
        relion_iter = row.get("relion_iteration")
        if relion_iter is None:
            continue
        stages = {key: float(value) for key, value in row.get("stages", {}).items()}
        deltas = _stage_deltas_from_cumulative(stages)
        summary["stage_cumulative_by_relion_iter"][str(relion_iter)] = stages
        summary["stage_delta_by_relion_iter"][str(relion_iter)] = deltas
        for key, value in deltas.items():
            summary["sum_stage_delta_s"][key] = float(summary["sum_stage_delta_s"].get(key, 0.0) + value)
    return summary


def _load_relion_mask_params(optimiser_star_path):
    """Extract RELION image-mask parameters from an optimiser STAR file."""
    text = Path(optimiser_star_path).read_text(errors="ignore")

    particle_match = re.search(r"rlnParticleDiameter\s+([0-9]+(?:\.[0-9]+)?)", text)
    if particle_match is None:
        particle_match = re.search(r"particle_diameter\s+([0-9]+(?:\.[0-9]+)?)", text)

    width_match = re.search(r"rlnWidthMaskEdge\s+([0-9]+(?:\.[0-9]+)?)", text)
    if width_match is None:
        width_match = re.search(r"width_mask_edge\s+([0-9]+(?:\.[0-9]+)?)", text)

    if particle_match is None or width_match is None:
        return None

    return float(particle_match.group(1)), float(width_match.group(1))


def _load_relion_max_significants(optimiser_star_path):
    """Extract RELION's maximum-significant-poses setting from an optimiser STAR."""
    text = Path(optimiser_star_path).read_text(errors="ignore")

    match = re.search(r"rlnMaximumSignificantPoses\s+(-?[0-9]+)", text)
    if match is None:
        match = re.search(r"maximum_significant_poses\s+(-?[0-9]+)", text)
    if match is None:
        return None
    return int(match.group(1))


def _parse_relion_cli_ini_high(text):
    """Extract a positive RELION ``--ini_high`` value from an optimiser STAR header."""
    cli_line = ""
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and "--" in stripped:
            cli_line = stripped.lstrip("#").strip()
            break
    match = re.search(r"(?:^|\s)--ini_high(?:\s+|=)(\S+)", cli_line)
    if match is None:
        return None
    val = float(match.group(1))
    if val <= 0.0:
        return None
    return val


def _parse_relion_tau2_fudge(text):
    """Extract RELION's tau2_fudge from a model or optimiser STAR text block.

    ``_rlnTau2FudgeFactor`` (model.star) is the value RELION actually used.
    ``_rlnTau2FudgeArg`` (optimiser.star) is the user's --tau2_fudge CLI
    value, or -1 when the user did not pass --tau2_fudge (RELION binary
    default kicks in: 1.0 for auto-refine, 4.0 for Class3D). Passing -1
    downstream inverts the Wiener regularization (``inv_tau = 1 /
    (pf^3 * tau2_fudge * tau)``) — that produces a corrupt iter-1
    reconstruction and collapses iter-2+ ``ave_Pmax`` even though iter-1
    Pmax is at RELION parity. Prefer ``Factor`` over ``Arg`` and treat
    a non-positive ``Arg`` as "unset" so ``_resolve_tau2_fudge`` falls
    back to the K-class default.
    """
    match = re.search(r"_?rlnTau2FudgeFactor\s+(\S+)", text)
    if match is not None:
        return float(match.group(1))
    match = re.search(r"_?rlnTau2FudgeArg\s+(\S+)", text)
    if match is None:
        return None
    val = float(match.group(1))
    if val <= 0.0:
        return None
    return val


def _resolve_tau2_fudge(n_classes, cli_tau2_fudge, relion_init_tau2_fudge):
    """Return the effective tau2_fudge and a human-readable source label."""
    if relion_init_tau2_fudge is not None:
        return float(relion_init_tau2_fudge), "RELION it000 optimiser"
    if cli_tau2_fudge is not None:
        return float(cli_tau2_fudge), "explicit CLI"
    if int(n_classes) > 1:
        return 4.0, "RELION Class3D default"
    return 1.0, "RELION auto-refine default"


def _load_relion_it000_model_stars(relion_init_dir, n_classes):
    """Load RELION iter-0 model STARs for strict cold-start replay.

    Class3D writes a shared ``run_it000_model.star``. AutoRefine writes
    half-specific ``run_it000_half{1,2}_model.star`` files instead; preserve
    the shared path when present, and fall back to the half pair for K=1.
    """
    import starfile as _starfile

    relion_init_dir = Path(relion_init_dir)
    shared_model_path = relion_init_dir / "run_it000_model.star"
    if shared_model_path.exists():
        model = _starfile.read(str(shared_model_path))
        return {
            "models": [model],
            "model_paths": [shared_model_path],
            "reference_model": model,
            "reference_model_path": shared_model_path,
            "source": "shared",
        }

    half_model_paths = [
        relion_init_dir / "run_it000_half1_model.star",
        relion_init_dir / "run_it000_half2_model.star",
    ]
    if int(n_classes) == 1 and all(path.exists() for path in half_model_paths):
        models = [_starfile.read(str(path)) for path in half_model_paths]
        return {
            "models": models,
            "model_paths": half_model_paths,
            "reference_model": models[0],
            "reference_model_path": half_model_paths[0],
            "source": "half-specific",
        }

    expected = [shared_model_path, *half_model_paths]
    missing = [str(path) for path in expected if not path.exists()]
    raise SystemExit(
        "--relion_init_dir given but no compatible iter-0 model STAR was found; "
        f"missing candidates: {', '.join(missing)}",
    )


def _resolve_replay_normcorr(perturb_replay_relion_dir, replay_relion_normcorr):
    """Default normcorr replay on only for explicit RELION replay runs."""
    if replay_relion_normcorr is not None:
        return bool(replay_relion_normcorr)
    return perturb_replay_relion_dir is not None


def _format_replay_mean_for_log(values) -> str:
    arr = np.asarray(values)
    if arr.size == 0:
        return "empty"
    return f"{float(arr.mean()):.4f}"


class NativeGroupLayout(NamedTuple):
    """RELION group labels in RECOVAR half order plus the full model axis."""

    group_ids_per_half: tuple[np.ndarray, np.ndarray]
    particle_ids_per_half: tuple[np.ndarray, np.ndarray]
    optics_group_ids_per_half: tuple[np.ndarray, np.ndarray]
    n_groups: int
    n_optics_groups: int
    source: str


def _relion_image_identity(name, *, label: str) -> tuple[int, str]:
    """Return the exact ``(<1-based index>, <stack>)`` RELION image identity."""

    match = re.fullmatch(r"(\d+)@(.+)", str(name))
    if match is None:
        raise ValueError(f"{label} image names must use the '<index>@<stack>' form; got {name!r}")
    return int(match.group(1)), match.group(2)


def _particle_identity_rows(particles, *, label: str) -> dict[tuple[int, str], int]:
    if "rlnImageName" not in particles.columns:
        raise ValueError(f"{label} is missing rlnImageName")
    identities = [
        _relion_image_identity(name, label=label)
        for name in np.asarray(particles["rlnImageName"]).reshape(-1)
    ]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{label} contains duplicate rlnImageName/stack identities")
    return {identity: row for row, identity in enumerate(identities)}


def _resolve_native_group_layout(
    our_particles,
    half1_idx,
    half2_idx,
    *,
    relion_particles=None,
) -> NativeGroupLayout | None:
    """Map RELION groups to RECOVAR rows without assuming equal STAR order.

    The supplied RELION data table is authoritative when it carries
    ``rlnGroupNumber``.  Otherwise this falls back to the RECOVAR input table.
    Group numbers remain on RELION's full model axis even when a half-set does
    not contain the highest-numbered group.
    """

    if relion_particles is not None and "rlnGroupNumber" in relion_particles.columns:
        group_particles = relion_particles
        source = "supplied RELION data STAR"
    elif "rlnGroupNumber" in our_particles.columns:
        group_particles = our_particles
        source = "RECOVAR input particles STAR"
    else:
        return None

    our_rows = _particle_identity_rows(our_particles, label="RECOVAR input STAR")
    group_rows = _particle_identity_rows(group_particles, label=source)
    if set(our_rows) != set(group_rows):
        missing = len(set(our_rows) - set(group_rows))
        extra = len(set(group_rows) - set(our_rows))
        raise ValueError(
            f"{source} and RECOVAR input STAR do not contain the same "
            f"rlnImageName/stack identities (missing={missing}, extra={extra})",
        )

    group_numbers_source = np.asarray(group_particles["rlnGroupNumber"], dtype=np.int64).reshape(-1)
    if group_numbers_source.shape != (len(group_particles),):
        raise ValueError(
            f"rlnGroupNumber length {group_numbers_source.size} does not match "
            f"{source} particle count {len(group_particles)}",
        )
    if group_numbers_source.size and int(np.min(group_numbers_source)) < 1:
        raise ValueError("RELION rlnGroupNumber values must be 1-based positive integers")

    group_ids_by_our_row = np.asarray(
        [group_numbers_source[group_rows[identity]] - 1 for identity in our_rows],
        dtype=np.int64,
    )
    particle_ids_by_our_row = np.asarray(
        [group_rows[identity] for identity in our_rows],
        dtype=np.int64,
    )
    if "rlnOpticsGroup" in group_particles.columns:
        optics_numbers_source = np.asarray(group_particles["rlnOpticsGroup"], dtype=np.int64).reshape(-1)
        if optics_numbers_source.shape != (len(group_particles),):
            raise ValueError(
                f"rlnOpticsGroup length {optics_numbers_source.size} does not match "
                f"{source} particle count {len(group_particles)}",
            )
        if optics_numbers_source.size and int(np.min(optics_numbers_source)) < 1:
            raise ValueError("RELION rlnOpticsGroup values must be 1-based positive integers")
    else:
        optics_numbers_source = np.ones(len(group_particles), dtype=np.int64)
    optics_group_ids_by_our_row = np.asarray(
        [optics_numbers_source[group_rows[identity]] - 1 for identity in our_rows],
        dtype=np.int64,
    )
    half_indices = []
    for label, values in (("half1_idx", half1_idx), ("half2_idx", half2_idx)):
        indices = np.asarray(values, dtype=np.int64).reshape(-1)
        if indices.size and (int(np.min(indices)) < 0 or int(np.max(indices)) >= len(our_particles)):
            raise ValueError(f"{label} contains an out-of-bounds RECOVAR particle row")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{label} contains duplicate RECOVAR particle rows")
        half_indices.append(indices)
    if np.intersect1d(half_indices[0], half_indices[1]).size:
        raise ValueError("half1_idx and half2_idx overlap")

    n_groups = int(np.max(group_numbers_source)) if group_numbers_source.size else 0
    n_optics_groups = int(np.max(optics_numbers_source)) if optics_numbers_source.size else 0
    return NativeGroupLayout(
        group_ids_per_half=(
            np.asarray(group_ids_by_our_row[half_indices[0]], dtype=np.int64),
            np.asarray(group_ids_by_our_row[half_indices[1]], dtype=np.int64),
        ),
        particle_ids_per_half=(
            np.asarray(particle_ids_by_our_row[half_indices[0]], dtype=np.int64),
            np.asarray(particle_ids_by_our_row[half_indices[1]], dtype=np.int64),
        ),
        optics_group_ids_per_half=(
            np.asarray(optics_group_ids_by_our_row[half_indices[0]], dtype=np.int64),
            np.asarray(optics_group_ids_by_our_row[half_indices[1]], dtype=np.int64),
        ),
        n_groups=n_groups,
        n_optics_groups=n_optics_groups,
        source=source,
    )


def _load_native_group_ids_per_half(particles_star, half1_idx, half2_idx):
    """Compatibility wrapper for a single particles STAR group layout."""

    import starfile as _starfile

    data = _starfile.read(str(particles_star))
    particles = data["particles"] if isinstance(data, dict) else data
    layout = _resolve_native_group_layout(particles, half1_idx, half2_idx)
    return None if layout is None else list(layout.group_ids_per_half)


def _load_replay_group_particles(relion_dir, *, init_relion_iteration=0):
    """Load the first authoritative replay data STAR carrying group labels."""

    if relion_dir is None:
        return None
    import starfile as _starfile

    relion_dir = Path(relion_dir).resolve()
    preferred = relion_dir / f"run_it{int(init_relion_iteration):03d}_data.star"
    iter0 = relion_dir / "run_it000_data.star"
    candidates = [preferred]
    if iter0 != preferred:
        candidates.append(iter0)
    candidates.extend(sorted(relion_dir.glob("run_it*_data.star")))
    seen = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        data = _starfile.read(str(path))
        particles = data["particles"] if isinstance(data, dict) else data
        if "rlnGroupNumber" in particles.columns:
            return particles, path
    return None


def _select_authoritative_group_particles(
    *,
    halfset_particles=None,
    halfset_source=None,
    replay_dirs=(),
    init_relion_iteration=0,
):
    """Select a particle table that actually carries RELION group labels."""

    if halfset_particles is not None and "rlnGroupNumber" in halfset_particles.columns:
        return halfset_particles, None if halfset_source is None else Path(halfset_source).resolve()
    for replay_dir in replay_dirs:
        replay_groups = _load_replay_group_particles(
            replay_dir,
            init_relion_iteration=init_relion_iteration,
        )
        if replay_groups is not None:
            return replay_groups
    return None, None


def _default_refinement_subsets(n_images, seed, n_classes):
    """Return default dataset splits for RELION-style refinement."""

    indices = np.arange(int(n_images), dtype=np.int64)
    if int(n_classes) > 1:
        return indices, np.empty(0, dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    return np.sort(indices[: int(n_images) // 2]), np.sort(indices[int(n_images) // 2 :])


def _relion_halfset_and_accuracy_layout(our_particles, relion_particles):
    """Map RELION's internal data rows onto RECOVAR's half-local ordering."""
    our_row_by_identity = _particle_identity_rows(
        our_particles,
        label="RECOVAR input STAR",
    )
    relion_row_by_identity = _particle_identity_rows(
        relion_particles,
        label="RELION data STAR",
    )
    if set(our_row_by_identity) != set(relion_row_by_identity):
        raise ValueError(
            "RELION and RECOVAR STAR files do not contain the same "
            "rlnImageName/stack identities"
        )
    our_identities = list(our_row_by_identity)
    relion_identities = list(relion_row_by_identity)

    relion_subsets = np.asarray(relion_particles["rlnRandomSubset"], dtype=np.int64)
    our_relion_rows = np.asarray(
        [relion_row_by_identity[identity] for identity in our_identities],
        dtype=np.int64,
    )
    our_subsets = relion_subsets[our_relion_rows]
    half1_idx = np.flatnonzero(our_subsets == 1).astype(np.int64)
    half2_idx = np.flatnonzero(our_subsets == 2).astype(np.int64)

    half1_local_by_our_row = {int(our_row): local for local, our_row in enumerate(half1_idx)}
    half1_base_order_local = np.asarray(
        [
            half1_local_by_our_row[our_row_by_identity[relion_identities[relion_row]]]
            for relion_row in np.flatnonzero(relion_subsets == 1)
        ],
        dtype=np.int64,
    )
    half1_particle_ids = our_relion_rows[half1_idx]
    if "rlnOpticsGroup" in relion_particles.columns:
        relion_optics = np.asarray(relion_particles["rlnOpticsGroup"], dtype=np.int64)
        half1_optics_group_ids = relion_optics[half1_particle_ids]
    else:
        half1_optics_group_ids = np.zeros(half1_idx.size, dtype=np.int64)
    return (
        half1_idx,
        half2_idx,
        half1_base_order_local,
        half1_optics_group_ids,
        half1_particle_ids,
    )


def _replay_complete_initial_particle_state(n_classes, init_relion_iteration):
    """Whether run_it000 poses/corrections seed the first expectation step.

    AutoRefine (K=1) searches around the run_it000 pre-centering offsets, so
    omitting that state changes the first winners.  Class3D (K>1) performs a
    fresh global first-iteration search and does not seed it from the input
    orientations; replaying those poses would incorrectly compose the global
    samples with the supplied particle orientations.
    """

    return int(n_classes) == 1 and int(init_relion_iteration) == 0


def _refine_sampling_kwargs(args, init_healpix_order):
    """Return sampling kwargs forwarded from the CLI into ``refine_single_volume``."""
    return {
        "translation_pixel_offset": args.offset_step if args.adaptive_oversampling > 0 else None,
        "init_healpix_order": init_healpix_order,
        "auto_local_healpix_order": args.auto_local_healpix_order,
        "init_translation_range": args.offset_range,
        "init_translation_step": args.offset_step,
    }


def _build_replay_iteration_overrides(
    relion_dir,
    half1_idx,
    half2_idx,
    max_iter,
    ds_voxel,
    ds_grid,
    *,
    include_normcorr,
    init_relion_iteration=0,
    particle_names=None,
    include_initial_state=False,
    strict=False,
):
    """Build per-iter replay overrides keyed on recovar iteration index.

    For each recovar iteration k >= 1 (i.e. iter 2 onwards in RELION terms),
    reads RELION's run_it{k:03d}_data.star + half1/half2 model.star
    (or the shared Class3D run_it{k:03d}_model.star) and builds an
    override dict containing:
      * image_corrections: per-image (avg_norm/normcorr) * group_scale
      * serialized_scale_corrections: per-image model-STAR group scale,
        retained as provenance rather than forced onto the live scorer
      * previous_best_translations / previous_best_rotation_eulers: RELION's
        previous hard assignments for local-search centering

    This matches scripts/run_multi_iter_parity.py::_load_relion_iteration_override
    (the proven replay logic). The recovar iter-k override is read from
    RELION iter-k's model+data (since recovar iter-k corresponds to RELION
    iter-(k+1), and the per-image scalings used at the start of RELION
    iter-(k+1) are the ones written by RELION iter-k's M-step).

    ``init_relion_iteration`` is normally zero. Diagnostic profile runs can
    set it to a later RELION iteration to jump directly into local search; in
    that case override slot 0 is sourced from the upstream RELION iteration
    instead of being left empty.

    When ``include_initial_state`` is true, slot 0 is loaded from RELION
    iteration 0 as well. This is required for a strict cold-start replay:
    run_it000 carries the particle pre-centering offsets, initial orientations,
    image/scale corrections, and direction prior that RELION uses in its first
    expectation step.
    """
    import re as _re
    from pathlib import Path as _Path

    import starfile as _sf

    relion_dir = _Path(relion_dir).resolve()

    def _model_has_class_direction_priors(model):
        return any(str(key).startswith("model_pdf_orient_class_") for key in model)

    def _read_model_direction_prior(model_path, model):
        if not _model_has_class_direction_priors(model):
            return None
        from recovar.em.sampling import read_relion_direction_prior, read_relion_direction_priors

        has_multiple_classes = any(
            str(key).startswith("model_pdf_orient_class_") and not str(key).endswith("_1")
            for key in model
        )
        if has_multiple_classes:
            return read_relion_direction_priors(model_path)
        return read_relion_direction_prior(model_path)

    def _read_model_noise_variance(model, *, image_shape):
        radial = _read_relion_single_optics_sigma2_noise(
            model,
            context="replay model",
        )
        if radial is None:
            return None
        radial = radial * float(ds_grid) ** 4
        return np.asarray(
            utils.make_radial_image(jnp.asarray(radial), image_shape, extend_last_frequency=True),
            dtype=np.float32,
        ).reshape(-1)

    def _read_model_class_tau2(model):
        if not isinstance(model, dict):
            return None
        class_tau2 = []
        for key, table in model.items():
            match = _re.fullmatch(r"model_class_(\d+)", str(key))
            if match is None:
                continue
            col = "rlnReferenceTau2" if "rlnReferenceTau2" in table.columns else None
            if col is None and "rlnReferenceSigma2" in table.columns:
                col = "rlnReferenceSigma2"
            if col is None:
                continue
            class_tau2.append(
                (
                    int(match.group(1)),
                    np.asarray(table[col], dtype=np.float64) * float(ds_grid) ** 4,
                )
            )
        if len(class_tau2) <= 1:
            return None
        class_tau2.sort(key=lambda item: item[0])
        return np.stack([tau2 for _, tau2 in class_tau2], axis=0)

    # Index i is consumed by iteration_loop for recovar iter i+1 during the
    # numbered refinement, and by the final all-data pass as len(current_sizes).
    # Allocate one extra slot so convergence on the last configured numbered
    # iteration can replay RELION run_it{max_iter:03d}_data.star.
    overrides = [None] * (max_iter + 1)
    init_relion_iteration = int(init_relion_iteration)
    for recovar_iter in range(0, max_iter + 1):
        # recovar iter k uses corrections computed by RELION iter k (which were
        # written into run_it{k}_data.star). Fresh non-replay runs retain the
        # historical empty slot 0; strict cold-start replay explicitly loads
        # run_it000 because it contains nonzero particle pre-centering offsets
        # and the other state consumed by RELION's first expectation step.
        relion_iter = init_relion_iteration + recovar_iter
        if relion_iter < 0 or (relion_iter == 0 and not include_initial_state):
            continue
        data_star = relion_dir / f"run_it{relion_iter:03d}_data.star"
        model_h1 = relion_dir / f"run_it{relion_iter:03d}_half1_model.star"
        model_h2 = relion_dir / f"run_it{relion_iter:03d}_half2_model.star"
        model_shared = relion_dir / f"run_it{relion_iter:03d}_model.star"
        if model_h1.exists() and model_h2.exists():
            model_paths = (model_h1, model_h2)
        elif model_shared.exists():
            model_paths = (model_shared, model_shared)
        else:
            model_paths = None
        if not data_star.exists() or model_paths is None:
            missing = []
            if not data_star.exists():
                missing.append(str(data_star))
            if model_paths is None:
                missing.append(f"{model_h1} + {model_h2} or {model_shared}")
            message = (
                f"Replay override for recovar iter {recovar_iter + 1} "
                f"(RELION iter {relion_iter:03d}) is missing {'; '.join(missing)}"
            )
            if strict:
                raise ValueError(message)
            logger.warning("%s — leaving unset", message)
            continue

        data = _sf.read(str(data_star))
        parts = data["particles"] if isinstance(data, dict) else data
        m1 = _sf.read(str(model_paths[0]))
        m2 = _sf.read(str(model_paths[1]))

        replay_identity_rows = _particle_identity_rows(
            parts,
            label=f"RELION replay STAR {data_star}",
        )

        nc = np.asarray(parts["rlnNormCorrection"], dtype=np.float64)

        def _scalar(table, key):
            v = table[key]
            return float(v if isinstance(v, (int, float)) else v.iloc[0] if hasattr(v, "iloc") else v[0])

        avg_norm_h1 = _scalar(m1["model_general"], "rlnNormCorrectionAverage")
        avg_norm_h2 = _scalar(m2["model_general"], "rlnNormCorrectionAverage")

        # rlnSigmaOffsetsAngst is RELION's per-iter translation sigma. RELION
        # iter (k+1) loads it from the iter-k model.star and uses it to build
        # pdf_offset (acc_ml_optimiser_impl.h::pdf_offset). recovar's iter-1
        # does not accumulate sigma2_offset moments (no per-image prior centers
        # exist yet), so without an explicit override the iter-2 E-step uses
        # the default init sigma (10 Å) instead of the data-driven RELION
        # value, which is ~6× too wide and depresses iter-2 Pmax by ~22%.
        sigma_offset_h1 = _scalar(m1["model_general"], "rlnSigmaOffsetsAngst")
        sigma_offset_h2 = _scalar(m2["model_general"], "rlnSigmaOffsetsAngst")
        sigma_offset_per_half = [float(sigma_offset_h1), float(sigma_offset_h2)]
        sigma_offset_avg = 0.5 * (sigma_offset_per_half[0] + sigma_offset_per_half[1])
        noise_h1 = _read_model_noise_variance(m1, image_shape=(int(ds_grid), int(ds_grid)))
        noise_h2 = _read_model_noise_variance(m2, image_shape=(int(ds_grid), int(ds_grid)))
        direction_prior_h1 = _read_model_direction_prior(model_paths[0], m1)
        direction_prior_h2 = _read_model_direction_prior(model_paths[1], m2)
        class_tau2 = _read_model_class_tau2(m1)

        groups_h1 = m1.get("model_groups")
        groups_h2 = m2.get("model_groups")
        scale_h1 = (
            np.asarray(groups_h1["rlnGroupScaleCorrection"], dtype=np.float64)
            if groups_h1 is not None and "rlnGroupScaleCorrection" in groups_h1.columns
            else np.array([1.0])
        )
        scale_h2 = (
            np.asarray(groups_h2["rlnGroupScaleCorrection"], dtype=np.float64)
            if groups_h2 is not None and "rlnGroupScaleCorrection" in groups_h2.columns
            else np.array([1.0])
        )
        group_no = (
            np.asarray(parts["rlnGroupNumber"], dtype=int)
            if "rlnGroupNumber" in parts.columns
            else np.ones(len(parts), dtype=int)
        )
        pp_scale_h1 = scale_h1[np.clip(group_no - 1, 0, len(scale_h1) - 1)]
        pp_scale_h2 = scale_h2[np.clip(group_no - 1, 0, len(scale_h2) - 1)]
        combined_h1 = (avg_norm_h1 / nc) * pp_scale_h1
        combined_h2 = (avg_norm_h2 / nc) * pp_scale_h2

        # Map RELION particle order to recovar's half1/half2 ordering.
        # half1_idx / half2_idx are row positions in RECOVAR's input STAR,
        # Match the complete ``(index, stack path)`` identity. Numeric stack
        # indices can repeat across multi-stack real-data STAR files.
        if particle_names is None:
            particle_identities = None
        else:
            particle_identities = [
                _relion_image_identity(name, label="RECOVAR input STAR")
                for name in particle_names
            ]
            if len(set(particle_identities)) != len(particle_identities):
                raise ValueError("RECOVAR input contains duplicate rlnImageName/stack identities")

        def _to_half(values, half_idx):
            rows = np.asarray(half_idx, dtype=np.int64)
            if particle_identities is None:
                return np.asarray(values, dtype=np.float32)[rows]
            identities = [particle_identities[int(row)] for row in rows]
            missing = sorted({identity for identity in identities if identity not in replay_identity_rows})
            if missing:
                preview = ", ".join(f"{index}@{stack}" for index, stack in missing[:8])
                raise ValueError(
                    f"RELION replay STAR is missing {len(missing)} RECOVAR particle identities "
                    f"(preview: {preview})"
                )
            return np.asarray(
                [values[replay_identity_rows[identity]] for identity in identities],
                dtype=np.float32,
            )

        corr_h1 = _to_half(combined_h1, half1_idx)
        corr_h2 = _to_half(combined_h2, half2_idx)
        scale_corr_h1 = _to_half(pp_scale_h1, half1_idx)
        scale_corr_h2 = _to_half(pp_scale_h2, half2_idx)

        trans_h1 = None
        trans_h2 = None
        if "rlnOriginXAngst" in parts.columns and "rlnOriginYAngst" in parts.columns:
            offsets = np.stack(
                [
                    np.asarray(parts["rlnOriginXAngst"], dtype=np.float64) / float(ds_voxel),
                    np.asarray(parts["rlnOriginYAngst"], dtype=np.float64) / float(ds_voxel),
                ],
                axis=1,
            )
            trans_h1 = _to_half(offsets, half1_idx)
            trans_h2 = _to_half(offsets, half2_idx)
        elif "rlnOriginX" in parts.columns and "rlnOriginY" in parts.columns:
            offsets = np.stack(
                [
                    np.asarray(parts["rlnOriginX"], dtype=np.float64),
                    np.asarray(parts["rlnOriginY"], dtype=np.float64),
                ],
                axis=1,
            )
            trans_h1 = _to_half(offsets, half1_idx)
            trans_h2 = _to_half(offsets, half2_idx)

        angle_cols = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
        rot_h1 = None
        rot_h2 = None
        euler_h1 = None
        euler_h2 = None
        if all(col in parts.columns for col in angle_cols):
            eulers = np.stack([np.asarray(parts[col], dtype=np.float64) for col in angle_cols], axis=1)
            rotations = utils.R_from_relion(eulers, degrees=True).astype(np.float32)
            rot_h1 = _to_half(rotations, half1_idx)
            rot_h2 = _to_half(rotations, half2_idx)
            euler_h1 = _to_half(eulers, half1_idx)
            euler_h2 = _to_half(eulers, half2_idx)

        override_k = {
            "translation_sigma_angstrom": sigma_offset_avg,
            "translation_sigma_angstrom_per_half": sigma_offset_per_half,
            "previous_best_translations": [trans_h1, trans_h2],
            "previous_best_rotations": [rot_h1, rot_h2],
            "previous_best_rotation_eulers": [euler_h1, euler_h2],
        }
        if noise_h1 is not None and noise_h2 is not None:
            override_k["noise_variance"] = _relion_mpi_process_start_scoring_noise_pair(
                noise_h1,
                noise_h2,
                # RELION performs this broadcast once in MPI initialise().
                # Later uninterrupted iterations update each follower's noise
                # independently, so only replay slot 0 is process-start state.
                split_random_halves=(recovar_iter == 0 and model_paths[0] != model_paths[1]),
            )
        if direction_prior_h1 is not None and direction_prior_h2 is not None:
            override_k["direction_prior"] = [direction_prior_h1, direction_prior_h2]
        if class_tau2 is not None:
            override_k["class_tau2"] = class_tau2
        if include_normcorr:
            override_k["image_corrections"] = [corr_h1, corr_h2]
            override_k["serialized_scale_corrections"] = [scale_corr_h1, scale_corr_h2]
        overrides[recovar_iter] = override_k
        if include_normcorr:
            logger.info(
                "Replay override recovar iter %d: image_corr means=(%s, %s), serialized_scale_corr means=(%s, %s), "
                "sigma_offset=(half1 %.4f Å, half2 %.4f Å, mean %.4f Å)",
                recovar_iter + 1,
                _format_replay_mean_for_log(corr_h1),
                _format_replay_mean_for_log(corr_h2),
                _format_replay_mean_for_log(scale_corr_h1),
                _format_replay_mean_for_log(scale_corr_h2),
                sigma_offset_per_half[0],
                sigma_offset_per_half[1],
                sigma_offset_avg,
            )
        else:
            logger.info(
                "Replay override recovar iter %d: sigma_offset=(half1 %.4f Å, half2 %.4f Å, mean %.4f Å) "
                "(normcorr replay disabled)",
                recovar_iter + 1,
                sigma_offset_per_half[0],
                sigma_offset_per_half[1],
                sigma_offset_avg,
            )

    return overrides


_FINAL_REPLAY_GROUP_KEYS = {
    "poses": {
        "previous_best_translations",
        "previous_best_rotations",
        "previous_best_rotation_eulers",
    },
    "sampling": {
        "translation_sigma_angstrom",
        "translation_sigma_angstrom_per_half",
    },
    "corrections": {
        "noise_variance",
        "direction_prior",
        "image_corrections",
        "serialized_scale_corrections",
    },
    "references": set(),
}


def _select_final_replay_override(source_override, requested_fields):
    """Select diagnostic final-boundary groups without touching numbered state."""
    requested_groups = {
        token.strip().lower()
        for token in str(requested_fields).split(",")
        if token.strip()
    }
    valid_groups = set(_FINAL_REPLAY_GROUP_KEYS) | {"all"}
    unknown_groups = sorted(requested_groups - valid_groups)
    if not requested_groups or unknown_groups:
        raise ValueError(
            "--final-replay-fields requires one or more of "
            "poses,sampling,corrections,references,all; "
            f"unknown={unknown_groups}"
        )
    if "all" in requested_groups:
        requested_groups = set(_FINAL_REPLAY_GROUP_KEYS)
    selected_keys = set().union(*(_FINAL_REPLAY_GROUP_KEYS[group] for group in requested_groups))
    selected_override = {
        key: value for key, value in source_override.items() if key in selected_keys
    }
    return requested_groups, selected_override


def _load_final_replay_reference_maps(relion_dir, source_iteration, volume_shape):
    """Load the exact K=1 half references consumed by RELION finalization."""
    from recovar.core import fourier_transform_utils
    from recovar.utils.helpers import load_relion_volume

    relion_dir = Path(relion_dir)
    references = []
    for half_number in (1, 2):
        map_path = relion_dir / (
            f"run_it{int(source_iteration):03d}_half{half_number}_class001.mrc"
        )
        if not map_path.is_file():
            raise ValueError(
                "diagnostic final-only reference substitution is missing "
                f"{map_path}"
            )
        reference_real = np.asarray(load_relion_volume(str(map_path)), dtype=np.float32)
        if tuple(reference_real.shape) != tuple(volume_shape):
            raise ValueError(
                f"diagnostic final-only reference {map_path} has shape "
                f"{reference_real.shape}, expected {tuple(volume_shape)}"
            )
        references.append(
            jnp.asarray(fourier_transform_utils.get_dft3(reference_real).reshape(-1))
        )
        logger.info(
            "Diagnostic final-only reference half %d <- %s",
            half_number,
            map_path,
        )
    return references


def _resolve_final_replay_source_iteration(
    *, configured_max_iter, explicit_source_iteration, complete_iterations
):
    """Bind a finite RELION oracle to one exact last-numbered boundary."""
    complete = sorted({int(value) for value in complete_iterations})
    if not complete:
        raise ValueError("final replay oracle has no complete numbered RELION states")
    source_iteration = (
        max(complete)
        if explicit_source_iteration is None
        else int(explicit_source_iteration)
    )
    if source_iteration not in complete:
        raise ValueError(
            f"requested final replay source iteration {source_iteration} is not a complete oracle state; "
            f"available={complete}"
        )
    if source_iteration > int(configured_max_iter):
        raise ValueError(
            f"final replay source iteration {source_iteration} exceeds configured max_iter={configured_max_iter}"
        )
    missing_prefix = sorted(set(range(0, source_iteration + 1)) - set(complete))
    if missing_prefix:
        raise ValueError(
            f"final replay oracle is not contiguous through iteration {source_iteration}; missing={missing_prefix}"
        )
    return source_iteration


def _complete_relion_numbered_state_iterations(relion_dir):
    """Return iterations with data, sampling, and half/shared model state."""
    import re

    relion_dir = Path(relion_dir).resolve()
    complete = []
    for data_path in relion_dir.glob("run_it[0-9][0-9][0-9]_data.star"):
        match = re.fullmatch(r"run_it([0-9]{3})_data\.star", data_path.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        sampling = relion_dir / f"run_it{iteration:03d}_sampling.star"
        half_models = (
            relion_dir / f"run_it{iteration:03d}_half1_model.star",
            relion_dir / f"run_it{iteration:03d}_half2_model.star",
        )
        shared_model = relion_dir / f"run_it{iteration:03d}_model.star"
        if sampling.is_file() and (all(path.is_file() for path in half_models) or shared_model.is_file()):
            complete.append(iteration)
    return sorted(complete)


def _attach_relion_projector_capture(
    replay_iteration_overrides,
    *,
    capture_dir,
    manifest_path,
    capture_iteration,
    init_relion_iteration,
    relion_replay_dir,
    volume_shape,
    n_classes,
):
    """Attach one sealed live projector to its exact numbered replay slot."""

    from recovar.em.sampling import read_relion_model_metadata

    capture_iteration = int(capture_iteration)
    init_relion_iteration = int(init_relion_iteration)
    if init_relion_iteration != 0:
        raise ValueError(
            "captured RELION projector replay currently requires an uninterrupted "
            "cold-start trajectory (init_relion_iteration=0); a later jump would "
            "reapply MPI process-start noise semantics without the sealed follower state"
        )
    replay_slot = capture_iteration - init_relion_iteration - 1
    if replay_iteration_overrides is None:
        raise ValueError("captured RELION projector requires trajectory replay overrides")
    if replay_slot < 0 or replay_slot >= len(replay_iteration_overrides):
        raise ValueError(
            "captured RELION projector iteration is outside the configured replay trajectory: "
            f"capture_iteration={capture_iteration}, init_relion_iteration={init_relion_iteration}, "
            f"replay_slots={len(replay_iteration_overrides)}"
        )
    existing = replay_iteration_overrides[replay_slot]
    if existing is None:
        raise ValueError(f"captured RELION projector replay slot {replay_slot} has no state override")
    if "relion_projector_state" in existing:
        raise ValueError(f"captured RELION projector replay slot {replay_slot} is already populated")

    relion_replay_dir = Path(relion_replay_dir).expanduser().resolve()
    model_candidates = (
        relion_replay_dir / f"run_it{capture_iteration:03d}_half1_model.star",
        relion_replay_dir / f"run_it{capture_iteration:03d}_model.star",
    )
    model_path = next((path for path in model_candidates if path.is_file()), None)
    if model_path is None:
        raise ValueError(
            "captured RELION projector has no matching replay control model: "
            + " or ".join(str(path) for path in model_candidates)
        )
    model_metadata = read_relion_model_metadata(model_path)
    current_size = int(model_metadata["current_image_size"])
    if current_size <= 0:
        raise ValueError(f"invalid captured-projector replay current size: {current_size}")

    capture_dir = Path(capture_dir).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    projector_state = build_relion_projector_replay_state(
        capture_dir,
        manifest_path=manifest_path,
        iteration=capture_iteration,
        current_size=current_size,
        volume_shape=tuple(int(value) for value in volume_shape),
        n_classes=int(n_classes),
    )
    replay_iteration_overrides[replay_slot] = {
        **existing,
        "relion_projector_state": projector_state,
    }
    logger.info(
        "STRICT-PARITY: attached captured RELION Projector::data iteration=%d "
        "replay_slot=%d current_size=%d manifest=%s",
        capture_iteration,
        replay_slot,
        current_size,
        projector_state["source_manifest_sha256"],
    )
    return replay_slot, projector_state


def _load_init_previous_best_poses_npz(path, pose_iter="last"):
    """Load previous best poses from a RECOVAR refinement_results.npz file.

    This is a diagnostic/debugging hook for starting directly in the local
    search branch. It does not affect the default GUI/CLI path.
    """

    pose_path = Path(path)
    with np.load(pose_path, allow_pickle=False) as npz:
        if str(pose_iter).lower() in {"last", "latest"}:
            pattern = re.compile(r"^best_rotation_eulers_iter_(\d{3})_half0$")
            available = sorted(
                int(match.group(1))
                for key in npz.files
                if (match := pattern.match(key)) is not None
                and f"best_rotation_eulers_iter_{match.group(1)}_half1" in npz.files
                and f"best_translations_iter_{match.group(1)}_half0" in npz.files
                and f"best_translations_iter_{match.group(1)}_half1" in npz.files
            )
            if not available:
                raise ValueError(f"No numbered per-half best-pose arrays found in {pose_path}")
            iter_label = f"{available[-1]:03d}"
        elif str(pose_iter).lower() in {"final_all_data", "final-all-data"}:
            euler_keys = [
                "best_rotation_eulers_final_all_data_half0",
                "best_rotation_eulers_final_all_data_half1",
            ]
            trans_keys = [
                "best_translations_final_all_data_half0",
                "best_translations_final_all_data_half1",
            ]
            missing = [key for key in euler_keys + trans_keys if key not in npz.files]
            if missing:
                raise ValueError(f"Missing final-all-data pose arrays in {pose_path}: {missing}")
            eulers = [np.asarray(npz[key], dtype=np.float32) for key in euler_keys]
            translations = [np.asarray(npz[key], dtype=np.float32) for key in trans_keys]
            return {
                "iteration": "final_all_data",
                "previous_best_rotation_eulers": eulers,
                "previous_best_translations": translations,
            }
        else:
            iter_label = f"{int(pose_iter):03d}"

        euler_keys = [
            f"best_rotation_eulers_iter_{iter_label}_half0",
            f"best_rotation_eulers_iter_{iter_label}_half1",
        ]
        trans_keys = [
            f"best_translations_iter_{iter_label}_half0",
            f"best_translations_iter_{iter_label}_half1",
        ]
        missing = [key for key in euler_keys + trans_keys if key not in npz.files]
        if missing:
            raise ValueError(f"Missing pose arrays for iter {iter_label} in {pose_path}: {missing}")
        eulers = [np.asarray(npz[key], dtype=np.float32) for key in euler_keys]
        translations = [np.asarray(npz[key], dtype=np.float32) for key in trans_keys]

    for half, (euler, translation) in enumerate(zip(eulers, translations), start=1):
        if euler.ndim != 2 or euler.shape[1] != 3:
            raise ValueError(f"half-{half} Euler array must have shape (N, 3), got {euler.shape}")
        if translation.ndim != 2 or translation.shape[1] != 2:
            raise ValueError(f"half-{half} translation array must have shape (N, 2), got {translation.shape}")
        if euler.shape[0] != translation.shape[0]:
            raise ValueError(
                f"half-{half} Euler/translation row mismatch: {euler.shape[0]} vs {translation.shape[0]}",
            )

    return {
        "iteration": iter_label,
        "previous_best_rotation_eulers": eulers,
        "previous_best_translations": translations,
    }


def _load_init_noise_radial_npz(path, noise_iter="last"):
    """Load a diagnostic initial noise spectrum from refinement_results.npz."""

    noise_path = Path(path)
    with np.load(noise_path, allow_pickle=False) as npz:
        if str(noise_iter).lower() in {"last", "latest"}:
            pattern = re.compile(r"^noise_radial_iter_(\d{3})$")
            available = sorted(
                int(match.group(1)) for key in npz.files if (match := pattern.match(key)) is not None
            )
            if not available:
                raise ValueError(f"No numbered noise_radial_iter arrays found in {noise_path}")
            iter_label = f"{available[-1]:03d}"
        else:
            iter_label = f"{int(noise_iter):03d}"
        key = f"noise_radial_iter_{iter_label}"
        if key not in npz.files:
            raise ValueError(f"Missing {key} in {noise_path}")
        noise_radial = np.asarray(npz[key], dtype=np.float64)

    if noise_radial.ndim != 1:
        raise ValueError(f"{key} must be a 1D radial spectrum, got shape {noise_radial.shape}")
    if not np.all(np.isfinite(noise_radial)):
        raise ValueError(f"{key} contains non-finite values")
    if np.any(noise_radial <= 0.0):
        raise ValueError(f"{key} must be strictly positive")
    return {
        "iteration": iter_label,
        "noise_radial": noise_radial,
    }


def _validate_initial_noise_radial(noise_radial, *, label: str):
    noise_radial = np.asarray(noise_radial, dtype=np.float64)
    if noise_radial.ndim != 1:
        raise ValueError(f"{label} must be a 1D radial spectrum, got shape {noise_radial.shape}")
    if not np.all(np.isfinite(noise_radial)):
        raise ValueError(f"{label} contains non-finite values")
    if np.any(noise_radial <= 0.0):
        raise ValueError(f"{label} must be strictly positive")
    return noise_radial


def _initial_noise_cache_key(ds, args, image_subset, *, batch_size: int, apply_image_mask: bool):
    """Build an exact-cache key for the deterministic bootstrap noise estimate."""

    data_dir = Path(args.data_dir).resolve()
    file_fingerprints = []
    if data_dir.exists():
        for path in sorted(data_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".star", ".mrc", ".mrcs", ".npz", ".pkl", ".cs"}:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            file_fingerprints.append(
                {
                    "name": path.name,
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    payload = {
        "version": 1,
        "data_dir": str(data_dir),
        "files": file_fingerprints,
        "n_units": int(ds.n_units),
        "image_shape": tuple(int(x) for x in ds.image_shape),
        "voxel_size": float(ds.voxel_size),
        "subset": np.asarray(image_subset, dtype=np.int32).tolist(),
        "batch_size": int(batch_size),
        "apply_image_mask": bool(apply_image_mask),
        "relion_mask_params": None if getattr(args, "_relion_mask_params", None) is None else tuple(
            float(x) for x in args._relion_mask_params
        ),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    image_mask = getattr(ds, "image_mask", None)
    if image_mask is not None:
        mask_arr = np.asarray(image_mask, dtype=np.float32)
        digest.update(str(mask_arr.shape).encode("utf-8"))
        digest.update(mask_arr.tobytes(order="C"))
    return digest.hexdigest()


def _load_initial_noise_cache(cache_dir, cache_key, image_shape):
    cache_path = Path(cache_dir) / f"initial_noise_{cache_key}.npz"
    if not cache_path.exists():
        return None, cache_path
    with np.load(cache_path, allow_pickle=False) as npz:
        stored_key = str(npz["cache_key"]) if "cache_key" in npz.files else ""
        if stored_key != str(cache_key):
            raise ValueError(f"Initial noise cache key mismatch in {cache_path}")
        stored_shape = tuple(int(x) for x in np.asarray(npz["image_shape"], dtype=np.int64))
        expected_shape = tuple(int(x) for x in image_shape)
        if stored_shape != expected_shape:
            raise ValueError(
                f"Initial noise cache image_shape mismatch in {cache_path}: {stored_shape} vs {expected_shape}"
            )
        noise_radial = _validate_initial_noise_radial(
            npz["noise_radial"],
            label=f"{cache_path}:noise_radial",
        )
    return noise_radial, cache_path


def _save_initial_noise_cache(cache_dir, cache_key, image_shape, noise_radial):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        (cache_dir / "SAFE_TO_DELETE").touch(exist_ok=True)
    except OSError:
        pass
    cache_path = cache_dir / f"initial_noise_{cache_key}.npz"
    tmp_path = cache_dir / f".{cache_path.name}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        tmp_path,
        cache_key=np.asarray(str(cache_key)),
        image_shape=np.asarray(image_shape, dtype=np.int64),
        noise_radial=np.asarray(noise_radial, dtype=np.float64),
    )
    os.replace(tmp_path, cache_path)
    return cache_path


def _find_relion_optimiser_star(args):
    """Locate a RELION run_optimiser.star to source mask + max_significants from.

    Searches an explicit ``--relion_optimiser`` arg first, then sibling
    directories of ``--relion_half_sets``, the ``--data_dir`` itself, and
    finally any ``relion_ref*/`` subdirectory of ``--data_dir`` (matching
    fixtures that name their RELION output ``relion_ref_os0/`` or similar).
    Picks the latest ``run_it{NNN}_optimiser.star`` if no plain
    ``run_optimiser.star`` is present in a candidate directory.
    """
    explicit = getattr(args, "relion_optimiser", None)
    if explicit:
        p = Path(explicit).resolve()
        if p.exists():
            return p

    search_dirs = []
    # Strict-parity --relion_init_dir / --perturb_replay_relion_dir point
    # directly at the RELION reference run; check those FIRST so the K=4
    # fixture (with `relion_pdb_k4_os0_ref/` subdir name that doesn't match
    # the `relion_ref*` glob) finds its optimiser star and recovar uses
    # RELION's particle-diameter mask instead of the dataset default.
    relion_init_dir = getattr(args, "relion_init_dir", None)
    if relion_init_dir:
        search_dirs.append(Path(relion_init_dir).resolve())
    perturb_replay_dir = getattr(args, "perturb_replay_relion_dir", None)
    if perturb_replay_dir:
        search_dirs.append(Path(perturb_replay_dir).resolve())
    if args.relion_half_sets is not None:
        search_dirs.append(Path(args.relion_half_sets).resolve().parent)
    data_dir = Path(args.data_dir).resolve()
    search_dirs.append(data_dir)
    search_dirs.append(data_dir / "relion_ref")
    # Match `relion_*ref*/` subdirs (covers `relion_ref_os0/`,
    # `relion_pdb_k4_os0_ref/`, `relion_pdb_k2_os0_ref/`, etc.).
    if data_dir.is_dir():
        for sub in sorted(list(data_dir.glob("relion_ref*")) + list(data_dir.glob("relion_*ref*"))):
            if sub.is_dir():
                search_dirs.append(sub)

    seen = set()
    for d in search_dirs:
        d = d.resolve()
        if d in seen:
            continue
        seen.add(d)
        plain = d / "run_optimiser.star"
        if plain.exists():
            return plain
        # Fall back to the last per-iter optimiser STAR in the directory.
        per_iter = sorted(d.glob("run_it*_optimiser.star"))
        if per_iter:
            return per_iter[-1]
    return None


def _resolve_optimizer_random_seed(explicit_seed, relion_optimiser_star):
    """Resolve the optimiser seed without silently diverging from RELION.

    An explicit CLI seed always wins.  For strict-parity runs whose RELION
    optimiser was explicitly supplied, inherit ``_rlnRandomSeed`` when the
    CLI seed is omitted.  Ordinary standalone runs retain the historical
    deterministic default of 42.
    """
    if explicit_seed is not None:
        return int(explicit_seed), "explicit CLI"

    if relion_optimiser_star is not None:
        from recovar.em.sampling import read_relion_optimiser_metadata

        metadata = read_relion_optimiser_metadata(relion_optimiser_star)
        relion_seed = metadata.get("random_seed")
        if relion_seed is not None:
            return int(relion_seed), f"RELION optimiser {Path(relion_optimiser_star).resolve()}"

    return 42, "standalone default"


def _explicit_relion_optimiser_for_seed(args):
    """Return a seed source only when the user explicitly selected RELION state.

    ``_find_relion_optimiser_star`` also performs convenient incidental
    discovery under ``data_dir``.  That discovery is useful for masks and
    support caps, but must not unexpectedly change a standalone run's RNG.
    """
    explicit_relion_state = any(
        getattr(args, name, None)
        for name in ("relion_optimiser", "relion_init_dir", "perturb_replay_relion_dir")
    )
    return _find_relion_optimiser_star(args) if explicit_relion_state else None


def _effective_perturb_seed(args):
    """Resolve the SamplingPerturbation seed used by the refinement loop.

    RELION uses the optimiser ``--random_seed`` for SamplingPerturbation, so
    the CLI-level ``--seed`` must drive the perturbation stream too. An explicit
    ``--perturb_seed`` overrides it; a negative explicit value keeps the legacy
    non-deterministic NumPy perturbation path for diagnostics.
    """
    explicit = getattr(args, "perturb_seed", None)
    if explicit is not None:
        return None if int(explicit) < 0 else int(explicit)
    seed = getattr(args, "seed", None)
    return None if seed is None else int(seed)


def _maybe_apply_relion_image_mask(ds, args):
    """Override the dataset scoring mask with RELION's particle-diameter mask."""
    explicit_particle_diameter = getattr(args, "particle_diameter_ang", None)
    explicit_width_mask_edge = getattr(args, "width_mask_edge_px", 5.0)
    if explicit_particle_diameter is not None:
        params = (float(explicit_particle_diameter), float(explicit_width_mask_edge))
        optimiser_star = "explicit CLI"
    else:
        optimiser_star = _find_relion_optimiser_star(args)
        if optimiser_star is None:
            logger.info("RELION optimiser STAR not found; keeping dataset image mask")
            return None

        params = _load_relion_mask_params(optimiser_star)
        if params is None:
            logger.info("No RELION mask parameters found in %s; keeping dataset image mask", optimiser_star)
            return None

    particle_diameter_ang, width_mask_edge_px = params

    if particle_diameter_ang <= 0:
        logger.info("Non-positive RELION particle diameter %.1f A; keeping dataset image mask", particle_diameter_ang)
        return None

    # Use the backend's set_relion_image_mask hook so we get RELION-exact
    # softMaskOutsideMap behavior (geometry + bg-fill mode), not just the
    # mask array overlaid on top of the default "multiply" mode. The
    # multiply mode silently zeros out pixels outside the mask, while
    # RELION blends them with the local background mean — which is what
    # the noise/likelihood downstream expects. See image_backends.py
    # ::set_relion_image_mask for the bit-exact equivalence note.
    backend = ds.image_source.backend
    if hasattr(backend, "set_relion_image_mask"):
        backend.set_relion_image_mask(
            pixel_size=ds.voxel_size,
            particle_diameter_ang=particle_diameter_ang,
            width_mask_edge_px=width_mask_edge_px,
        )
    else:
        from recovar.core import mask as core_mask

        relion_mask = core_mask.relion_soft_image_mask(
            image_size=ds.image_shape[0],
            pixel_size=ds.voxel_size,
            particle_diameter_ang=particle_diameter_ang,
            width_mask_edge_px=width_mask_edge_px,
        )
        backend.image_mask = relion_mask
    if hasattr(ds.image_source, "image_mask"):
        ds.image_source.image_mask = backend.image_mask

    radius_px = particle_diameter_ang / (2.0 * ds.voxel_size)
    logger.info(
        "Applied RELION scoring mask from %s: particle_diameter=%.1f A, width_mask_edge=%.1f px, radius=%.2f px",
        optimiser_star,
        particle_diameter_ang,
        width_mask_edge_px,
        radius_px,
    )
    return params


def main():
    _assert_expected_repo_imports()
    parser = argparse.ArgumentParser(description="Run full EM refinement on synthetic data")
    parser.add_argument(
        "--data_dir",
        default="/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data",
        help="Directory containing particles.star, reference_init.mrc, etc.",
    )
    parser.add_argument(
        "--output",
        default="/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/our_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--skip-large-outputs",
        action="store_true",
        help=(
            "Skip refinement_results.npz and final MRC writes. Intended only for "
            "short timing probes where run logs and benchmark ledgers are sufficient."
        ),
    )
    parser.add_argument("--max_iter", type=int, default=10, help="Maximum EM iterations")
    parser.add_argument(
        "--healpix_order",
        type=int,
        default=3,
        help="RELION coarse pass-1 HEALPix order. With adaptive oversampling, "
        "pass 2 evaluates healpix_order + adaptive_oversampling.",
    )
    parser.add_argument(
        "--max_healpix_order",
        type=int,
        default=None,
        help=(
            "Maximum coarse HEALPix order for RECOVAR's RELION-style sampling "
            "updates. If omitted, K>1 Class3D stays fixed at --healpix_order "
            "to match RELION's _rlnDoAutoSampling=0 command path; K=1 keeps "
            "the historical auto-refine cap of 7. Set explicitly to allow "
            "Class3D coarse-grid refinement."
        ),
    )
    parser.add_argument(
        "--auto_local_healpix_order",
        type=int,
        default=4,
        help="RELION --auto_local_healpix_order threshold for switching from "
        "global to local angular searches. RELION's binary default is 4; "
        "set to 3 when comparing against runs launched with "
        "--auto_local_healpix_order 3.",
    )
    parser.add_argument("--offset_range", type=float, default=3.0, help="Translation search range (pixels)")
    parser.add_argument("--offset_step", type=float, default=1.0, help="Translation step (pixels)")
    parser.add_argument(
        "--offset_sigma_angstrom",
        type=float,
        default=10.0,
        help="RELION-style Gaussian translation-prior sigma in Angstrom.",
    )
    parser.add_argument("--adaptive_oversampling", type=int, default=1, help="Oversampling levels (0=off, 1=2x)")
    parser.add_argument(
        "--max_significants",
        type=int,
        default=None,
        help="Max significant samples per image. Use <=0 for RELION-style uncapped mode. "
        "If omitted, read _rlnMaximumSignificantPoses from the optimiser STAR.",
    )
    parser.add_argument(
        "--tau2_fudge",
        type=float,
        default=None,
        help="RELION tau2_fudge regularization strength. If omitted, use "
        "RELION's mode default: 1.0 for K=1 auto-refine and 4.0 for K>1 "
        "Class3D. If --relion_init_dir has run_it000_optimiser.star, its "
        "rlnTau2FudgeFactor/rlnTau2FudgeArg value takes precedence. Higher "
        "values produce smoother volumes (stronger prior).",
    )
    parser.add_argument(
        "--perturb_factor",
        type=float,
        default=0.5,
        help="RELION SamplingPerturbation factor (default 0.5 matching "
        "RELION GUI `--perturb 0.5`). Applies a per-iter random rigid "
        "rotation of the SO(3) trial grid and translation shift, ported "
        "from healpix_sampling.cpp:167-174 / 1909-1934 / 1810-1820. "
        "Set to 0 to disable.",
    )
    parser.add_argument(
        "--perturb_seed",
        type=int,
        default=None,
        help="Optional deterministic seed for the SamplingPerturbation RNG. "
        "If unset, defaults to --seed to match RELION's --random_seed. "
        "Use a negative value for the legacy non-reproducible NumPy path.",
    )
    parser.add_argument(
        "--perturb_replay_relion_dir",
        default=None,
        help="Controlled RELION trajectory replay: read SamplingPerturbInstance "
        "and per-iteration particle/model overrides from run_it{NNN}_* files in "
        "this directory. This is not an autonomous trajectory; omit it and use "
        "--perturb_seed for same-seed autonomous refinement.",
    )
    parser.add_argument(
        "--final-replay-relion-dir",
        default=None,
        help=(
            "Diagnostic-only final-boundary substitution source. Numbered refinement remains "
            "autonomous; selected last-numbered state fields and/or the unnumbered final sampling "
            "state are loaded only after convergence."
        ),
    )
    parser.add_argument(
        "--final-replay-fields",
        default="all",
        help=(
            "Comma-separated diagnostic final-only groups: poses, sampling, corrections, "
            "references, or all. "
            "poses replaces previous rotations/translations; sampling replaces translation sigma "
            "and final sampling grid/perturbation; corrections replaces noise, direction prior, "
            "image correction, and serialized scale correction; references replaces only the "
            "two input half-reference maps used by the final all-data expectation."
        ),
    )
    parser.add_argument(
        "--final-replay-source-iteration",
        type=int,
        default=None,
        help=(
            "Exact last-numbered RELION iteration used by --final-replay-relion-dir. "
            "Defaults to the latest contiguous complete numbered state and fails closed "
            "if it exceeds --max_iter or lacks final convergence provenance."
        ),
    )
    parser.add_argument(
        "--perturb-replay-restart-state-iterations",
        default="",
        help=(
            "Comma-separated numbered RELION sampling-state iterations where a "
            "provenance-qualified continuation restarted its sampling object. "
            "The next expectation reconstructs the unrounded perturbation from "
            "RELION's seed-1 restart state and random_seed+iteration; STAR values "
            "remain strict consistency guards. Example: 11 for a rescue whose "
            "first continued expectation is numbered iteration 12."
        ),
    )
    parser.add_argument(
        "--relion-projector-capture-dir",
        default=None,
        help=(
            "Directory containing a validated live RELION Projector::data capture. "
            "Requires --perturb_replay_relion_dir and "
            "--relion-projector-capture-iteration."
        ),
    )
    parser.add_argument(
        "--relion-projector-capture-manifest",
        default=None,
        help=(
            "Validated SHA-256 manifest for --relion-projector-capture-dir. "
            "Defaults to iterN_VALIDATED_SHA256SUMS inside that directory."
        ),
    )
    parser.add_argument(
        "--relion-projector-capture-iteration",
        type=int,
        default=None,
        help="Numbered RELION expectation iteration represented by the live capture.",
    )
    parser.add_argument(
        "--perturb-replay-restart-provenance",
        default=None,
        help=(
            "Required provenance file for --perturb-replay-restart-state-iterations. "
            "Its resolved path and SHA256 are recorded in refinement_results.npz and "
            "the benchmark ledger."
        ),
    )
    parser.add_argument(
        "--relion-scale-followers",
        type=int,
        default=None,
        help=(
            "Strict Class3D parity topology for RELION's follower-local group-scale state. "
            "Requires --relion-dispatch-schedule because RELION assigns --pool chunks "
            "dynamically. Pass 0 explicitly only for a non-parity diagnostic."
        ),
    )
    parser.add_argument(
        "--relion-dispatch-schedule",
        default=None,
        help=(
            "NPZ containing per-iteration dynamic MPI follower ownership captured from "
            "the same RELION oracle run. Its state-file and ordered-particle hashes are "
            "verified against the active replay/init directory. Required by default for "
            "strict K>1 replay/init."
        ),
    )
    parser.add_argument(
        "--relion-follower-scale-replay",
        default=None,
        help=(
            "Diagnostic NPZ containing complete follower-scale matrices at selected "
            "numbered RELION iterations. Requires strict K>1 follower topology and "
            "a matching captured dispatch schedule. The first numbered iteration is "
            "rejected because its resident image-normalization state does not yet exist."
        ),
    )
    parser.add_argument(
        "--init_relion_iteration",
        type=int,
        default=0,
        help=(
            "Diagnostic replay offset: treat the first RECOVAR loop iteration "
            "as continuing after this RELION iteration. This is mainly for "
            "profile-only jumps into later local-search iterations."
        ),
    )
    parser.add_argument(
        "--replay_relion_normcorr",
        dest="replay_relion_normcorr",
        action="store_true",
        default=None,
        help="If set together with --perturb_replay_relion_dir, also inject "
        "RELION's per-iter rlnNormCorrection / rlnGroupScaleCorrection into "
        "recovar's E-step at iter 2+. This is the default when "
        "--perturb_replay_relion_dir is set.",
    )
    parser.add_argument(
        "--no-replay_relion_normcorr",
        dest="replay_relion_normcorr",
        action="store_false",
        help="Disable RELION normCorrection / group-scale replay while still "
        "using other per-iteration replay overrides.",
    )
    parser.add_argument("--init_resolution", type=float, default=30.0, help="Initial resolution (Angstrom)")
    parser.add_argument(
        "--image-fourier-backend",
        choices=("host_numpy", "jax_gpu", "relion_cuda"),
        default="host_numpy",
        help=(
            "Fourier preprocessing backend for RELION-masked particle images. "
            "The default preserves the established host NumPy path; relion_cuda "
            "selects the source-faithful CUDA normalization, translation, and mask path."
        ),
    )
    parser.add_argument("--image_batch_size", type=int, default=500, help="Images per GPU batch")
    parser.add_argument(
        "--rotation_block_size",
        type=int,
        default=40000,
        help="Rotations per block (larger = faster, less Python overhead)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for half-set splitting, SamplingPerturbation, and optimiser sampling. "
            "If omitted with explicit RELION optimiser/init state, inherit _rlnRandomSeed; "
            "otherwise use 42."
        ),
    )
    parser.add_argument(
        "--relion_half_sets",
        default=None,
        help="Path to a RELION data STAR file with rlnRandomSubset column. "
        "If given, use RELION's half-set assignments instead of random seed.",
    )
    parser.add_argument(
        "--relion_optimiser",
        default=None,
        help="Explicit path to a RELION run_optimiser.star (or "
        "run_it{NNN}_optimiser.star). Used to source the particle-diameter "
        "mask + max_significants. If unset, searches data_dir and any "
        "relion_ref*/ subdirectory.",
    )
    parser.add_argument(
        "--particle_diameter_ang",
        type=float,
        default=None,
        help="Explicit RELION particle diameter in Angstrom for the scoring "
        "mask. Overrides mask discovery from --relion_optimiser.",
    )
    parser.add_argument(
        "--width_mask_edge_px",
        type=float,
        default=5.0,
        help="RELION softMaskOutsideMap edge width in pixels when --particle_diameter_ang is provided.",
    )
    parser.add_argument(
        "--relion_current_sizes",
        default=None,
        help="Comma-separated list of per-iteration current_sizes from RELION "
        "(oracle mode). Example: '0,56,30,50,70,98,98,92,88,90'",
    )
    parser.add_argument(
        "--relion_healpix_orders",
        default=None,
        help="Diagnostic oracle mode: comma-separated base HEALPix order for "
        "every numbered iteration. Suppresses autonomous angular-sampling "
        "transitions while leaving maps, posteriors, noise, and poses autonomous.",
    )
    parser.add_argument(
        "--firstiter_cc",
        action="store_true",
        default=False,
        help="Enable RELION --firstiter_cc emulation: iter-1 uses normalized "
        "cross-correlation scoring + winner-take-all reconstruction + ini_high "
        "low-pass on the iter-1 reference. Required for parity with RELION "
        "fixtures that were built with --firstiter_cc (Class3D defaults to it; "
        "auto_refine 3D-Auto-refine uses Gaussian scoring at iter 1 by default).",
    )
    parser.add_argument(
        "--apply-initial-lowpass",
        dest="apply_initial_lowpass",
        action="store_true",
        default=False,
        help="Apply RELION's ``initialLowPassFilterReferences`` to the init "
        "reference at ``--init_resolution`` before iter-1 expectation. "
        "RELION's ml_optimiser.cpp::initialLowPassFilterReferences runs "
        "whenever ``--ini_high > 0`` regardless of --firstiter_cc. recovar "
        "previously only mirrored that under --firstiter_cc, which left an "
        "iter-1 reconstruction gap on K=1 auto-refine fixtures built with "
        "``--ini_high 30`` and no ``--firstiter_cc``. Default off for backward "
        "compatibility; turn on for RELION-parity runs against such fixtures.",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=1,
        help="Number of K-class references for Class3D-style refinement. K=1 "
        "is the auto-refine path; K>1 enables joint class×pose EM. With K>1, "
        "either --init_class_volumes must be provided or "
        "<data_dir>/reference_init_class00K.mrc must exist for each K.",
    )
    parser.add_argument(
        "--relion_init_dir",
        default=None,
        help="Strict-parity cold-start: load RELION run_it000_model.star, "
        "or AutoRefine run_it000_half{1,2}_model.star for K=1, "
        "sigma2_noise spectrum + per-class rlnReferenceTau2 spectra + "
        "rlnTau2FudgeFactor/rlnTau2FudgeArg + rlnSigmaOffsetsAngst from this "
        "directory and use them as recovar's iter-0 state (instead of "
        "bootstrapping from images). "
        "Eliminates the ~1e-3 relative drift between recovar's bootstrapped "
        "sigma2_noise and RELION's, which is what flips ~22%% of K=4 iter-1 "
        "class assignments and caps mean_corr at 0.94 in pure cold-start. "
        "Combine with --perturb_replay_relion_dir to also match RELION's "
        "per-iter HEALPix grid jitter; that pair lifts K=4 cold-start to "
        "≥ 0.99 mean_corr (kernel-level parity, gated by "
        "test_em_parity_fast_kclass_strict_coldstart).",
    )
    parser.add_argument(
        "--init_class_volumes",
        default=None,
        help="Comma-separated paths to K initial reference volumes "
        "(K must equal --n_classes). Defaults to "
        "<data_dir>/reference_init_class00{1..K}.mrc when omitted.",
    )
    parser.add_argument(
        "--init_volume",
        default=None,
        help=(
            "Initial reference volume for K=1. Defaults to "
            "<data_dir>/reference_init.mrc when omitted."
        ),
    )
    parser.add_argument(
        "--init_previous_best_poses_npz",
        default=None,
        help=(
            "Diagnostic only: seed local-search priors from a RECOVAR "
            "refinement_results.npz file containing per-half best Euler and "
            "translation arrays."
        ),
    )
    parser.add_argument(
        "--init_previous_best_poses_iter",
        default="last",
        help=(
            "Iteration selector for --init_previous_best_poses_npz. Use an "
            "integer, 'last' for the latest numbered iteration, or "
            "'final_all_data'."
        ),
    )
    parser.add_argument(
        "--init_noise_from_npz",
        default=None,
        help=(
            "Diagnostic only: initialize sigma2_noise from a RECOVAR "
            "refinement_results.npz noise_radial_iter_### array instead of "
            "estimating it from images."
        ),
    )
    parser.add_argument(
        "--init_noise_iter",
        default="last",
        help="Iteration selector for --init_noise_from_npz. Use an integer or 'last'.",
    )
    parser.add_argument(
        "--initial_noise_cache_dir",
        default=None,
        help=(
            "Diagnostic speed cache for the masked bootstrap initial sigma2_noise "
            "estimate. On a cache miss the estimate is computed normally and saved; "
            "on a hit the exact cached radial spectrum is reused."
        ),
    )
    parser.add_argument(
        "--skip_final_iteration",
        action="store_true",
        help="Diagnostic only: skip the final all-data Nyquist iteration.",
    )
    parser.add_argument(
        "--timing_dir",
        default=None,
        help=(
            "Optional directory for lightweight per-iteration timing NPZs. "
            "This uses RECOVAR_PARITY_TIMING_DIR internally and does not "
            "write full parity tensor/volume dumps."
        ),
    )
    parser.add_argument(
        "--save_intermediates_dir",
        default=None,
        help=(
            "Optional debug directory for per-iteration regularized/unregularized "
            "class maps, Fourier accumulators, assignments, and metadata."
        ),
    )
    parser.add_argument(
        "--save_intermediates_skip_unregularized",
        action="store_true",
        help=(
            "When --save_intermediates_dir is set, save regularized maps and "
            "metadata but skip diagnostic unregularized maps. This preserves "
            "regularized-map FSC debugging while avoiding an extra "
            "reconstruction pass per iteration."
        ),
    )
    parser.add_argument(
        "--local_search_profile",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Control exact-local profile collection. The default profiles when "
            "--save_intermediates_dir is provided; use 'on' for timing ledgers "
            "without full intermediate dumps."
        ),
    )
    parser.add_argument(
        "--stop_after_local_search_profile",
        action="store_true",
        help=(
            "Diagnostic mode: stop after the first local-search E-step has "
            "written timing profiles, without running tau/FSC/reconstruction."
        ),
    )
    parser.add_argument(
        "--stop_after_local_search",
        action="store_true",
        help=(
            "Diagnostic mode: stop after the first local-search E-step without "
            "forcing detailed per-bucket profile collection."
        ),
    )
    parser.add_argument(
        "--stop_after_local_search_score_only",
        action="store_true",
        help=(
            "Diagnostic mode: stop after local pass-2 scoring while skipping "
            "the local M-step/noise accumulators. This is for pose/Pmax "
            "debugging only and does not produce maps or FSC-quality outputs."
        ),
    )
    parser.add_argument(
        "--diagnostic_single_half",
        action="store_true",
        help=(
            "Diagnostic local-search speed mode: with a local-search stop flag, "
            "run only half 1 and leave half 2 empty. This is invalid for map/FSC "
            "runs because it bypasses the gold-standard second half."
        ),
    )
    parser.add_argument(
        "--benchmark_ledger_json",
        default=None,
        help="Optional JSON path for an auto-refine quality/performance ledger.",
    )
    args = parser.parse_args()

    seed_optimiser_star = _explicit_relion_optimiser_for_seed(args)
    args.seed, optimizer_seed_source = _resolve_optimizer_random_seed(args.seed, seed_optimiser_star)
    logger.info("Optimiser random seed: %d (%s)", args.seed, optimizer_seed_source)

    if args.timing_dir:
        timing_dir_path = Path(args.timing_dir)
        timing_dir_path.mkdir(parents=True, exist_ok=True)
        os.environ["RECOVAR_PARITY_TIMING_DIR"] = str(timing_dir_path)
    else:
        timing_dir_path = None

    # Verify GPU
    devices = jax.devices()
    logger.info("JAX devices: %s", devices)
    if not any(getattr(d, "platform", "") in {"gpu", "cuda"} for d in devices):
        logger.error("No GPU available. Aborting.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # ---- Load dataset ----
    logger.info("Loading dataset from %s", args.data_dir)
    from recovar.data_io.cryoem_dataset import load_dataset

    ds = load_dataset(
        os.path.join(args.data_dir, "particles.star"),
        lazy=False,
    )
    relion_mask_params = _maybe_apply_relion_image_mask(ds, args)
    args._relion_mask_params = relion_mask_params
    particle_diameter_ang = None if relion_mask_params is None else float(relion_mask_params[0])
    logger.info("Dataset: %d images, image_shape=%s, voxel_size=%.3f A/px", ds.n_units, ds.image_shape, ds.voxel_size)

    # ---- Create half-sets ----
    n_images = ds.n_units
    if args.n_classes < 1:
        raise SystemExit(f"--n_classes must be >= 1, got {args.n_classes}")

    import starfile as _starfile

    our_star = _starfile.read(os.path.join(args.data_dir, "particles.star"))
    our_particles = our_star["particles"] if isinstance(our_star, dict) else our_star
    # Keep the input-STAR particle identities available for replay mapping.
    # RELION data STAR rows can be permuted relative to this table, so callers
    # must map by rlnImageName rather than assuming row positions coincide.
    our_names = np.asarray(our_particles["rlnImageName"])
    expected_accuracy_half1_base_order_local = None
    expected_accuracy_half1_optics_group_ids = None
    expected_accuracy_half1_particle_ids = None
    expected_accuracy_half1_ctf_params = None
    expected_accuracy_do_ctf_correction = None
    relion_particles = None
    relion_group_particles = None
    relion_group_source = None

    if args.relion_half_sets is not None:
        # Use RELION's half-set split from rlnRandomSubset
        logger.info("Loading RELION half-set assignments from %s", args.relion_half_sets)
        relion_data = _starfile.read(args.relion_half_sets)
        relion_particles = relion_data["particles"]
        (
            half1_idx,
            half2_idx,
            expected_accuracy_half1_base_order_local,
            expected_accuracy_half1_optics_group_ids,
            expected_accuracy_half1_particle_ids,
        ) = _relion_halfset_and_accuracy_layout(our_particles, relion_particles)
        from recovar.data_io import metadata_readers

        relion_ctf_with_apix = metadata_readers.parse_ctf_from_star(
            args.relion_half_sets,
            ds.grid_size,
        )
        expected_accuracy_half1_ctf_params = np.asarray(
            relion_ctf_with_apix[expected_accuracy_half1_particle_ids, 1:],
            dtype=np.float64,
        )
        logger.info("Using RELION half-set split: %d (subset=1) + %d (subset=2)", len(half1_idx), len(half2_idx))
    else:
        half1_idx, half2_idx = _default_refinement_subsets(n_images, args.seed, args.n_classes)
        if args.n_classes > 1:
            logger.info(
                "Using RELION Class3D all-data split: %d particles + empty second accumulator",
                len(half1_idx),
            )

    local_stop_requested = (
        bool(args.stop_after_local_search_profile)
        or bool(args.stop_after_local_search)
        or bool(args.stop_after_local_search_score_only)
    )
    if args.diagnostic_single_half:
        if not local_stop_requested:
            raise SystemExit(
                "--diagnostic_single_half is only valid with --stop_after_local_search, "
                "--stop_after_local_search_profile, or --stop_after_local_search_score_only"
            )
        if args.n_classes != 1:
            raise SystemExit("--diagnostic_single_half is K=1-only")
        logger.warning(
            "Diagnostic single-half local-search probe: running half 1 only (%d images); "
            "half 2 is empty. Do not use this for map/FSC quality.",
            int(half1_idx.size),
        )
        half2_idx = np.empty(0, dtype=np.int64)

    ds_half1 = ds.subset(half1_idx)
    ds_half2 = ds.subset(half2_idx)
    logger.info("Half-sets: %d + %d images", ds_half1.n_units, ds_half2.n_units)
    relion_group_particles, relion_group_source = _select_authoritative_group_particles(
        halfset_particles=relion_particles,
        halfset_source=args.relion_half_sets,
        replay_dirs=(args.perturb_replay_relion_dir, args.relion_init_dir),
        init_relion_iteration=args.init_relion_iteration,
    )
    native_group_layout = _resolve_native_group_layout(
        our_particles,
        half1_idx,
        half2_idx,
        relion_particles=relion_group_particles,
    )
    native_group_ids_per_half = (
        None if native_group_layout is None else list(native_group_layout.group_ids_per_half)
    )
    native_group_count = None if native_group_layout is None else native_group_layout.n_groups
    strict_relion_scale_context = bool(
        args.n_classes > 1
        and (args.perturb_replay_relion_dir is not None or args.relion_init_dir is not None)
    )
    relion_dispatch_schedule = None
    if args.relion_dispatch_schedule is not None:
        if not strict_relion_scale_context:
            raise SystemExit(
                "--relion-dispatch-schedule is strict K>1 RELION replay/init state only"
            )
        try:
            relion_dispatch_schedule = load_relion_dispatch_schedule(
                args.relion_dispatch_schedule
            )
            oracle_dirs = []
            for candidate in (args.perturb_replay_relion_dir, args.relion_init_dir):
                if candidate is None:
                    continue
                resolved = Path(candidate).expanduser().resolve()
                if resolved not in oracle_dirs:
                    oracle_dirs.append(resolved)
                    verify_relion_dispatch_schedule_oracle(
                        relion_dispatch_schedule,
                        resolved,
                    )
            def _require_manifested_oracle_file(path, *, label):
                resolved_path = Path(path).expanduser().resolve()
                for oracle_dir in oracle_dirs:
                    try:
                        relative = resolved_path.relative_to(oracle_dir).as_posix()
                    except ValueError:
                        continue
                    if relative not in relion_dispatch_schedule.oracle_artifact_paths:
                        raise ValueError(
                            f"{label} is not included in the verified RELION oracle manifest: "
                            f"{resolved_path}"
                        )
                    return
                raise ValueError(
                    f"{label} must belong to a verified RELION oracle directory: {resolved_path}"
                )

            discovered_optimiser = _find_relion_optimiser_star(args)
            if discovered_optimiser is not None:
                _require_manifested_oracle_file(
                    discovered_optimiser,
                    label="consumed RELION optimiser",
                )
            for oracle_dir in oracle_dirs:
                sampling_candidates = list(oracle_dir.glob("run_it*_sampling.star"))
                final_sampling = oracle_dir / "run_sampling.star"
                if final_sampling.exists():
                    sampling_candidates.append(final_sampling)
                for sampling_path in sampling_candidates:
                    _require_manifested_oracle_file(
                        sampling_path,
                        label="consumed RELION sampling state",
                    )
            observed_group_order = relion_ordered_particle_sha256(relion_group_particles)
            if observed_group_order != relion_dispatch_schedule.particle_order_sha256:
                raise ValueError(
                    "authoritative RELION group/half-set particle order does not match "
                    "the dispatch schedule"
                )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Invalid --relion-dispatch-schedule: {exc}") from exc
    if args.relion_scale_followers is None:
        if strict_relion_scale_context and relion_dispatch_schedule is None:
            raise SystemExit(
                "Strict K>1 RELION replay requires --relion-dispatch-schedule captured "
                "from the same oracle run: expectation follower ownership is a dynamic "
                "MPI work queue and cannot be reconstructed from --seed. Pass "
                "--relion-scale-followers 0 only for an explicit non-parity diagnostic."
            )
        relion_scale_followers = (
            0 if relion_dispatch_schedule is None else relion_dispatch_schedule.n_followers
        )
    else:
        relion_scale_followers = int(args.relion_scale_followers)
        if relion_scale_followers < 0:
            raise SystemExit("--relion-scale-followers must be non-negative")
        if relion_scale_followers > 0 and not strict_relion_scale_context:
            raise SystemExit(
                "--relion-scale-followers is strict K>1 RELION replay/init state only; "
                "provide --relion_init_dir or --perturb_replay_relion_dir"
            )
    if relion_scale_followers > 0 and relion_dispatch_schedule is None:
        raise SystemExit(
            "--relion-scale-followers > 0 requires --relion-dispatch-schedule; "
            "seed-only/static ownership is not RELION-exact"
        )
    if (
        relion_dispatch_schedule is not None
        and relion_scale_followers != relion_dispatch_schedule.n_followers
    ):
        raise SystemExit(
            "--relion-scale-followers disagrees with --relion-dispatch-schedule "
            f"({relion_scale_followers} != {relion_dispatch_schedule.n_followers})"
        )
    if relion_scale_followers > 0 and native_group_layout is None:
        raise SystemExit("Strict RELION follower-scale emulation requires an authoritative group layout")
    if relion_scale_followers > 0 and int(args.init_relion_iteration) > 0:
        raise SystemExit(
            "Strict RELION follower-scale emulation cannot cold-start after iteration 0 from "
            "a leader-serialized STAR; rerun from iteration 0 (full follower-scale checkpoint "
            "input is not implemented)"
        )
    relion_follower_scale_replay = None
    if args.relion_follower_scale_replay is not None:
        if not strict_relion_scale_context or relion_scale_followers < 1:
            raise SystemExit(
                "--relion-follower-scale-replay requires strict K>1 RELION follower topology"
            )
        try:
            relion_follower_scale_replay = load_relion_follower_scale_replay(
                args.relion_follower_scale_replay
            )
            validate_relion_follower_scale_replay(
                relion_follower_scale_replay,
                n_followers=relion_scale_followers,
                n_groups=native_group_layout.n_groups,
                schedule_iterations=relion_dispatch_schedule.relion_iterations,
                schedule_oracle_id=relion_dispatch_schedule.oracle_id,
                schedule_artifact_paths=relion_dispatch_schedule.oracle_artifact_paths,
                oracle_dir=oracle_dirs[0],
                numbered_iterations=range(
                    int(args.init_relion_iteration) + 1,
                    int(args.init_relion_iteration) + int(args.max_iter) + 1,
                ),
                first_numbered_iteration=int(args.init_relion_iteration) + 1,
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Invalid --relion-follower-scale-replay: {exc}") from exc
        logger.info(
            "Diagnostic RELION follower-scale replay: source=%s oracle_id=%s "
            "iterations=%s shape=%s",
            relion_follower_scale_replay.source,
            relion_follower_scale_replay.oracle_id,
            relion_follower_scale_replay.relion_iterations.tolist(),
            relion_follower_scale_replay.follower_scales.shape,
        )
    if native_group_layout is not None:
        logger.info(
            "Native RELION group layout: source=%s full_groups=%d half1_present=%d half2_present=%d",
            native_group_layout.source,
            native_group_count,
            int(np.unique(native_group_ids_per_half[0]).size),
            int(np.unique(native_group_ids_per_half[1]).size),
        )
        if relion_group_source is not None:
            logger.info("Native RELION group layout provenance: %s", relion_group_source)
    if relion_scale_followers > 0:
        logger.info(
            "Strict RELION follower-scale topology: followers=%d physical_groups=%d "
            "optics_groups=%d dynamic_schedule=%s",
            relion_scale_followers,
            native_group_layout.n_groups,
            native_group_layout.n_optics_groups,
            relion_dispatch_schedule.source,
        )
        logger.info(
            "Verified RELION dispatch oracle: oracle_id=%s artifacts=%d particle_star=%s",
            relion_dispatch_schedule.oracle_id,
            len(relion_dispatch_schedule.oracle_artifact_paths),
            relion_dispatch_schedule.particle_star_relative_path,
        )
    relion_scale_follower_owners_by_iteration = None
    if relion_scale_followers > 0:
        relion_scale_follower_owners_by_iteration = {}
        for relion_iteration_value in relion_dispatch_schedule.relion_iterations:
            relion_iteration = int(relion_iteration_value)
            try:
                owners_half1 = relion_class3d_follower_owners_from_schedule(
                    relion_dispatch_schedule,
                    particle_ids_by_image=native_group_layout.particle_ids_per_half[0],
                    optics_group_ids_by_image=native_group_layout.optics_group_ids_per_half[0],
                    random_seed=int(args.seed),
                    relion_iteration=relion_iteration,
                )
            except (RuntimeError, ValueError) as exc:
                raise SystemExit(
                    f"RELION dispatch schedule cannot supply iteration {relion_iteration}: {exc}"
                ) from exc
            relion_scale_follower_owners_by_iteration[relion_iteration] = [
                owners_half1,
                np.zeros(native_group_layout.particle_ids_per_half[1].size, dtype=np.int64),
            ]
        required_numbered_iterations = set(
            range(
                int(args.init_relion_iteration) + 1,
                int(args.init_relion_iteration) + int(args.max_iter) + 1,
            )
        )
        missing_numbered_iterations = sorted(
            required_numbered_iterations - set(relion_scale_follower_owners_by_iteration)
        )
        if missing_numbered_iterations:
            raise SystemExit(
                "RELION dispatch schedule cannot supply requested numbered iterations: "
                f"{missing_numbered_iterations}"
            )

    optimiser_star = _find_relion_optimiser_star(args)
    relion_firstiter_ini_high_angstrom = None
    if optimiser_star is not None:
        from recovar.em.sampling import read_relion_optimiser_metadata

        expected_accuracy_do_ctf_correction = read_relion_optimiser_metadata(
            optimiser_star,
        ).get("do_correct_ctf")
        if expected_accuracy_do_ctf_correction is not None:
            expected_accuracy_do_ctf_correction = bool(expected_accuracy_do_ctf_correction)
            logger.info(
                "RELION expected-accuracy CTF correction: %s (from %s)",
                expected_accuracy_do_ctf_correction,
                optimiser_star,
            )
        optimiser_text = Path(optimiser_star).read_text(errors="ignore")
        relion_firstiter_ini_high_angstrom = _parse_relion_cli_ini_high(optimiser_text)
        if args.firstiter_cc:
            if relion_firstiter_ini_high_angstrom is None:
                logger.info(
                    "RELION firstiter_cc: no positive --ini_high found in %s; "
                    "not applying post-iter1 ini_high low-pass",
                    optimiser_star,
                )
            else:
                logger.info(
                    "RELION firstiter_cc: using --ini_high %.2f A from %s for post-iter1 low-pass",
                    float(relion_firstiter_ini_high_angstrom),
                    optimiser_star,
                )
    if args.max_significants is None and optimiser_star is not None:
        relion_max_significants = _load_relion_max_significants(optimiser_star)
        if relion_max_significants is not None:
            args.max_significants = relion_max_significants
            logger.info(
                "Using RELION max_significants from %s: %d",
                optimiser_star,
                args.max_significants,
            )
    if args.max_significants is None:
        args.max_significants = 500

    # ---- Load initial volume ----
    # CANONICAL recovar idiom for loading a volume: load_mrc + get_dft3.
    # See recovar/output/output.py:980-984 and recovar/simulation/simulator.py:425.
    # NEVER use raw `mrcfile.open` + `np.fft.fftn(np.fft.ifftshift(...))` here:
    # that produces a Fourier volume with the right values but at WRONG array
    # indices (DC at corner instead of center), so `slice_volume` reads
    # Nyquist as if it were DC and projections are off by ~2400x in amplitude
    # at low frequencies.
    from recovar.utils.helpers import load_mrc as _load_mrc

    # RELION's ``initialLowPassFilterReferences`` (ml_optimiser.cpp:3556) low-
    # pass-filters mymodel.Iref in place at startup, gated only on
    # ``ini_high > 0`` (not on ``--firstiter_cc``). With ``--apply-initial-
    # lowpass``, mirror that behavior: apply LP at ``--init_resolution`` to
    # the reference before iter-1 expectation. The Fourier mask edge is
    # RELION's ``WIDTH_FMASK_EDGE = 2`` (see ml_optimiser.h:91), NOT the
    # real-space ``--maskedge = 5`` — those are distinct quantities.
    _RELION_FMASK_EDGE = 2

    def _apply_ini_high_lowpass_real(volume_real, volume_shape, voxel_size, ini_high):
        from recovar.em.initial_model.bootstrap_iref import initial_low_pass_filter_references

        filtered = initial_low_pass_filter_references(
            np.asarray(volume_real, dtype=np.float64)[None, ...],
            ori_size=int(volume_shape[0]),
            pixel_size=float(voxel_size),
            ini_high_ang=float(ini_high),
            filter_edgewidth=float(_RELION_FMASK_EDGE),
        )[0]
        return filtered.astype(np.float32, copy=False)

    _apply_ini_lowpass = bool(getattr(args, "apply_initial_lowpass", False))
    _ini_high_for_lowpass = (
        float(args.init_resolution)
        if _apply_ini_lowpass and float(args.init_resolution) > 0.0
        else None
    )

    if args.n_classes == 1:
        init_mrc_path = args.init_volume or os.path.join(args.data_dir, "reference_init.mrc")
        init_vol_real = _load_mrc(init_mrc_path).astype(np.float32)
        assert init_vol_real.shape == ds.volume_shape, (
            f"Volume shape mismatch: {init_vol_real.shape} vs {ds.volume_shape}"
        )
        if _ini_high_for_lowpass is not None:
            init_vol_real = _apply_ini_high_lowpass_real(
                init_vol_real, ds.volume_shape, ds.voxel_size, _ini_high_for_lowpass,
            )
            logger.info(
                "Applied RELION initialLowPassFilterReferences to init reference: ini_high=%.2f A, fmask_edge=%d shells",
                _ini_high_for_lowpass, _RELION_FMASK_EDGE,
            )
        init_vol_ft = np.array(ftu.get_dft3(jnp.asarray(init_vol_real))).astype(np.complex64).reshape(-1)
        logger.info("Initial volume loaded from %s: shape=%s", init_mrc_path, init_vol_real.shape)
    else:
        if args.init_class_volumes:
            class_paths = [p.strip() for p in args.init_class_volumes.split(",")]
        else:
            class_paths = [
                os.path.join(args.data_dir, f"reference_init_class{k + 1:03d}.mrc") for k in range(args.n_classes)
            ]
        if len(class_paths) != args.n_classes:
            raise SystemExit(f"--init_class_volumes count {len(class_paths)} != --n_classes {args.n_classes}")
        per_class_ft = []
        for k, p in enumerate(class_paths):
            vol_real = _load_mrc(p).astype(np.float32)
            assert vol_real.shape == ds.volume_shape, (
                f"Class {k + 1} volume shape mismatch at {p}: {vol_real.shape} vs {ds.volume_shape}"
            )
            if _ini_high_for_lowpass is not None:
                vol_real = _apply_ini_high_lowpass_real(
                    vol_real, ds.volume_shape, ds.voxel_size, _ini_high_for_lowpass,
                )
            vol_ft = np.array(ftu.get_dft3(jnp.asarray(vol_real))).astype(np.complex64).reshape(-1)
            per_class_ft.append(vol_ft)
            logger.info("Class %d initial volume loaded from %s", k + 1, p)
        if _ini_high_for_lowpass is not None:
            logger.info(
                "Applied RELION initialLowPassFilterReferences to %d init references: ini_high=%.2f A, fmask_edge=%d shells",
                args.n_classes, _ini_high_for_lowpass, _RELION_FMASK_EDGE,
            )
        # Stack to (K, V); refine_single_volume._normalize_initial_means handles the
        # per-half broadcast.
        init_vol_ft = np.stack(per_class_ft, axis=0)
        # For downstream init_PS estimation, use class-1 as the representative
        # (K-class noise/prior bootstrap currently uses a single spectrum).
        init_vol_real = _load_mrc(class_paths[0]).astype(np.float32)

    # ---- Set up rotation and translation grids ----
    from recovar.em.sampling import get_relion_rotation_grid, get_translation_grid

    init_healpix_order, finest_healpix_order = _resolve_relion_sampling_orders(
        args.healpix_order,
        args.adaptive_oversampling,
    )
    effective_max_healpix_order, max_healpix_order_source = _resolve_effective_max_healpix_order(
        n_classes=args.n_classes,
        healpix_order=init_healpix_order,
        max_healpix_order=args.max_healpix_order,
    )
    rotation_grid_order = init_healpix_order
    logger.info(
        "RELION grid orders: coarse/pass1=%d, fine/pass2=%d (adaptive_oversampling=%d)",
        init_healpix_order,
        finest_healpix_order,
        args.adaptive_oversampling,
    )
    logger.info(
        "Max coarse HEALPix order: %d (%s)",
        effective_max_healpix_order,
        max_healpix_order_source,
    )

    rotations = get_relion_rotation_grid(rotation_grid_order).astype(np.float32)
    translations = get_translation_grid(args.offset_range, args.offset_step).astype(np.float32)
    logger.info("Rotation grid: %d rotations (healpix_order=%d)", rotations.shape[0], rotation_grid_order)
    logger.info(
        "Translation grid: %d translations (range=%.1f, step=%.1f)",
        translations.shape[0],
        args.offset_range,
        args.offset_step,
    )

    # ---- Initialize noise and prior ----
    # Use a RELION-style initial sigma2 estimate from particle power spectra
    # instead of a flat unit spectrum, so iteration 1 starts on a comparable
    # likelihood scale.
    image_size = ds.image_size
    volume_size = ds.volume_size

    from recovar.reconstruction import noise as recon_noise

    if args.init_noise_from_npz is not None:
        init_noise = _load_init_noise_radial_npz(args.init_noise_from_npz, args.init_noise_iter)
        initial_noise_radial = init_noise["noise_radial"]
        noise_variance = recon_noise.make_radial_noise(initial_noise_radial, ds.image_shape)
        logger.info(
            "Diagnostic init: loaded sigma2_noise from %s iter=%s: min=%.3e median=%.3e max=%.3e",
            args.init_noise_from_npz,
            init_noise["iteration"],
            float(np.min(np.asarray(initial_noise_radial))),
            float(np.median(np.asarray(initial_noise_radial))),
            float(np.max(np.asarray(initial_noise_radial))),
        )
    else:
        initial_noise_subset = np.arange(min(1000, ds.n_units), dtype=np.int32)
        initial_noise_batch_size = min(args.image_batch_size, initial_noise_subset.size)
        initial_noise_cache_key = None
        initial_noise_cache_path = None
        initial_noise_radial = None
        if args.initial_noise_cache_dir is not None:
            initial_noise_cache_key = _initial_noise_cache_key(
                ds,
                args,
                initial_noise_subset,
                batch_size=initial_noise_batch_size,
                apply_image_mask=True,
            )
            initial_noise_radial, initial_noise_cache_path = _load_initial_noise_cache(
                args.initial_noise_cache_dir,
                initial_noise_cache_key,
                ds.image_shape,
            )
            if initial_noise_radial is not None:
                logger.info(
                    "Initial sigma2_noise cache hit: %s",
                    initial_noise_cache_path,
                )
        # In RELION mode the E-step scores masked images, so the bootstrap noise
        # MUST come from masked images too — otherwise sigma2 is dominated by the
        # solvent area and the iter-1 chi² is ~3.3-6× too small (verified
        # 2026-04-08 against the tiny parity dataset, see tmp/check_sigma2_mask.py).
        if initial_noise_radial is None:
            initial_noise_radial = recon_noise.estimate_initial_noise_spectrum_from_unaligned_images(
                ds,
                initial_noise_subset,
                batch_size=initial_noise_batch_size,
                apply_image_mask=True,
            )
            if args.initial_noise_cache_dir is not None:
                initial_noise_cache_path = _save_initial_noise_cache(
                    args.initial_noise_cache_dir,
                    initial_noise_cache_key,
                    ds.image_shape,
                    initial_noise_radial,
                )
                logger.info("Initial sigma2_noise cache saved: %s", initial_noise_cache_path)
        noise_variance = recon_noise.make_radial_noise(initial_noise_radial, ds.image_shape)
        logger.info(
            "Initial sigma2_noise estimate from %d images: min=%.3e median=%.3e max=%.3e",
            initial_noise_subset.size,
            float(np.min(np.asarray(initial_noise_radial))),
            float(np.median(np.asarray(initial_noise_radial))),
            float(np.max(np.asarray(initial_noise_radial))),
        )

    # Compute initial signal prior from init volume (weak prior). For K>1
    # use class-1 as the representative volume; the engine derives per-class
    # tau2 trajectories from the per-class FSCs once the loop starts.
    from recovar.reconstruction.regularization import average_over_shells

    if args.n_classes > 1:
        init_PS_source = jnp.asarray(per_class_ft[0])
    else:
        init_PS_source = jnp.asarray(init_vol_ft)
    init_PS = average_over_shells(jnp.abs(init_PS_source) ** 2, ds.volume_shape)
    from recovar import utils

    init_prior = utils.make_radial_image(init_PS, ds.volume_shape, extend_last_frequency=True)
    # Scale by a factor to provide regularization without being too strong
    mean_variance = jnp.asarray(init_prior * 0.5 + jnp.max(init_prior) * 1e-4)

    # ---- STRICT-PARITY: --relion_init_dir override of bootstrapped iter-0 state ----
    # When set, replace the image-bootstrap sigma2_noise + power-spectrum-bootstrap
    # tau2 with RELION's exact iter-0 values from run_it000_model.star. This
    # eliminates the ~1e-3 relative drift between bootstraps that flips ~22%
    # of K=4 iter-1 class assignments and caps cold-start mean_corr at 0.94.
    relion_init_sigma_offset_angstrom = None
    relion_init_tau2_fudge = None
    if args.relion_init_dir is not None:
        import re as _re
        from pathlib import Path as _Path

        _relion_init_dir = _Path(args.relion_init_dir)
        _it0_optim_path = _relion_init_dir / "run_it000_optimiser.star"
        _it0_model_bundle = _load_relion_it000_model_stars(_relion_init_dir, args.n_classes)
        _it0_models = _it0_model_bundle["models"]
        _it0_model = _it0_model_bundle["reference_model"]
        _it0_model_path = _it0_model_bundle["reference_model_path"]
        # sigma2_noise spectrum (× N⁴ for recovar's unit convention; matches
        # run_k_class_parity.py:715-717).
        _n4 = ds.grid_size**4
        _relion_sigma2_per_model = [
            _read_relion_single_optics_sigma2_noise(
                _model,
                context=f"RELION iteration-0 model {model_index + 1}",
            )
            for model_index, _model in enumerate(_it0_models)
        ]
        if any(sigma2 is None for sigma2 in _relion_sigma2_per_model):
            raise ValueError("RELION iteration-0 model is missing rlnSigma2Noise")
        if _it0_model_bundle["source"] == "half-specific":
            # RELION MPI follower rank 1 broadcasts its scoring noise spectrum
            # to the half-2 follower during initialisation.
            _relion_sigma2_per_model[1] = _relion_sigma2_per_model[0].copy()
            logger.info(
                "STRICT-PARITY: emulating RELION MPI rank-1 sigma2_noise broadcast "
                "for both AutoRefine half-sets",
            )
        if len(_relion_sigma2_per_model) == 1:
            _relion_sigma2 = _relion_sigma2_per_model[0]
            _relion_noise_radial = jnp.asarray(_relion_sigma2 * _n4)
            noise_variance = recon_noise.make_radial_noise(_relion_noise_radial, ds.image_shape)
            logger.info(
                "STRICT-PARITY: replaced bootstrapped sigma2_noise with RELION it000 "
                "shared spectrum (× N^4=%.3e). RELION shape=%s, head=%s",
                float(_n4),
                _relion_sigma2.shape,
                np.asarray(_relion_sigma2[:5]),
            )
        else:
            noise_variance = [
                recon_noise.make_radial_noise(jnp.asarray(_relion_sigma2 * _n4), ds.image_shape)
                for _relion_sigma2 in _relion_sigma2_per_model
            ]
            logger.info(
                "STRICT-PARITY: replaced bootstrapped sigma2_noise with RELION it000 "
                "per-half spectra (× N^4=%.3e). half1 shape=%s head=%s half2 shape=%s head=%s",
                float(_n4),
                _relion_sigma2_per_model[0].shape,
                np.asarray(_relion_sigma2_per_model[0][:5]),
                _relion_sigma2_per_model[1].shape,
                np.asarray(_relion_sigma2_per_model[1][:5]),
            )
        # Per-class tau2 spectra (rlnReferenceTau2 × N⁴ for recovar units).
        if args.n_classes > 1:
            _per_class_tau2 = []
            for _k in range(args.n_classes):
                _tab = _it0_model[f"model_class_{_k + 1}"]
                _col = "rlnReferenceTau2" if "rlnReferenceTau2" in _tab.columns else "rlnReferenceSigma2"
                _per_class_tau2.append(np.asarray(_tab[_col], dtype=np.float64) * _n4)
            mean_variance = jnp.stack(
                [
                    jnp.asarray(utils.make_radial_image(_t, ds.volume_shape, extend_last_frequency=True)).reshape(-1)
                    for _t in _per_class_tau2
                ],
                axis=0,
            )
            logger.info(
                "STRICT-PARITY: replaced bootstrapped per-class tau2 with RELION it000 spectra (K=%d)",
                args.n_classes,
            )
        else:
            _tab = _it0_model["model_class_1"]
            _col = "rlnReferenceTau2" if "rlnReferenceTau2" in _tab.columns else "rlnReferenceSigma2"
            _relion_tau2 = np.asarray(_tab[_col], dtype=np.float64) * _n4
            mean_variance = jnp.asarray(
                utils.make_radial_image(_relion_tau2, ds.volume_shape, extend_last_frequency=True)
            ).reshape(-1)
            logger.info("STRICT-PARITY: replaced bootstrapped tau2 with RELION it000 spectrum (K=1)")
        # Tau2 fudge factor: prefer _rlnTau2FudgeFactor from model.star
        # (the actual value RELION used) over _rlnTau2FudgeArg from
        # optimiser.star (the CLI flag, which is -1 when the user did not
        # pass --tau2_fudge and RELION applied the binary default).
        _it0_model_text = _it0_model_path.read_text()
        relion_init_tau2_fudge = _parse_relion_tau2_fudge(_it0_model_text)
        if relion_init_tau2_fudge is not None:
            logger.info(
                "STRICT-PARITY: tau2_fudge from RELION it000 model.star: %.3f",
                relion_init_tau2_fudge,
            )
        if _it0_optim_path.exists():
            _opt_text = _it0_optim_path.read_text()
            if relion_init_tau2_fudge is None:
                relion_init_tau2_fudge = _parse_relion_tau2_fudge(_opt_text)
                if relion_init_tau2_fudge is not None:
                    logger.info(
                        "STRICT-PARITY: --tau2_fudge override from RELION it000 optimiser: %.3f",
                        relion_init_tau2_fudge,
                    )
            _m_so = _re.search(r"_rlnSigmaOffsetsAngst\s+(\S+)", _opt_text)
            if _m_so is not None:
                relion_init_sigma_offset_angstrom = float(_m_so.group(1))
                logger.info(
                    "STRICT-PARITY: --offset_sigma_angstrom override from RELION it000: %.3f Å",
                    relion_init_sigma_offset_angstrom,
                )

    # Compute initial current_size from init_resolution
    init_current_size = max(32, int(2 * ds.voxel_size * ds.grid_size / args.init_resolution))
    logger.info("Initial current_size from resolution %.1f A: %d pixels", args.init_resolution, init_current_size)

    # ---- Run refinement ----
    from recovar.em.dense_single_volume.iteration_loop import refine_single_volume

    experiment_datasets = [ds_half1, ds_half2]
    translations_jnp = jnp.asarray(translations)

    logger.info("=" * 70)
    logger.info(
        "Starting RELION-parity refinement: max_iter=%d, adaptive_oversampling=%d",
        args.max_iter,
        args.adaptive_oversampling,
    )
    logger.info("=" * 70)

    # Parse oracle current_sizes if provided
    oracle_current_sizes = None
    if args.relion_current_sizes is not None:
        oracle_current_sizes = [int(x) for x in args.relion_current_sizes.split(",")]
        logger.info("Oracle mode: using RELION current_sizes=%s", oracle_current_sizes)
    oracle_healpix_orders = None
    if args.relion_healpix_orders is not None:
        oracle_healpix_orders = [int(x) for x in args.relion_healpix_orders.split(",")]
        logger.info("Oracle mode: using RELION healpix_orders=%s", oracle_healpix_orders)

    # Build per-iter replay overrides from RELION's per-iter data.star +
    # model.star when --perturb_replay_relion_dir is set. The override always
    # injects RELION's per-iter sigma_offset (parity-critical: recovar's iter-1
    # does not run the C1 sigma_offset update so iter-2 would otherwise use the
    # 10 Å default — 6× too wide vs RELION ~1.6 Å — depressing iter-2 Pmax by
    # ~22%). Per-image normCorrection / group-scale replay is part of strict
    # RELION replay and can be disabled with --no-replay_relion_normcorr for
    # diagnostics.
    replay_iteration_overrides = None
    if args.perturb_replay_relion_dir is not None:
        replay_normcorr = _resolve_replay_normcorr(
            args.perturb_replay_relion_dir,
            args.replay_relion_normcorr,
        )
        replay_iteration_overrides = _build_replay_iteration_overrides(
            args.perturb_replay_relion_dir,
            half1_idx,
            half2_idx,
            # Numbered expectation k consumes the state written before it, so
            # iterations 1..N use run_it000..run_it{N-1}.  After convergence,
            # RELION's unnumbered all-data expectation consumes the state just
            # written by iteration N and therefore needs run_it{N} as the
            # extra final-only override.
            int(args.max_iter),
            ds_voxel=ds.voxel_size,
            ds_grid=ds.grid_size,
            include_normcorr=replay_normcorr,
            init_relion_iteration=args.init_relion_iteration,
            particle_names=our_names,
            strict=True,
        )

    final_replay_override = None
    final_replay_reference_maps = None
    final_replay_source_iteration = None
    final_sampling_replay_relion_dir = None
    if args.final_replay_relion_dir is not None:
        final_replay_dir = Path(args.final_replay_relion_dir).resolve()
        complete_iterations = _complete_relion_numbered_state_iterations(final_replay_dir)
        source_iteration = _resolve_final_replay_source_iteration(
            configured_max_iter=args.max_iter,
            explicit_source_iteration=args.final_replay_source_iteration,
            complete_iterations=complete_iterations,
        )
        final_replay_source_iteration = source_iteration
        final_optimiser_path = final_replay_dir / "run_optimiser.star"
        final_sampling_path = final_replay_dir / "run_sampling.star"
        if not final_optimiser_path.is_file() or not final_sampling_path.is_file():
            raise ValueError(
                "diagnostic final-only substitution requires unnumbered run_optimiser.star "
                f"and run_sampling.star in {final_replay_dir}"
            )
        from recovar.em.sampling import read_relion_optimiser_metadata

        final_optimiser_metadata = read_relion_optimiser_metadata(final_optimiser_path)
        if not bool(final_optimiser_metadata.get("has_converged", False)):
            raise ValueError(
                f"diagnostic final-only oracle does not report convergence: {final_optimiser_path}"
            )
        final_overrides = _build_replay_iteration_overrides(
            final_replay_dir,
            half1_idx,
            half2_idx,
            source_iteration,
            ds_voxel=ds.voxel_size,
            ds_grid=ds.grid_size,
            include_normcorr=True,
            init_relion_iteration=args.init_relion_iteration,
            particle_names=our_names,
            strict=True,
        )
        source_override = final_overrides[-1]
        if source_override is None:
            raise ValueError("diagnostic final-only substitution did not load a last-numbered override")
        requested_groups, final_replay_override = _select_final_replay_override(
            source_override,
            args.final_replay_fields,
        )
        if "references" in requested_groups:
            if args.n_classes != 1:
                raise ValueError(
                    "diagnostic final-only reference substitution currently requires --n-classes=1"
                )
            final_replay_reference_maps = _load_final_replay_reference_maps(
                final_replay_dir,
                source_iteration,
                ds.volume_shape,
            )
        if "sampling" in requested_groups:
            final_sampling_replay_relion_dir = str(final_replay_dir)
        logger.info(
            "Diagnostic final-only substitution: source_iteration=%d groups=%s fields=%s source=%s",
            source_iteration,
            sorted(requested_groups),
            sorted(final_replay_override),
            final_replay_dir,
        )

    # ``--relion_init_dir`` is the strict cold-start contract, not merely a
    # noise/tau bootstrap. RELION's run_it000 particle/model state includes
    # large pre-centering offsets on real data; omitting them makes iter-1
    # search around zero and changes the hard firstiter-CC winners even though
    # the starting reference and Pmax values appear to match.
    if args.relion_init_dir is not None and _replay_complete_initial_particle_state(
        args.n_classes,
        args.init_relion_iteration,
    ):
        initial_overrides = _build_replay_iteration_overrides(
            args.relion_init_dir,
            half1_idx,
            half2_idx,
            0,
            ds_voxel=ds.voxel_size,
            ds_grid=ds.grid_size,
            include_normcorr=True,
            init_relion_iteration=0,
            particle_names=our_names,
            include_initial_state=True,
            strict=True,
        )
        if initial_overrides[0] is not None:
            if replay_iteration_overrides is None:
                replay_iteration_overrides = [None] * (args.max_iter + 1)
            replay_iteration_overrides[0] = initial_overrides[0]
            logger.info(
                "STRICT-PARITY: loaded complete RELION run_it000 cold-start state "
                "for the first expectation step",
            )
        else:
            logger.warning(
                "STRICT-PARITY: %s did not provide a complete run_it000 data/model "
                "state; first-iteration particle pre-centering remains unset",
                args.relion_init_dir,
            )
    elif args.relion_init_dir is not None and int(args.n_classes) > 1:
        logger.info(
            "STRICT-PARITY: Class3D first iteration uses a fresh global search; "
            "not replaying run_it000 input poses/corrections",
        )

    relion_projector_replay_slot = None
    relion_projector_source_manifest_sha256 = None
    relion_projector_capture_dir_resolved = None
    relion_projector_capture_manifest_resolved = None
    if args.relion_projector_capture_dir is not None:
        if args.perturb_replay_relion_dir is None:
            raise SystemExit(
                "--relion-projector-capture-dir requires --perturb_replay_relion_dir"
            )
        if args.relion_projector_capture_iteration is None:
            raise SystemExit(
                "--relion-projector-capture-dir requires "
                "--relion-projector-capture-iteration"
            )
        capture_dir = Path(args.relion_projector_capture_dir).expanduser().resolve()
        capture_manifest = (
            Path(args.relion_projector_capture_manifest).expanduser().resolve()
            if args.relion_projector_capture_manifest is not None
            else capture_dir
            / f"iter{int(args.relion_projector_capture_iteration)}_VALIDATED_SHA256SUMS"
        )
        try:
            relion_projector_replay_slot, projector_state = _attach_relion_projector_capture(
                replay_iteration_overrides,
                capture_dir=capture_dir,
                manifest_path=capture_manifest,
                capture_iteration=args.relion_projector_capture_iteration,
                init_relion_iteration=args.init_relion_iteration,
                relion_replay_dir=args.perturb_replay_relion_dir,
                volume_shape=ds.volume_shape,
                n_classes=args.n_classes,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid captured RELION projector replay: {exc}") from exc
        relion_projector_source_manifest_sha256 = projector_state[
            "source_manifest_sha256"
        ]
        relion_projector_capture_dir_resolved = capture_dir
        relion_projector_capture_manifest_resolved = capture_manifest
    elif (
        args.relion_projector_capture_manifest is not None
        or args.relion_projector_capture_iteration is not None
    ):
        raise SystemExit(
            "--relion-projector-capture-manifest/iteration require "
            "--relion-projector-capture-dir"
        )

    effective_tau2_fudge, tau2_fudge_source = _resolve_tau2_fudge(
        args.n_classes,
        args.tau2_fudge,
        relion_init_tau2_fudge,
    )
    logger.info("Using tau2_fudge=%.3f (%s)", float(effective_tau2_fudge), tau2_fudge_source)

    t_start = time.time()

    effective_perturb_seed = _effective_perturb_seed(args)
    perturb_replay_restart_state_iterations = tuple(
        sorted(
            {
                int(token.strip())
                for token in args.perturb_replay_restart_state_iterations.split(",")
                if token.strip()
            }
        )
    )
    if any(value < 0 for value in perturb_replay_restart_state_iterations):
        raise SystemExit("--perturb-replay-restart-state-iterations values must be non-negative")
    if perturb_replay_restart_state_iterations and args.perturb_replay_relion_dir is None:
        raise SystemExit(
            "--perturb-replay-restart-state-iterations requires --perturb_replay_relion_dir"
        )
    perturb_replay_restart_provenance_path = None
    perturb_replay_restart_provenance_sha256 = None
    if perturb_replay_restart_state_iterations:
        if args.perturb_replay_restart_provenance is None:
            raise SystemExit(
                "--perturb-replay-restart-state-iterations requires "
                "--perturb-replay-restart-provenance"
            )
        perturb_replay_restart_provenance_path = Path(
            args.perturb_replay_restart_provenance
        ).expanduser().resolve()
        if not perturb_replay_restart_provenance_path.is_file():
            raise SystemExit(
                "--perturb-replay-restart-provenance is not a file: "
                f"{perturb_replay_restart_provenance_path}"
            )
        perturb_replay_restart_provenance_sha256 = _sha256_file(
            perturb_replay_restart_provenance_path
        )
        logger.info(
            "SamplingPerturbation restart provenance: iterations=%s path=%s sha256=%s",
            list(perturb_replay_restart_state_iterations),
            perturb_replay_restart_provenance_path,
            perturb_replay_restart_provenance_sha256,
        )
    elif args.perturb_replay_restart_provenance is not None:
        raise SystemExit(
            "--perturb-replay-restart-provenance requires "
            "--perturb-replay-restart-state-iterations"
        )
    logger.info(
        "SamplingPerturbation seed: %s%s",
        "unseeded" if effective_perturb_seed is None else str(effective_perturb_seed),
        " (explicit)" if args.perturb_seed is not None else " (from --seed)",
    )
    init_previous_best_poses = None
    if args.init_previous_best_poses_npz is not None:
        init_previous_best_poses = _load_init_previous_best_poses_npz(
            args.init_previous_best_poses_npz,
            args.init_previous_best_poses_iter,
        )
        logger.info(
            "Diagnostic local-search seed: loaded previous best poses from %s (iter=%s; half sizes=%s)",
            args.init_previous_best_poses_npz,
            init_previous_best_poses["iteration"],
            [
                int(arr.shape[0])
                for arr in init_previous_best_poses["previous_best_rotation_eulers"]
            ],
        )

    result = refine_single_volume(
        experiment_datasets=experiment_datasets,
        init_volume=jnp.asarray(init_vol_ft),
        init_noise_variance=noise_variance,
        init_mean_variance=mean_variance,
        rotations=rotations,
        translations=translations_jnp,
        disc_type=os.environ.get("RECOVAR_DISC_TYPE_OVERRIDE", "linear_interp"),
        max_iter=args.max_iter,
        image_batch_size=args.image_batch_size,
        rotation_block_size=args.rotation_block_size,
        relion_current_sizes=oracle_current_sizes,
        relion_healpix_orders=oracle_healpix_orders,
        init_current_size=init_current_size,
        fsc_threshold=1.0 / 7.0,
        adaptive_oversampling=args.adaptive_oversampling,
        max_significants=args.max_significants,
        nside_level=rotation_grid_order if args.adaptive_oversampling > 0 else None,
        **_refine_sampling_kwargs(args, init_healpix_order),
        max_healpix_order=effective_max_healpix_order,
        init_translation_sigma_angstrom=(
            relion_init_sigma_offset_angstrom
            if relion_init_sigma_offset_angstrom is not None
            else args.offset_sigma_angstrom
        ),
        particle_diameter_ang=particle_diameter_ang,
        tau2_fudge=effective_tau2_fudge,
        perturb_factor=args.perturb_factor,
        perturb_seed=effective_perturb_seed,
        optimizer_random_seed=args.seed,
        expected_accuracy_half1_base_order_local=expected_accuracy_half1_base_order_local,
        expected_accuracy_half1_optics_group_ids=expected_accuracy_half1_optics_group_ids,
        expected_accuracy_half1_particle_ids=expected_accuracy_half1_particle_ids,
        expected_accuracy_half1_ctf_params=expected_accuracy_half1_ctf_params,
        expected_accuracy_do_ctf_correction=expected_accuracy_do_ctf_correction,
        perturb_replay_relion_dir=args.perturb_replay_relion_dir,
        perturb_replay_restart_state_iterations=perturb_replay_restart_state_iterations,
        final_sampling_replay_relion_dir=final_sampling_replay_relion_dir,
        replay_iteration_overrides=replay_iteration_overrides,
        final_replay_override=final_replay_override,
        final_replay_reference_maps=final_replay_reference_maps,
        final_replay_source_iteration=final_replay_source_iteration,
        init_relion_iteration=args.init_relion_iteration,
        n_classes=args.n_classes,
        image_fourier_backend=args.image_fourier_backend,
        emulate_relion_firstiter_cc=bool(args.firstiter_cc),
        relion_firstiter_ini_high_angstrom=(
            relion_firstiter_ini_high_angstrom if args.firstiter_cc else None
        ),
        init_group_ids=native_group_ids_per_half,
        init_group_count=native_group_count,
        relion_scale_follower_count=relion_scale_followers,
        relion_scale_follower_owners_by_iteration=relion_scale_follower_owners_by_iteration,
        relion_follower_scale_replay=relion_follower_scale_replay,
        init_relion_particle_ids=(
            None if native_group_layout is None else list(native_group_layout.particle_ids_per_half)
        ),
        init_relion_optics_group_ids=(
            None if native_group_layout is None else list(native_group_layout.optics_group_ids_per_half)
        ),
        init_relion_optics_group_count=(
            None if native_group_layout is None else native_group_layout.n_optics_groups
        ),
        init_previous_best_translations=(
            None
            if init_previous_best_poses is None
            else init_previous_best_poses["previous_best_translations"]
        ),
        init_previous_best_rotation_eulers=(
            None
            if init_previous_best_poses is None
            else init_previous_best_poses["previous_best_rotation_eulers"]
        ),
        skip_final_iteration=bool(args.skip_final_iteration),
        save_intermediates_dir=args.save_intermediates_dir,
        save_intermediates_skip_unregularized=bool(args.save_intermediates_skip_unregularized),
        local_search_profile_mode=args.local_search_profile,
        stop_after_local_search_profile=bool(args.stop_after_local_search_profile),
        stop_after_local_search=bool(args.stop_after_local_search),
        stop_after_local_search_score_only=bool(args.stop_after_local_search_score_only),
    )

    total_time = time.time() - t_start
    logger.info("=" * 70)
    logger.info("Refinement complete in %.1fs (%d iterations)", total_time, args.max_iter)
    logger.info("=" * 70)

    if result.get("profile_only"):
        local_profile_rows = _jsonable_profile_rows(result.get("local_profile_history", []))
        global_profile_rows = _jsonable_profile_rows(result.get("global_profile_history", []))
        setup_phase_seconds = {
            str(key): float(value) for key, value in result.get("setup_phase_seconds", {}).items()
        }
        timing_rows = _collect_timing_rows(timing_dir_path)
        timing_summary = _summarize_timing_rows(timing_rows)
        profile_summary = {
            "profile_only": True,
            "stop_after_local_search_score_only": bool(result.get("stop_after_local_search_score_only", False)),
            "git_commit": _safe_git_commit(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "jax_version": getattr(jax, "__version__", None),
            "jaxlib_version": getattr(jaxlib, "__version__", None),
            "jax_devices": [str(device) for device in jax.devices()],
            "data_dir": str(Path(args.data_dir).resolve()),
            "output_dir": str(Path(args.output).resolve()),
            "timing_dir": str(timing_dir_path.resolve()) if timing_dir_path is not None else None,
            "total_time_s": float(total_time),
            "current_sizes": [int(x) for x in result.get("current_sizes", [])],
            "wall_times_trajectory": [float(x) for x in result.get("wall_times", [])],
            "n_images": int(n_images),
            "image_shape": [int(x) for x in ds.image_shape],
            "volume_shape": [int(x) for x in ds.volume_shape],
            "voxel_size": float(ds.voxel_size),
            "healpix_order": int(args.healpix_order),
            "auto_local_healpix_order": int(args.auto_local_healpix_order),
            "adaptive_oversampling": int(args.adaptive_oversampling),
            "max_significants": int(args.max_significants),
            "diagnostic_single_half": bool(args.diagnostic_single_half),
            "setup_phase_seconds": setup_phase_seconds,
            "local_profile_rows": local_profile_rows,
            "global_profile_rows": global_profile_rows,
            "timing_rows": timing_rows,
            "timing_summary": timing_summary,
            "perturb_replay_restart_state_iterations": list(
                perturb_replay_restart_state_iterations
            ),
            "perturb_replay_restart_provenance_path": (
                str(perturb_replay_restart_provenance_path)
                if perturb_replay_restart_provenance_path is not None
                else None
            ),
            "perturb_replay_restart_provenance_sha256": (
                perturb_replay_restart_provenance_sha256
            ),
            "relion_projector_replay_slot": relion_projector_replay_slot,
            "relion_projector_source_manifest_sha256": (
                relion_projector_source_manifest_sha256
            ),
            "relion_projector_capture_dir": (
                None
                if relion_projector_capture_dir_resolved is None
                else str(relion_projector_capture_dir_resolved)
            ),
            "relion_projector_capture_manifest": (
                None
                if relion_projector_capture_manifest_resolved is None
                else str(relion_projector_capture_manifest_resolved)
            ),
        }
        profile_path = Path(args.output) / "local_search_profile_only.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with profile_path.open("w", encoding="utf-8") as f:
            json.dump(profile_summary, f, indent=2, sort_keys=True)
        logger.info("Profile-only summary saved to %s", profile_path)
        if args.benchmark_ledger_json:
            ledger_path = Path(args.benchmark_ledger_json)
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with ledger_path.open("w", encoding="utf-8") as f:
                json.dump(profile_summary, f, indent=2, sort_keys=True)
            logger.info("Benchmark ledger saved to %s", ledger_path)
        print("\n" + "=" * 70)
        print("LOCAL SEARCH PROFILE ONLY")
        print("=" * 70)
        print(f"Profiles: {len(local_profile_rows)}")
        print(f"Total wall time: {total_time:.1f}s")
        if result.get("current_sizes"):
            print(f"Current size: {result['current_sizes'][-1]}")
        print(f"Summary JSON: {profile_path}")
        print("=" * 70)
        return

    # ---- Save results ----
    save_dict = {
        "current_sizes": np.array(result["current_sizes"]),
        "pixel_resolutions": np.array(result["pixel_resolutions"]),
        "wall_times": np.array(result["wall_times"]),
        "total_time": total_time,
        "n_iterations": args.max_iter,
        "healpix_order": args.healpix_order,
        "coarse_healpix_order": init_healpix_order,
        "finest_healpix_order": finest_healpix_order,
        "max_healpix_order": effective_max_healpix_order,
        "max_healpix_order_source": np.asarray(max_healpix_order_source),
        "n_rotations": rotations.shape[0],
        "n_translations": translations.shape[0],
        "n_images": n_images,
        "image_shape": np.array(ds.image_shape),
        "volume_shape": np.array(ds.volume_shape),
        "voxel_size": ds.voxel_size,
        "adaptive_oversampling": args.adaptive_oversampling,
        "max_significants": args.max_significants,
        "offset_sigma_angstrom": args.offset_sigma_angstrom,
        "tau2_fudge": np.float64(effective_tau2_fudge),
        "tau2_fudge_source": np.asarray(tau2_fudge_source),
        "particle_diameter_ang": (np.float64(particle_diameter_ang) if particle_diameter_ang is not None else np.nan),
        "firstiter_cc_effective": np.bool_(bool(args.firstiter_cc)),
        "half1_indices": half1_idx,
        "half2_indices": half2_idx,
        "perturb_replay_restart_state_iterations": np.asarray(
            perturb_replay_restart_state_iterations,
            dtype=np.int64,
        ),
        "perturb_replay_restart_provenance_path": np.asarray(
            ""
            if perturb_replay_restart_provenance_path is None
            else str(perturb_replay_restart_provenance_path)
        ),
        "perturb_replay_restart_provenance_sha256": np.asarray(
            perturb_replay_restart_provenance_sha256 or ""
        ),
        "relion_projector_replay_slot": np.int64(
            -1 if relion_projector_replay_slot is None else relion_projector_replay_slot
        ),
        "relion_projector_source_manifest_sha256": np.asarray(
            relion_projector_source_manifest_sha256 or ""
        ),
        "relion_projector_capture_dir": np.asarray(
            ""
            if relion_projector_capture_dir_resolved is None
            else str(relion_projector_capture_dir_resolved)
        ),
        "relion_projector_capture_manifest": np.asarray(
            ""
            if relion_projector_capture_manifest_resolved is None
            else str(relion_projector_capture_manifest_resolved)
        ),
    }
    if relion_follower_scale_replay is not None:
        save_dict["relion_follower_scale_replay_iterations"] = np.asarray(
            relion_follower_scale_replay.relion_iterations,
            dtype=np.int64,
        )
        save_dict["relion_follower_scale_replay_source"] = np.asarray(
            relion_follower_scale_replay.source
        )
        save_dict["relion_follower_scale_replay_oracle_id"] = np.asarray(
            relion_follower_scale_replay.oracle_id
        )
        save_dict["relion_follower_scale_replay_boundary"] = np.asarray(
            relion_follower_scale_replay.boundary
        )
        save_dict["relion_follower_scale_replay_source_artifacts"] = np.asarray(
            relion_follower_scale_replay.source_artifact_relative_paths
        )
    if relion_dispatch_schedule is not None:
        save_dict["relion_dispatch_oracle_id"] = np.asarray(
            relion_dispatch_schedule.oracle_id
        )
        save_dict["relion_dispatch_oracle_manifest_sha256"] = np.asarray(
            relion_dispatch_schedule.oracle_manifest_sha256
        )
        save_dict["relion_dispatch_particle_order_sha256"] = np.asarray(
            relion_dispatch_schedule.particle_order_sha256
        )

    if "healpix_order_trajectory" in result:
        save_dict["healpix_order_trajectory"] = np.asarray(
            result["healpix_order_trajectory"],
            dtype=np.int32,
        )
    for key, dtype in (
        ("relion_follower_scale_replay_requested_iterations", np.int64),
        ("relion_follower_scale_replay_applied_iterations", np.int64),
        ("relion_scale_follower_scales", np.float64),
        ("relion_scale_rank1_serialized", np.float64),
        ("relion_scale_follower_owners_half1", np.int64),
        ("relion_scale_follower_owners_half1_trajectory", np.int64),
        ("relion_scale_follower_scales_numbered_pre_score_trajectory", np.float64),
        ("relion_scale_follower_scales_numbered_post_mstep_trajectory", np.float64),
    ):
        if result.get(key) is not None:
            save_dict[key] = np.asarray(result[key], dtype=dtype)
    if "ave_Pmax_trajectory" in result:
        save_dict["ave_Pmax_trajectory"] = np.asarray(
            result["ave_Pmax_trajectory"],
            dtype=np.float64,
        )
    for trajectory_key in (
        "frac_changed_trajectory",
        "acc_rot_trajectory",
        "acc_trans_trajectory",
        "smallest_change_angles_trajectory",
        "smallest_change_offsets_trajectory",
    ):
        if trajectory_key in result:
            save_dict[trajectory_key] = np.asarray(result[trajectory_key], dtype=np.float64)
    for trajectory_key, trajectory_dtype in (
        ("acc_rot_per_class_trajectory", np.float64),
        ("acc_trans_per_class_trajectory", np.float64),
        ("expected_accuracy_class_counts_trajectory", np.int64),
    ):
        if trajectory_key in result:
            save_dict[trajectory_key] = np.asarray(result[trajectory_key], dtype=trajectory_dtype)
    if "expected_accuracy_status_trajectory" in result:
        save_dict["expected_accuracy_status_trajectory"] = np.asarray(
            result["expected_accuracy_status_trajectory"],
            dtype=np.str_,
        )
    for indices_key in (
        "expected_accuracy_trial_local_indices",
        "expected_accuracy_trial_particle_ids",
    ):
        if result.get(indices_key) is not None:
            save_dict[indices_key] = np.asarray(result[indices_key], dtype=np.int64)
    for final_accuracy_key, final_accuracy_dtype in (
        ("final_all_data_acc_rot", np.float64),
        ("final_all_data_acc_trans", np.float64),
        ("final_all_data_acc_rot_per_class", np.float64),
        ("final_all_data_acc_trans_per_class", np.float64),
        ("final_all_data_expected_accuracy_class_counts", np.int64),
    ):
        if result.get(final_accuracy_key) is not None:
            save_dict[final_accuracy_key] = np.asarray(
                result[final_accuracy_key],
                dtype=final_accuracy_dtype,
            )
    if result.get("final_all_data_expected_accuracy_status") is not None:
        save_dict["final_all_data_expected_accuracy_status"] = np.asarray(
            result["final_all_data_expected_accuracy_status"],
            dtype=np.str_,
        )
    if "sigma_offset_trajectory" in result:
        save_dict["sigma_offset_trajectory"] = np.asarray(
            result["sigma_offset_trajectory"],
            dtype=np.float64,
        )
    if "sigma_offset_per_half_trajectory" in result:
        save_dict["sigma_offset_per_half_trajectory"] = np.asarray(
            result["sigma_offset_per_half_trajectory"],
            dtype=object,
        )
    if "sigma_offset_used_trajectory" in result:
        save_dict["sigma_offset_used_trajectory"] = np.asarray(
            result["sigma_offset_used_trajectory"],
            dtype=np.float64,
        )
    if "sigma_offset_used_per_half_trajectory" in result:
        save_dict["sigma_offset_used_per_half_trajectory"] = np.asarray(
            result["sigma_offset_used_per_half_trajectory"],
            dtype=object,
        )
    if result.get("direction_prior_trajectory_per_half") is not None:
        save_dict["direction_prior_trajectory_per_half"] = np.asarray(
            result["direction_prior_trajectory_per_half"], dtype=object
        )
    if "convergence_state" in result:
        state = result["convergence_state"]
        save_dict["convergence_iteration"] = np.int32(state.iteration)
        save_dict["convergence_current_resolution"] = np.float64(state.current_resolution)
        save_dict["convergence_ave_Pmax"] = np.float64(state.ave_Pmax)
        save_dict["convergence_healpix_order"] = np.int32(state.healpix_order)
        save_dict["convergence_has_converged"] = np.bool_(state.has_converged)

    # Save K-class metadata when available (n_classes>1).
    if result.get("class_weights") is not None:
        save_dict["class_weights"] = np.asarray(result["class_weights"], dtype=np.float64)
    if result.get("class_weight_trajectory") is not None:
        save_dict["class_weight_trajectory"] = np.asarray(result["class_weight_trajectory"], dtype=np.float64)
    if result.get("class_mstep_weight_trajectory") is not None:
        save_dict["class_mstep_weight_trajectory"] = np.asarray(
            result["class_mstep_weight_trajectory"], dtype=np.float64
        )
    if result.get("class_full_posterior_weight_trajectory") is not None:
        save_dict["class_full_posterior_weight_trajectory"] = np.asarray(
            result["class_full_posterior_weight_trajectory"], dtype=np.float64
        )
    if result.get("class_assignments") is not None and any(c is not None for c in result["class_assignments"]):
        for k, ca in enumerate(result["class_assignments"]):
            if ca is not None:
                save_dict[f"class_assignments_half{k}"] = np.asarray(ca, dtype=np.int32)
    if result.get("class_assignment_history") is not None:
        class_half_order_indices = np.concatenate(
            [np.asarray(half1_idx, dtype=np.int64), np.asarray(half2_idx, dtype=np.int64)],
        )
        for i, classes in enumerate(result["class_assignment_history"]):
            classes_half_order = np.asarray(classes, dtype=np.int32).reshape(-1)
            save_dict[f"class_assignments_iter_{i:03d}"] = classes_half_order
            save_dict[f"class_assignments_half_order_iter_{i:03d}"] = classes_half_order
            if classes_half_order.shape[0] == class_half_order_indices.shape[0]:
                classes_by_image = np.full(int(n_images), -1, dtype=np.int32)
                classes_by_image[class_half_order_indices] = classes_half_order
                save_dict[f"class_assignments_by_image_iter_{i:03d}"] = classes_by_image
    if result.get("per_class_sigma_offset_trajectory") is not None:
        # Per-iter K-vector or None; serialize as object array via dtype=object.
        save_dict["per_class_sigma_offset_trajectory"] = np.asarray(
            result["per_class_sigma_offset_trajectory"], dtype=object
        )
    local_profile_rows = _jsonable_profile_rows(result.get("local_profile_history", []))
    global_profile_rows = _jsonable_profile_rows(result.get("global_profile_history", []))
    setup_phase_seconds = {str(key): float(value) for key, value in result.get("setup_phase_seconds", {}).items()}

    # Save FSC curves per iteration
    for i, fsc in enumerate(result["fsc_history"]):
        save_dict[f"fsc_iter_{i:03d}"] = np.asarray(fsc)

    # Save significant counts per iteration (if available). The refinement
    # loop concatenates half 1 then half 2, which is not generally image order.
    _add_significant_count_artifacts(
        save_dict,
        result["significant_counts"],
        [half1_idx, half2_idx],
        n_images,
    )

    if "data_vs_prior_trajectory" in result:
        for i, dvp in enumerate(result["data_vs_prior_trajectory"]):
            save_dict[f"data_vs_prior_iter_{i:03d}"] = np.asarray(dvp)

    # Per-iter per-shell sigma2_noise and tau2 (added 2026-04 for RELION parity diff)
    if "noise_radial_trajectory" in result:
        for i, nr in enumerate(result["noise_radial_trajectory"]):
            if nr is not None:
                save_dict[f"noise_radial_iter_{i:03d}"] = np.asarray(nr, dtype=np.float64)
    if "noise_radial_per_half_trajectory" in result:
        for i, nr_half in enumerate(result["noise_radial_per_half_trajectory"]):
            if nr_half is not None:
                save_dict[f"noise_radial_per_half_iter_{i:03d}"] = np.asarray(nr_half, dtype=np.float64)
    if "tau2_radial_trajectory" in result:
        for i, t2 in enumerate(result["tau2_radial_trajectory"]):
            if t2 is not None:
                save_dict[f"tau2_radial_iter_{i:03d}"] = np.asarray(t2, dtype=np.float64)
    for result_key, prefix in [
        ("tau2_sigma2_trajectory", "tau2_sigma2_iter"),
        ("tau2_avg_weight_trajectory", "tau2_avg_weight_iter"),
        ("tau2_shell_sum_trajectory", "tau2_shell_sum_iter"),
        ("tau2_shell_count_trajectory", "tau2_shell_count_iter"),
        ("tau2_fsc_used_trajectory", "tau2_fsc_used_iter"),
        ("tau2_ssnr_trajectory", "tau2_ssnr_iter"),
    ]:
        if result_key in result:
            for i, arr in enumerate(result[result_key]):
                if arr is not None:
                    save_dict[f"{prefix}_{i:03d}"] = np.asarray(arr, dtype=np.float64)

    # Save per-image Pmax per iteration (if available)
    if "pmax_per_image_history" in result:
        pmax_half_order_indices = np.concatenate(
            [np.asarray(half1_idx, dtype=np.int64), np.asarray(half2_idx, dtype=np.int64)],
        )
        for i, pmax in enumerate(result["pmax_per_image_history"]):
            pmax_half_order = np.asarray(pmax, dtype=np.float32).reshape(-1)
            save_dict[f"pmax_per_image_iter_{i:03d}"] = pmax_half_order
            save_dict[f"pmax_per_half_order_iter_{i:03d}"] = pmax_half_order
            if pmax_half_order.shape[0] == pmax_half_order_indices.shape[0]:
                pmax_by_image = np.full(int(n_images), np.nan, dtype=np.float32)
                pmax_by_image[pmax_half_order_indices] = pmax_half_order
                save_dict[f"pmax_per_image_by_image_iter_{i:03d}"] = pmax_by_image
    if result.get("final_all_data_fsc") is not None:
        save_dict["fsc_final_all_data"] = np.asarray(result["final_all_data_fsc"], dtype=np.float32)
    if "final_all_data_ran" in result:
        save_dict["final_all_data_ran"] = np.asarray(result["final_all_data_ran"], dtype=np.bool_)
    for result_key, save_key in (
        ("tau2_radial_final_all_data", "tau2_radial_final_all_data"),
        ("tau2_fsc_used_final_all_data", "tau2_fsc_used_final_all_data"),
        ("tau2_ssnr_final_all_data", "tau2_ssnr_final_all_data"),
    ):
        if result.get(result_key) is not None:
            save_dict[save_key] = np.asarray(result[result_key], dtype=np.float64)
    if "final_all_data_sampling_perturbation" in result:
        save_dict["final_all_data_sampling_perturbation"] = np.asarray(
            result["final_all_data_sampling_perturbation"],
            dtype=np.float32,
        )
    if "final_all_data_sampling_perturbation_applied" in result:
        save_dict["final_all_data_sampling_perturbation_applied"] = np.asarray(
            result["final_all_data_sampling_perturbation_applied"],
            dtype=np.bool_,
        )
    if "final_all_data_sampling_relion_iteration" in result:
        save_dict["final_all_data_sampling_relion_iteration"] = np.asarray(
            result["final_all_data_sampling_relion_iteration"],
            dtype=np.int32,
        )
    if result.get("final_all_data_sampling_star") is not None:
        save_dict["final_all_data_sampling_star"] = np.asarray(str(result["final_all_data_sampling_star"]))
    if result.get("final_all_data_sampling_star_source") is not None:
        save_dict["final_all_data_sampling_star_source"] = np.asarray(
            str(result["final_all_data_sampling_star_source"])
        )
    if "final_all_data_sampling_offset_range" in result:
        save_dict["final_all_data_sampling_offset_range"] = np.asarray(
            result["final_all_data_sampling_offset_range"],
            dtype=np.float32,
        )
    if "final_all_data_sampling_offset_step" in result:
        save_dict["final_all_data_sampling_offset_step"] = np.asarray(
            result["final_all_data_sampling_offset_step"],
            dtype=np.float32,
        )
    if "final_all_data_grid_correct" in result:
        save_dict["final_all_data_grid_correct"] = np.asarray(
            result["final_all_data_grid_correct"],
            dtype=np.bool_,
        )
    if result.get("final_all_data_gridding_correct") is not None:
        save_dict["final_all_data_gridding_correct"] = np.asarray(
            str(result["final_all_data_gridding_correct"])
        )
    if result.get("tau2_weight_combination_final_all_data") is not None:
        save_dict["tau2_weight_combination_final_all_data"] = np.asarray(
            str(result["tau2_weight_combination_final_all_data"])
        )

    half_indices = [
        np.asarray(half1_idx, dtype=np.int64),
        np.asarray(half2_idx, dtype=np.int64),
    ]
    for i, iter_eulers in enumerate(result.get("best_rotation_eulers_history", [])):
        half_arrays = _pose_history_half_arrays(iter_eulers, dtype=np.float32)
        if half_arrays is None or all(arr is None for arr in half_arrays):
            continue
        compact = []
        for k, arr in enumerate(half_arrays):
            if arr is None:
                continue
            save_dict[f"best_rotation_eulers_iter_{i:03d}_half{k}"] = arr
            compact.append(arr)
        if compact:
            save_dict[f"best_rotation_eulers_iter_{i:03d}"] = np.concatenate(compact, axis=0)
        by_image = _pose_history_by_image(iter_eulers, half_indices, n_images, (3,), dtype=np.float32)
        if by_image is not None:
            save_dict[f"best_rotation_eulers_by_image_iter_{i:03d}"] = by_image
            save_dict["best_rotation_eulers_final_by_image"] = by_image

    for i, iter_trans in enumerate(result.get("best_translations_history", [])):
        half_arrays = _pose_history_half_arrays(iter_trans, dtype=np.float32)
        if half_arrays is None or all(arr is None for arr in half_arrays):
            continue
        compact = []
        for k, arr in enumerate(half_arrays):
            if arr is None:
                continue
            save_dict[f"best_translations_iter_{i:03d}_half{k}"] = arr
            compact.append(arr)
        if compact:
            save_dict[f"best_translations_iter_{i:03d}"] = np.concatenate(compact, axis=0)
        by_image = _pose_history_by_image(iter_trans, half_indices, n_images, (2,), dtype=np.float32)
        if by_image is not None:
            save_dict[f"best_translations_by_image_iter_{i:03d}"] = by_image
            save_dict["best_translations_final_by_image"] = by_image

    final_all_data_eulers = result.get("final_all_data_best_rotation_eulers")
    final_all_data_euler_halves = _pose_history_half_arrays(final_all_data_eulers, dtype=np.float32)
    if final_all_data_euler_halves is not None and not all(arr is None for arr in final_all_data_euler_halves):
        compact = []
        for k, arr in enumerate(final_all_data_euler_halves):
            if arr is None:
                continue
            save_dict[f"best_rotation_eulers_final_all_data_half{k}"] = arr
            compact.append(arr)
        if compact:
            save_dict["best_rotation_eulers_final_all_data"] = np.concatenate(compact, axis=0)
        by_image = _pose_history_by_image(final_all_data_eulers, half_indices, n_images, (3,), dtype=np.float32)
        if by_image is not None:
            save_dict["best_rotation_eulers_final_all_data_by_image"] = by_image

    final_all_data_trans = result.get("final_all_data_best_translations")
    final_all_data_trans_halves = _pose_history_half_arrays(final_all_data_trans, dtype=np.float32)
    if final_all_data_trans_halves is not None and not all(arr is None for arr in final_all_data_trans_halves):
        compact = []
        for k, arr in enumerate(final_all_data_trans_halves):
            if arr is None:
                continue
            save_dict[f"best_translations_final_all_data_half{k}"] = arr
            compact.append(arr)
        if compact:
            save_dict["best_translations_final_all_data"] = np.concatenate(compact, axis=0)
        by_image = _pose_history_by_image(final_all_data_trans, half_indices, n_images, (2,), dtype=np.float32)
        if by_image is not None:
            save_dict["best_translations_final_all_data_by_image"] = by_image

    final_all_data_pmax = result.get("final_all_data_max_posterior")
    final_all_data_pmax_halves = _pose_history_half_arrays(final_all_data_pmax, dtype=np.float32)
    if final_all_data_pmax_halves is not None and not all(arr is None for arr in final_all_data_pmax_halves):
        compact = []
        for k, arr in enumerate(final_all_data_pmax_halves):
            if arr is None:
                continue
            save_dict[f"pmax_final_all_data_half{k}"] = arr
            compact.append(arr)
        if compact:
            save_dict["pmax_final_all_data"] = np.concatenate(compact, axis=0)
        by_image = _pose_history_by_image(final_all_data_pmax, half_indices, n_images, (), dtype=np.float32)
        if by_image is not None:
            save_dict["pmax_final_all_data_by_image"] = by_image

    git_provenance = git_worktree_provenance()
    save_dict["git_commit"] = np.asarray(git_provenance["head"])
    save_dict["git_branch"] = np.asarray(git_provenance["branch"])
    save_dict["git_dirty_count"] = np.asarray(git_provenance["dirty_count"], dtype=np.int64)
    save_dict["git_diff_sha256"] = np.asarray(git_provenance["diff_sha256"])
    save_dict["git_worktree_fingerprint_sha256"] = np.asarray(git_provenance["worktree_fingerprint_sha256"])
    save_dict["git_status_porcelain"] = np.asarray(git_provenance["status_porcelain"])
    save_dict["git_untracked_file_hashes"] = np.asarray(git_provenance["untracked_file_hashes"])

    # Save final merged volume (Fourier space)
    save_dict["final_mean_ft"] = np.asarray(result["mean"])
    if setup_phase_seconds:
        save_dict["setup_phase_names"] = np.asarray(list(setup_phase_seconds.keys()))
        save_dict["setup_phase_cumulative_s"] = np.asarray(list(setup_phase_seconds.values()), dtype=np.float64)

    # Save per-half-set means
    for k in range(2):
        save_dict[f"half{k}_mean_ft"] = np.asarray(result["means"][k])

    # Save hard assignments
    for k in range(2):
        if result["hard_assignments"][k] is not None:
            save_dict[f"hard_assignments_half{k}"] = np.asarray(result["hard_assignments"][k])

    out_path = os.path.join(args.output, "refinement_results.npz")
    if args.skip_large_outputs:
        logger.info("Skipping large refinement result archive (--skip-large-outputs): %s", out_path)
    else:
        np.savez_compressed(out_path, **save_dict)
        logger.info("Results saved to %s", out_path)

    timing_rows = _collect_timing_rows(timing_dir_path)
    timing_summary = _summarize_timing_rows(timing_rows)
    if args.benchmark_ledger_json:
        ledger_path = Path(args.benchmark_ledger_json)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = {
            "git_commit": _safe_git_commit(),
            "git_provenance": git_provenance,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "jax_version": getattr(jax, "__version__", None),
            "jaxlib_version": getattr(jaxlib, "__version__", None),
            "jax_devices": [str(device) for device in jax.devices()],
            "data_dir": str(Path(args.data_dir).resolve()),
            "output_dir": str(Path(args.output).resolve()),
            "timing_dir": str(timing_dir_path.resolve()) if timing_dir_path is not None else None,
            "max_iter": int(args.max_iter),
            "n_iterations_emitted": int(len(result.get("current_sizes", []))),
            "n_wall_times": int(len(result.get("wall_times", []))),
            "total_time_s": float(total_time),
            "wall_times_trajectory": [float(x) for x in result.get("wall_times", [])],
            "current_sizes": [int(x) for x in result.get("current_sizes", [])],
            "pixel_resolutions": [float(x) for x in result.get("pixel_resolutions", [])],
            "ave_Pmax_trajectory": [float(x) for x in result.get("ave_Pmax_trajectory", [])],
            "n_images": int(n_images),
            "image_shape": [int(x) for x in ds.image_shape],
            "volume_shape": [int(x) for x in ds.volume_shape],
            "voxel_size": float(ds.voxel_size),
            "n_rotations": int(rotations.shape[0]),
            "n_translations": int(translations.shape[0]),
            "healpix_order": int(args.healpix_order),
            "coarse_healpix_order": int(init_healpix_order),
            "finest_healpix_order": int(finest_healpix_order),
            "max_healpix_order": int(effective_max_healpix_order),
            "max_healpix_order_source": str(max_healpix_order_source),
            "auto_local_healpix_order": int(args.auto_local_healpix_order),
            "adaptive_oversampling": int(args.adaptive_oversampling),
            "max_significants": int(args.max_significants),
            "setup_phase_seconds": setup_phase_seconds,
            "local_profile_rows": local_profile_rows,
            "global_profile_rows": global_profile_rows,
            "timing_rows": timing_rows,
            "timing_summary": timing_summary,
            "perturb_replay_restart_state_iterations": list(
                perturb_replay_restart_state_iterations
            ),
            "perturb_replay_restart_provenance_path": (
                str(perturb_replay_restart_provenance_path)
                if perturb_replay_restart_provenance_path is not None
                else None
            ),
            "perturb_replay_restart_provenance_sha256": (
                perturb_replay_restart_provenance_sha256
            ),
            "relion_projector_replay_slot": relion_projector_replay_slot,
            "relion_projector_source_manifest_sha256": (
                relion_projector_source_manifest_sha256
            ),
            "relion_projector_capture_dir": (
                None
                if relion_projector_capture_dir_resolved is None
                else str(relion_projector_capture_dir_resolved)
            ),
            "relion_projector_capture_manifest": (
                None
                if relion_projector_capture_manifest_resolved is None
                else str(relion_projector_capture_manifest_resolved)
            ),
        }
        with ledger_path.open("w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2, sort_keys=True)
        logger.info("Benchmark ledger saved to %s", ledger_path)

    if args.skip_large_outputs:
        logger.info("Skipping final MRC volume writes (--skip-large-outputs)")
    else:
        # Also save final merged volume as MRC for visual inspection.
        # Use the canonical idiom: get_idft3 + write_mrc (handles axis transpose).
        from recovar.utils.helpers import write_mrc as _write_mrc

        def _ft_to_real_volume(ft_array):
            ft_reshape = np.asarray(ft_array).reshape(ds.volume_shape)
            return np.real(np.array(ftu.get_idft3(jnp.asarray(ft_reshape)))).astype(np.float32)

        if args.n_classes == 1:
            final_mean_real = _ft_to_real_volume(result["mean"])
            _write_mrc(os.path.join(args.output, "final_merged.mrc"), final_mean_real, voxel_size=ds.voxel_size)
            logger.info("Final merged volume saved to final_merged.mrc")
            for k in range(2):
                half_real = _ft_to_real_volume(result["means"][k])
                _write_mrc(
                    os.path.join(args.output, f"final_half{k + 1}.mrc"),
                    half_real,
                    voxel_size=ds.voxel_size,
                )
                logger.info("Half-%d volume saved", k + 1)
            unfiltered_means = result.get("unfiltered_means")
            if unfiltered_means is not None:
                for k in range(2):
                    unfiltered_real = _ft_to_real_volume(unfiltered_means[k])
                    _write_mrc(
                        os.path.join(args.output, f"final_half{k + 1}_unfil.mrc"),
                        unfiltered_real,
                        voxel_size=ds.voxel_size,
                    )
                    logger.info("Unfiltered half-%d volume saved", k + 1)
        else:
            # K-class: result["means"][k] has shape (K, V); result["class_means"]
            # has shape (K, V) for the merged final iter; result["mean"] is the
            # class-weighted merged volume.
            final_mean_real = _ft_to_real_volume(result["mean"])
            _write_mrc(os.path.join(args.output, "final_merged.mrc"), final_mean_real, voxel_size=ds.voxel_size)
            if result.get("class_means") is not None:
                class_means_arr = np.asarray(result["class_means"])
                for c in range(args.n_classes):
                    vol_real = _ft_to_real_volume(class_means_arr[c])
                    _write_mrc(
                        os.path.join(args.output, f"final_class{c + 1:03d}.mrc"),
                        vol_real,
                        voxel_size=ds.voxel_size,
                    )
                logger.info("Saved %d per-class merged final volumes", args.n_classes)

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("REFINEMENT SUMMARY")
    print("=" * 70)
    print(f"{'Iter':>4s}  {'CurSize':>8s}  {'PixRes':>8s}  {'ResA':>8s}  {'Time(s)':>8s}", end="")
    if any(c is not None for c in result["significant_counts"]):
        print(f"  {'MedSig':>8s}", end="")
    print()
    print("-" * 70)

    for i in range(len(result["current_sizes"])):
        cs = result["current_sizes"][i]
        pr = result["pixel_resolutions"][i]
        res_a = _shell_index_to_resolution_angstrom(pr, ds.image_shape[0], ds.voxel_size)
        wt = result["wall_times"][i]
        line = f"{i + 1:4d}  {cs:8d}  {pr:8.1f}  {res_a:8.2f}  {wt:8.1f}"
        if result["significant_counts"][i] is not None:
            med_sig = int(np.median(np.asarray(result["significant_counts"][i])))
            line += f"  {med_sig:8d}"
        print(line)

    print("-" * 70)
    print(f"Total wall time: {total_time:.1f}s")
    print(f"Final current_size: {result['current_sizes'][-1]}")
    print(f"Final pixel resolution: {result['pixel_resolutions'][-1]:.1f}")
    print(
        "Final resolution: "
        f"{_shell_index_to_resolution_angstrom(result['pixel_resolutions'][-1], ds.image_shape[0], ds.voxel_size):.2f} A"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
