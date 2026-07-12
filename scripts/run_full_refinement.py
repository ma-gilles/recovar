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
import json
import logging
import os
import platform
import re
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.utils.parity_provenance import _safe_git_commit, git_worktree_provenance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def _shell_index_to_resolution_angstrom(shell_index, grid_size, voxel_size):
    if voxel_size <= 0:
        return float(shell_index)
    shell_index = float(shell_index)
    if shell_index <= 0:
        return float("inf")
    return float(grid_size) * float(voxel_size) / shell_index


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


def _load_native_group_ids_per_half(particles_star, half1_idx, half2_idx):
    """Return 0-based RELION group IDs per half when particles.star provides them."""

    import starfile as _starfile

    data = _starfile.read(str(particles_star))
    particles = data["particles"] if isinstance(data, dict) else data
    if "rlnGroupNumber" not in particles.columns:
        return None
    group_numbers = np.asarray(particles["rlnGroupNumber"], dtype=np.int64).reshape(-1)
    if group_numbers.size != len(particles):
        raise ValueError(
            f"rlnGroupNumber length {group_numbers.size} does not match particles table length {len(particles)}"
        )
    if group_numbers.size and int(np.min(group_numbers)) < 1:
        raise ValueError("RELION rlnGroupNumber values must be 1-based positive integers")
    group_ids = group_numbers - 1
    return [
        np.asarray(group_ids[np.asarray(half1_idx, dtype=np.int64)], dtype=np.int64),
        np.asarray(group_ids[np.asarray(half2_idx, dtype=np.int64)], dtype=np.int64),
    ]


def _default_refinement_subsets(n_images, seed, n_classes):
    """Return default dataset splits for RELION-style refinement."""

    indices = np.arange(int(n_images), dtype=np.int64)
    if int(n_classes) > 1:
        return indices, np.empty(0, dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    return np.sort(indices[: int(n_images) // 2]), np.sort(indices[int(n_images) // 2 :])


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
):
    """Build per-iter replay overrides keyed on recovar iteration index.

    For each recovar iteration k >= 1 (i.e. iter 2 onwards in RELION terms),
    reads RELION's run_it{k:03d}_data.star + half1/half2 model.star
    (or the shared Class3D run_it{k:03d}_model.star) and builds an
    override dict containing:
      * image_corrections: per-image (avg_norm/normcorr) * group_scale
      * scale_corrections: per-image group_scale alone
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

    def _idx(name):
        m = _re.match(r"(\d+)@", str(name))
        return int(m.group(1)) - 1 if m else -1

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
        optics = model.get("model_optics_group_1") if isinstance(model, dict) else None
        if optics is None or "rlnSigma2Noise" not in optics:
            return None
        radial = np.asarray(optics["rlnSigma2Noise"], dtype=np.float64) * float(ds_grid) ** 4
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
            logger.warning(
                "Replay override for recovar iter %d (RELION iter %03d): missing %s — leaving unset",
                recovar_iter + 1,
                relion_iter,
                "; ".join(missing),
            )
            continue

        data = _sf.read(str(data_star))
        parts = data["particles"] if isinstance(data, dict) else data
        m1 = _sf.read(str(model_paths[0]))
        m2 = _sf.read(str(model_paths[1]))

        names = list(parts["rlnImageName"])
        idx_to_pos = {_idx(names[i]): i for i in range(len(names))}

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
        # whereas idx_to_pos is keyed by the original stack index encoded in
        # rlnImageName.  These coincide for synthetic contiguous fixtures but
        # not for real-data subsets that retain their original stack indices.
        if particle_names is None:
            particle_stack_indices = None
        else:
            particle_stack_indices = np.asarray([_idx(name) for name in particle_names], dtype=np.int64)
            if np.any(particle_stack_indices < 0):
                raise ValueError("RECOVAR input contains rlnImageName values without a '<index>@' prefix")
            if np.unique(particle_stack_indices).size != particle_stack_indices.size:
                raise ValueError("RECOVAR input contains duplicate particle stack indices")

        def _to_half(values, half_idx):
            rows = np.asarray(half_idx, dtype=np.int64)
            stack_indices = rows if particle_stack_indices is None else particle_stack_indices[rows]
            missing = sorted({int(i) for i in stack_indices if int(i) not in idx_to_pos})
            if missing:
                preview = ", ".join(str(i + 1) for i in missing[:8])
                raise ValueError(
                    f"RELION replay STAR is missing {len(missing)} RECOVAR particle stack indices "
                    f"(1-based preview: {preview})"
                )
            return np.asarray([values[idx_to_pos[int(i)]] for i in stack_indices], dtype=np.float32)

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
            override_k["noise_variance"] = [noise_h1, noise_h2]
        if direction_prior_h1 is not None and direction_prior_h2 is not None:
            override_k["direction_prior"] = [direction_prior_h1, direction_prior_h2]
        if class_tau2 is not None:
            override_k["class_tau2"] = class_tau2
        if include_normcorr:
            override_k["image_corrections"] = [corr_h1, corr_h2]
            override_k["scale_corrections"] = [scale_corr_h1, scale_corr_h2]
        overrides[recovar_iter] = override_k
        if include_normcorr:
            logger.info(
                "Replay override recovar iter %d: image_corr means=(%s, %s), scale_corr means=(%s, %s), "
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
        help="If set, read SamplingPerturbInstance per iteration from RELION's "
        "run_it{NNN}_sampling.star in this directory and use that exact value "
        "instead of recovar's RNG. Required for bit-exact ab-initio replay "
        "against a RELION reference run.",
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
    parser.add_argument("--image_batch_size", type=int, default=500, help="Images per GPU batch")
    parser.add_argument(
        "--rotation_block_size",
        type=int,
        default=40000,
        help="Rotations per block (larger = faster, less Python overhead)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for half-set split")
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
    our_names = list(our_particles["rlnImageName"])

    if args.relion_half_sets is not None:
        # Use RELION's half-set split from rlnRandomSubset
        logger.info("Loading RELION half-set assignments from %s", args.relion_half_sets)
        import re

        relion_data = _starfile.read(args.relion_half_sets)
        relion_particles = relion_data["particles"]
        relion_subsets = np.array(relion_particles["rlnRandomSubset"])
        relion_names = list(relion_particles["rlnImageName"])

        # Build mapping: particle stack index -> subset
        def _image_name_to_stack_idx(name):
            m = re.match(r"(\d+)@", name)
            return int(m.group(1)) if m else -1

        relion_idx_to_subset = {}
        for i in range(len(relion_names)):
            stack_idx = _image_name_to_stack_idx(relion_names[i])
            relion_idx_to_subset[stack_idx] = relion_subsets[i]

        # Our dataset loads in stack order 1,2,3,...
        # Map to RELION's subset assignments
        our_subsets = np.array([relion_idx_to_subset[_image_name_to_stack_idx(name)] for name in our_names])

        half1_idx = np.where(our_subsets == 1)[0]
        half2_idx = np.where(our_subsets == 2)[0]
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
    native_group_ids_per_half = _load_native_group_ids_per_half(
        os.path.join(args.data_dir, "particles.star"),
        half1_idx,
        half2_idx,
    )
    if native_group_ids_per_half is not None:
        logger.info(
            "Native RELION group IDs from particles.star: half1 groups=%s half2 groups=%s",
            np.unique(native_group_ids_per_half[0]).tolist(),
            np.unique(native_group_ids_per_half[1]).tolist(),
        )

    optimiser_star = _find_relion_optimiser_star(args)
    relion_firstiter_ini_high_angstrom = None
    if optimiser_star is not None:
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
            np.asarray(_model["model_optics_group_1"]["rlnSigma2Noise"], dtype=np.float64)
            for _model in _it0_models
        ]
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
            args.max_iter,
            ds_voxel=ds.voxel_size,
            ds_grid=ds.grid_size,
            include_normcorr=replay_normcorr,
            init_relion_iteration=args.init_relion_iteration,
            particle_names=our_names,
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

    effective_tau2_fudge, tau2_fudge_source = _resolve_tau2_fudge(
        args.n_classes,
        args.tau2_fudge,
        relion_init_tau2_fudge,
    )
    logger.info("Using tau2_fudge=%.3f (%s)", float(effective_tau2_fudge), tau2_fudge_source)

    t_start = time.time()

    effective_perturb_seed = _effective_perturb_seed(args)
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
        perturb_replay_relion_dir=args.perturb_replay_relion_dir,
        replay_iteration_overrides=replay_iteration_overrides,
        init_relion_iteration=args.init_relion_iteration,
        n_classes=args.n_classes,
        emulate_relion_firstiter_cc=bool(args.firstiter_cc),
        relion_firstiter_ini_high_angstrom=(
            relion_firstiter_ini_high_angstrom if args.firstiter_cc else None
        ),
        init_group_ids=native_group_ids_per_half,
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
        "half1_indices": half1_idx,
        "half2_indices": half2_idx,
    }

    if "healpix_order_trajectory" in result:
        save_dict["healpix_order_trajectory"] = np.asarray(
            result["healpix_order_trajectory"],
            dtype=np.int32,
        )
    if "ave_Pmax_trajectory" in result:
        save_dict["ave_Pmax_trajectory"] = np.asarray(
            result["ave_Pmax_trajectory"],
            dtype=np.float64,
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

    # Save significant counts per iteration (if available)
    for i, counts in enumerate(result["significant_counts"]):
        if counts is not None:
            save_dict[f"sig_counts_iter_{i:03d}"] = np.asarray(counts)

    if "data_vs_prior_trajectory" in result:
        for i, dvp in enumerate(result["data_vs_prior_trajectory"]):
            save_dict[f"data_vs_prior_iter_{i:03d}"] = np.asarray(dvp)

    # Per-iter per-shell sigma2_noise and tau2 (added 2026-04 for RELION parity diff)
    if "noise_radial_trajectory" in result:
        for i, nr in enumerate(result["noise_radial_trajectory"]):
            if nr is not None:
                save_dict[f"noise_radial_iter_{i:03d}"] = np.asarray(nr, dtype=np.float64)
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
