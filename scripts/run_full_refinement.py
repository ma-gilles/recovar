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

from recovar.core import fourier_transform_utils as ftu
from recovar.utils.parity_provenance import _safe_git_commit

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


def _parse_relion_tau2_fudge(text):
    """Extract RELION's tau2_fudge from an optimiser STAR text block."""
    match = re.search(r"_(?:rlnTau2FudgeFactor|rlnTau2FudgeArg)\s+(\S+)", text)
    if match is None:
        match = re.search(r"(?:rlnTau2FudgeFactor|rlnTau2FudgeArg)\s+(\S+)", text)
    if match is None:
        return None
    return float(match.group(1))


def _resolve_tau2_fudge(n_classes, cli_tau2_fudge, relion_init_tau2_fudge):
    """Return the effective tau2_fudge and a human-readable source label."""
    if relion_init_tau2_fudge is not None:
        return float(relion_init_tau2_fudge), "RELION it000 optimiser"
    if cli_tau2_fudge is not None:
        return float(cli_tau2_fudge), "explicit CLI"
    if int(n_classes) > 1:
        return 4.0, "RELION Class3D default"
    return 1.0, "RELION auto-refine default"


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
):
    """Build per-iter replay overrides keyed on recovar iteration index.

    For each recovar iteration k >= 1 (i.e. iter 2 onwards in RELION terms),
    reads RELION's run_it{k:03d}_data.star + half1/half2 model.star and
    builds an override dict containing:
      * image_corrections: per-image (avg_norm/normcorr) * group_scale
      * scale_corrections: per-image group_scale alone

    This matches scripts/run_multi_iter_parity.py::_load_relion_iteration_override
    (the proven replay logic). The recovar iter-k override is read from
    RELION iter-k's model+data (since recovar iter-k corresponds to RELION
    iter-(k+1), and the per-image scalings used at the start of RELION
    iter-(k+1) are the ones written by RELION iter-k's M-step).
    """
    import re as _re
    from pathlib import Path as _Path

    import starfile as _sf

    relion_dir = _Path(relion_dir).resolve()

    def _idx(name):
        m = _re.match(r"(\d+)@", str(name))
        return int(m.group(1)) - 1 if m else -1

    overrides = [None] * max_iter
    for recovar_iter in range(1, max_iter):
        # recovar iter k uses corrections computed by RELION iter k (which were
        # written into run_it{k}_data.star). recovar iter 1 (the first iter) has
        # no upstream RELION normcorr — leave that override as None so the
        # E-step uses image_corrections=None (=1.0 for all particles, matching
        # RELION iter-0 nc=1.0).
        relion_iter = recovar_iter  # recovar iter k <-> RELION iter k for normcorr source
        data_star = relion_dir / f"run_it{relion_iter:03d}_data.star"
        model_h1 = relion_dir / f"run_it{relion_iter:03d}_half1_model.star"
        model_h2 = relion_dir / f"run_it{relion_iter:03d}_half2_model.star"
        if not data_star.exists() or not model_h1.exists() or not model_h2.exists():
            logger.warning(
                "Replay override for recovar iter %d: missing %s — leaving unset", recovar_iter + 1, data_star
            )
            continue

        data = _sf.read(str(data_star))
        parts = data["particles"] if isinstance(data, dict) else data
        m1 = _sf.read(str(model_h1))
        m2 = _sf.read(str(model_h2))

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
        sigma_offset_avg = 0.5 * (float(sigma_offset_h1) + float(sigma_offset_h2))

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
        # recovar's half1_idx / half2_idx are stack-row indices into the same
        # particles.star as RELION's data.star.
        def _to_half(combined, half_idx):
            return np.asarray([combined[idx_to_pos[int(i)]] for i in half_idx], dtype=np.float32)

        corr_h1 = _to_half(combined_h1, half1_idx)
        corr_h2 = _to_half(combined_h2, half2_idx)
        scale_corr_h1 = _to_half(pp_scale_h1, half1_idx)
        scale_corr_h2 = _to_half(pp_scale_h2, half2_idx)

        override_k = {"translation_sigma_angstrom": sigma_offset_avg}
        if include_normcorr:
            override_k["image_corrections"] = [corr_h1, corr_h2]
            override_k["scale_corrections"] = [scale_corr_h1, scale_corr_h2]
        overrides[recovar_iter] = override_k
        if include_normcorr:
            logger.info(
                "Replay override recovar iter %d: image_corr means=(%.4f, %.4f), scale_corr means=(%.4f, %.4f), sigma_offset=%.4f Å",
                recovar_iter + 1,
                float(corr_h1.mean()),
                float(corr_h2.mean()),
                float(scale_corr_h1.mean()),
                float(scale_corr_h2.mean()),
                sigma_offset_avg,
            )
        else:
            logger.info(
                "Replay override recovar iter %d: sigma_offset=%.4f Å (normcorr replay disabled)",
                recovar_iter + 1,
                sigma_offset_avg,
            )

    return overrides


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
    parser.add_argument("--max_iter", type=int, default=10, help="Maximum EM iterations")
    parser.add_argument(
        "--healpix_order",
        type=int,
        default=3,
        help="RELION coarse pass-1 HEALPix order. With adaptive oversampling, "
        "pass 2 evaluates healpix_order + adaptive_oversampling.",
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
        "--replay_relion_normcorr",
        action="store_true",
        default=False,
        help="If set together with --perturb_replay_relion_dir, also inject "
        "RELION's per-iter rlnNormCorrection / rlnGroupScaleCorrection into "
        "recovar's E-step at iter 2+. Useful for parity diagnostics; not yet "
        "demonstrated to improve quality on the 5k 128² fixture.",
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
        help="Strict-parity cold-start: load RELION run_it000_model.star "
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
        "--timing_dir",
        default=None,
        help=(
            "Optional directory for lightweight per-iteration timing NPZs. "
            "This uses RECOVAR_PARITY_TIMING_DIR internally and does not "
            "write full parity tensor/volume dumps."
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
    if not any(d.platform == "gpu" for d in devices):
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
    particle_diameter_ang = None if relion_mask_params is None else float(relion_mask_params[0])
    logger.info("Dataset: %d images, image_shape=%s, voxel_size=%.3f A/px", ds.n_units, ds.image_shape, ds.voxel_size)

    # ---- Create half-sets ----
    n_images = ds.n_units

    if args.relion_half_sets is not None:
        # Use RELION's half-set split from rlnRandomSubset
        logger.info("Loading RELION half-set assignments from %s", args.relion_half_sets)
        import re

        import starfile as _starfile

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
        our_star = _starfile.read(os.path.join(args.data_dir, "particles.star"))
        our_particles = our_star["particles"] if isinstance(our_star, dict) else our_star
        our_names = list(our_particles["rlnImageName"])
        our_subsets = np.array([relion_idx_to_subset[_image_name_to_stack_idx(name)] for name in our_names])

        half1_idx = np.where(our_subsets == 1)[0]
        half2_idx = np.where(our_subsets == 2)[0]
        logger.info("Using RELION half-set split: %d (subset=1) + %d (subset=2)", len(half1_idx), len(half2_idx))
    else:
        indices = np.arange(n_images)
        rng = np.random.RandomState(args.seed)
        rng.shuffle(indices)
        half1_idx = np.sort(indices[: n_images // 2])
        half2_idx = np.sort(indices[n_images // 2 :])

    ds_half1 = ds.subset(half1_idx)
    ds_half2 = ds.subset(half2_idx)
    logger.info("Half-sets: %d + %d images", ds_half1.n_units, ds_half2.n_units)

    optimiser_star = _find_relion_optimiser_star(args)
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

    if args.n_classes < 1:
        raise SystemExit(f"--n_classes must be >= 1, got {args.n_classes}")
    if args.n_classes == 1:
        init_mrc_path = args.init_volume or os.path.join(args.data_dir, "reference_init.mrc")
        init_vol_real = _load_mrc(init_mrc_path).astype(np.float32)
        assert init_vol_real.shape == ds.volume_shape, (
            f"Volume shape mismatch: {init_vol_real.shape} vs {ds.volume_shape}"
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
            vol_ft = np.array(ftu.get_dft3(jnp.asarray(vol_real))).astype(np.complex64).reshape(-1)
            per_class_ft.append(vol_ft)
            logger.info("Class %d initial volume loaded from %s", k + 1, p)
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
    rotation_grid_order = init_healpix_order
    logger.info(
        "RELION grid orders: coarse/pass1=%d, fine/pass2=%d (adaptive_oversampling=%d)",
        init_healpix_order,
        finest_healpix_order,
        args.adaptive_oversampling,
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

    initial_noise_subset = np.arange(min(1000, ds.n_units), dtype=np.int32)
    # In RELION mode the E-step scores masked images, so the bootstrap noise
    # MUST come from masked images too — otherwise sigma2 is dominated by the
    # solvent area and the iter-1 chi² is ~3.3-6× too small (verified
    # 2026-04-08 against the tiny parity dataset, see tmp/check_sigma2_mask.py).
    initial_noise_radial = recon_noise.estimate_initial_noise_spectrum_from_unaligned_images(
        ds,
        initial_noise_subset,
        batch_size=min(args.image_batch_size, initial_noise_subset.size),
        apply_image_mask=True,
    )
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

        import starfile as _starfile

        _relion_init_dir = _Path(args.relion_init_dir)
        _it0_model_path = _relion_init_dir / "run_it000_model.star"
        _it0_optim_path = _relion_init_dir / "run_it000_optimiser.star"
        if not _it0_model_path.exists():
            raise SystemExit(f"--relion_init_dir given but {_it0_model_path} not found")
        _it0_model = _starfile.read(str(_it0_model_path))
        # sigma2_noise spectrum (× N⁴ for recovar's unit convention; matches
        # run_k_class_parity.py:715-717).
        _n4 = ds.grid_size**4
        _relion_sigma2 = np.asarray(_it0_model["model_optics_group_1"]["rlnSigma2Noise"], dtype=np.float64)
        _relion_noise_radial = jnp.asarray(_relion_sigma2 * _n4)
        noise_variance = recon_noise.make_radial_noise(_relion_noise_radial, ds.image_shape)
        logger.info(
            "STRICT-PARITY: replaced bootstrapped sigma2_noise with RELION it000 "
            "spectrum (× N^4=%.3e). RELION shape=%s, head=%s",
            float(_n4),
            _relion_sigma2.shape,
            np.asarray(_relion_sigma2[:5]),
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
        # Tau2 fudge factor + sigma_offset from optimiser.star.
        if _it0_optim_path.exists():
            _opt_text = _it0_optim_path.read_text()
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
    # ~22%). Per-image normCorrection / group scale replay remains opt-in via
    # --replay_relion_normcorr (it can hurt corr_vs_GT in some configurations).
    replay_iteration_overrides = None
    if args.perturb_replay_relion_dir is not None:
        replay_iteration_overrides = _build_replay_iteration_overrides(
            args.perturb_replay_relion_dir,
            half1_idx,
            half2_idx,
            args.max_iter,
            ds_voxel=ds.voxel_size,
            ds_grid=ds.grid_size,
            include_normcorr=bool(args.replay_relion_normcorr),
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

    result = refine_single_volume(
        experiment_datasets=experiment_datasets,
        init_volume=jnp.asarray(init_vol_ft),
        init_noise_variance=noise_variance,
        init_mean_variance=mean_variance,
        rotations=rotations,
        translations=translations_jnp,
        disc_type="linear_interp",
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
        n_classes=args.n_classes,
        emulate_relion_firstiter_cc=bool(args.firstiter_cc),
        relion_firstiter_ini_high_angstrom=(args.init_resolution if args.firstiter_cc else None),
    )

    total_time = time.time() - t_start
    logger.info("=" * 70)
    logger.info("Refinement complete in %.1fs (%d iterations)", total_time, args.max_iter)
    logger.info("=" * 70)

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
    if "sigma_offset_used_trajectory" in result:
        save_dict["sigma_offset_used_trajectory"] = np.asarray(
            result["sigma_offset_used_trajectory"],
            dtype=np.float64,
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
    if result.get("class_assignments") is not None and any(c is not None for c in result["class_assignments"]):
        for k, ca in enumerate(result["class_assignments"]):
            if ca is not None:
                save_dict[f"class_assignments_half{k}"] = np.asarray(ca, dtype=np.int32)
    if result.get("per_class_sigma_offset_trajectory") is not None:
        # Per-iter K-vector or None; serialize as object array via dtype=object.
        save_dict["per_class_sigma_offset_trajectory"] = np.asarray(
            result["per_class_sigma_offset_trajectory"], dtype=object
        )
    local_profile_rows = [
        {
            key: (np.asarray(value).item() if np.asarray(value).ndim == 0 else np.asarray(value).tolist())
            for key, value in row.items()
        }
        for row in result.get("local_profile_history", [])
    ]
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
        for i, pmax in enumerate(result["pmax_per_image_history"]):
            save_dict[f"pmax_per_image_iter_{i:03d}"] = np.asarray(pmax, dtype=np.float32)

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
    np.savez_compressed(out_path, **save_dict)
    logger.info("Results saved to %s", out_path)

    timing_rows = _collect_timing_rows(timing_dir_path)
    timing_summary = _summarize_timing_rows(timing_rows)
    if args.benchmark_ledger_json:
        ledger_path = Path(args.benchmark_ledger_json)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = {
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
            "auto_local_healpix_order": int(args.auto_local_healpix_order),
            "adaptive_oversampling": int(args.adaptive_oversampling),
            "max_significants": int(args.max_significants),
            "setup_phase_seconds": setup_phase_seconds,
            "local_profile_rows": local_profile_rows,
            "timing_rows": timing_rows,
            "timing_summary": timing_summary,
        }
        with ledger_path.open("w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2, sort_keys=True)
        logger.info("Benchmark ledger saved to %s", ledger_path)

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
