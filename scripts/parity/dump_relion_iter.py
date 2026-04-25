#!/usr/bin/env python
"""Dump RELION reference iters into the same .npz schema as parity_dump.

Usage:
    pixi run python scripts/parity/dump_relion_iter.py \
        --relion-dir /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0 \
        --out _agent_scratch/parity/relion \
        --max-iter 14
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mrcfile
import numpy as np
import starfile


def _read_text(p: Path) -> str:
    return p.read_text()


def _grep_value(text: str, key: str, cast=float, default=None):
    m = re.search(rf"^{re.escape(key)}\s+([\-\.\deE\+]+)", text, re.MULTILINE)
    if not m:
        return default
    try:
        return cast(m.group(1))
    except (TypeError, ValueError):
        return default


def parse_optimiser(p: Path) -> dict:
    text = _read_text(p)
    return {
        "current_iteration": _grep_value(text, "_rlnCurrentIteration", int, -1),
        "n_iterations": _grep_value(text, "_rlnNumberOfIterations", int, -1),
        "adaptive_oversample_order": _grep_value(text, "_rlnAdaptiveOversampleOrder", int, 0),
        "adaptive_oversample_fraction": _grep_value(text, "_rlnAdaptiveOversampleFraction", float, 0.999),
        "random_seed": _grep_value(text, "_rlnRandomSeed", int, 0),
        "particle_diameter": _grep_value(text, "_rlnParticleDiameter", float, 0.0),
        "do_zero_mask": _grep_value(text, "_rlnDoZeroMask", int, 1),
        "do_solvent_flattening": _grep_value(text, "_rlnDoSolventFlattening", int, 1),
        "offset_range_x": _grep_value(text, "_rlnOffsetRangeX", float, -1.0),
        "has_converged": _grep_value(text, "_rlnHasConverged", int, 0),
    }


def parse_sampling(p: Path) -> dict:
    text = _read_text(p)
    return {
        "healpix_order": _grep_value(text, "_rlnHealpixOrder", int, 3),
        "healpix_order_original": _grep_value(text, "_rlnHealpixOrderOriginal", int, 3),
        "offset_range": _grep_value(text, "_rlnOffsetRange", float, 0.0),
        "offset_step": _grep_value(text, "_rlnOffsetStep", float, 0.0),
        "offset_range_original": _grep_value(text, "_rlnOffsetRangeOriginal", float, 0.0),
        "offset_step_original": _grep_value(text, "_rlnOffsetStepOriginal", float, 0.0),
        "perturb_instance": _grep_value(text, "_rlnSamplingPerturbInstance", float, 0.0),
        "perturb_factor": _grep_value(text, "_rlnSamplingPerturbFactor", float, 0.0),
    }


def parse_model(p: Path) -> dict:
    """Parse RELION model.star: control state + per-shell sigma2_noise."""
    text = _read_text(p)
    out = {
        "current_image_size": _grep_value(text, "_rlnCurrentImageSize", int, -1),
        "current_resolution": _grep_value(text, "_rlnCurrentResolution", float, -1.0),
        "tau2_fudge_factor": _grep_value(text, "_rlnTau2FudgeFactor", float, 1.0),
        "sigma_offsets_angst": _grep_value(text, "_rlnSigmaOffsetsAngst", float, 0.0),
        "ave_pmax": _grep_value(text, "_rlnAveragePmax", float, 0.0),
        "log_likelihood": _grep_value(text, "_rlnLogLikelihood", float, 0.0),
    }

    sigma2_noise: list[float] = []
    in_block = False
    seen_header = False
    for line in text.splitlines():
        s = line.strip()
        if not in_block:
            if "_rlnSigma2Noise" in s:
                in_block = True
                seen_header = True
                continue
            continue
        if not seen_header:
            seen_header = True
            continue
        if not s:
            if sigma2_noise:
                break
            continue
        if s.startswith("data_") or s.startswith("loop_") or s.startswith("_rln"):
            if sigma2_noise:
                break
            continue
        parts = s.split()
        if len(parts) >= 2:
            try:
                sigma2_noise.append(float(parts[-1]))
            except ValueError:
                if sigma2_noise:
                    break
    out["sigma2_noise"] = np.asarray(sigma2_noise, dtype=np.float64)
    return out


def stack_index_from_image_name(name: str) -> int:
    m = re.match(r"(\d+)@", str(name))
    return int(m.group(1)) - 1 if m else -1


def parse_data_star(p: Path) -> dict:
    blocks = starfile.read(str(p), always_dict=True)
    particles = blocks.get("particles") if isinstance(blocks, dict) else blocks
    if particles is None:
        raise RuntimeError(f"No particles block in {p}")
    img_names = particles["rlnImageName"].to_numpy()
    stack_idx = np.asarray([stack_index_from_image_name(n) for n in img_names], dtype=np.int64)
    return {
        "image_name": img_names.astype("U"),
        "stack_index": stack_idx,
        "rlnAngleRot": particles["rlnAngleRot"].to_numpy(dtype=np.float64),
        "rlnAngleTilt": particles["rlnAngleTilt"].to_numpy(dtype=np.float64),
        "rlnAnglePsi": particles["rlnAnglePsi"].to_numpy(dtype=np.float64),
        "rlnOriginXAngst": particles["rlnOriginXAngst"].to_numpy(dtype=np.float64),
        "rlnOriginYAngst": particles["rlnOriginYAngst"].to_numpy(dtype=np.float64),
        "rlnRandomSubset": particles["rlnRandomSubset"].to_numpy(dtype=np.int32),
        "rlnLogLikeliContribution": particles["rlnLogLikeliContribution"].to_numpy(dtype=np.float64)
        if "rlnLogLikeliContribution" in particles.columns
        else None,
        "rlnMaxValueProbDistribution": particles["rlnMaxValueProbDistribution"].to_numpy(dtype=np.float32)
        if "rlnMaxValueProbDistribution" in particles.columns
        else None,
        "rlnNrOfSignificantSamples": particles["rlnNrOfSignificantSamples"].to_numpy(dtype=np.int32)
        if "rlnNrOfSignificantSamples" in particles.columns
        else None,
        "rlnNormCorrection": particles["rlnNormCorrection"].to_numpy(dtype=np.float64)
        if "rlnNormCorrection" in particles.columns
        else None,
    }


def _shell_indices_3d(n: int) -> np.ndarray:
    idx = np.arange(n)
    iz = np.where(idx <= n // 2, idx, idx - n)
    iy = np.where(idx <= n // 2, idx, idx - n)
    ix = np.where(idx <= n // 2, idx, idx - n)
    Z, Y, X = np.meshgrid(iz, iy, ix, indexing="ij")
    r = np.sqrt(Z * Z + Y * Y + X * X)
    return np.clip(r.astype(np.int32), 0, n // 2).reshape(-1)


def _downsample_real_volume(real_vol: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return real_vol.reshape(-1).astype(np.float32)
    n = real_vol.shape[0]
    crop = n // factor
    if crop < 4:
        return real_vol.reshape(-1).astype(np.float32)
    start = (n - crop) // 2
    end = start + crop
    return real_vol[start:end, start:end, start:end].reshape(-1).astype(np.float32)


def parse_half_map(p: Path) -> tuple[np.ndarray, float]:
    with mrcfile.open(str(p), mode="r") as m:
        return np.asarray(m.data, dtype=np.float32), float(m.voxel_size.x)


def dump_iter(relion_dir: Path, iteration: int, out_dir: Path, downsample_factor: int = 2) -> Path:
    """Dump one RELION iter to .npz mirroring parity_dump.dump_iteration schema."""

    data_star = relion_dir / f"run_it{iteration:03d}_data.star"
    optimiser_star = relion_dir / f"run_it{iteration:03d}_optimiser.star"
    sampling_star = relion_dir / f"run_it{iteration:03d}_sampling.star"
    model1 = relion_dir / f"run_it{iteration:03d}_half1_model.star"
    model2 = relion_dir / f"run_it{iteration:03d}_half2_model.star"
    half1_map = relion_dir / f"run_it{iteration:03d}_half1_class001.mrc"
    half2_map = relion_dir / f"run_it{iteration:03d}_half2_class001.mrc"

    opt = parse_optimiser(optimiser_star)
    samp = parse_sampling(sampling_star)
    m1 = parse_model(model1)
    m2 = parse_model(model2)
    data = parse_data_star(data_star)
    h1, voxel = parse_half_map(half1_map)
    h2, _ = parse_half_map(half2_map)

    # FSC: not in model.star — would need fsc.star or postprocess. Leave NaN for now.
    fsc = np.full(h1.shape[0] // 2 + 1, np.nan, dtype=np.float64)

    payload: dict[str, np.ndarray] = {
        "iteration": np.int32(iteration),
        "current_image_size": np.int32(m1["current_image_size"]),
        "current_resolution_a": np.float64(m1["current_resolution"]),
        "tau2_fudge": np.float64(m1["tau2_fudge_factor"]),
        "sigma_offset": np.float64(m1["sigma_offsets_angst"]),
        "ave_pmax_model": np.float64(m1["ave_pmax"]),
        "log_likelihood": np.float64(m1["log_likelihood"]),
        "voxel_size": np.float64(voxel),
        "healpix_order": np.int32(samp["healpix_order"]),
        "healpix_order_original": np.int32(samp["healpix_order_original"]),
        "offset_range": np.float64(samp["offset_range"]),
        "offset_step": np.float64(samp["offset_step"]),
        "perturb_instance": np.float64(samp["perturb_instance"]),
        "perturb_factor": np.float64(samp["perturb_factor"]),
        "adaptive_oversample_order": np.int32(opt["adaptive_oversample_order"]),
        "adaptive_oversample_fraction": np.float64(opt["adaptive_oversample_fraction"]),
        "random_seed": np.int64(opt["random_seed"]),
        "fsc": fsc,
        "half1_sigma2_noise": m1["sigma2_noise"],
        "half2_sigma2_noise": m2["sigma2_noise"],
    }

    n = h1.shape[0]
    payload["half1_mean_real_ds"] = _downsample_real_volume(h1, downsample_factor)
    payload["half2_mean_real_ds"] = _downsample_real_volume(h2, downsample_factor)
    payload["half1_mean_real_full_norm"] = np.float64(np.linalg.norm(h1))
    payload["half2_mean_real_full_norm"] = np.float64(np.linalg.norm(h2))
    payload["half_volume_n"] = np.int32(n)

    payload["particle_image_name"] = data["image_name"]
    payload["particle_stack_index"] = data["stack_index"]
    payload["particle_random_subset"] = data["rlnRandomSubset"]
    payload["particle_eulers"] = np.stack(
        [data["rlnAngleRot"], data["rlnAngleTilt"], data["rlnAnglePsi"]], axis=1
    ).astype(np.float64)
    payload["particle_origin_angst_xy"] = np.stack([data["rlnOriginXAngst"], data["rlnOriginYAngst"]], axis=1).astype(
        np.float64
    )
    payload["particle_origin_pixel_xy"] = payload["particle_origin_angst_xy"] / float(voxel)
    if data["rlnMaxValueProbDistribution"] is not None:
        payload["particle_max_pmax"] = data["rlnMaxValueProbDistribution"]
    if data["rlnLogLikeliContribution"] is not None:
        payload["particle_log_likelihood"] = data["rlnLogLikeliContribution"]
    if data["rlnNrOfSignificantSamples"] is not None:
        payload["particle_n_significant"] = data["rlnNrOfSignificantSamples"]
    if data["rlnNormCorrection"] is not None:
        payload["particle_norm_correction"] = data["rlnNormCorrection"]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"iter_{iteration:03d}.npz"
    np.savez_compressed(out_path, **payload)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relion-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-iter", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=14)
    ap.add_argument("--downsample-factor", type=int, default=2)
    args = ap.parse_args()
    for it in range(args.min_iter, args.max_iter + 1):
        try:
            p = dump_iter(args.relion_dir, it, args.out, args.downsample_factor)
            print(f"wrote {p}")
        except FileNotFoundError as e:
            print(f"skip iter {it}: {e}")


if __name__ == "__main__":
    main()
