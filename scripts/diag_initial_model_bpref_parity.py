#!/usr/bin/env python
"""Diagnose RELION InitialModel iter-1 E-step -> BPref parity.

The active InitialModel path uses ``gpu_pipeline.run_iter_gpu_vdam`` which
drives ``dense_single_volume.em_engine.run_em`` directly. This diagnostic uses
the same production E-step ingredients, but exposes the suspected divergence
points as switches:

- halfset routing
- fine versus RELION coarse-expanded translation prior
- full-volume versus RELION half-volume M-step backprojection layout

It writes the compared BPref arrays, full returned accumulators, posterior
Pmax values, and shell-wise metrics under the requested output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import time
from pathlib import Path

import numpy as np


DEFAULT_FIXTURE_DIR = Path("/scratch/gpfs/GILLES/mg6942/tmp/relion_initialmodel_64_20260420_121428_8956_run")
DEFAULT_PARTICLES_STAR = None
DEFAULT_RELION_DUMP_DIR = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_debug_dump")
DEFAULT_RELION_ESTEP_DUMP = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_estep_dump_small")


def _read_bin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        nz, ny, nx = struct.unpack("qqq", f.read(24))
        pos = f.tell()
        f.seek(0, 2)
        rem = f.tell() - pos
        f.seek(pos)
        bp = rem // (nz * ny * nx) if nz * ny * nx else 8
        dt = np.complex128 if bp == 16 else np.float64
        return np.fromfile(f, dtype=dt, count=nz * ny * nx).reshape(nz, ny, nx)


def _read_headered_bin_auto(path: Path) -> np.ndarray:
    """Read RELION debug arrays with either int64 or int32 shape headers."""

    size = path.stat().st_size
    with open(path, "rb") as f:
        for fmt, header_size in (("qqq", 24), ("iii", 12)):
            f.seek(0)
            dims = struct.unpack(fmt, f.read(header_size))
            if any(int(d) <= 0 for d in dims):
                continue
            n = int(dims[0]) * int(dims[1]) * int(dims[2])
            if n <= 0:
                continue
            rem = size - header_size
            if rem == n * np.dtype(np.complex128).itemsize:
                dtype = np.complex128
            elif rem == n * np.dtype(np.float64).itemsize:
                dtype = np.float64
            else:
                continue
            return np.fromfile(f, dtype=dtype, count=n).reshape(tuple(int(d) for d in dims))
    raise ValueError(f"could not infer RELION debug-array header for {path}")


def _read_raw_scalar(path: Path) -> float | None:
    if not path.exists():
        return None
    data = np.fromfile(path, dtype=np.float64)
    if data.size:
        return float(data.reshape(-1)[0])
    data32 = np.fromfile(path, dtype=np.float32)
    return float(data32.reshape(-1)[0]) if data32.size else None


def _read_iter0_sigma2(fixture_dir: Path, n_shells: int) -> np.ndarray:
    txt = (fixture_dir / "run_it000_model.star").read_text()
    m = re.search(r"data_model_optics_group_1\n(.*?)(?:\ndata_)", txt, re.DOTALL)
    if not m:
        raise RuntimeError("could not find data_model_optics_group_1")
    values = np.zeros(n_shells, dtype=np.float64)
    for line in m.group(1).strip().split("\n"):
        toks = line.split()
        if len(toks) == 3:
            try:
                values[int(toks[0])] = float(toks[2])
            except ValueError:
                continue
    return values


def _infer_particles_star(fixture_dir: Path) -> Path:
    optimiser = fixture_dir / "run_it000_optimiser.star"
    txt = optimiser.read_text()
    m = re.search(r"(?:^|\s)--i\s+(\S+)", txt)
    if not m:
        raise RuntimeError(f"could not infer --i particles STAR from {optimiser}")
    return Path(m.group(1))


def _cc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    aa = np.asarray(a)
    bb = np.asarray(b)
    if mask is not None:
        aa = aa[mask]
        bb = bb[mask]
    af = aa.reshape(-1) - aa.reshape(-1).mean()
    bf = bb.reshape(-1) - bb.reshape(-1).mean()
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    if denom == 0:
        return float("nan")
    return float(np.real(np.vdot(af, bf)) / denom)


def _norm_ratio(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    aa = np.asarray(a)
    bb = np.asarray(b)
    if mask is not None:
        aa = aa[mask]
        bb = bb[mask]
    return float(np.linalg.norm(aa.reshape(-1)) / max(float(np.linalg.norm(bb.reshape(-1))), 1e-30))


def _max_rel_err(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    aa = np.asarray(a)
    bb = np.asarray(b)
    if mask is not None:
        aa = aa[mask]
        bb = bb[mask]
    return float(np.max(np.abs(aa - bb)) / max(float(np.max(np.abs(bb))), 1e-30))


def _bpref_shells(shape: tuple[int, int, int], r_max: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.arange(shape[0], dtype=np.int32) - (shape[0] // 2)
    y = np.arange(shape[1], dtype=np.int32) - (shape[1] // 2)
    x = np.arange(shape[2], dtype=np.int32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    shell = np.rint(np.sqrt(zz * zz + yy * yy + xx * xx)).astype(np.int32)
    return shell, (zz * zz + yy * yy + xx * xx) <= int(r_max) ** 2


def _array_metrics(ours: np.ndarray, target: np.ndarray, r_max: int) -> dict[str, object]:
    shell, sphere = _bpref_shells(target.shape, r_max)
    out: dict[str, object] = {
        "shape": list(target.shape),
        "cc_all": _cc(ours, target),
        "cc_sphere": _cc(ours, target, sphere),
        "cc_outside_sphere": _cc(ours, target, ~sphere),
        "norm_ratio_all": _norm_ratio(ours, target),
        "norm_ratio_sphere": _norm_ratio(ours, target, sphere),
        "max_rel_err_all": _max_rel_err(ours, target),
        "max_rel_err_sphere": _max_rel_err(ours, target, sphere),
        "sphere_count": int(np.sum(sphere)),
        "outside_sphere_count": int(np.sum(~sphere)),
    }
    shell_rows = []
    for s in range(int(shell.max()) + 1):
        mask = shell == s
        if not np.any(mask):
            continue
        shell_rows.append(
            {
                "shell": int(s),
                "count": int(np.sum(mask)),
                "cc": _cc(ours, target, mask),
                "norm_ratio": _norm_ratio(ours, target, mask),
                "max_rel_err": _max_rel_err(ours, target, mask),
                "target_norm": float(np.linalg.norm(np.asarray(target)[mask].reshape(-1))),
                "ours_norm": float(np.linalg.norm(np.asarray(ours)[mask].reshape(-1))),
            }
        )
    out["shells"] = shell_rows
    return out


def _read_relion_sorted_idx(path: Path, n_images: int | None = None) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        arr = _read_bin(path).reshape(-1)
    except Exception:
        raw = np.fromfile(path, dtype=np.int64)
        if raw.size == 0:
            return None
        arr = raw
    arr = np.asarray(arr, dtype=np.int64).reshape(-1)
    if n_images is not None:
        for off in range(max(0, arr.size - n_images + 1)):
            candidate = arr[off : off + n_images]
            if candidate.size == n_images and np.array_equal(np.sort(candidate), np.arange(n_images)):
                return candidate
        raise ValueError(f"could not find {n_images}-entry permutation in {path}; got {arr.size} int64 values")
    if arr.size and arr[0] == arr.size - 1 and np.unique(arr[1:]).size == arr.size - 1:
        arr = arr[1:]
    return arr


def _halfset_ids_for_mode(mode: str, n_images: int, micrograph_names: np.ndarray, relion_sorted_idx: np.ndarray | None):
    ids = np.arange(n_images, dtype=np.int64)
    half = np.zeros(n_images, dtype=np.int8)
    if mode == "natural_id_parity":
        half = (ids % 2).astype(np.int8)
    elif mode == "micrograph_sorted_position_parity":
        order = np.argsort(np.asarray(micrograph_names), kind="stable")
        half[order] = (np.arange(n_images, dtype=np.int64) % 2).astype(np.int8)
    elif mode == "relion_sorted_position_parity":
        if relion_sorted_idx is None:
            raise ValueError("relion_sorted_position_parity requires sorted_idx_iter001.bin")
        order = np.asarray(relion_sorted_idx, dtype=np.int64)
        if order.size != n_images:
            raise ValueError(f"sorted_idx size {order.size} != n_images {n_images}")
        half[order] = (np.arange(n_images, dtype=np.int64) % 2).astype(np.int8)
    elif mode == "relion_sorted_value_parity":
        if relion_sorted_idx is None:
            raise ValueError("relion_sorted_value_parity requires sorted_idx_iter001.bin")
        order = np.asarray(relion_sorted_idx, dtype=np.int64)
        if order.size != n_images:
            raise ValueError(f"sorted_idx size {order.size} != n_images {n_images}")
        half[order] = (order % 2).astype(np.int8)
    else:
        raise ValueError(f"unknown halfset mode {mode!r}")
    return half


def _load_sampling(fixture_dir: Path, estep_dump_dir: Path):
    from recovar.em.sampling import (
        apply_relion_translation_perturbation,
        get_oversampled_rotation_grid_from_samples,
        get_oversampled_translation_grid,
        get_translation_grid,
        read_relion_perturbation_from_sampling_star,
    )

    sampling_star = fixture_dir / "run_it001_sampling.star"
    random_perturbation, _perturbation_factor = read_relion_perturbation_from_sampling_star(str(sampling_star))
    coarse_translations = get_translation_grid(max_pixel=6, pixel_offset=2).astype(np.float32)
    eulers_bin = estep_dump_dir / "p0_oversampled_eulers.bin"
    trans_bin = estep_dump_dir / "p0_oversampled_translations.bin"
    if eulers_bin.exists() and trans_bin.exists():
        from recovar.utils.helpers import R_from_relion

        with open(eulers_bin, "rb") as f:
            h = struct.unpack("qqq", f.read(24))
            eulers = np.fromfile(f, dtype=np.float64, count=h[0] * h[1] * h[2]).reshape(-1, 3)
        rotations = R_from_relion(eulers).astype(np.float32)
        with open(trans_bin, "rb") as f:
            h = struct.unpack("qqq", f.read(24))
            trans = np.fromfile(f, dtype=np.float64, count=h[0] * h[1] * h[2]).reshape(-1, 3)
        translations = trans[:, :2].astype(np.float32)
        if translations.shape[0] % coarse_translations.shape[0] != 0:
            raise RuntimeError(
                "cannot infer oversampled translation parent map: "
                f"{translations.shape[0]} fine translations, {coarse_translations.shape[0]} coarse translations",
            )
        n_over_trans = translations.shape[0] // coarse_translations.shape[0]
        fine_translation_parent = np.repeat(np.arange(coarse_translations.shape[0]), n_over_trans).astype(np.int32)
        source = "relion_dump_p0_oversampled"
    else:
        coarse_indices = np.arange(48 * 12, dtype=np.int64)
        rotations, _ = get_oversampled_rotation_grid_from_samples(
            coarse_indices,
            parent_nside_level=1,
            oversampling_order=1,
            random_perturbation=random_perturbation,
        )
        translations, fine_translation_parent = get_oversampled_translation_grid(
            coarse_translations,
            pixel_offset=2,
            oversampling_order=1,
        )
        translations = apply_relion_translation_perturbation(
            translations.astype(np.float32),
            random_perturbation,
            offset_step_pixels=2.0,
        )
        fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
        source = "constructed_from_sampling_star"
    return (
        np.asarray(rotations, dtype=np.float32),
        np.asarray(translations, dtype=np.float32),
        np.asarray(coarse_translations, dtype=np.float32),
        np.asarray(fine_translation_parent, dtype=np.int32),
        source,
    )


def _build_estep_inputs(
    ds,
    fixture_dir: Path,
    estep_dump_dir: Path,
    current_size: int,
    reference_source: str,
    translation_prior_mode: str,
    gridding_mode: str,
    iref_scale_mode: str,
    iref_extra_scale: float,
    iref_sign: float,
    noise_scale_mode: str,
    noise_extra_scale: float,
):
    import jax.numpy as jnp

    from recovar.core import fourier_transform_utils as ftu
    from recovar.core.relion_project import gridding_correct_volume_real
    from recovar.reconstruction.relion_functions import griddingCorrect
    from recovar.reconstruction.noise import make_radial_noise
    from recovar.utils.helpers import load_relion_volume, relion_volume_to_recovar

    ori = int(ds.grid_size)
    if reference_source == "mrc":
        iref_real = np.asarray(load_relion_volume(str(fixture_dir / "run_it000_class001.mrc")), dtype=np.float64)
    elif reference_source == "relion_iref_dump_raw":
        iref_real = np.asarray(_read_headered_bin_auto(estep_dump_dir / "iref_c0_pre_setup.bin"), dtype=np.float64)
    elif reference_source == "relion_iref_dump_recovar":
        iref_relion = np.asarray(_read_headered_bin_auto(estep_dump_dir / "iref_c0_pre_setup.bin"), dtype=np.float64)
        iref_real = np.asarray(relion_volume_to_recovar(iref_relion), dtype=np.float64)
    elif reference_source == "relion_projector_dump":
        ppref = np.asarray(_read_headered_bin_auto(estep_dump_dir / "ppref_c0_data_post_setup.bin"))
        half_ps = ppref.shape[0] // 2
        half = np.zeros((ori, ori, ori // 2 + 1), dtype=np.complex128)
        c = ori // 2
        half[c - half_ps : c + half_ps + 1, c - half_ps : c + half_ps + 1, : ppref.shape[2]] = ppref
        iref_ft = np.asarray(ftu.half_volume_to_full_volume(jnp.asarray(half), (ori, ori, ori))).reshape(-1)
        if iref_scale_mode != "one" or gridding_mode != "none":
            raise ValueError("relion_projector_dump is already gridded/scaled; use --gridding-mode none --iref-scale-mode one")
        iref_ft = iref_ft * float(iref_sign) * float(iref_extra_scale)
        sigma2 = _read_iter0_sigma2(fixture_dir, ori // 2 + 1)
        n4 = ori**4
        if noise_scale_mode == "one":
            noise_scale = 1.0
        elif noise_scale_mode == "n4":
            noise_scale = float(n4)
        elif noise_scale_mode == "inv_n4":
            noise_scale = 1.0 / float(n4)
        else:
            raise ValueError(f"unknown noise_scale_mode={noise_scale_mode!r}")
        noise_scale = noise_scale * float(noise_extra_scale)
        noise_variance = np.asarray(make_radial_noise(sigma2 * noise_scale, (ori, ori))).astype(np.float32).reshape(-1)
        rotations, translations, coarse_translations, fine_translation_parent, sampling_source = _load_sampling(
            fixture_dir,
            estep_dump_dir,
        )
        translation_log_prior, translation_meta = _build_translation_prior(
            ds,
            translations,
            coarse_translations,
            fine_translation_parent,
            translation_prior_mode,
        )
        meta = {
            "ori_size": ori,
            "current_size": int(current_size),
            "r_max": int(current_size // 2),
            "reference_source": reference_source,
            "sampling_source": sampling_source,
            "gridding_mode": gridding_mode,
            "iref_scale_mode": iref_scale_mode,
            "iref_scale": 1.0,
            "iref_extra_scale": float(iref_extra_scale),
            "iref_sign": float(iref_sign),
            "noise_scale_mode": noise_scale_mode,
            "noise_scale": float(noise_scale),
            "n_rotations": int(rotations.shape[0]),
            "n_translations": int(translations.shape[0]),
            "n_coarse_translations": int(coarse_translations.shape[0]),
            **translation_meta,
        }
        return {
            "mean": iref_ft.astype(np.complex64),
            "mean_variance": (np.abs(iref_ft) ** 2).astype(np.float32),
            "noise_variance": noise_variance,
            "rotations": rotations,
            "translations": translations,
            "translation_log_prior": translation_log_prior,
            "meta": meta,
        }
    else:
        raise ValueError(f"unknown reference_source={reference_source!r}")
    sigma2 = _read_iter0_sigma2(fixture_dir, ori // 2 + 1)
    if gridding_mode == "core":
        iref_real_for_ft = np.asarray(gridding_correct_volume_real(jnp.asarray(iref_real), ori, 1))
    elif gridding_mode == "legacy":
        iref_real_for_ft, _ = griddingCorrect(jnp.asarray(iref_real), ori, padding_factor=1, order=1)
        iref_real_for_ft = np.asarray(iref_real_for_ft)
    elif gridding_mode == "none":
        iref_real_for_ft = np.asarray(iref_real)
    else:
        raise ValueError(f"unknown gridding_mode={gridding_mode!r}")
    iref_ft = np.asarray(ftu.get_dft3(jnp.asarray(iref_real_for_ft))).reshape(-1)
    if iref_scale_mode == "one":
        iref_scale = 1.0
    elif iref_scale_mode == "inv_n2":
        iref_scale = 1.0 / (ori**2)
    elif iref_scale_mode == "inv_n3":
        iref_scale = 1.0 / (ori**3)
    elif iref_scale_mode == "n2":
        iref_scale = float(ori) ** 2
    else:
        raise ValueError(f"unknown iref_scale_mode={iref_scale_mode!r}")
    iref_ft = iref_ft * float(iref_sign) * float(iref_scale)
    iref_ft = iref_ft * float(iref_extra_scale)
    n4 = ori**4
    if noise_scale_mode == "one":
        noise_scale = 1.0
    elif noise_scale_mode == "n4":
        noise_scale = float(n4)
    elif noise_scale_mode == "inv_n4":
        noise_scale = 1.0 / float(n4)
    else:
        raise ValueError(f"unknown noise_scale_mode={noise_scale_mode!r}")
    noise_scale = noise_scale * float(noise_extra_scale)
    noise_variance = np.asarray(make_radial_noise(sigma2 * noise_scale, (ori, ori))).astype(np.float32).reshape(-1)
    rotations, translations, coarse_translations, fine_translation_parent, sampling_source = _load_sampling(
        fixture_dir,
        estep_dump_dir,
    )

    translation_log_prior, translation_meta = _build_translation_prior(
        ds,
        translations,
        coarse_translations,
        fine_translation_parent,
        translation_prior_mode,
    )

    meta = {
        "ori_size": ori,
        "current_size": int(current_size),
        "r_max": int(current_size // 2),
        "reference_source": reference_source,
        "sampling_source": sampling_source,
        "gridding_mode": gridding_mode,
        "iref_scale_mode": iref_scale_mode,
        "iref_scale": float(iref_scale),
        "iref_extra_scale": float(iref_extra_scale),
        "iref_sign": float(iref_sign),
        "noise_scale_mode": noise_scale_mode,
        "noise_scale": float(noise_scale),
        "n_rotations": int(rotations.shape[0]),
        "n_translations": int(translations.shape[0]),
        "n_coarse_translations": int(coarse_translations.shape[0]),
        **translation_meta,
    }
    return {
        "mean": iref_ft.astype(np.complex64),
        "mean_variance": (np.abs(iref_ft) ** 2).astype(np.float32),
        "noise_variance": noise_variance,
        "rotations": rotations,
        "translations": translations,
        "translation_log_prior": translation_log_prior,
        "meta": meta,
    }


def _build_translation_prior(
    ds,
    translations: np.ndarray,
    coarse_translations: np.ndarray,
    fine_translation_parent: np.ndarray,
    translation_prior_mode: str,
):
    sigma_offset_ang = 6.398173
    if translation_prior_mode == "fine":
        t_dist2_ang2 = (translations[:, 0] ** 2 + translations[:, 1] ** 2) * (float(ds.voxel_size) ** 2)
        translation_log_prior = (-0.5 * t_dist2_ang2 / (sigma_offset_ang**2)).astype(np.float32)
    elif translation_prior_mode == "coarse_expanded":
        t_dist2_ang2 = (coarse_translations[:, 0] ** 2 + coarse_translations[:, 1] ** 2) * (
            float(ds.voxel_size) ** 2
        )
        coarse_prior = (-0.5 * t_dist2_ang2 / (sigma_offset_ang**2)).astype(np.float32)
        translation_log_prior = coarse_prior[fine_translation_parent].astype(np.float32, copy=False)
    elif translation_prior_mode == "none":
        translation_log_prior = None
    else:
        raise ValueError(f"unknown translation_prior_mode={translation_prior_mode!r}")

    meta = {
        "translation_prior_mode": translation_prior_mode,
        "translation_log_prior_min": None if translation_log_prior is None else float(np.min(translation_log_prior)),
        "translation_log_prior_max": None if translation_log_prior is None else float(np.max(translation_log_prior)),
    }
    return translation_log_prior, meta


def _run_halfset_estep(args, subset_ds, inputs: dict[str, object], *, relion_half_volume_mstep: bool):
    import jax
    import jax.numpy as jnp

    from recovar.em.dense_single_volume.em_engine import run_em

    n = int(subset_ds.n_images)
    engine_kwargs = {
        "score_with_masked_images": True,
        "relion_firstiter_score_mode": "gaussian",
        "image_pre_shifts": np.zeros((n, 2), dtype=np.float32),
        "sparse_pass2": bool(args.sparse_pass2),
        "relion_half_volume_mstep": bool(relion_half_volume_mstep),
    }
    translation_log_prior = inputs["translation_log_prior"]
    if translation_log_prior is not None:
        engine_kwargs["translation_log_prior"] = np.asarray(translation_log_prior, dtype=np.float32)
        engine_kwargs["translation_prior_centers"] = np.zeros((n, 2), dtype=np.float32)

    result = run_em(
        subset_ds,
        mean=jnp.asarray(inputs["mean"], dtype=jnp.complex64),
        mean_variance=jnp.asarray(inputs["mean_variance"], dtype=jnp.float32),
        noise_variance=jnp.asarray(inputs["noise_variance"], dtype=jnp.float32),
        rotations=jnp.asarray(inputs["rotations"], dtype=jnp.float32),
        translations=jnp.asarray(inputs["translations"], dtype=jnp.float32),
        disc_type="linear_interp",
        image_batch_size=int(args.image_batch_size),
        rotation_block_size=int(args.rotation_block_size),
        current_size=int(args.current_size),
        projection_padding_factor=1,
        reconstruction_padding_factor=1,
        half_spectrum_scoring=True,
        return_stats=True,
        **engine_kwargs,
    )
    jax.block_until_ready(result[2])
    return np.asarray(result[2]), np.asarray(result[3]), result[4]


def run_mode(args, ds, main_in, relion_sorted_idx, mode: str, out_dir: Path, layout_name: str) -> dict[str, object]:
    from recovar.em.initial_model.gpu_pipeline import run_em_output_to_bpref

    micrograph_names = np.asarray(main_in["_rlnMicrographName"].astype(str).to_numpy())
    halfset_ids = _halfset_ids_for_mode(mode, int(ds.n_images), micrograph_names, relion_sorted_idx)
    h0 = np.where(halfset_ids == 0)[0]
    h1 = np.where(halfset_ids == 1)[0]
    if not hasattr(ds, "subset"):
        raise RuntimeError("dataset has no subset()")
    ds_h0 = ds.subset(h0)
    ds_h1 = ds.subset(h1)
    inputs = _build_estep_inputs(
        ds,
        args.fixture_dir,
        args.relion_estep_dump_dir,
        args.current_size,
        args.reference_source,
        args.translation_prior_mode,
        args.gridding_mode,
        args.iref_scale_mode,
        args.iref_extra_scale,
        args.iref_sign,
        args.noise_scale_mode,
        args.noise_extra_scale,
    )
    relion_half_volume_mstep = layout_name == "half"

    t0 = time.time()
    Ft_y_h0, Ft_ctf_h0, stats_h0 = _run_halfset_estep(
        args,
        ds_h0,
        inputs,
        relion_half_volume_mstep=relion_half_volume_mstep,
    )
    Ft_y_h1, Ft_ctf_h1, stats_h1 = _run_halfset_estep(
        args,
        ds_h1,
        inputs,
        relion_half_volume_mstep=relion_half_volume_mstep,
    )
    elapsed_s = time.time() - t0

    ori = int(ds.grid_size)
    r_max = int(args.current_size // 2)
    n2_frame = float(ori) ** 2
    n4_frame = float(ori) ** 4
    ours_h0, ours_w0 = run_em_output_to_bpref(Ft_y_h0, Ft_ctf_h0, ori, r_max, 1)
    ours_h1, ours_w1 = run_em_output_to_bpref(Ft_y_h1, Ft_ctf_h1, ori, r_max, 1)
    ours_h0 = -np.asarray(ours_h0) * n2_frame
    ours_h1 = -np.asarray(ours_h1) * n2_frame
    ours_w0 = np.asarray(ours_w0) * n4_frame
    ours_w1 = np.asarray(ours_w1) * n4_frame

    target_h0 = _read_bin(args.relion_dump_dir / "pipe_it1_c0_bp_data_pre_reweight.bin")
    target_h1 = _read_bin(args.relion_dump_dir / "pipe_it1_c0_bp_data_h_pre_reweight.bin")
    target_w0 = _read_bin(args.relion_dump_dir / "pipe_it1_c0_bp_weight.bin")
    target_w1 = _read_bin(args.relion_dump_dir / "pipe_it1_c0_bp_weight_h.bin")

    direct = {
        "h0_bp_data": _array_metrics(ours_h0, target_h0, r_max),
        "h1_bp_data": _array_metrics(ours_h1, target_h1, r_max),
        "h0_bp_weight": _array_metrics(ours_w0, target_w0, r_max),
        "h1_bp_weight": _array_metrics(ours_w1, target_w1, r_max),
    }
    swapped = {
        "h0_bp_data": _array_metrics(ours_h1, target_h0, r_max),
        "h1_bp_data": _array_metrics(ours_h0, target_h1, r_max),
        "h0_bp_weight": _array_metrics(ours_w1, target_w0, r_max),
        "h1_bp_weight": _array_metrics(ours_w0, target_w1, r_max),
    }

    pmax = np.concatenate(
        [
            np.asarray(stats_h0.max_posterior_per_image, dtype=np.float32),
            np.asarray(stats_h1.max_posterior_per_image, dtype=np.float32),
        ]
    )

    npz_path = out_dir / f"{mode}_{layout_name}_arrays.npz"
    np.savez_compressed(
        npz_path,
        halfset_ids=halfset_ids.astype(np.int8),
        halfset_h0_ids=h0.astype(np.int64),
        halfset_h1_ids=h1.astype(np.int64),
        ours_h0_bp_data=ours_h0,
        ours_h1_bp_data=ours_h1,
        ours_h0_bp_weight=ours_w0,
        ours_h1_bp_weight=ours_w1,
        target_h0_bp_data=target_h0,
        target_h1_bp_data=target_h1,
        target_h0_bp_weight=target_w0,
        target_h1_bp_weight=target_w1,
        Ft_y_h0=Ft_y_h0,
        Ft_y_h1=Ft_y_h1,
        Ft_ctf_h0=Ft_ctf_h0,
        Ft_ctf_h1=Ft_ctf_h1,
        max_posterior=pmax,
    )

    return {
        "mode": mode,
        "mstep_layout": layout_name,
        "elapsed_s": float(elapsed_s),
        "npz_path": str(npz_path),
        "halfset_counts": {
            "h0": int(np.sum(halfset_ids == 0)),
            "h1": int(np.sum(halfset_ids == 1)),
        },
        "config": inputs["meta"],
        "direct": direct,
        "swapped": swapped,
        "pmax_mean": float(np.mean(pmax)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--particles-star", type=Path, default=DEFAULT_PARTICLES_STAR)
    parser.add_argument("--relion-dump-dir", type=Path, default=DEFAULT_RELION_DUMP_DIR)
    parser.add_argument("--relion-estep-dump-dir", type=Path, default=DEFAULT_RELION_ESTEP_DUMP)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--current-size", type=int, default=28)
    parser.add_argument("--image-batch-size", type=int, default=50)
    parser.add_argument("--rotation-block-size", type=int, default=100)
    parser.add_argument("--sparse-pass2", action="store_true")
    parser.add_argument(
        "--reference-source",
        choices=["mrc", "relion_iref_dump_raw", "relion_iref_dump_recovar", "relion_projector_dump"],
        default="mrc",
        help="Reference volume/state used to score projections.",
    )
    parser.add_argument(
        "--mstep-layout",
        choices=["full", "half", "both"],
        default="half",
        help="Backprojection accumulator layout to test.",
    )
    parser.add_argument(
        "--translation-prior-mode",
        choices=["coarse_expanded", "fine", "none"],
        default="coarse_expanded",
    )
    parser.add_argument(
        "--gridding-mode",
        choices=["core", "legacy", "none"],
        default="core",
    )
    parser.add_argument(
        "--iref-scale-mode",
        choices=["one", "inv_n2", "inv_n3", "n2"],
        default="inv_n2",
    )
    parser.add_argument("--iref-extra-scale", type=float, default=1.0)
    parser.add_argument("--iref-sign", type=float, default=-1.0)
    parser.add_argument(
        "--noise-scale-mode",
        choices=["one", "n4", "inv_n4"],
        default="n4",
    )
    parser.add_argument("--noise-extra-scale", type=float, default=1.0)
    parser.add_argument(
        "--halfset-mode",
        choices=[
            "all",
            "natural_id_parity",
            "micrograph_sorted_position_parity",
            "relion_sorted_position_parity",
            "relion_sorted_value_parity",
        ],
        default="all",
    )
    args = parser.parse_args()

    if args.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = Path("_agent_scratch") / f"initial_model_bpref_diag_{stamp}_{os.getpid()}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.particles_star is None:
        args.particles_star = _infer_particles_star(args.fixture_dir)

    import jax

    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.data_io.starfile import read_star

    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("No JAX GPU device visible")

    ds = load_dataset(str(args.particles_star), lazy=False)
    backend = ds.image_source.backend if hasattr(ds.image_source, "backend") else None
    if backend is not None and hasattr(backend, "set_relion_image_mask"):
        backend.set_relion_image_mask(
            pixel_size=float(ds.voxel_size),
            particle_diameter_ang=544.0,
            width_mask_edge_px=5.0,
        )
    main_in, _ = read_star(str(args.particles_star))
    relion_sorted_idx = _read_relion_sorted_idx(args.relion_dump_dir / "sorted_idx_iter001.bin", int(ds.n_images))

    modes = [
        "natural_id_parity",
        "micrograph_sorted_position_parity",
        "relion_sorted_position_parity",
        "relion_sorted_value_parity",
    ]
    if args.halfset_mode != "all":
        modes = [args.halfset_mode]
    layouts = ["full", "half"] if args.mstep_layout == "both" else [args.mstep_layout]

    all_summary = {
        "out_dir": str(args.out_dir),
        "fixture_dir": str(args.fixture_dir),
        "particles_star": str(args.particles_star),
        "relion_dump_dir": str(args.relion_dump_dir),
        "relion_estep_dump_dir": str(args.relion_estep_dump_dir),
        "reference_source": args.reference_source,
        "mstep_layout": args.mstep_layout,
        "translation_prior_mode": args.translation_prior_mode,
        "gridding_mode": args.gridding_mode,
        "iref_scale_mode": args.iref_scale_mode,
        "iref_extra_scale": float(args.iref_extra_scale),
        "iref_sign": float(args.iref_sign),
        "noise_scale_mode": args.noise_scale_mode,
        "noise_extra_scale": float(args.noise_extra_scale),
        "sparse_pass2": bool(args.sparse_pass2),
        "jax_devices": [str(d) for d in devices],
        "n_images": int(ds.n_images),
        "relion_sorted_idx_available": relion_sorted_idx is not None,
        "relion_sorted_idx_head": (
            [] if relion_sorted_idx is None else [int(x) for x in relion_sorted_idx[:20].tolist()]
        ),
        "relion_scalar_Pmax": _read_raw_scalar(args.relion_dump_dir / "Pmax.bin"),
        "modes": [],
    }

    for layout_name in layouts:
        for mode in modes:
            print(f"[diag] running layout={layout_name} mode={mode}", flush=True)
            summary = run_mode(args, ds, main_in, relion_sorted_idx, mode, args.out_dir, layout_name)
            all_summary["modes"].append(summary)
            direct = summary["direct"]
            swapped = summary["swapped"]
            print(
                "[diag] "
                f"{layout_name}/{mode}: direct data CC h0={direct['h0_bp_data']['cc_all']:+.6f} "
                f"h1={direct['h1_bp_data']['cc_all']:+.6f}; "
                f"direct weight CC h0={direct['h0_bp_weight']['cc_all']:+.6f} "
                f"h1={direct['h1_bp_weight']['cc_all']:+.6f}; "
                f"swapped data h0={swapped['h0_bp_data']['cc_all']:+.6f} "
                f"h1={swapped['h1_bp_data']['cc_all']:+.6f}; "
                f"pmax_mean={summary['pmax_mean']}",
                flush=True,
            )

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_summary, indent=2, sort_keys=True))
    print(f"[diag] wrote {summary_path}")


if __name__ == "__main__":
    main()
