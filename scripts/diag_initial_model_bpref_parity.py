#!/usr/bin/env python
"""Diagnose RELION InitialModel iter-1 E-step -> BPref parity.

This intentionally mirrors
``tests/unit/initial_model/test_estep_fixture.py::test_estep_bpref_forward_parity``
but writes reusable diagnostics instead of only asserting a regression floor.

The first question this answers is whether the low BPref CC is caused by the
BackProjector/layout conversion itself, or by upstream choices such as
pseudo-halfset routing and posterior weights.
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


def _read_relion_3d_real_dump(path: Path) -> np.ndarray:
    """Read RELION debug dumps written as ``int32 z, y, x, RFLOAT data``."""

    with open(path, "rb") as f:
        zdim, ydim, xdim = struct.unpack("iii", f.read(12))
        return np.fromfile(f, dtype=np.float64, count=zdim * ydim * xdim).reshape(zdim, ydim, xdim)


def _read_relion_3d_complex_dump(path: Path) -> np.ndarray:
    """Read RELION debug dumps written as ``int32 z, y, x, Complex data``."""

    with open(path, "rb") as f:
        zdim, ydim, xdim = struct.unpack("iii", f.read(12))
        return np.fromfile(f, dtype=np.complex128, count=zdim * ydim * xdim).reshape(zdim, ydim, xdim)


def _read_relion_2d_dump(path: Path, *, complex_values: bool) -> np.ndarray:
    """Read RELION debug dumps written as ``int32 ydim, int32 xdim, data``."""

    with open(path, "rb") as f:
        ydim, xdim = struct.unpack("ii", f.read(8))
        dtype = np.complex128 if complex_values else np.float64
        return np.fromfile(f, dtype=dtype, count=ydim * xdim).reshape(ydim, xdim)


def _read_raw_scalar(path: Path) -> float | None:
    if not path.exists():
        return None
    data = np.fromfile(path, dtype=np.float64)
    if data.size:
        return float(data.reshape(-1)[0])
    data32 = np.fromfile(path, dtype=np.float32)
    return float(data32.reshape(-1)[0]) if data32.size else None


def _infer_particles_star(fixture_dir: Path) -> Path:
    optimiser_star = fixture_dir / "run_it000_optimiser.star"
    if not optimiser_star.exists():
        raise FileNotFoundError(f"cannot infer particles STAR; missing {optimiser_star}")
    text = optimiser_star.read_text()
    match = re.search(r"(?:^|\s)--i\s+(\S+)", text)
    if match is None:
        raise RuntimeError(f"cannot infer particles STAR from {optimiser_star}")
    return Path(match.group(1))


def _configure_relion_image_mask(dataset, *, particle_diameter_ang: float = 544.0, width_mask_edge_px: float = 5.0) -> None:
    from recovar.core import mask as core_mask

    source = dataset.image_source
    backend = getattr(source, "backend", source)
    if backend is None:
        return
    image_mask = core_mask.relion_soft_image_mask(
        int(dataset.grid_size),
        float(dataset.voxel_size),
        float(particle_diameter_ang),
        float(width_mask_edge_px),
    )
    if hasattr(backend, "image_mask"):
        backend.image_mask = image_mask
    if hasattr(backend, "mask"):
        backend.mask = image_mask
    if hasattr(backend, "image_mask_mode"):
        backend.image_mask_mode = "relion_background_fill"


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


def _read_iter0_sigma_offset_angstrom(fixture_dir: Path, estep_dump_dir: Path) -> float:
    sigma2_offset = _read_raw_scalar(estep_dump_dir / "pass0_over0_sigma2_offset.bin")
    if sigma2_offset is None:
        sigma2_offset = _read_raw_scalar(estep_dump_dir / "sigma2_offset.bin")
    if sigma2_offset is not None and sigma2_offset > 0.0:
        return float(np.sqrt(sigma2_offset))
    text = (fixture_dir / "run_it000_model.star").read_text()
    match = re.search(r"_rlnSigmaOffsetsAngst\s+(\S+)", text)
    if match is not None:
        return float(match.group(1))
    raise RuntimeError("could not determine RELION sigma_offset for iter-0 translation prior")


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


def _read_relion_sorted_idx(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.fromfile(path, dtype=np.int64)
    if arr.size == 0:
        return None
    # RELION debug writer stores: n, sentinel, then the n-entry permutation.
    if arr.size >= 2 and int(arr[0]) == arr.size - 2:
        arr = arr[2:]
    elif arr.size >= 1 and int(arr[0]) == arr.size - 1:
        arr = arr[1:]
    return arr


def _read_dumped_perturbation(estep_dump_dir: Path, fallback_sampling_star: Path) -> tuple[float, float, str]:
    """Return RELION's full-precision in-memory perturbation when dumped."""

    for name in ("p0_perturbation.txt", "p1_perturbation.txt", "p2_perturbation.txt"):
        path = estep_dump_dir / name
        if not path.exists():
            continue
        values: dict[str, float] = {}
        for line in path.read_text().splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            try:
                values[key.strip()] = float(value)
            except ValueError:
                continue
        if "random_perturbation" in values:
            return (
                float(values["random_perturbation"]),
                float(values.get("perturbation_factor", 0.0)),
                str(path),
            )

    from recovar.em.sampling import read_relion_perturbation_from_sampling_star

    random_perturbation, perturbation_factor = read_relion_perturbation_from_sampling_star(str(fallback_sampling_star))
    return float(random_perturbation), float(perturbation_factor), str(fallback_sampling_star)


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
    )

    sampling_star = fixture_dir / "run_it001_sampling.star"
    random_perturbation, _perturbation_factor, perturbation_source = _read_dumped_perturbation(
        estep_dump_dir,
        sampling_star,
    )
    eulers_bin = estep_dump_dir / "p0_oversampled_eulers.bin"
    trans_bin = estep_dump_dir / "p0_oversampled_translations.bin"
    base_coarse_translations = get_translation_grid(max_pixel=6, pixel_offset=2).astype(np.float32)
    coarse_translations = apply_relion_translation_perturbation(
        base_coarse_translations.astype(np.float32, copy=False),
        random_perturbation,
        offset_step_pixels=2.0,
    ).astype(np.float32, copy=False)
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
        source = f"relion_dump_p0_oversampled; perturbation={perturbation_source}"
    else:
        coarse_indices = np.arange(48 * 12, dtype=np.int64)
        rotations, _ = get_oversampled_rotation_grid_from_samples(
            coarse_indices,
            parent_nside_level=1,
            oversampling_order=1,
            random_perturbation=random_perturbation,
            index_order="relion_hidden",
        )
        translations, _ = get_oversampled_translation_grid(
            base_coarse_translations,
            pixel_offset=2,
            oversampling_order=1,
        )
        translations = apply_relion_translation_perturbation(
            translations.astype(np.float32),
            random_perturbation,
            offset_step_pixels=2.0,
        )
        source = f"constructed_from_perturbation; perturbation={perturbation_source}"
    return (
        np.asarray(rotations, dtype=np.float32),
        np.asarray(translations, dtype=np.float32),
        np.asarray(coarse_translations, dtype=np.float32),
        float(random_perturbation),
        source,
    )


def _relion_projector_dense_volume_from_dump(ppref: np.ndarray, ori_size: int) -> np.ndarray:
    """Embed RELION ``Projector::data`` into dense full-centered Fourier layout."""

    from recovar.core import fourier_transform_utils as ftu

    ppref = np.asarray(ppref, dtype=np.complex128)
    if ppref.ndim != 3:
        raise ValueError(f"ppref must be 3D, got {ppref.shape}")
    n = int(ori_size)
    if n % 2:
        raise ValueError(f"expected even ori_size, got {ori_size}")
    center = n // 2
    half = np.zeros((n, n, center + 1), dtype=np.complex128)
    slab = ppref[::-1, :, :]
    zdim, ydim, xdim = slab.shape
    z_center = zdim // 2
    y_center = ydim // 2
    if xdim > center + 1:
        raise ValueError(f"ppref x half-axis {xdim} does not fit ori_size={ori_size}")
    for iz in range(zdim):
        for iy in range(ydim):
            half[(iz - z_center) + center, (iy - y_center) + center, :xdim] = slab[iz, iy, :]
    return np.asarray(ftu.half_volume_to_full_volume(half, (n, n, n)), dtype=np.complex128)


def _relion_projector_dense_rotations(rotations: np.ndarray) -> np.ndarray:
    """Map RELION rotation matrices to the dense frame for embedded projector data."""

    rotations = np.asarray(rotations, dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"rotations must have shape (R, 3, 3), got {rotations.shape}")
    swap_xz = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    flip_x = np.diag([-1.0, 1.0, 1.0]).astype(np.float64)
    inv_t = np.linalg.inv(rotations).transpose(0, 2, 1)
    return np.einsum("rij,jk,kl->ril", inv_t, swap_xz, flip_x).astype(np.float32)


def _build_config(args, ds, fixture_dir: Path, estep_dump_dir: Path, current_size: int):
    import jax.numpy as jnp

    from recovar.core import fourier_transform_utils as ftu
    from recovar.em.dense_single_volume.helpers.orientation_priors import make_relion_translation_log_prior
    from recovar.em.sampling import get_translation_grid
    from recovar.em.initial_model.dense_adapter import DenseInitialModelEstepConfig
    from recovar.reconstruction.noise import make_radial_noise
    from recovar.reconstruction.relion_functions import griddingCorrect
    from recovar.utils.helpers import load_relion_volume, relion_volume_to_recovar

    ori = int(ds.grid_size)
    rotations, translations, coarse_translations, random_perturbation, sampling_source = _load_sampling(
        fixture_dir,
        estep_dump_dir,
    )

    reference_frame = "recovar_dense_real_dft"
    relion_projector_shape = None
    if args.reference_source == "relion_iref_dump":
        iref_relion = _read_relion_3d_real_dump(estep_dump_dir / "iref_c0_pre_setup.bin")
        iref_real = np.asarray(relion_volume_to_recovar(iref_relion), dtype=np.float64)
        iref_real_for_ft, _ = griddingCorrect(jnp.asarray(iref_real), ori, padding_factor=1, order=1)
        iref_ft = np.asarray(ftu.get_dft3(jnp.asarray(np.asarray(iref_real_for_ft)))).reshape(-1)
        iref_ft = iref_ft * (-1.0 / (ori**2))
    elif args.reference_source == "relion_projector_dump":
        ppref = _read_relion_3d_complex_dump(estep_dump_dir / "ppref_c0_data_post_setup.bin")
        # Keep RELION Projector::data in its native (z, y, x>=0) layout and
        # use the native projector path in dense scoring. RECOVAR's CTF sign is
        # opposite RELION's dumped CTF convention, so preserve the existing
        # -N^2 scorer-frame scale.
        relion_projector_shape = tuple(int(x) for x in ppref.shape)
        iref_ft = (np.asarray(ppref, dtype=np.complex128) * (-(ori**2))).reshape(-1)
        reference_frame = "relion_projector_data_native_scaled_minus_n2"
    else:
        iref_real = np.asarray(load_relion_volume(str(fixture_dir / "run_it000_class001.mrc")), dtype=np.float64)
        iref_real_for_ft, _ = griddingCorrect(jnp.asarray(iref_real), ori, padding_factor=1, order=1)
        iref_ft = np.asarray(ftu.get_dft3(jnp.asarray(np.asarray(iref_real_for_ft)))).reshape(-1)
        iref_ft = iref_ft * (-1.0 / (ori**2))
    sigma2 = _read_iter0_sigma2(fixture_dir, ori // 2 + 1)
    n4 = ori**4
    noise_variance = np.asarray(make_radial_noise(sigma2 * n4, (ori, ori))).astype(np.float32).reshape(-1)

    sigma_offset_ang = _read_iter0_sigma_offset_angstrom(fixture_dir, estep_dump_dir)
    n_coarse_trans = int(coarse_translations.shape[0])
    n_fine_trans = int(translations.shape[0])
    if n_coarse_trans <= 0 or n_fine_trans % n_coarse_trans != 0:
        raise ValueError(f"cannot infer fine translation parents from {n_fine_trans=} and {n_coarse_trans=}")
    over_trans = n_fine_trans // n_coarse_trans
    fine_translation_parent = np.repeat(np.arange(n_coarse_trans, dtype=np.int32), over_trans)
    coarse_for_prior = get_translation_grid(max_pixel=6, pixel_offset=2).astype(np.float32)

    if args.translation_prior_mode == "none":
        translation_log_prior = None
        coarse_translation_log_prior = None
    elif args.translation_prior_mode == "fine":
        translation_log_prior = make_relion_translation_log_prior(
            translations,
            float(ds.voxel_size),
            sigma_offset_ang,
        )
        coarse_translation_log_prior = make_relion_translation_log_prior(
            coarse_for_prior,
            float(ds.voxel_size),
            sigma_offset_ang,
        )
    elif args.translation_prior_mode == "coarse_expanded":
        coarse_translation_log_prior = make_relion_translation_log_prior(
            coarse_for_prior,
            float(ds.voxel_size),
            sigma_offset_ang,
        )
        translation_log_prior = coarse_translation_log_prior[fine_translation_parent].astype(np.float32, copy=False)
    else:
        raise ValueError(f"unknown translation_prior_mode={args.translation_prior_mode!r}")

    engine_kwargs = {
        "score_with_masked_images": True,
        "reconstruct_with_masked_images": bool(args.reconstruct_with_masked_images),
        "relion_firstiter_score_mode": "gaussian",
        "image_pre_shifts": np.zeros((ds.n_images, 2), dtype=np.float32),
        "sparse_pass2": bool(args.sparse_pass2),
    }
    if relion_projector_shape is not None:
        engine_kwargs["relion_projector_shape"] = relion_projector_shape
    if translation_log_prior is not None:
        engine_kwargs["translation_log_prior"] = translation_log_prior
    if args.sparse_pass2:
        engine_kwargs.update(
            {
                "healpix_order": 1,
                "oversampling_order": 1,
                "translation_step": 2.0,
                "random_perturbation": random_perturbation,
                "coarse_translations": coarse_translations,
                "coarse_translation_log_prior": coarse_translation_log_prior,
                "particle_diameter_ang": 544.0,
                "adaptive_fraction": float(args.adaptive_fraction),
                "max_significants": int(args.max_significants),
                "return_profile": True,
            }
        )

    config = DenseInitialModelEstepConfig(
        means=iref_ft[None, :].astype(np.complex64),
        mean_variance=(np.abs(iref_ft) ** 2)[None, :].astype(np.float32),
        noise_variance=noise_variance,
        rotations=rotations,
        translations=translations,
        image_batch_size=50,
        rotation_block_size=100,
        padding_factor=1,
        relion_bpref_frame=True,
        relion_projector_frame=args.reference_source == "relion_projector_dump",
        engine_kwargs=engine_kwargs,
    )
    meta = {
        "ori_size": ori,
        "current_size": int(current_size),
        "r_max": int(current_size // 2),
        "sampling_source": sampling_source,
        "reference_source": args.reference_source,
        "reference_frame": reference_frame,
        "relion_projector_frame": bool(args.reference_source == "relion_projector_dump"),
        "n_rotations": int(rotations.shape[0]),
        "n_translations": int(translations.shape[0]),
        "n_coarse_translations": int(coarse_translations.shape[0]),
        "translation_prior_mode": args.translation_prior_mode,
        "reconstruct_with_masked_images": bool(args.reconstruct_with_masked_images),
        "sigma_offset_angstrom": float(sigma_offset_ang),
        "sparse_pass2": bool(args.sparse_pass2),
        "adaptive_fraction": float(args.adaptive_fraction),
        "max_significants": int(args.max_significants),
        "translation_log_prior_min": None if translation_log_prior is None else float(np.min(translation_log_prior)),
        "translation_log_prior_max": None if translation_log_prior is None else float(np.max(translation_log_prior)),
    }
    return config, meta


def run_mode(args, ds, main_in, relion_sorted_idx, mode: str, out_dir: Path) -> dict[str, object]:
    from recovar.em.initial_model import initialise_denovo_state
    from recovar.em.initial_model.dense_adapter import run_dense_initial_model_estep

    config, config_meta = _build_config(args, ds, args.fixture_dir, args.relion_estep_dump_dir, args.current_size)
    state = initialise_denovo_state(
        ori_size=int(ds.grid_size),
        pixel_size=float(ds.voxel_size),
        K=1,
        nr_iter=1,
        n_directions=int(config.rotations.shape[0]),
        pseudo_halfsets=True,
    )
    state.current_size = int(args.current_size)

    micrograph_names = np.asarray(main_in["_rlnMicrographName"].astype(str).to_numpy())
    halfset_ids = _halfset_ids_for_mode(mode, int(ds.n_images), micrograph_names, relion_sorted_idx)

    t0 = time.time()
    estep = run_dense_initial_model_estep(
        ds,
        state,
        config,
        particle_ids=np.arange(ds.n_images, dtype=np.int64),
        halfset_ids=halfset_ids,
    )
    elapsed_s = time.time() - t0

    ours_h0 = np.asarray(estep.accumulators[0].data)
    ours_h1 = np.asarray(estep.accumulators[1].data)
    ours_w0 = np.asarray(estep.accumulators[0].weight)
    ours_w1 = np.asarray(estep.accumulators[1].weight)
    target_h0 = _read_bin(args.relion_dump_dir / "pipe_it1_c0_bp_data_pre_reweight.bin")
    target_h1 = _read_bin(args.relion_dump_dir / "pipe_it1_c0_bp_data_h_pre_reweight.bin")
    target_w0 = _read_bin(args.relion_dump_dir / "pipe_it1_c0_bp_weight.bin")
    target_w1 = _read_bin(args.relion_dump_dir / "pipe_it1_c0_bp_weight_h.bin")

    r_max = int(args.current_size // 2)
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
    flip_axis0 = {
        "h0_bp_data": _array_metrics(ours_h0[::-1, :, :], target_h0, r_max),
        "h1_bp_data": _array_metrics(ours_h1[::-1, :, :], target_h1, r_max),
        "h0_bp_weight": _array_metrics(ours_w0[::-1, :, :], target_w0, r_max),
        "h1_bp_weight": _array_metrics(ours_w1[::-1, :, :], target_w1, r_max),
    }
    swapped_flip_axis0 = {
        "h0_bp_data": _array_metrics(ours_h1[::-1, :, :], target_h0, r_max),
        "h1_bp_data": _array_metrics(ours_h0[::-1, :, :], target_h1, r_max),
        "h0_bp_weight": _array_metrics(ours_w1[::-1, :, :], target_w0, r_max),
        "h1_bp_weight": _array_metrics(ours_w0[::-1, :, :], target_w1, r_max),
    }

    result_obj = next(iter(estep.halfset_results.values())) if estep.halfset_results else None
    grouped_Ft_y = None if result_obj is None else getattr(result_obj, "grouped_Ft_y", None)
    grouped_Ft_ctf = None if result_obj is None else getattr(result_obj, "grouped_Ft_ctf", None)

    npz_path = out_dir / f"{mode}_arrays.npz"
    np.savez_compressed(
        npz_path,
        halfset_ids=halfset_ids.astype(np.int8),
        ours_h0_bp_data=ours_h0,
        ours_h1_bp_data=ours_h1,
        ours_h0_bp_weight=ours_w0,
        ours_h1_bp_weight=ours_w1,
        ours_h0_bp_data_flip_axis0=ours_h0[::-1, :, :],
        ours_h1_bp_data_flip_axis0=ours_h1[::-1, :, :],
        ours_h0_bp_weight_flip_axis0=ours_w0[::-1, :, :],
        ours_h1_bp_weight_flip_axis0=ours_w1[::-1, :, :],
        target_h0_bp_data=target_h0,
        target_h1_bp_data=target_h1,
        target_h0_bp_weight=target_w0,
        target_h1_bp_weight=target_w1,
        grouped_Ft_y=np.asarray(grouped_Ft_y) if grouped_Ft_y is not None else np.zeros(0, dtype=np.complex64),
        grouped_Ft_ctf=np.asarray(grouped_Ft_ctf) if grouped_Ft_ctf is not None else np.zeros(0, dtype=np.complex64),
        max_posterior=np.asarray(estep.meta.get("max_posterior_per_image", []), dtype=np.float32),
        selected_particle_ids=np.asarray(estep.meta.get("selected_particle_ids", []), dtype=np.int64),
    )

    summary = {
        "mode": mode,
        "elapsed_s": float(elapsed_s),
        "npz_path": str(npz_path),
        "halfset_counts": {
            "h0": int(np.sum(halfset_ids == 0)),
            "h1": int(np.sum(halfset_ids == 1)),
        },
        "config": config_meta,
        "direct": direct,
        "swapped": swapped,
        "flip_axis0": flip_axis0,
        "swapped_flip_axis0": swapped_flip_axis0,
        "pmax_mean": (
            float(np.mean(np.asarray(estep.meta["max_posterior_per_image"])))
            if "max_posterior_per_image" in estep.meta
            else None
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument(
        "--particles-star",
        type=Path,
        default=None,
        help="Particle STAR. Defaults to the --i path recorded in run_it000_optimiser.star.",
    )
    parser.add_argument("--relion-dump-dir", type=Path, default=DEFAULT_RELION_DUMP_DIR)
    parser.add_argument("--relion-estep-dump-dir", type=Path, default=DEFAULT_RELION_ESTEP_DUMP)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--current-size", type=int, default=28)
    parser.add_argument(
        "--reference-source",
        choices=["mrc", "relion_iref_dump", "relion_projector_dump"],
        default="mrc",
        help=(
            "Reference volume source for dense scoring. relion_iref_dump uses RELION's in-memory Iref dump; "
            "relion_projector_dump uses RELION Projector::data plus the projector-frame rotation bridge."
        ),
    )
    parser.add_argument("--sparse-pass2", action="store_true")
    parser.add_argument("--reconstruct-with-masked-images", action="store_true")
    parser.add_argument("--adaptive-fraction", type=float, default=0.999)
    parser.add_argument("--max-significants", type=int, default=-1)
    parser.add_argument(
        "--translation-prior-mode",
        choices=["fine", "coarse_expanded", "none"],
        default="fine",
        help=(
            "How to apply the RELION offset prior on an oversampled translation grid. "
            "RELION computes offset priors over the active pass translation grid; "
            "use fine for dense/fine pass-2 parity and coarse_expanded only as an ablation."
        ),
    )
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

    import jax

    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.data_io.starfile import read_star

    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("No JAX GPU device visible")

    if args.particles_star is None:
        args.particles_star = _infer_particles_star(args.fixture_dir)

    ds = load_dataset(str(args.particles_star), lazy=False)
    _configure_relion_image_mask(ds)
    main_in, _ = read_star(str(args.particles_star))
    relion_sorted_idx = _read_relion_sorted_idx(args.relion_dump_dir / "sorted_idx_iter001.bin")

    modes = [
        "natural_id_parity",
        "micrograph_sorted_position_parity",
        "relion_sorted_position_parity",
        "relion_sorted_value_parity",
    ]
    if args.halfset_mode != "all":
        modes = [args.halfset_mode]

    all_summary = {
        "out_dir": str(args.out_dir),
        "fixture_dir": str(args.fixture_dir),
        "particles_star": str(args.particles_star),
        "relion_dump_dir": str(args.relion_dump_dir),
        "relion_estep_dump_dir": str(args.relion_estep_dump_dir),
        "jax_devices": [str(d) for d in devices],
        "n_images": int(ds.n_images),
        "relion_sorted_idx_available": relion_sorted_idx is not None,
        "relion_sorted_idx_head": (
            [] if relion_sorted_idx is None else [int(x) for x in relion_sorted_idx[:20].tolist()]
        ),
        "relion_scalar_Pmax": _read_raw_scalar(args.relion_dump_dir / "Pmax.bin"),
        "modes": [],
    }

    for mode in modes:
        print(f"[diag] running mode={mode}", flush=True)
        summary = run_mode(args, ds, main_in, relion_sorted_idx, mode, args.out_dir)
        all_summary["modes"].append(summary)
        direct = summary["direct"]
        swapped = summary["swapped"]
        flip_axis0 = summary["flip_axis0"]
        print(
            "[diag] "
            f"{mode}: direct data CC h0={direct['h0_bp_data']['cc_all']:+.6f} "
            f"h1={direct['h1_bp_data']['cc_all']:+.6f}; "
            f"swapped h0={swapped['h0_bp_data']['cc_all']:+.6f} "
            f"h1={swapped['h1_bp_data']['cc_all']:+.6f}; "
            f"flip0 h0={flip_axis0['h0_bp_data']['cc_all']:+.6f} "
            f"h1={flip_axis0['h1_bp_data']['cc_all']:+.6f}; "
            f"pmax_mean={summary['pmax_mean']}",
            flush=True,
        )

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_summary, indent=2, sort_keys=True))
    print(f"[diag] wrote {summary_path}")


if __name__ == "__main__":
    main()
