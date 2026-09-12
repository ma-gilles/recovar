"""E-step records and accumulator helpers shared by the dense and sparse InitialModel E-steps.

The E-step configuration and result records, the accumulator packing, the
per-image row selection and the metadata assembly live here so the dense
adapter and the sparse pass-2 owner import them without importing each other.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from recovar.em.vdam.layout import relion_bpref_frame_scales, run_em_output_to_bpref
from recovar.em.vdam.mstep_accumulator import VdamAccumulator
from recovar.em.vdam.relion_layout import relion_x_public_output_to_bpref
from recovar.em.vdam.state import InitialModelState

ProjectorSetupBackend = Literal["native", "jax"]


@dataclass(frozen=True, kw_only=True)
class DenseInitialModelEstepConfig:
    """Configuration for one InitialModel dense K-class E-step."""

    means: Any | None = None
    mean_variance: Any | None = None
    noise_variance: Any
    rotations: Any
    translations: Any
    disc_type: str = "linear_interp"
    image_batch_size: int = 500
    rotation_block_size: int = 5000
    pass2_engine: str = "auto"
    relion_wavg_sequential_cuda: bool = True
    exact_local_bucket_radix: int = 4
    exact_local_physical_order_chunk_size: int = 0
    stable_fourier_window_shapes: bool = False
    padding_factor: int = 1
    class_log_priors: Any | None = None
    relion_bpref_frame: bool = True
    relion_projector_frame: bool = False
    projector_setup_backend: ProjectorSetupBackend = "native"
    relion_projector_half_by_class: Any | None = None
    relion_projector_r_max: int | None = None
    engine_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DenseInitialModelEstepResult:
    """Output consumed by ``iteration_loop.run_vdam_iterations``."""

    accumulators: list[VdamAccumulator]
    meta: dict[str, Any]
    halfset_results: dict[int, Any]


def _engine_kwargs_for_image_indices(
    engine_kwargs: dict[str, Any],
    image_indices: np.ndarray,
    *,
    n_images: int,
) -> dict[str, Any]:
    """Slice ``translation_log_prior`` (selected-image axis) for the dense engine."""
    out = dict(engine_kwargs)
    prior = out.get("translation_log_prior")
    if prior is None:
        return out
    prior_np = np.asarray(prior)
    if prior_np.ndim != 2:
        return out

    image_indices = np.asarray(image_indices, dtype=np.int64)
    if prior_np.shape[0] == int(n_images):
        out["translation_log_prior"] = prior_np[image_indices]
    elif prior_np.shape[0] != int(image_indices.size):
        raise ValueError(
            "translation_log_prior must be shared, selected-image, or full-dataset shaped; "
            f"got first axis {prior_np.shape[0]} for {image_indices.size} selected images and {n_images} total images"
        )
    return out


def _select_image_rows(value, image_indices: np.ndarray, *, n_images: int, name: str):
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        return value
    image_indices = np.asarray(image_indices, dtype=np.int64)
    if array.shape[0] == int(n_images):
        return array[image_indices]
    if array.shape[0] == int(image_indices.size):
        return value
    raise ValueError(
        f"{name} must be shared, selected-image, or full-dataset shaped; "
        f"got first axis {array.shape[0]} for {image_indices.size} selected images and {n_images} total images"
    )


def _group_local_kwargs(
    engine_kwargs: dict[str, Any],
    image_indices: np.ndarray,
    *,
    n_images: int,
) -> dict[str, Any]:
    """Return dense/local kwargs in the compact row space of a dataset subset."""

    out = _engine_kwargs_for_image_indices(engine_kwargs, image_indices, n_images=n_images)
    for name in ("image_pre_shifts", "image_corrections", "scale_corrections", "translation_prior_centers"):
        out[name] = _select_image_rows(out.get(name), image_indices, n_images=n_images, name=name)
    return out


def _relion_projector_dense_rotations(rotations: np.ndarray) -> np.ndarray:
    """Map RELION rotation matrices to dense slicing rotations for Projector::data."""

    rotations = np.asarray(rotations, dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"rotations must have shape (R, 3, 3), got {rotations.shape}")
    swap_xz = np.array(
        (
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    flip_x = np.diag((-1.0, 1.0, 1.0)).astype(np.float64)
    inv_t = np.linalg.inv(rotations).transpose(0, 2, 1)
    return np.einsum("rij,jk,kl->ril", inv_t, swap_xz, flip_x).astype(np.float32)


def _add_accumulator_weight_meta(meta: dict[str, Any], accumulators: list[VdamAccumulator], K: int) -> None:
    """Record RELION BPref weight totals used to normalize class prior updates."""

    sums = np.zeros(int(K), dtype=np.float64)
    halfset_sums: dict[int, np.ndarray] = {}
    for accum in accumulators:
        class_idx = int(accum.class_idx)
        halfset_idx = int(accum.halfset_idx)
        value = float(np.sum(np.asarray(accum.weight, dtype=np.float64)))
        sums[class_idx] += value
        halfset_sums.setdefault(halfset_idx, np.zeros(int(K), dtype=np.float64))[class_idx] += value
    meta["class_bpref_weight_sums"] = sums
    for halfset_idx, values in sorted(halfset_sums.items()):
        meta[f"halfset_{halfset_idx}_class_bpref_weight_sums"] = values


def _empty_accumulator(state: InitialModelState, class_idx: int, halfset_idx: int) -> VdamAccumulator:
    r_max = state.ori_size // 2 if state.current_size <= 0 else state.current_size // 2
    if r_max >= state.ori_size // 2:
        shape = (state.ori_size, state.ori_size, state.ori_size // 2 + 1)
    else:
        half_ps = r_max + 1
        shape = (2 * half_ps + 1, 2 * half_ps + 1, half_ps + 1)
    return VdamAccumulator(
        data=np.zeros(shape, dtype=np.complex128),
        weight=np.zeros(shape, dtype=np.float64),
        class_idx=class_idx,
        halfset_idx=halfset_idx,
    )


def _arrays_to_accumulators(
    Ft_y_by_class,
    Ft_ctf_by_class,
    state: InitialModelState,
    *,
    halfset_idx: int | None,
    reconstruction_group_count: int | None = None,
    relion_bpref_frame: bool,
    relion_projector_frame: bool,
    padding_factor: int,
) -> list[VdamAccumulator]:
    r_max = state.ori_size // 2 if state.current_size <= 0 else state.current_size // 2
    data_scale, weight_scale = (1.0, 1.0)
    if relion_bpref_frame:
        data_scale, weight_scale = relion_bpref_frame_scales(state.ori_size)
    dump_dir = os.environ.get("RECOVAR_INITIAL_MODEL_ACCUM_DUMP_DIR")

    grouped = halfset_idx is None
    if grouped:
        if reconstruction_group_count is None or int(reconstruction_group_count) <= 0:
            raise ValueError(
                "reconstruction_group_count is required for grouped accumulators"
            )
        output_halfsets = range(int(reconstruction_group_count))
    else:
        if reconstruction_group_count not in (None, 1):
            raise ValueError(
                "reconstruction_group_count is only valid for grouped accumulators"
            )
        output_halfsets = (int(halfset_idx),)

    accumulators: list[VdamAccumulator] = []
    for k in range(state.K):
        class_data = np.asarray(Ft_y_by_class[k])
        class_weight = np.asarray(Ft_ctf_by_class[k])
        if grouped and (
            class_data.shape[0] != int(reconstruction_group_count)
            or class_weight.shape[0] != int(reconstruction_group_count)
        ):
            raise ValueError(
                "grouped accumulator arrays do not match reconstruction_group_count"
            )
        for output_halfset in output_halfsets:
            public_data = class_data[output_halfset] if grouped else class_data
            public_weight = class_weight[output_halfset] if grouped else class_weight
            converter = relion_x_public_output_to_bpref if relion_bpref_frame else run_em_output_to_bpref
            bp_data, bp_weight = converter(
                public_data,
                public_weight,
                state.ori_size,
                r_max,
                padding_factor=padding_factor,
            )
            if relion_projector_frame and not relion_bpref_frame:
                bp_data = bp_data[::-1, :, :]
                bp_weight = bp_weight[::-1, :, :]
            if dump_dir:
                path = Path(dump_dir)
                path.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    path / f"accum_h{int(output_halfset)}_k{int(k)}.npz",
                    Ft_y=np.asarray(public_data),
                    Ft_ctf=np.asarray(public_weight),
                    bp_data_unscaled=np.asarray(bp_data),
                    bp_weight_unscaled=np.asarray(bp_weight),
                    bp_data_scaled=np.asarray(bp_data * data_scale),
                    bp_weight_scaled=np.asarray(bp_weight * weight_scale),
                    data_scale=np.float64(data_scale),
                    weight_scale=np.float64(weight_scale),
                    relion_projector_frame=np.bool_(relion_projector_frame),
                    relion_bpref_frame=np.bool_(relion_bpref_frame),
                    padding_factor=np.int32(padding_factor),
                    ori_size=np.int32(state.ori_size),
                    current_size=np.int32(state.current_size),
                )
            accumulators.append(
                VdamAccumulator(
                    data=bp_data * data_scale,
                    weight=bp_weight * weight_scale,
                    class_idx=k,
                    halfset_idx=int(output_halfset),
                )
            )
    return accumulators


def _estep_meta(halfset_results: dict[int, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {"halfset_ids": tuple(sorted(halfset_results))}
    class_posterior_sums = None
    class_posterior_sums_full = None
    class_reconstruction_support_sums = None
    class_direction_posterior_sums = None
    noise_totals: dict[str, Any] | None = None
    for h, result in halfset_results.items():
        if getattr(result, "class_posterior_sums", None) is not None:
            full_sums = np.asarray(result.class_posterior_sums, dtype=np.float64)
            sums = np.asarray(
                getattr(result, "class_mstep_posterior_sums", full_sums),
                dtype=np.float64,
            )
            meta[f"halfset_{h}_class_posterior_sums"] = sums
            meta[f"halfset_{h}_class_posterior_sums_full"] = full_sums
            class_posterior_sums = sums if class_posterior_sums is None else class_posterior_sums + sums
            class_posterior_sums_full = (
                full_sums if class_posterior_sums_full is None else class_posterior_sums_full + full_sums
            )
        per_class_noise = getattr(result, "noise_stats", None)
        if per_class_noise is not None:
            support = np.asarray([float(stats.sumw) for stats in per_class_noise], dtype=np.float64)
            meta[f"halfset_{h}_class_reconstruction_support_sums"] = support
            class_reconstruction_support_sums = (
                support if class_reconstruction_support_sums is None else class_reconstruction_support_sums + support
            )
        if getattr(result, "class_assignments", None) is not None:
            meta[f"halfset_{h}_class_assignments"] = np.asarray(result.class_assignments, dtype=np.int32)
        stats = getattr(result, "stats", None)
        if stats is not None and getattr(stats, "max_posterior_per_image", None) is not None:
            meta[f"halfset_{h}_pmax_mean"] = float(np.mean(np.asarray(stats.max_posterior_per_image)))
        profile_summary = getattr(result, "profile_summary", None)
        if profile_summary is not None:
            meta[f"halfset_{h}_profile_summary"] = dict(profile_summary)
        noise_stats = getattr(result, "aggregate_noise_stats", None)
        if noise_stats is not None:
            half = {
                "wsum_sigma2_offset": float(noise_stats.wsum_sigma2_offset),
                "sigma2_offset_sumw": float(noise_stats.sumw),
                "wsum_sigma2_noise": np.asarray(noise_stats.wsum_sigma2_noise, dtype=np.float64),
                "wsum_img_power": np.asarray(noise_stats.wsum_img_power, dtype=np.float64),
                "noise_sumw": float(noise_stats.sumw),
            }
            if getattr(noise_stats, "wsum_noise_a2", None) is not None:
                half["wsum_noise_a2"] = np.asarray(noise_stats.wsum_noise_a2, dtype=np.float64)
            if getattr(noise_stats, "wsum_noise_xa", None) is not None:
                half["wsum_noise_xa"] = np.asarray(noise_stats.wsum_noise_xa, dtype=np.float64)
            for k, v in half.items():
                meta[f"halfset_{h}_{k}"] = v
            if noise_totals is None:
                noise_totals = dict(half)
            else:
                for k, v in half.items():
                    noise_totals[k] = noise_totals.get(k, 0.0) + v
        per_class_stats = getattr(result, "per_class_stats", None)
        if per_class_stats is not None:
            direction_sums = np.stack(
                [np.asarray(cs.rotation_posterior_sums, dtype=np.float64) for cs in per_class_stats],
                axis=0,
            )
            class_direction_posterior_sums = (
                direction_sums
                if class_direction_posterior_sums is None
                else class_direction_posterior_sums + direction_sums
            )
    if class_posterior_sums is not None:
        meta["class_posterior_sums"] = class_posterior_sums
    if class_posterior_sums_full is not None:
        meta["class_posterior_sums_full"] = class_posterior_sums_full
    if class_reconstruction_support_sums is not None:
        meta["class_reconstruction_support_sums"] = class_reconstruction_support_sums
    if class_direction_posterior_sums is not None:
        meta["class_direction_posterior_sums"] = class_direction_posterior_sums
    if noise_totals is not None:
        meta.update(noise_totals)
    return meta
