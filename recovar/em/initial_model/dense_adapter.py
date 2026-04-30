"""InitialModel E-step adapter backed by native dense K-class EM.

This is the ab-initio bridge we want long-term: the hidden variable axis is
``class x pose`` inside the dense K-class engine. There is no outer loop over
classes here. Pseudo-halfsets are currently separate dense passes because the
VDAM M-step needs independent halfset BackProjectors; the performance parity
work can fuse those accumulators later without changing this public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from recovar.em.dense_single_volume.k_class import run_dense_k_class_em

from .layout import relion_bpref_frame_scales, run_em_output_to_bpref
from .m_step import VdamAccumulator
from .state import InitialModelState

_ENGINE_DEFAULTS: dict[str, Any] = {
    "current_size": None,
    "projection_padding_factor": None,
    "reconstruction_padding_factor": None,
    "half_spectrum_scoring": True,
    "score_with_masked_images": True,
    "sparse_pass2": False,
}


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
    padding_factor: int = 1
    class_log_priors: Any | None = None
    relion_bpref_frame: bool = True
    engine_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DenseInitialModelEstepResult:
    """Output consumed by ``iteration_loop.run_vdam_iterations``."""

    accumulators: list[VdamAccumulator]
    meta: dict[str, Any]
    halfset_results: dict[int, Any]


def split_pseudo_halfset_particle_ids(
    n_images: int,
    micrograph_names: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return RELION-style pseudo-halfset image indices.

    RELION alternates halfset ids after sorting particles by micrograph name.
    If names are unavailable, natural-order alternation is the best local
    fallback and is exact when the dataset is already RELION-sorted.
    """
    ids = np.arange(int(n_images), dtype=np.int64)
    if micrograph_names is None:
        return ids[0::2], ids[1::2]
    order = np.argsort(np.asarray(micrograph_names), kind="stable")
    return order[0::2].astype(np.int64, copy=False), order[1::2].astype(np.int64, copy=False)


def class_log_priors_from_state(state: InitialModelState) -> np.ndarray:
    """Return finite log class priors from ``state.pdf_class``."""
    weights = np.asarray(state.pdf_class, dtype=np.float64)
    if weights.shape != (state.K,):
        raise ValueError(f"state.pdf_class must have shape ({state.K},), got {weights.shape}")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("state.pdf_class must contain positive finite class probabilities")
    weights = weights / np.sum(weights)
    return np.log(weights)


def _image_groups(
    particle_ids: np.ndarray | None,
    halfset_ids: np.ndarray | None,
    *,
    n_images: int,
    pseudo_halfsets: bool,
) -> list[tuple[int, np.ndarray]]:
    ids = np.arange(n_images, dtype=np.int64) if particle_ids is None else np.asarray(particle_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError(f"particle_ids must be 1D, got {ids.shape}")
    if np.any(ids < 0) or np.any(ids >= n_images):
        raise ValueError("particle_ids contains entries outside the dataset")

    if not pseudo_halfsets:
        return [(0, ids)]

    if halfset_ids is None:
        h0, h1 = ids[0::2], ids[1::2]
    else:
        halves = np.asarray(halfset_ids, dtype=np.int8)
        if halves.shape != ids.shape:
            raise ValueError(f"halfset_ids shape {halves.shape} must match particle_ids shape {ids.shape}")
        if np.any((halves != 0) & (halves != 1)):
            raise ValueError("halfset_ids must contain only 0/1 values")
        h0, h1 = ids[halves == 0], ids[halves == 1]
    return [(0, h0), (1, h1)]


def _dense_engine_kwargs(state: InitialModelState, config: DenseInitialModelEstepConfig) -> dict[str, Any]:
    engine_kwargs = dict(_ENGINE_DEFAULTS)
    engine_kwargs["current_size"] = None if state.current_size <= 0 else state.current_size
    engine_kwargs["projection_padding_factor"] = config.padding_factor
    engine_kwargs["reconstruction_padding_factor"] = config.padding_factor
    engine_kwargs.update(config.engine_kwargs)

    if engine_kwargs["sparse_pass2"]:
        raise NotImplementedError(
            "InitialModel dense E-step currently requires dense K-class pass2; "
            "sparse_pass2 must be disabled until the K-class sparse pass is joint over class x pose."
        )
    if engine_kwargs["projection_padding_factor"] != engine_kwargs["reconstruction_padding_factor"]:
        raise ValueError("InitialModel dense E-step requires matching projection/reconstruction padding factors")
    return engine_kwargs


def reference_to_dense_means(references: np.ndarray) -> np.ndarray:
    """Convert RELION-frame real-space InitialModel references to dense EM means.

    VDAM stores ``Iref`` as RELION real-space volumes. The dense EM engine
    scores centered Fourier volumes in recovar convention. The conversion
    mirrors the fixture-backed InitialModel E-step recipe: apply RELION's
    gridding correction at ``pad=1``, FFT, and use RELION's sign/scale
    convention for backprojector-frame references.
    """
    import jax.numpy as jnp

    from recovar.core import fourier_transform_utils as ftu
    from recovar.reconstruction.relion_functions import griddingCorrect

    refs = np.asarray(references)
    if refs.ndim != 4:
        raise ValueError(f"references must have shape (K, N, N, N), got {refs.shape}")
    n = int(refs.shape[-1])
    means = []
    for ref in refs:
        corrected, _ = griddingCorrect(jnp.asarray(ref), n, padding_factor=1, order=1)
        ft = ftu.get_dft3(corrected).reshape(-1)
        means.append(np.asarray(ft) * (-1.0 / float(n**2)))
    return np.asarray(means, dtype=np.complex64)


def _resolve_class_inputs(
    state: InitialModelState,
    config: DenseInitialModelEstepConfig,
) -> tuple[Any, Any]:
    means = config.means if config.means is not None else reference_to_dense_means(state.Iref)
    if config.mean_variance is not None:
        mean_variance = config.mean_variance
    else:
        mean_variance = np.abs(np.asarray(means)) ** 2
    return means, mean_variance


def _empty_accumulator(state: InitialModelState, class_idx: int, halfset_idx: int) -> VdamAccumulator:
    if state.current_size <= 0:
        r_max = state.ori_size // 2
    else:
        r_max = state.current_size // 2
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


def _result_to_accumulators(
    result,
    state: InitialModelState,
    *,
    halfset_idx: int,
    relion_bpref_frame: bool,
    padding_factor: int,
) -> list[VdamAccumulator]:
    r_max = state.ori_size // 2 if state.current_size <= 0 else state.current_size // 2
    data_scale, weight_scale = (1.0, 1.0)
    if relion_bpref_frame:
        data_scale, weight_scale = relion_bpref_frame_scales(state.ori_size)

    accumulators: list[VdamAccumulator] = []
    for k in range(state.K):
        bp_data, bp_weight = run_em_output_to_bpref(
            np.asarray(result.Ft_y[k]),
            np.asarray(result.Ft_ctf[k]),
            state.ori_size,
            r_max,
            padding_factor=padding_factor,
        )
        accumulators.append(
            VdamAccumulator(
                data=bp_data * data_scale,
                weight=bp_weight * weight_scale,
                class_idx=k,
                halfset_idx=halfset_idx,
            )
        )
    return accumulators


def _estep_meta(halfset_results: dict[int, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {"halfset_ids": tuple(sorted(halfset_results))}
    for h, result in halfset_results.items():
        if getattr(result, "class_posterior_sums", None) is not None:
            meta[f"halfset_{h}_class_posterior_sums"] = np.asarray(result.class_posterior_sums, dtype=np.float64)
        if getattr(result, "class_assignments", None) is not None:
            meta[f"halfset_{h}_class_assignments"] = np.asarray(result.class_assignments, dtype=np.int32)
        stats = getattr(result, "stats", None)
        if stats is not None and getattr(stats, "max_posterior_per_image", None) is not None:
            meta[f"halfset_{h}_pmax_mean"] = float(np.mean(np.asarray(stats.max_posterior_per_image)))
    return meta


def run_dense_initial_model_estep(
    experiment_dataset,
    state: InitialModelState,
    config: DenseInitialModelEstepConfig,
    *,
    particle_ids: np.ndarray | None = None,
    halfset_ids: np.ndarray | None = None,
) -> DenseInitialModelEstepResult:
    """Run the InitialModel E-step and return VDAM accumulators.

    The dense K-class engine sees all classes at once. For pseudo-halfsets we
    run one dense pass per halfset to keep the VDAM moment update exact.
    """
    class_log_priors = (
        class_log_priors_from_state(state) if config.class_log_priors is None else np.asarray(config.class_log_priors)
    )
    groups = _image_groups(
        particle_ids,
        halfset_ids,
        n_images=int(experiment_dataset.n_images),
        pseudo_halfsets=state.pseudo_halfsets,
    )
    engine_kwargs = _dense_engine_kwargs(state, config)
    means, mean_variance = _resolve_class_inputs(state, config)

    halfset_results: dict[int, Any] = {}
    by_halfset: dict[int, list[VdamAccumulator]] = {}
    for halfset_idx, image_indices in groups:
        if image_indices.size == 0:
            by_halfset[halfset_idx] = [_empty_accumulator(state, k, halfset_idx) for k in range(state.K)]
            continue
        result = run_dense_k_class_em(
            experiment_dataset,
            means,
            mean_variance,
            config.noise_variance,
            config.rotations,
            config.translations,
            config.disc_type,
            class_log_priors=class_log_priors,
            image_batch_size=config.image_batch_size,
            rotation_block_size=config.rotation_block_size,
            image_indices=image_indices,
            **engine_kwargs,
        )
        halfset_results[halfset_idx] = result
        by_halfset[halfset_idx] = _result_to_accumulators(
            result,
            state,
            halfset_idx=halfset_idx,
            relion_bpref_frame=config.relion_bpref_frame,
            padding_factor=config.padding_factor,
        )

    accumulators: list[VdamAccumulator] = []
    for halfset_idx in sorted(by_halfset):
        accumulators.extend(by_halfset[halfset_idx])
    return DenseInitialModelEstepResult(
        accumulators=accumulators,
        meta=_estep_meta(halfset_results),
        halfset_results=halfset_results,
    )


def dense_initial_model_expectation_step(
    experiment_dataset,
    config: DenseInitialModelEstepConfig,
) -> Callable[[InitialModelState, np.ndarray, np.ndarray], tuple[list[VdamAccumulator], dict[str, Any]]]:
    """Build an ``iteration_loop`` expectation-step callback."""

    def _expectation_step(state: InitialModelState, particle_ids: np.ndarray, halfset_ids: np.ndarray):
        result = run_dense_initial_model_estep(
            experiment_dataset,
            state,
            config,
            particle_ids=particle_ids,
            halfset_ids=halfset_ids,
        )
        return result.accumulators, result.meta

    return _expectation_step
