"""InitialModel E-step adapter on the dense K-class engine.

The hidden variable axis is ``class x pose``; pseudo-halfsets share one E-step
(reconstruction accumulators are split per halfset) so projection/scoring isn't
duplicated while the VDAM M-step still gets independent halfset BackProjectors.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from recovar.em.classification.k_class import run_dense_k_class_em
from recovar.em.vdam.estep_common import (
    _PARTICLE_RESULT_FIELDS,
    DenseInitialModelEstepConfig,
    DenseInitialModelEstepResult,
    ProjectorSetupBackend,
    _add_accumulator_weight_meta,
    _arrays_to_accumulators,
    _empty_accumulator,
    _estep_meta,
    _group_local_kwargs,
    _relion_projector_dense_rotations,
)
from recovar.em.vdam.sparse_pass2_estep import _run_sparse_pass2_initial_model_estep
from recovar.em.vdam.state import InitialModelState, VdamAccumulator

_ENGINE_DEFAULTS: dict[str, Any] = {
    "current_size": None,
    "projection_padding_factor": None,
    "reconstruction_padding_factor": None,
    "half_spectrum_scoring": True,
    "score_with_masked_images": True,
    "reconstruct_with_masked_images": True,
    "sparse_pass2": False,
    # RELION InitialModel BPref uses the rounded radial reconstruction support
    # encoded by Minvsigma2, not the full square Fourier crop.
    "recon_square_window": False,
    "recon_exact_radius": False,
    "reconstruction_subtract_projected_reference": True,
    # RELION InitialModel scores the full rounded Fourier crop emitted by its
    # CUDA projector, including the few crop-corner pixels outside r_max.
    "projection_mask_current_image_disk": False,
}
_INACTIVE_CLASS_LOG_PRIOR = -1.0e30
_EXACT_RELION_PROJECTOR_ENV = "RECOVAR_INITIAL_MODEL_EXACT_RELION_PROJECTOR"
_RELION_PROJECTOR_DUMP_DIR_ENV = "RECOVAR_INITIAL_MODEL_PROJECTOR_DUMP_DIR"


logger = logging.getLogger(__name__)


def class_log_priors_from_state(state: InitialModelState) -> np.ndarray:
    """Log class priors from ``state.pdf_class`` (collapsed classes get a finite sentinel)."""
    weights = np.asarray(state.pdf_class, dtype=np.float64)
    if weights.shape != (state.K,):
        raise ValueError(f"state.pdf_class must have shape ({state.K},), got {weights.shape}")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("state.pdf_class must contain non-negative finite class probabilities")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("state.pdf_class must contain at least one positive class probability")
    out = np.full(state.K, _INACTIVE_CLASS_LOG_PRIOR, dtype=np.float64)
    positive = weights > 0.0
    out[positive] = np.log(weights[positive] / total)
    return out


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

    controlled = ("image_indices", "reconstruction_group_ids", "reconstruction_group_count")
    present = sorted(name for name in controlled if name in config.engine_kwargs)
    if present:
        raise ValueError(f"InitialModel dense E-step controls these dense-engine arguments: {', '.join(present)}")
    if engine_kwargs["projection_padding_factor"] != engine_kwargs["reconstruction_padding_factor"]:
        raise ValueError("InitialModel dense E-step requires matching projection/reconstruction padding factors")
    return engine_kwargs


_DENSE_RUN_EM_REJECT = frozenset(
    {
        # InitialModel-only kwargs the dense run_em wrapper doesn't accept.
        "reconstruct_with_masked_images",
        "recon_square_window",
        "recon_exact_radius",
        "reconstruction_subtract_projected_reference",
        "projection_mask_current_image_disk",
        "relion_projector_shape",
        # Sparse/local engine kwargs that run_dense_k_class_em rejects.
        "return_profile",
        "return_best_pose_details",
        "return_stats",
        "disable_adjoint_y",
        "disable_adjoint_ctf",
        "normalization_log_evidence",
    }
)


def _dense_run_em_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs unsupported by the dense run_em wrapper (sparse-only escape hatch)."""
    return {k: v for k, v in kwargs.items() if k not in _DENSE_RUN_EM_REJECT}


# Kept as this module's patch point for tests; the rule lives with the preprocessing helpers.


def _relion_projector_to_dense_volume(projector_data: np.ndarray, ori_size: int) -> np.ndarray:
    """Embed cropped RELION ``Projector::data`` half-complex slab into dense full-centered Fourier cube.

    Out-of-range coordinates are clipped (any-size input accepted; high frequencies truncated).
    """
    import recovar.core.fourier_transform_utils as ftu

    ppref = np.asarray(projector_data, dtype=np.complex128)
    if ppref.ndim != 3:
        raise ValueError(f"projector_data must be 3D, got {ppref.shape}")
    n = int(ori_size)
    center = n // 2
    half = np.zeros((n, n, center + 1), dtype=np.complex128)
    slab = ppref[::-1, :, :]
    z_center = slab.shape[0] // 2
    y_center = slab.shape[1] // 2
    x_max = min(slab.shape[2], center + 1)
    for iz in range(slab.shape[0]):
        z = (iz - z_center) + center
        if z < 0 or z >= n:
            continue
        for iy in range(slab.shape[1]):
            y = (iy - y_center) + center
            if 0 <= y < n:
                half[z, y, :x_max] = slab[iz, iy, :x_max]
    return np.asarray(ftu.half_volume_to_full_volume(half, (n, n, n)), dtype=np.complex128)


def reference_to_relion_projector_half_maps(
    references: np.ndarray,
    *,
    current_size: int,
    padding_factor: int = 1,
    interpolator: int = 1,
    projector_setup_backend: ProjectorSetupBackend = "native",
) -> tuple[np.ndarray, int]:
    """Convert references to RELION half maps without retaining their spectrum."""
    half_maps, _power, r_max = reference_to_relion_projector_half_maps_and_power(
        references,
        current_size=current_size,
        padding_factor=padding_factor,
        interpolator=interpolator,
        projector_setup_backend=projector_setup_backend,
    )
    return half_maps, r_max


def reference_to_relion_projector_half_maps_and_power(
    references: np.ndarray,
    *,
    current_size: int,
    padding_factor: int = 1,
    interpolator: int = 1,
    projector_setup_backend: ProjectorSetupBackend = "native",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Convert references to native-layout half maps and their corrected spectrum.

    The opt-in JAX backend keeps its FP64 FFT at full capacity as current_size
    changes. Only the logical crop and complex64 consumer conversion vary.
    Unsupported projector geometry retains the native implementation.
    """
    from recovar.utils.helpers import recovar_volume_to_relion

    if projector_setup_backend not in {"native", "jax"}:
        raise ValueError(f"Unknown projector_setup_backend: {projector_setup_backend!r}")
    refs = np.asarray(references)
    if refs.ndim != 4:
        raise ValueError(f"references must have shape (K, N, N, N), got {refs.shape}")
    n = int(refs.shape[-1])
    use_jax = (
        projector_setup_backend == "jax"
        and n > 0 and n % 2 == 0
        and refs.shape[1:] == (n, n, n)
        and int(padding_factor) in {1, 2}
        and int(interpolator) == 1
    )
    if use_jax:
        import jax
        import jax.numpy as jnp

        from recovar.em.relion.relion_projector_setup import setup_relion_projector
    else:
        from recovar.relion_bind import _relion_bind_core as bind

    halves = []
    power_spectra = []
    r_max_values = []
    for ref in refs:
        ref_relion = np.asarray(recovar_volume_to_relion(ref), dtype=np.float64)
        if use_jax:
            # Projector::initialiseData uses a negative size for full resolution;
            # zero means radius zero here (state wrappers retain their defaults).
            r_max = n // 2 if int(current_size) < 0 else min(int(current_size) // 2, n // 2)
            projector_data, power = setup_relion_projector(
                ref_relion, np.int32(r_max), ori_size=n,
                padding_factor=int(padding_factor),
            )
            logical_size = 2 * (int(padding_factor) * r_max + 1) + 1
            start = projector_data.shape[0] // 2 - logical_size // 2
            projector_data = projector_data[
                start : start + logical_size, start : start + logical_size,
                : logical_size // 2 + 1,
            ].astype(jnp.complex64)
            projector_data, power = jax.device_get((projector_data, power))
        else:
            (
                projector_data, power, _ori_size, _padding_factor_out,
                r_max, _r_min_nn, _interpolator_out,
            ) = bind.compute_fourier_transform_map(
                ref_relion,
                n,
                int(padding_factor),
                int(interpolator),
                int(current_size),
                True,
                2,
            )
        halves.append(np.asarray(projector_data))
        power_spectra.append(np.asarray(power, dtype=np.float64))
        r_max_values.append(int(r_max))
    if len(set(r_max_values)) != 1:
        raise ValueError(f"RELION projector maps disagree on r_max: {r_max_values}")
    return (
        np.asarray(halves),
        np.asarray(power_spectra, dtype=np.float64),
        int(r_max_values[0]),
    )


def relion_projector_half_maps_to_dense_means(projector_half_maps: np.ndarray, ori_size: int) -> np.ndarray:
    """Embed RELION ``Projector::data`` maps into dense recovar scoring volumes."""

    n = int(ori_size)
    means = []
    for projector_data in np.asarray(projector_half_maps):
        dense = _relion_projector_to_dense_volume(np.asarray(projector_data), n)
        # RECOVAR_DENSE_MEANS_SCALE diag override (see project_k2_c2_cc_root_cause_2026_05_03).
        tok = (os.environ.get("RECOVAR_DENSE_MEANS_SCALE") or "-N2").strip()
        scale = {"-N2": -(n**2), "N2": float(n**2)}.get(tok)
        if scale is None:
            scale = float(tok)
        means.append(dense.reshape(-1) * scale)
    return np.asarray(means, dtype=np.complex64)


def _dense_rotations_for_config(rotations: Any, config: DenseInitialModelEstepConfig) -> np.ndarray:
    rotations_np = np.asarray(rotations, dtype=np.float32)
    if not config.relion_projector_frame:
        return rotations_np
    return _relion_projector_dense_rotations(rotations_np)


# (attr_name_on_result, dtype) for fields harvested per halfset and concatenated.


def reference_to_dense_means(references: np.ndarray) -> np.ndarray:
    """Convert recovar-frame InitialModel references to unnormalised centered FFTs for dense scoring."""
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
        means.append(np.asarray(ftu.get_dft3(corrected).reshape(-1)))
    return np.asarray(means, dtype=np.complex64)


def prepare_relion_projector_class_inputs(
    state: InitialModelState,
    *,
    padding_factor: int,
    projector_setup_backend: ProjectorSetupBackend = "native",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build InitialModel's production RELION projector once per iteration."""
    projector_half_by_class, projector_r_max = reference_to_relion_projector_half_maps(
        state.Iref,
        current_size=state.current_size if state.current_size > 0 else state.ori_size,
        padding_factor=padding_factor,
        projector_setup_backend=projector_setup_backend,
    )
    return _finish_relion_projector_class_inputs(
        state, padding_factor, projector_half_by_class, projector_r_max
    )


def prepare_relion_projector_class_inputs_and_power(
    state: InitialModelState,
    *,
    padding_factor: int,
    projector_setup_backend: ProjectorSetupBackend = "native",
    interpolator: int = 1,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]:
    """Produce scoring operands and tau2 from the identical corrected FFT."""
    half_maps, power, r_max = reference_to_relion_projector_half_maps_and_power(
        state.Iref,
        current_size=state.current_size if state.current_size > 0 else state.ori_size,
        padding_factor=padding_factor,
        projector_setup_backend=projector_setup_backend,
        interpolator=interpolator,
    )
    inputs = _finish_relion_projector_class_inputs(state, padding_factor, half_maps, r_max)
    return inputs, power


def _finish_relion_projector_class_inputs(
    state: InitialModelState,
    padding_factor: int,
    projector_half_by_class: np.ndarray,
    projector_r_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    projector_dump_dir = os.environ.get(_RELION_PROJECTOR_DUMP_DIR_ENV, "").strip()
    if projector_dump_dir:
        os.makedirs(projector_dump_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(projector_dump_dir, f"iter{int(state.iter):03d}_relion_projector_half.npz"),
            projector_half=np.asarray(projector_half_by_class),
            projector_r_max=np.int64(projector_r_max),
            current_size=np.int64(
                state.current_size if state.current_size > 0 else state.ori_size
            ),
            padding_factor=np.int64(padding_factor),
            iteration=np.int64(state.iter),
        )
    means = relion_projector_half_maps_to_dense_means(
        projector_half_by_class,
        int(state.ori_size),
    )
    mean_variance = np.abs(np.asarray(means)) ** 2
    return means, mean_variance, projector_half_by_class, int(projector_r_max)


def _resolve_class_inputs(
    state: InitialModelState,
    config: DenseInitialModelEstepConfig,
) -> tuple[Any, Any, np.ndarray | None, int | None]:
    relion_projector_half_by_class = None
    relion_projector_r_max = None
    if config.relion_projector_half_by_class is not None:
        if config.relion_projector_r_max is None:
            raise ValueError(
                "relion_projector_r_max is required with relion_projector_half_by_class"
            )
        relion_projector_half_by_class = np.asarray(config.relion_projector_half_by_class)
        relion_projector_r_max = int(config.relion_projector_r_max)
        means = (
            config.means
            if config.means is not None
            else relion_projector_half_maps_to_dense_means(
                relion_projector_half_by_class,
                int(state.ori_size),
            )
        )
    elif config.means is not None:
        means = config.means
    elif config.relion_projector_frame:
        means, _prepared_variance, projector_half_by_class, projector_r_max = (
            prepare_relion_projector_class_inputs(
                state,
                padding_factor=config.padding_factor,
                projector_setup_backend=config.projector_setup_backend,
            )
        )
        exact_projector_setting = os.environ.get(_EXACT_RELION_PROJECTOR_ENV, "1").strip().lower()
        if exact_projector_setting not in {"0", "false", "no", "off"}:
            relion_projector_half_by_class = projector_half_by_class
            relion_projector_r_max = projector_r_max
    else:
        means = reference_to_dense_means(state.Iref)
    mean_variance = config.mean_variance if config.mean_variance is not None else np.abs(np.asarray(means)) ** 2
    return means, mean_variance, relion_projector_half_by_class, relion_projector_r_max


def run_dense_initial_model_estep(
    experiment_dataset,
    state: InitialModelState,
    config: DenseInitialModelEstepConfig,
    *,
    particle_ids: np.ndarray | None = None,
    halfset_ids: np.ndarray | None = None,
) -> DenseInitialModelEstepResult:
    """Run the InitialModel E-step with RELION-compatible pseudo-halfset routing."""
    class_log_priors = (
        class_log_priors_from_state(state) if config.class_log_priors is None else np.asarray(config.class_log_priors)
    )
    groups = _image_groups(
        particle_ids,
        halfset_ids,
        n_images=int(experiment_dataset.n_images),
        pseudo_halfsets=state.pseudo_halfsets,
    )
    selected_particle_ids = (
        np.arange(int(experiment_dataset.n_images), dtype=np.int64)
        if particle_ids is None
        else np.asarray(particle_ids, dtype=np.int64)
    )
    if state.pseudo_halfsets:
        selected_halfset_ids = (
            np.arange(selected_particle_ids.size, dtype=np.int32) % 2
            if halfset_ids is None
            else np.asarray(halfset_ids, dtype=np.int32)
        )
    else:
        selected_halfset_ids = None
    engine_kwargs = _dense_engine_kwargs(state, config)
    means, mean_variance, relion_projector_half_by_class, relion_projector_r_max = _resolve_class_inputs(
        state,
        config,
    )
    if bool(engine_kwargs.get("sparse_pass2", False)):
        return _run_sparse_pass2_initial_model_estep(
            experiment_dataset,
            state,
            config,
            class_log_priors=class_log_priors,
            groups=groups,
            joint_particle_ids=selected_particle_ids,
            joint_halfset_ids=selected_halfset_ids,
            means=means,
            mean_variance=mean_variance,
            relion_projector_half_by_class=relion_projector_half_by_class,
            relion_projector_r_max=relion_projector_r_max,
            engine_kwargs=engine_kwargs,
        )

    # Sparse execution constructs its own coarse/local rotation operands.
    if config.rotations is None:
        raise ValueError("Dense execution requires materialized rotations")
    dense_rotations = _dense_rotations_for_config(config.rotations, config)
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
            dense_rotations,
            config.translations,
            config.disc_type,
            class_log_priors=class_log_priors,
            image_batch_size=config.image_batch_size,
            rotation_block_size=config.rotation_block_size,
            image_indices=image_indices,
            accumulate_noise=True,
            **_dense_run_em_kwargs(
                _group_local_kwargs(
                    engine_kwargs,
                    image_indices,
                    n_images=int(experiment_dataset.n_images),
                )
            ),
        )
        halfset_results[halfset_idx] = result
        by_halfset[halfset_idx] = _arrays_to_accumulators(
            result.Ft_y,
            result.Ft_ctf,
            state,
            halfset_idx=halfset_idx,
            relion_bpref_frame=config.relion_bpref_frame,
            relion_projector_frame=config.relion_projector_frame,
            padding_factor=config.padding_factor,
        )

    accumulators: list[VdamAccumulator] = []
    for halfset_idx in sorted(by_halfset):
        accumulators.extend(by_halfset[halfset_idx])

    selected_particle_ids: list[np.ndarray] = []
    field_lists: dict[str, list[np.ndarray]] = {attr: [] for attr, _ in _PARTICLE_RESULT_FIELDS}
    max_posterior: list[np.ndarray] = []
    for halfset_idx, image_indices in groups:
        result = halfset_results.get(halfset_idx)
        if result is None:
            continue
        attrs = {attr: getattr(result, attr, None) for attr, _ in _PARTICLE_RESULT_FIELDS}
        stats = getattr(result, "stats", None)
        pmax = None if stats is None else getattr(stats, "max_posterior_per_image", None)
        if pmax is None and all(v is None for v in attrs.values()):
            continue
        selected_particle_ids.append(np.asarray(image_indices, dtype=np.int64))
        for attr, dtype in _PARTICLE_RESULT_FIELDS:
            if attrs[attr] is not None:
                field_lists[attr].append(np.asarray(attrs[attr], dtype=dtype))
        if pmax is not None:
            max_posterior.append(np.asarray(pmax, dtype=np.float32))

    meta = _estep_meta(halfset_results)
    _add_accumulator_weight_meta(meta, accumulators, state.K)
    if selected_particle_ids:
        meta["selected_particle_ids"] = np.concatenate(selected_particle_ids).astype(np.int64, copy=False)
    for attr, dtype in _PARTICLE_RESULT_FIELDS:
        if field_lists[attr]:
            meta[attr] = np.concatenate(field_lists[attr]).astype(dtype, copy=False)
    if max_posterior:
        meta["max_posterior_per_image"] = np.concatenate(max_posterior).astype(np.float32, copy=False)
    return DenseInitialModelEstepResult(
        accumulators=accumulators,
        meta=meta,
        halfset_results=halfset_results,
    )
