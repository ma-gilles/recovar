#!/usr/bin/env python3
"""Capture one RECOVAR K=1 fine-score boundary on native RELION support.

This diagnostic consumes a completed RECOVAR iteration boundary and a native
RELION fine-score/factor capture.  It evaluates only the selected particle and
only the coarse pose pairs that RELION retained.  The resulting pass-2 NPZ is
written by RECOVAR's production dump path, including exact raw-score operands.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from recovar.core import fourier_transform_utils as ftu
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed
from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_spec,
)
from recovar.em.dense_single_volume.helpers.oversampling import (
    compute_pass2_stats_sparse,
)
from recovar.em.dense_single_volume.iteration_loop import (
    _relion_projector_half_maps_for_scoring,
)
from recovar.reconstruction import noise as reconstruction_noise
from recovar.utils.helpers import load_mrc
from scripts.analyze_k1_exact_ppref_fine_boundary import _load_ppref
from scripts.analyze_em_k1_live_reference_counterfactual import (
    relion_values_on_recovar_window,
)
from scripts.run_full_refinement import _maybe_apply_relion_image_mask
from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import (
    ACTIVE,
    load_fine_score_capture,
)
from scripts.validate_relion_preprocess_capture import (
    load_artifact as load_preprocess_capture,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class _NativeScoreWindowDataset:
    """Forward a dataset while replacing only masked score-window Fourier rows."""

    def __init__(self, base, score_indices: np.ndarray, score_values: np.ndarray):
        self._base = base
        self._score_indices = np.asarray(score_indices, dtype=np.int32)
        self._score_values = np.asarray(score_values, dtype=np.complex64)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def process_images_half(self, images, apply_image_mask=False, **kwargs):
        processed = self._base.process_images_half(
            images,
            apply_image_mask=apply_image_mask,
            **kwargs,
        )
        if not apply_image_mask:
            return processed
        expected_half_size = int(
            self._base.image_shape[0] * (self._base.image_shape[1] // 2 + 1)
        )
        _require(
            processed.shape[0] == 1 and processed.shape[1] == expected_half_size,
            "focused preprocessing override requires one full half-spectrum image",
        )
        return jax.numpy.asarray(processed).at[:, self._score_indices].set(
            jax.numpy.asarray(self._score_values[None], dtype=jax.numpy.complex64)
        )


def _constant_by_key(
    keys: np.ndarray,
    values: np.ndarray,
    *,
    size: int,
    name: str,
) -> np.ndarray:
    result = np.zeros(int(size), dtype=np.float32)
    keys = np.asarray(keys, dtype=np.int64).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    _require(keys.shape == values.shape, f"{name} key/value shapes differ")
    _require(
        np.all((keys >= 0) & (keys < int(size))),
        f"{name} key is outside [0, {int(size)})",
    )
    for key in np.unique(keys):
        selected = values[keys == key]
        _require(
            np.all(selected.view(np.uint32) == selected[0].view(np.uint32)),
            f"{name} is not bitwise constant for key {int(key)}",
        )
        result[int(key)] = selected[0]
    return result


def native_support_geometry(
    *,
    factor,
    score,
    physical_image_size: int,
    n_coarse_rotations: int,
    n_coarse_translations: int,
) -> dict[str, np.ndarray]:
    """Translate a native active table into RECOVAR sparse-pass geometry."""

    active = score.candidates[(score.candidates["flags"] & ACTIVE) != 0]
    _require(active.size > 0, "native fine-score capture has no active candidates")
    rotation_local = np.asarray(active["rotation_local"], dtype=np.int64)
    _require(
        np.all((rotation_local >= 0) & (rotation_local < factor.rotations.size)),
        "native active rotation-local index is out of range",
    )
    factor_rotation_keys = np.asarray(
        factor.rotations["orientation_class_key"],
        dtype=np.int64,
    )
    factor_children = np.asarray(
        factor.rotations["oversampled_rotation"],
        dtype=np.int64,
    )
    _require(
        np.array_equal(
            np.asarray(active["rotation_id"], dtype=np.int64),
            factor_rotation_keys[rotation_local],
        ),
        "native fine-score rotation ids disagree with factor geometry",
    )
    _require(
        np.all((factor_rotation_keys >= 0) & (factor_rotation_keys < n_coarse_rotations)),
        "native coarse rotation key is outside the RECOVAR grid",
    )
    _require(
        np.all((factor_children >= 0) & (factor_children < 8)),
        "native oversampled rotation child is outside [0, 8)",
    )

    n_fine_rotations = int(n_coarse_rotations) * 8
    fine_rotations = np.zeros((n_fine_rotations, 3, 3), dtype=np.float32)
    fine_rotation_present = np.zeros(n_fine_rotations, dtype=bool)
    native_matrices = (
        np.asarray(
            factor.rotations["matrix"],
            dtype=np.float32,
        )
        .reshape(-1, 3, 3)
        .transpose(0, 2, 1)
    )
    global_fine_rotation = factor_rotation_keys * 8 + factor_children
    _require(
        np.unique(global_fine_rotation).size == factor.rotations.size,
        "native factor rotations do not map one-to-one to global fine rows",
    )
    fine_rotations[global_fine_rotation] = native_matrices
    fine_rotation_present[global_fine_rotation] = True

    native_translation_ids = np.asarray(
        factor.translations["translation"],
        dtype=np.int64,
    )
    _require(
        np.array_equal(native_translation_ids, np.arange(factor.translations.size)),
        "native fine translations are not stored in translation-id order",
    )
    _require(
        factor.translations.size == int(n_coarse_translations) * 4,
        "native fine translation count is not four children per coarse row",
    )
    native_phase = np.column_stack((factor.translations["x"], factor.translations["y"])).astype(np.float64)
    fine_translations = -native_phase * float(physical_image_size) / (2.0 * np.pi)
    fine_translation_parent = native_translation_ids // 4

    coarse_rotation = np.asarray(active["rotation_id"], dtype=np.int64)
    coarse_translation = np.asarray(active["coarse_translation"], dtype=np.int64)
    _require(
        np.all((coarse_translation >= 0) & (coarse_translation < n_coarse_translations)),
        "native coarse translation key is outside the RECOVAR grid",
    )
    support_pairs = np.unique(
        np.column_stack((coarse_rotation, coarse_translation)),
        axis=0,
    )
    support = (support_pairs[:, 0] * int(n_coarse_translations) + support_pairs[:, 1]).astype(np.int32)

    selected_fine_rows = np.unique(global_fine_rotation[rotation_local])
    _require(
        np.all(fine_rotation_present[selected_fine_rows]),
        "native active rotations are missing from the focused fine grid",
    )
    rotation_log_prior = _constant_by_key(
        coarse_rotation,
        active["orientation_log_prior"],
        size=n_coarse_rotations,
        name="orientation log prior",
    )
    translation_log_prior = _constant_by_key(
        coarse_translation,
        active["translation_log_prior"],
        size=n_coarse_translations,
        name="translation log prior",
    )

    return {
        "support": support,
        "support_pairs": support_pairs.astype(np.int64),
        "fine_rotations": fine_rotations,
        "fine_rotation_parent": np.arange(n_fine_rotations, dtype=np.int64) // 8,
        "fine_rotation_present": fine_rotation_present,
        "fine_translations": fine_translations,
        "fine_translation_parent": fine_translation_parent.astype(np.int32),
        "rotation_log_prior": rotation_log_prior,
        "translation_log_prior": translation_log_prior,
        "native_active_count": np.asarray(active.size, dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--recovar-run", type=Path, required=True)
    parser.add_argument("--boundary-index", type=int, required=True)
    parser.add_argument("--consumer-iteration", type=int, required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--half", type=int, choices=(1, 2), required=True)
    parser.add_argument("--current-size", type=int, required=True)
    parser.add_argument("--native-factor", type=Path, required=True)
    parser.add_argument("--native-fine-score", type=Path, required=True)
    parser.add_argument(
        "--normalization-factor-override",
        type=float,
        help="Optional exact float32 image-normalization factor for a one-particle counterfactual.",
    )
    parser.add_argument(
        "--native-ppref",
        type=Path,
        help="Optional exact native Projector::data slab for a reference-only counterfactual.",
    )
    parser.add_argument(
        "--native-preprocess-capture",
        type=Path,
        help=(
            "Diagnostic-only replacement of the masked score-window Fourier "
            "operand with an exact native RELION preprocessing capture."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--particle-diameter-ang", type=float, default=200.0)
    parser.add_argument(
        "--native-lane-softmask-reduction",
        action="store_true",
        help="Replay RELION's lane-across-blocks soft-mask reduction tree.",
    )
    parser.add_argument(
        "--native-atomic-softmask-reduction",
        action="store_true",
        help="Diagnostic replay of RELION's schedule-dependent atomic soft-mask reduction.",
    )
    args = parser.parse_args()

    _require(
        not (
            args.native_lane_softmask_reduction
            and args.native_atomic_softmask_reduction
        ),
        "native-lane and native-atomic soft-mask reductions are mutually exclusive",
    )

    _require(args.consumer_iteration == args.boundary_index + 1, "consumer must follow boundary")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _require(
        not any(args.output_dir.iterdir()),
        "focused output directory must be empty",
    )

    data_star = args.data_dir / "particles.star"
    results_path = args.recovar_run / "output/refinement_results.npz"
    parity_path = args.recovar_run / f"parity/iter_{args.boundary_index:03d}.npz"
    consumer_parity_path = args.recovar_run / f"parity/iter_{args.consumer_iteration:03d}.npz"
    intermediate_dir = args.recovar_run / "output/intermediates"
    zero_based_intermediate = args.boundary_index - 1
    _require(zero_based_intermediate >= 0, "boundary index must be positive")

    ds = load_dataset(str(data_star), lazy=False)
    _maybe_apply_relion_image_mask(
        ds,
        SimpleNamespace(
            particle_diameter_ang=float(args.particle_diameter_ang),
            width_mask_edge_px=5.0,
        ),
    )
    image_backend = ds.image_source.backend
    _require(
        hasattr(image_backend, "set_relion_fourier_backend"),
        "dataset image backend cannot select RELION CUDA preprocessing",
    )
    image_backend.set_relion_fourier_backend("relion_cuda")
    if args.native_lane_softmask_reduction:
        _require(
            hasattr(image_backend, "set_relion_native_lane_reduction"),
            "dataset image backend cannot select native-lane soft-mask reduction",
        )
        image_backend.set_relion_native_lane_reduction(True)
    if args.native_atomic_softmask_reduction:
        os.environ["RECOVAR_RELION_NATIVE_ATOMIC_SOFTMASK_REDUCTION"] = "1"
    _require(0 <= args.source_index < ds.n_units, "source index is outside the dataset")
    subset = ds.subset(np.asarray([args.source_index], dtype=np.int64))

    factor = load_factor_capture(args.native_factor)
    score = load_fine_score_capture(args.native_fine_score)
    _require(factor.stack_index == args.source_index + 1, "factor capture identity changed")
    _require(score.stack_index == args.source_index + 1, "score capture identity changed")
    _require(
        int(score.header[4]) == args.consumer_iteration,
        "score capture iteration changed",
    )

    native_preprocess_source = None
    if args.native_preprocess_capture is not None:
        native_preprocess = load_preprocess_capture(args.native_preprocess_capture)
        _require(
            native_preprocess.stack_index == args.source_index + 1,
            "preprocessing capture identity changed",
        )
        _require(
            native_preprocess.iteration == args.consumer_iteration,
            "preprocessing capture iteration changed",
        )
        n_half = int(ds.image_shape[0] * (ds.image_shape[1] // 2 + 1))
        window_spec = make_fourier_window_spec(
            ds.image_shape,
            args.current_size,
            n_half,
            square=False,
            include_recon_window=True,
        )
        score_indices = np.asarray(window_spec.score_indices_np, dtype=np.int32)
        native_window = relion_values_on_recovar_window(
            np.asarray(
                native_preprocess.masked_fourier_post_optics[0],
                dtype=np.complex64,
            ).reshape(1, -1),
            score_indices,
            full_image_size=int(ds.image_shape[0]),
            current_size=args.current_size,
        )[0]
        native_window_internal = np.asarray(
            native_window * np.float32(int(ds.image_shape[0]) ** 2),
            dtype=np.complex64,
        )
        subset = _NativeScoreWindowDataset(
            subset,
            score_indices,
            native_window_internal,
        )
        native_preprocess_source = str(args.native_preprocess_capture.resolve())

    coarse_rotations = np.load(intermediate_dir / f"it{args.consumer_iteration - 1:03d}_rotations.npy")
    coarse_translations = np.load(intermediate_dir / f"it{args.consumer_iteration - 1:03d}_translations.npy")
    geometry = native_support_geometry(
        factor=factor,
        score=score,
        physical_image_size=int(ds.image_shape[0]),
        n_coarse_rotations=int(coarse_rotations.shape[0]),
        n_coarse_translations=int(coarse_translations.shape[0]),
    )

    with np.load(parity_path, allow_pickle=False) as parity:
        _require(int(parity["relion_iteration"]) == args.boundary_index, "parity boundary changed")
        original_indices = np.asarray(
            parity[f"half{args.half}_original_image_indices"],
            dtype=np.int64,
        )
        rows = np.flatnonzero(original_indices == args.source_index)
        _require(rows.size == 1, "source particle is not unique in the requested half")
        row = int(rows[0])
        image_corrections = np.asarray(
            [parity[f"half{args.half}_image_corrections"][row]],
            dtype=np.float64,
        )
        scale_corrections = np.asarray(
            [parity[f"half{args.half}_scale_corrections"][row]],
            dtype=np.float64,
        )
    image_correction_source = f"parity/iter_{args.boundary_index:03d}.npz"
    if args.normalization_factor_override is not None:
        image_corrections = np.asarray(
            [np.float32(args.normalization_factor_override)],
            dtype=np.float64,
        )
        image_correction_source = "--normalization-factor-override (float32)"
    with np.load(consumer_parity_path, allow_pickle=False) as consumer_parity:
        _require(
            int(consumer_parity["relion_iteration"]) == args.consumer_iteration,
            "consumer parity iteration changed",
        )
        consumer_original_indices = np.asarray(
            consumer_parity[f"half{args.half}_original_image_indices"],
            dtype=np.int64,
        )
        consumer_rows = np.flatnonzero(consumer_original_indices == args.source_index)
        _require(consumer_rows.size == 1, "source particle is not unique at the consumer boundary")
        image_pre_shifts = np.asarray(
            consumer_parity[f"half{args.half}_translation_search_base"][
                int(consumer_rows[0]) : int(consumer_rows[0]) + 1
            ],
            dtype=np.float32,
        )

    map_path = intermediate_dir / (f"it{zero_based_intermediate:03d}_half{args.half}_reg.mrc")
    map_real = load_mrc(str(map_path)).astype(np.float32)
    mean_ft = np.asarray(ftu.get_dft3(jax.numpy.asarray(map_real))).astype(np.complex64).reshape(-1)
    mean_variance = np.load(intermediate_dir / f"it{zero_based_intermediate:03d}_tau2.npy")
    noise_key = f"noise_radial_per_half_iter_{zero_based_intermediate:03d}"
    with np.load(results_path, allow_pickle=True) as results:
        _require(noise_key in results.files, f"refinement results miss {noise_key}")
        noise_radial = np.asarray(results[noise_key], dtype=np.float64)
    _require(noise_radial.shape[0] == 2, "post-update noise must contain two halves")
    noise_variance = np.asarray(
        reconstruction_noise.make_radial_noise(
            noise_radial[args.half - 1],
            ds.image_shape,
        ),
        dtype=np.float64,
    ).reshape(-1)
    os.environ["RECOVAR_RELION_PROJECTOR_DUMP_DIR"] = str(args.output_dir.resolve())
    projector_dump_path = (
        args.output_dir
        / f"focused_stack{args.source_index + 1}_it{args.consumer_iteration}_relion_projector_half.npz"
    )
    if args.native_ppref is None:
        projector, projector_rmax = _relion_projector_half_maps_for_scoring(
            mean_ft,
            volume_shape=ds.volume_shape,
            current_size=args.current_size,
            padding_factor=2,
            n_classes=1,
            real_references=map_real[None],
            dump_label=f"focused_stack{args.source_index + 1}_it{args.consumer_iteration}",
        )
        projector = np.asarray(projector)
        if projector.ndim == 4 and projector.shape[0] == 1:
            projector = projector[0]
        projector_source = str(map_path.resolve())
    else:
        projector, projector_metadata = _load_ppref(args.native_ppref.resolve())
        _require(
            int(projector_metadata["iteration"]) == args.consumer_iteration,
            "native PPref iteration differs from the consumer iteration",
        )
        _require(
            int(projector_metadata["current_size"]) == args.current_size,
            "native PPref current size differs",
        )
        _require(
            int(projector_metadata["padding_factor"]) == 2,
            "native PPref padding factor differs",
        )
        projector_rmax = int(projector_metadata["r_max"])
        projector_source = str(args.native_ppref.resolve())
        np.savez_compressed(
            projector_dump_path,
            projector_half=np.asarray(projector, dtype=np.complex64)[None],
            projector_r_max=np.asarray(projector_rmax, dtype=np.int64),
            current_size=np.asarray(args.current_size, dtype=np.int64),
            padding_factor=np.asarray(2, dtype=np.int64),
            volume_shape=np.asarray(ds.volume_shape, dtype=np.int64),
            n_classes=np.asarray(1, dtype=np.int64),
        )
    _require(
        projector_dump_path.is_file() and projector_dump_path.stat().st_size > 0,
        "focused RECOVAR PPref dump missing",
    )

    os.environ["RECOVAR_PASS2_DUMP_DIR"] = str(args.output_dir.resolve())
    os.environ["RECOVAR_PASS2_DUMP_ORIGINAL_INDICES"] = str(args.source_index)
    os.environ["RECOVAR_PASS2_DUMP_CURRENT_SIZE"] = str(args.current_size)
    os.environ["RECOVAR_PASS2_DUMP_ITERATION"] = str(args.consumer_iteration)
    os.environ["RECOVAR_PASS2_DUMP_RAW_OPERANDS"] = "1"
    os.environ["RECOVAR_SPARSE_PASS2_PROJECTION_CACHE"] = "off"
    os.environ["RECOVAR_RELION_FINE_ROTATION_EXECUTION_ORDER"] = "1"
    os.environ["RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS"] = "1"
    os.environ["RECOVAR_K1_RELION_EXACT_CTF_STAR"] = str(data_star.resolve())

    sparse_pass2_bucketed.set_bpref_contribution_dump_context(
        iteration=args.consumer_iteration,
        half=args.half,
    )
    try:
        compute_pass2_stats_sparse(
            subset,
            mean_ft,
            mean_variance,
            noise_variance,
            coarse_translations,
            [geometry["support"]],
            nside_level=3,
            disc_type="linear_interp",
            oversampling_order=1,
            current_size=args.current_size,
            rotation_log_prior=geometry["rotation_log_prior"],
            score_with_masked_images=True,
            return_stats=True,
            translation_log_prior=geometry["translation_log_prior"][None],
            accumulate_noise=False,
            half_spectrum_scoring=True,
            projection_padding_factor=2,
            reconstruction_padding_factor=2,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            use_float64_scoring=False,
            do_gridding_correction=True,
            square_window=False,
            random_perturbation=0.0,
            return_score_log_z=True,
            fine_rotations_override=geometry["fine_rotations"],
            fine_mstep_rotations_override=geometry["fine_rotations"],
            fine_rotation_parent_override=geometry["fine_rotation_parent"],
            fine_translations_override=geometry["fine_translations"],
            fine_translation_parent_override=geometry["fine_translation_parent"],
            relion_x_half_mstep=True,
            relion_fine_mstep_prune=True,
            relion_firstiter_score_mode="gaussian",
            relion_firstiter_winner_take_all=False,
            relion_exact_fine_gaussian=True,
            relion_projector_half=projector,
            relion_projector_r_max=projector_rmax,
            adaptive_fraction=0.999,
        )
    finally:
        sparse_pass2_bucketed.clear_bpref_contribution_dump_context()

    dump_path = args.output_dir / (f"pass2_orig{args.source_index:06d}_cs{args.current_size:03d}.npz")
    _require(dump_path.is_file() and dump_path.stat().st_size > 0, "focused pass-2 dump missing")
    report = {
        "schema": "recovar.em.k1_focused_native_support.v1",
        "status": "complete",
        "source_index_zero_based": args.source_index,
        "stack_index_one_based": args.source_index + 1,
        "half_one_based": args.half,
        "boundary_iteration": args.boundary_index,
        "consumer_iteration": args.consumer_iteration,
        "current_size": args.current_size,
        "native_active_candidate_count": int(geometry["native_active_count"]),
        "native_support_pair_count": int(geometry["support"].size),
        "exact_relion_bpref_operands": True,
        "image_fourier_backend": "relion_cuda",
        "native_lane_softmask_reduction": bool(
            args.native_lane_softmask_reduction
        ),
        "native_atomic_softmask_reduction": bool(
            args.native_atomic_softmask_reduction
        ),
        "native_preprocess_score_window_source": native_preprocess_source,
        "noise_source": f"refinement_results.npz:{noise_key}:half{args.half}",
        "image_pre_shift_source": f"parity/iter_{args.consumer_iteration:03d}.npz",
        "image_correction": float(image_corrections[0]),
        "image_correction_source": image_correction_source,
        "projector_dump_path": str(projector_dump_path.resolve()),
        "projector_source": projector_source,
        "dump_path": str(dump_path.resolve()),
        "device": str(jax.devices()[0]),
    }
    (args.output_dir / "FOCUSED_CAPTURE.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
