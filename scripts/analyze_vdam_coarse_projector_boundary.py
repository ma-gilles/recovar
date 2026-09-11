#!/usr/bin/env python3
"""Replay a RECOVAR coarse boundary with RELION's captured in-memory PPref."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovar import cuda_backproject  # noqa: E402
from recovar.em.dense_single_volume.helpers.projection import (  # noqa: E402
    compute_relion_projector_projections_block,
    relion_projector_half_to_texture_full,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_translation_angles_f32,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_scoring import (
    _relion_cuda_fine_full_to_compact_lookup,
)
from scripts.validate_relion_coarse_lane_capture import (  # noqa: E402
    possible_atomic_sums,
)


def _flat_dump(path: Path, dtype: np.dtype) -> np.ndarray:
    """Read a count-prefixed RELION debug vector."""

    path = Path(path)
    with path.open("rb") as stream:
        count = np.fromfile(stream, dtype=np.int32, count=1)
        if count.size != 1 or int(count[0]) < 0:
            raise ValueError(f"invalid RELION vector header: {path}")
        values = np.fromfile(stream, dtype=dtype, count=int(count[0]))
        trailing = stream.read(1)
    if values.size != int(count[0]) or trailing:
        raise ValueError(f"invalid RELION vector payload: {path}")
    return values


def _load_projector(native_directory: Path, prefix: str) -> tuple[np.ndarray, int]:
    root = Path(native_directory) / prefix
    dims = _flat_dump(Path(f"{root}dims.bin"), np.dtype("<i4")).astype(np.int64)
    if dims.size != 7:
        raise ValueError(f"RELION PPref dims must have 7 entries, got {dims.size}")
    xdim, ydim, zdim, _xinit, _yinit, _zinit, r_max = map(int, dims)
    real = _flat_dump(Path(f"{root}real.bin"), np.dtype("<f8"))
    imag = _flat_dump(Path(f"{root}imag.bin"), np.dtype("<f8"))
    if real.size != imag.size or real.size != xdim * ydim * zdim:
        raise ValueError("RELION PPref payload does not match its dimensions")
    projector = (real + np.complex128(1j) * imag).astype(np.complex64)
    return projector.reshape(zdim, ydim, xdim), r_max


def _load_captured_recovar_projector(
    coarse: dict[str, np.ndarray],
) -> tuple[np.ndarray, int, int] | None:
    """Load the exact in-memory RECOVAR projector saved by a targeted dump."""

    values = np.asarray(
        coarse.get("relion_projector_half_per_class", np.empty((0,), np.complex64)),
        dtype=np.complex64,
    )
    if values.size == 0:
        return None
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValueError(
            "captured RECOVAR projector must have shape (1, z, y, x), got "
            f"{values.shape}",
        )
    r_max = int(np.asarray(coarse["relion_projector_r_max"]).reshape(-1)[0])
    padding_factor = int(
        np.asarray(coarse["projection_padding_factor"]).reshape(-1)[0]
    )
    if r_max < 0 or padding_factor < 1:
        raise ValueError(
            "captured RECOVAR projector metadata is invalid: "
            f"r_max={r_max}, padding_factor={padding_factor}",
        )
    return values[0], r_max, padding_factor


def _centered_metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.shape != candidate.shape:
        raise ValueError(f"score shape mismatch: {reference.shape} != {candidate.shape}")
    reference = reference - np.min(reference)
    candidate = candidate - np.min(candidate)
    residual = candidate.astype(np.float64) - reference.astype(np.float64)
    return {
        "count": int(reference.size),
        "exact_count": int(np.count_nonzero(reference == candidate)),
        "max_abs": float(np.max(np.abs(residual))),
        "rms": float(np.sqrt(np.mean(residual * residual))),
        "p95_abs": float(np.percentile(np.abs(residual), 95.0)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _complex_metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    reference = np.asarray(reference, dtype=np.complex64)
    candidate = np.asarray(candidate, dtype=np.complex64)
    if reference.shape != candidate.shape:
        raise ValueError(f"complex shape mismatch: {reference.shape} != {candidate.shape}")
    residual = candidate.astype(np.complex128) - reference.astype(np.complex128)
    denominator = float(np.linalg.norm(reference.astype(np.complex128).reshape(-1)))
    return {
        "count": int(reference.size),
        "exact_count": int(np.count_nonzero(reference == candidate)),
        "max_abs": float(np.max(np.abs(residual))),
        "relative_l2": float(np.linalg.norm(residual.reshape(-1)) / denominator),
    }


def _native_current_fft_rows(*, full_size: int, current_size: int) -> np.ndarray:
    logical_y = np.where(
        np.arange(current_size) <= current_size // 2,
        np.arange(current_size),
        np.arange(current_size) - current_size,
    )
    return (
        (logical_y[:, None] + full_size // 2) * (full_size // 2 + 1)
        + np.arange(current_size // 2 + 1)[None, :]
    ).astype(np.int32).reshape(-1)


def _coarse_lane_cutoff_report(
    *,
    scores_with_prior: np.ndarray,
    lane_partials: np.ndarray,
    initial_diff2: np.float32,
    production_diff2: np.ndarray,
    fused_diff2: np.ndarray,
    native_diff2: np.ndarray,
    max_significants: int,
) -> dict[str, object]:
    """Classify a cutoff pair against every legal four-lane atomic order."""

    total_scores = np.asarray(scores_with_prior, dtype=np.float32)
    if total_scores.ndim != 2:
        raise ValueError(f"scores_with_prior must be (R,T), got {total_scores.shape}")
    n_rotations, n_translations = total_scores.shape
    lanes = np.asarray(lane_partials, dtype=np.float32)
    if lanes.shape != (n_rotations, 128):
        raise ValueError(
            f"lane_partials must have shape {(n_rotations, 128)}, got {lanes.shape}"
        )
    if max_significants < 1 or max_significants >= total_scores.size:
        raise ValueError(f"invalid max_significants={max_significants}")
    active_lanes = 128 // n_translations
    if active_lanes < 1 or active_lanes > 8:
        raise ValueError(
            f"refusing atomic enumeration for {active_lanes} active lanes"
        )

    order = np.argsort(-total_scores.reshape(-1), kind="stable")
    flats = order[[max_significants - 1, max_significants]].astype(np.int64)
    raw_arrays = {
        "production": np.asarray(production_diff2, dtype=np.float32).reshape(-1),
        "fused_replay": np.asarray(fused_diff2, dtype=np.float32).reshape(-1),
        "native": np.asarray(native_diff2, dtype=np.float32).reshape(-1),
    }
    for name, values in raw_arrays.items():
        if values.size != total_scores.size:
            raise ValueError(
                f"{name} diff2 has {values.size} values, expected {total_scores.size}"
            )

    candidates = []
    possible_total_scores = []
    for rank, flat in zip((max_significants, max_significants + 1), flats):
        rotation, translation = divmod(int(flat), n_translations)
        thread_ids = translation + np.arange(active_lanes) * n_translations
        partials = lanes[rotation, thread_ids]
        outcomes = possible_atomic_sums(partials, initial=np.float32(initial_diff2))
        outcome_bits = set(np.asarray(outcomes).view(np.uint32).astype(int).tolist())
        prior = np.subtract(
            total_scores[rotation, translation],
            -raw_arrays["production"][flat],
            dtype=np.float32,
        )
        total_outcomes = np.add(-outcomes, prior, dtype=np.float32)
        possible_total_scores.append(total_outcomes)
        candidates.append(
            {
                "rank": int(rank),
                "flat_index": int(flat),
                "rotation_index": int(rotation),
                "translation_index": int(translation),
                "thread_ids": thread_ids.astype(int).tolist(),
                "lane_partials": partials.astype(float).tolist(),
                "atomic_outcome_count": int(outcomes.size),
                "atomic_diff2_min": float(np.min(outcomes)),
                "atomic_diff2_max": float(np.max(outcomes)),
                "production_diff2": float(raw_arrays["production"][flat]),
                "production_reachable": int(
                    raw_arrays["production"][flat].view(np.uint32)
                )
                in outcome_bits,
                "fused_replay_diff2": float(raw_arrays["fused_replay"][flat]),
                "fused_replay_reachable": int(
                    raw_arrays["fused_replay"][flat].view(np.uint32)
                )
                in outcome_bits,
                "native_diff2": float(raw_arrays["native"][flat]),
                "native_diff2_reachable_from_candidate_lanes": int(
                    raw_arrays["native"][flat].view(np.uint32)
                )
                in outcome_bits,
                "candidate_log_prior": float(prior),
            }
        )

    gap_outcomes = np.subtract(
        possible_total_scores[0][:, None],
        possible_total_scores[1][None, :],
        dtype=np.float32,
    ).reshape(-1)
    gap_bits = set(gap_outcomes.view(np.uint32).astype(int).tolist())
    observed_gaps = {}
    priors = np.asarray(
        [candidate["candidate_log_prior"] for candidate in candidates],
        dtype=np.float32,
    )
    for name, values in raw_arrays.items():
        observed_total = np.add(-values[flats], priors, dtype=np.float32)
        gap = np.subtract(observed_total[0], observed_total[1], dtype=np.float32)
        residual = np.abs(gap_outcomes.astype(np.float64) - float(gap))
        observed_gaps[name] = {
            "rank_maxsig_minus_next": float(gap),
            "exactly_reachable_from_candidate_lanes": int(gap.view(np.uint32))
            in gap_bits,
            "nearest_legal_gap_abs_difference": float(np.min(residual)),
        }

    return {
        "max_significants": int(max_significants),
        "active_lanes_per_translation": int(active_lanes),
        "candidates": candidates,
        "legal_total_score_gap_count": int(np.unique(gap_outcomes.view(np.uint32)).size),
        "legal_total_score_gap_min": float(np.min(gap_outcomes)),
        "legal_total_score_gap_max": float(np.max(gap_outcomes)),
        "observed_total_score_gaps": observed_gaps,
    }


def _fine_projector_checks(
    native_directory: Path,
    projector: np.ndarray,
    *,
    r_max: int,
    physical_image_size: int,
) -> dict[str, dict[str, float | int]]:
    native_directory = Path(native_directory)
    eulers = _flat_dump(
        native_directory / "pass1_class0_fine_eulers.bin", np.dtype("<f8")
    ).astype(np.float32).reshape(-1, 3, 3)
    rotation_indices = _flat_dump(
        native_directory / "pass1_acc_rot_idx.bin", np.dtype("<i4")
    ).astype(np.int32)
    current_size_values = np.fromfile(
        native_directory / "pass1_img0_exp_current_image_size.bin", dtype=np.float64
    )
    if current_size_values.shape != (1,):
        raise ValueError("native fine current-size scalar has the wrong shape")
    fine_size = int(round(float(current_size_values[0])))
    native_pixel_count = fine_size * (fine_size // 2 + 1)
    native_real = _flat_dump(
        native_directory / "pass1_class0_fine_ref_real.bin", np.dtype("<f8")
    )
    native_imag = _flat_dump(
        native_directory / "pass1_class0_fine_ref_imag.bin", np.dtype("<f8")
    )
    if native_real.size != rotation_indices.size * native_pixel_count:
        raise ValueError("native fine-reference payload has the wrong topology")
    scale = np.float32(physical_image_size**2)
    native_reference = -scale * (
        native_real.astype(np.float32) + np.complex64(1j) * native_imag.astype(np.float32)
    ).reshape(rotation_indices.size, native_pixel_count)
    rows = _native_current_fft_rows(
        full_size=physical_image_size,
        current_size=fine_size,
    )
    comparisons = {}
    for label, rotations in (
        ("native_eulers", eulers),
        ("native_eulers_transposed", np.swapaxes(eulers, -1, -2)),
    ):
        projected, _ = compute_relion_projector_projections_block(
            jnp.asarray(projector, dtype=jnp.complex64),
            jnp.asarray(rotations, dtype=jnp.float32),
            (physical_image_size, physical_image_size),
            r_max=int(r_max),
            padding_factor=1,
            return_abs2=False,
            centered_rows=True,
            dense_scale=True,
            projector_output_size=fine_size,
            relion_texture_interp=True,
            mask_current_image_disk=False,
        )
        projected = np.asarray(jax.block_until_ready(projected), dtype=np.complex64)
        replay_reference = projected[:, rows][rotation_indices]
        comparisons[label] = _complex_metric(native_reference, replay_reference)
    return comparisons


def analyze(
    native_directory: Path,
    recovar_coarse_dump: Path,
    *,
    projector_prefix: str = "pass1_class0_ppref_",
    native_score_name: str = "pass0_coarse_raw_diff2.bin",
    physical_image_size: int = 128,
    debug_npz: Path | None = None,
) -> dict[str, object]:
    projector, r_max = _load_projector(native_directory, projector_prefix)
    with np.load(recovar_coarse_dump, allow_pickle=False) as payload:
        coarse = {name: np.asarray(payload[name]) for name in payload.files}
    recovar_projector_capture = _load_captured_recovar_projector(coarse)

    current_size = int(np.asarray(coarse["current_size"]).reshape(-1)[0])
    rotations = np.asarray(coarse["rotations"], dtype=np.float32)
    translations = np.asarray(coarse["translation_phase_source"], dtype=np.float32)
    score_indices = np.asarray(coarse["coarse_gaussian_score_indices"], dtype=np.int32)
    unshifted = np.asarray(coarse["coarse_gaussian_unshifted_corrected"], dtype=np.complex64)
    pixel_weight = np.asarray(coarse["coarse_gaussian_pixel_weight"], dtype=np.float32)
    initial_diff2 = np.asarray(coarse["coarse_gaussian_initial_diff2"], dtype=np.float32)

    projected, _ = compute_relion_projector_projections_block(
        jnp.asarray(projector, dtype=jnp.complex64),
        jnp.asarray(rotations, dtype=jnp.float32),
        (physical_image_size, physical_image_size),
        r_max=int(r_max),
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        dense_scale=True,
        projector_output_size=current_size,
        relion_texture_interp=True,
    )
    projected = projected[:, jnp.asarray(score_indices, dtype=jnp.int32)]
    translation_angles = _relion_translation_angles_f32(
        translations,
        (physical_image_size, physical_image_size),
    )
    shifted = cuda_backproject.relion_translate_score_f32(
        jnp.asarray(unshifted[None], dtype=jnp.complex64),
        jnp.asarray(translation_angles, dtype=jnp.float32),
        jnp.asarray(score_indices, dtype=jnp.int32),
        (physical_image_size, physical_image_size),
    )
    full_to_compact = _relion_cuda_fine_full_to_compact_lookup(
        (physical_image_size, physical_image_size),
        current_size,
        score_indices,
    )
    replay_diff2 = cuda_backproject.relion_coarse_diff2_rectangular_f32(
        jnp.asarray(projected, dtype=jnp.complex64),
        jnp.asarray(shifted[None], dtype=jnp.complex64),
        jnp.asarray(pixel_weight[None], dtype=jnp.float32),
        jnp.asarray(initial_diff2.reshape(1), dtype=jnp.float32),
        jnp.asarray(full_to_compact, dtype=jnp.int32),
    )
    fused_replay_diff2 = cuda_backproject.relion_coarse_diff2_projector_f32(
        relion_projector_half_to_texture_full(
            jnp.asarray(projector, dtype=jnp.complex64)
        ),
        jnp.asarray(rotations, dtype=jnp.float32),
        jnp.asarray(unshifted[None], dtype=jnp.complex64),
        jnp.asarray(translation_angles, dtype=jnp.float32),
        jnp.asarray(pixel_weight[None], dtype=jnp.float32),
        jnp.asarray(initial_diff2.reshape(1), dtype=jnp.float32),
        jnp.asarray(full_to_compact, dtype=jnp.int32),
        current_size=current_size,
        physical_image_size=physical_image_size,
        model_max_r=r_max,
    )
    projected, shifted, replay_diff2, fused_replay_diff2 = (
        np.asarray(value)
        for value in jax.block_until_ready(
            (projected, shifted, replay_diff2, fused_replay_diff2)
        )
    )
    recovar_projector_fused_replay_diff2 = None
    recovar_projector_fused_lane_partials = None
    if recovar_projector_capture is not None:
        recovar_projector, recovar_r_max, recovar_padding_factor = (
            recovar_projector_capture
        )
        if recovar_padding_factor != 1:
            raise ValueError(
                "fused RECOVAR projector replay currently requires padding factor 1, "
                f"got {recovar_padding_factor}",
            )
        (
            recovar_projector_fused_replay_diff2,
            recovar_projector_fused_lane_partials,
        ) = (
            np.asarray(value)
            for value in jax.block_until_ready(
                cuda_backproject.relion_coarse_diff2_projector_lanes_f32(
                    relion_projector_half_to_texture_full(
                        jnp.asarray(recovar_projector, dtype=jnp.complex64)
                    ),
                    jnp.asarray(rotations, dtype=jnp.float32),
                    jnp.asarray(unshifted[None], dtype=jnp.complex64),
                    jnp.asarray(translation_angles, dtype=jnp.float32),
                    jnp.asarray(pixel_weight[None], dtype=jnp.float32),
                    jnp.asarray(initial_diff2.reshape(1), dtype=jnp.float32),
                    jnp.asarray(full_to_compact, dtype=jnp.int32),
                    current_size=current_size,
                    physical_image_size=physical_image_size,
                    model_max_r=recovar_r_max,
                )
            )
        )
        recovar_projector_fused_replay_diff2 = np.asarray(
            recovar_projector_fused_replay_diff2,
            dtype=np.float32,
        ).reshape(-1)
        recovar_projector_fused_lane_partials = np.asarray(
            recovar_projector_fused_lane_partials,
            dtype=np.float32,
        ).reshape(rotations.shape[0], 128)

    native_diff2 = _flat_dump(
        Path(native_directory) / native_score_name,
        # RELION's AccPtr diagnostic promotes XFLOAT payloads to binary64.
        np.dtype("<f8"),
    ).astype(np.float32)
    replay_diff2 = np.asarray(replay_diff2, dtype=np.float32).reshape(-1)
    fused_replay_diff2 = np.asarray(fused_replay_diff2, dtype=np.float32).reshape(-1)
    if native_diff2.shape != replay_diff2.shape:
        raise ValueError(
            f"native/replay score topology differs: {native_diff2.shape} != {replay_diff2.shape}"
        )
    production_diff2 = -np.asarray(
        coarse["scores_pre_prior_per_class"], dtype=np.float32
    ).reshape(-1)
    coarse_lane_cutoff = None
    if recovar_projector_fused_lane_partials is not None:
        coarse_lane_cutoff = _coarse_lane_cutoff_report(
            scores_with_prior=np.asarray(
                coarse["scores_with_prior_per_class"],
                dtype=np.float32,
            ).reshape(rotations.shape[0], translations.shape[0]),
            lane_partials=recovar_projector_fused_lane_partials,
            initial_diff2=np.float32(initial_diff2),
            production_diff2=production_diff2,
            fused_diff2=recovar_projector_fused_replay_diff2,
            native_diff2=native_diff2,
            max_significants=int(np.asarray(coarse["max_significants"]).item()),
        )
    fine_projector_checks = _fine_projector_checks(
        native_directory,
        projector,
        r_max=r_max,
        physical_image_size=physical_image_size,
    )
    if debug_npz is not None:
        debug_npz = Path(debug_npz)
        debug_npz.parent.mkdir(parents=True, exist_ok=True)
        debug_values = dict(
            native_diff2=native_diff2,
            production_diff2=production_diff2,
            preprojected_replay_diff2=replay_diff2,
            fused_replay_diff2=fused_replay_diff2,
            projected=projected,
            shifted=shifted,
            rotations=rotations,
            translation_angles=np.asarray(translation_angles),
            score_indices=score_indices,
            full_to_compact=np.asarray(full_to_compact),
            unshifted=unshifted,
            pixel_weight=pixel_weight,
        )
        if recovar_projector_fused_replay_diff2 is not None:
            debug_values["recovar_projector_fused_replay_diff2"] = (
                recovar_projector_fused_replay_diff2
            )
            debug_values["recovar_projector_fused_lane_partials"] = (
                recovar_projector_fused_lane_partials
            )
        np.savez_compressed(debug_npz, **debug_values)

    comparisons = {
        "native_vs_recovar_production_centered_diff2": _centered_metric(
            native_diff2,
            production_diff2,
        ),
        "native_vs_native_ppref_replay_centered_diff2": _centered_metric(
            native_diff2,
            replay_diff2,
        ),
        "recovar_production_vs_native_ppref_replay_centered_diff2": _centered_metric(
            production_diff2,
            replay_diff2,
        ),
        "native_vs_fused_native_ppref_replay_centered_diff2": _centered_metric(
            native_diff2,
            fused_replay_diff2,
        ),
        "preprojected_vs_fused_native_ppref_replay_centered_diff2": _centered_metric(
            replay_diff2,
            fused_replay_diff2,
        ),
        "native_fine_projector_replay": fine_projector_checks,
    }
    if recovar_projector_fused_replay_diff2 is not None:
        comparisons[
            "recovar_production_vs_fused_recovar_ppref_replay_centered_diff2"
        ] = _centered_metric(
            production_diff2,
            recovar_projector_fused_replay_diff2,
        )
        comparisons[
            "native_vs_fused_recovar_ppref_replay_centered_diff2"
        ] = _centered_metric(
            native_diff2,
            recovar_projector_fused_replay_diff2,
        )

    return {
        "schema": "recovar.vdam_coarse_projector_boundary.v2",
        "status": "ok",
        "identity": {
            "current_size": current_size,
            "physical_image_size": int(physical_image_size),
            "rotation_count": int(rotations.shape[0]),
            "translation_count": int(translations.shape[0]),
            "projector_shape": list(projector.shape),
            "projector_r_max": int(r_max),
            "captured_recovar_projector": recovar_projector_capture is not None,
        },
        "comparisons": comparisons,
        "coarse_lane_cutoff": coarse_lane_cutoff,
        "artifacts": {
            "native_directory": str(Path(native_directory).resolve()),
            "recovar_coarse_dump": str(Path(recovar_coarse_dump).resolve()),
            "recovar_coarse_dump_sha256": _sha256(recovar_coarse_dump),
        },
        "device": str(jax.devices()[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--recovar-coarse-dump", type=Path, required=True)
    parser.add_argument("--projector-prefix", default="pass1_class0_ppref_")
    parser.add_argument("--native-score-name", default="pass0_coarse_raw_diff2.bin")
    parser.add_argument("--physical-image-size", type=int, default=128)
    parser.add_argument("--debug-npz", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.native_directory,
        args.recovar_coarse_dump,
        projector_prefix=args.projector_prefix,
        native_score_name=args.native_score_name,
        physical_image_size=args.physical_image_size,
        debug_npz=args.debug_npz,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
