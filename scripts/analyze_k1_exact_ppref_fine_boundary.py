#!/usr/bin/env python3
"""Replay one K=1 fine panel from an exact RELION in-memory PPref capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovar.em.dense_single_volume.helpers.projection import (
    compute_relion_projector_projections_block,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_diff2_sum,
    _relion_cuda_fine_full_to_compact_lookup,
)
from scripts.analyze_k1_bpref_contributor_membership import match_rotations
from scripts.analyze_k1_fine_direction_boundary import (
    _float32_key,
    _integer,
    _real,
    _scalar,
)
from scripts.validate_relion_fine_operand_capture import load_fine_operand_capture
from scripts.validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture


PPREF_MAGIC = b"RLNPPREFV1"
PPREF_HEADER_WORDS = 16


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_ppref(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with path.open("rb") as stream:
        magic = stream.read(16).split(b"\0", 1)[0]
        _require(magic == PPREF_MAGIC, f"invalid PPref magic: {magic!r}")
        header = np.frombuffer(
            stream.read(PPREF_HEADER_WORDS * 8), dtype="<u8"
        ).copy()
        payload = np.frombuffer(stream.read(), dtype="<f4").copy()
    _require(header.size == PPREF_HEADER_WORDS, "truncated PPref header")
    _require(int(header[0]) == 1, "unsupported PPref schema")
    _require(int(header[14]) == 4, "PPref scalar is not float32")
    count = int(header[13])
    _require(payload.size == 2 * count, "PPref payload size differs from header")
    shape = (int(header[7]), int(header[6]), int(header[5]))
    _require(int(np.prod(shape)) == count, "PPref dimensions differ from element count")
    ppref = (payload[0::2] + 1j * payload[1::2]).astype(np.complex64).reshape(shape)
    signed = lambda value: int(np.asarray(value, dtype=np.uint64).view(np.int64).item())
    metadata = {
        "version": int(header[0]),
        "iteration": int(header[1]),
        "rank": signed(header[2]),
        "model": int(header[3]),
        "current_size": int(header[4]),
        "shape_zyx": list(shape),
        "origin_xyz": [signed(header[8]), signed(header[9]), signed(header[10])],
        "r_max": int(header[11]),
        "padding_factor": float(
            struct.unpack("<f", struct.pack("<I", int(header[12]) & 0xFFFFFFFF))[0]
        ),
        "complex_count": count,
    }
    return ppref, metadata


def _centered_stats(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    _require(candidate.shape == reference.shape, "score topology mismatch")
    delta = (
        candidate.astype(np.float64) - float(np.min(candidate))
        - (reference.astype(np.float64) - float(np.min(reference)))
    )
    absolute = np.abs(delta)
    return {
        "count": int(delta.size),
        "bitwise_equal_count_after_float32_centering": int(
            np.count_nonzero(
                (candidate - np.min(candidate)).view(np.uint32)
                == (reference - np.min(reference)).view(np.uint32)
            )
        ),
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "max_abs": float(np.max(absolute)),
        "rms": float(np.sqrt(np.mean(delta * delta))),
    }


def _normalised_probability(raw: np.ndarray, prior: np.ndarray, mask: np.ndarray) -> np.ndarray:
    log_weight = np.where(mask, -raw.astype(np.float64) + prior.astype(np.float64), -np.inf)
    log_weight -= np.max(log_weight)
    probability = np.where(mask, np.exp(log_weight), 0.0)
    probability /= np.sum(probability)
    return probability


def _posterior_stats(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "total_variation": float(0.5 * np.sum(np.abs(delta))),
        "l1": float(np.sum(np.abs(delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "sum": float(np.sum(candidate)),
    }


def _projection_stats(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.complex64)
    reference = np.asarray(reference, dtype=np.complex64)
    delta = candidate.astype(np.complex128) - reference.astype(np.complex128)
    return {
        "count": int(candidate.size),
        "bitwise_equal_count": int(np.count_nonzero(candidate == reference)),
        "relative_l2": float(
            np.linalg.norm(delta) / max(np.linalg.norm(reference.astype(np.complex128)), np.finfo(float).tiny)
        ),
        "max_abs": float(np.max(np.abs(delta))),
    }


def _largest_score_residuals(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    native_rotation_rows: np.ndarray,
    recovar_rotation_rows: np.ndarray,
    recovar_translation_rows: np.ndarray,
    recovar_global_rotations: np.ndarray,
    translations: np.ndarray,
    limit: int = 12,
) -> list[dict[str, Any]]:
    candidate_centered = np.asarray(candidate, dtype=np.float32) - np.min(candidate)
    reference_centered = np.asarray(reference, dtype=np.float32) - np.min(reference)
    residual = candidate_centered.astype(np.float64) - reference_centered.astype(np.float64)
    order = np.argsort(np.abs(residual), kind="stable")[::-1][:limit]
    return [
        {
            "native_candidate_row": int(row),
            "native_fine_rotation_local": int(native_rotation_rows[row]),
            "recovar_rotation_row": int(recovar_rotation_rows[row]),
            "recovar_global_fine_rotation": int(
                recovar_global_rotations[recovar_rotation_rows[row]]
            ),
            "recovar_translation_row": int(recovar_translation_rows[row]),
            "translation_pixels": translations[recovar_translation_rows[row]].astype(float).tolist(),
            "native_raw": float(reference[row]),
            "candidate_raw": float(candidate[row]),
            "centered_delta_candidate_minus_native": float(residual[row]),
        }
        for row in order
    ]


def analyze(
    *,
    ppref_path: Path,
    recovar_path: Path,
    native_directory: Path,
    fine_operand_path: Path,
    fine_operand_rotation_row: int,
    chunk_size: int,
    output_npz: Path | None = None,
    native_fine_score_path: Path | None = None,
) -> dict[str, Any]:
    _require(jax.default_backend() == "gpu", "exact PPref replay requires a GPU")
    ppref, ppref_metadata = _load_ppref(ppref_path)
    with np.load(recovar_path, allow_pickle=False) as archive:
        recovar = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "current_size",
        "fine_translations",
        "rotations",
        "oversampled_rot_indices",
        "candidate_mask",
        "rotation_log_prior",
        "translation_log_prior",
        "shifted_corrected",
        "ctf2_over_nv_score",
        "half_weights",
        "window_indices",
        "proj_half",
        "relion_highres_xi2_half",
    }
    _require(required <= set(recovar), f"RECOVAR dump misses {sorted(required - set(recovar))}")
    current_size = int(np.asarray(recovar["current_size"]).item())
    _require(current_size == ppref_metadata["current_size"], "current-size mismatch")
    _require(ppref_metadata["padding_factor"] == 2.0, "padding-factor mismatch")
    _require(ppref_metadata["r_max"] == current_size // 2, "r_max mismatch")

    rotations = np.asarray(recovar["rotations"], dtype=np.float32)
    translations = np.asarray(recovar["fine_translations"], dtype=np.float32)
    candidate_mask = np.asarray(recovar["candidate_mask"], dtype=bool)
    shifted = np.asarray(recovar["shifted_corrected"], dtype=np.complex64)
    pixel_weight = np.multiply(
        np.asarray(recovar["ctf2_over_nv_score"], dtype=np.float32),
        np.asarray(recovar["half_weights"], dtype=np.float32),
        dtype=np.float32,
    )
    window_indices = np.asarray(recovar["window_indices"], dtype=np.int32)
    full_to_compact = _relion_cuda_fine_full_to_compact_lookup(
        (128, 128), current_size, window_indices
    )
    highres = np.float32(np.asarray(recovar["relion_highres_xi2_half"]).item())
    rotation_prior = np.asarray(recovar["rotation_log_prior"], dtype=np.float64)
    translation_prior = np.asarray(recovar["translation_log_prior"], dtype=np.float64)
    combined_prior = rotation_prior[:, None] + translation_prior[None, :]

    n_rotations = rotations.shape[0]
    _require(candidate_mask.shape == (n_rotations, translations.shape[0]), "candidate-mask shape mismatch")
    native_ppref_raw = np.empty(candidate_mask.shape, dtype=np.float32)
    recovar_ppref_raw = np.empty(candidate_mask.shape, dtype=np.float32)
    selected_native_ppref_projection = None
    selected_recovar_ppref_projection = None

    @jax.jit
    def score_block(projection: jax.Array) -> jax.Array:
        raw = _relion_cuda_fine_diff2_sum(
            projection[:, None, :],
            jnp.asarray(shifted)[None, :, :],
            jnp.asarray(pixel_weight)[None, None, :],
            jnp.asarray(full_to_compact, dtype=jnp.int32),
        )
        return raw + jnp.asarray(highres, dtype=jnp.float32)

    for start in range(0, n_rotations, chunk_size):
        stop = min(start + chunk_size, n_rotations)
        actual = stop - start
        rotation_block = rotations[start:stop]
        if actual < chunk_size:
            rotation_block = np.concatenate(
                [rotation_block, np.repeat(rotation_block[-1:], chunk_size - actual, axis=0)],
                axis=0,
            )
        native_projection, _ = compute_relion_projector_projections_block(
            jnp.asarray(ppref),
            jnp.asarray(rotation_block),
            (128, 128),
            r_max=ppref_metadata["r_max"],
            padding_factor=2,
            return_abs2=False,
            centered_rows=True,
            dense_scale=True,
            projector_output_size=current_size,
            pixel_indices=jnp.asarray(window_indices),
            relion_texture_interp=True,
        )
        native_projection = np.asarray(jax.block_until_ready(native_projection), dtype=np.complex64)[:actual]
        recovar_projection = np.asarray(recovar["proj_half"][start:stop], dtype=np.complex64)
        native_ppref_raw[start:stop] = np.asarray(
            jax.block_until_ready(score_block(jnp.asarray(native_projection))), dtype=np.float32
        )
        recovar_ppref_raw[start:stop] = np.asarray(
            jax.block_until_ready(score_block(jnp.asarray(recovar_projection))), dtype=np.float32
        )
        if start <= fine_operand_rotation_row < stop:
            local = fine_operand_rotation_row - start
            selected_native_ppref_projection = native_projection[local].copy()
            selected_recovar_ppref_projection = recovar_projection[local].copy()
        print(f"scored rotations {stop}/{n_rotations}", flush=True)

    native_rotations = _real(native_directory, "pass1_class0_fine_eulers.bin").reshape(-1, 3, 3)
    native_rotations = native_rotations.astype(np.float32).transpose(0, 2, 1)
    native_rotation_rows = _integer(native_directory, "pass1_acc_rot_idx.bin")
    native_parent_rotations = _integer(native_directory, "pass1_acc_rot_id.bin")
    native_translation_x = _real(native_directory, "pass1_candidate_translation_x.bin").astype(np.float32)
    native_translation_y = _real(native_directory, "pass1_candidate_translation_y.bin").astype(np.float32)
    native_raw = _real(native_directory, "pass1_exp_Mweight_raw_preprior.bin").astype(np.float32)
    native_prior = _real(native_directory, "pass1_candidate_combined_log_prior.bin")
    native_probability = _real(native_directory, "pass1_candidate_weight_normalized.bin")
    native_significant = _integer(native_directory, "pass1_candidate_in_reconstruction_set.bin").astype(bool)
    _require(
        int(round(_scalar(native_directory, "pass1_acc_stack_index.bin")))
        == int(np.asarray(recovar["original_index"]).item()) + 1,
        "particle identity mismatch",
    )
    rotation_matches = match_rotations(native_rotations, rotations, tolerance=0.0)
    _require(rotation_matches.pairs.shape[0] == rotations.shape[0], "rotation sets differ")
    native_to_recovar_rotation = np.empty(rotations.shape[0], dtype=np.int64)
    native_to_recovar_rotation[rotation_matches.pairs[:, 0]] = rotation_matches.pairs[:, 1]
    translation_lookup = {
        _float32_key(x, y): index for index, (x, y) in enumerate(translations.tolist())
    }
    native_to_recovar_translation = np.asarray(
        [translation_lookup[_float32_key(x, y)] for x, y in zip(native_translation_x, native_translation_y, strict=True)],
        dtype=np.int64,
    )
    native_fine_score = None
    if native_fine_score_path is not None:
        native_fine_score = load_fine_score_capture(native_fine_score_path)
        selected = native_fine_score.candidates[
            (native_fine_score.candidates["flags"] & ACTIVE) != 0
        ]
        _require(selected.size == native_raw.size, "same-run fine-score candidate count differs")
        _require(
            np.array_equal(selected["rotation_local"], native_rotation_rows.astype(np.uint64)),
            "same-run fine-score rotation order differs",
        )
        native_translation_ids = _integer(native_directory, "pass1_acc_trans_idx.bin")
        _require(
            np.array_equal(selected["translation_id"], native_translation_ids.astype(np.uint64)),
            "same-run fine-score translation order differs",
        )
        native_raw = np.asarray(selected["raw_diff2"], dtype=np.float32)
        native_prior = np.add(
            selected["orientation_log_prior"],
            selected["translation_log_prior"],
            dtype=np.float32,
        ).astype(np.float64)
        native_weight = np.asarray(selected["post_exponent_weight"], dtype=np.float64)
        native_probability = native_weight / np.sum(native_weight)
    recovar_rotation_rows = native_to_recovar_rotation[native_rotation_rows]
    mapped_native_ppref_raw = native_ppref_raw[
        recovar_rotation_rows, native_to_recovar_translation
    ]
    mapped_recovar_ppref_raw = recovar_ppref_raw[
        recovar_rotation_rows, native_to_recovar_translation
    ]

    native_ppref_probability_all = _normalised_probability(
        native_ppref_raw, combined_prior, candidate_mask
    )
    recovar_ppref_probability_all = _normalised_probability(
        recovar_ppref_raw, combined_prior, candidate_mask
    )
    native_ppref_probability = native_ppref_probability_all[
        recovar_rotation_rows, native_to_recovar_translation
    ]
    recovar_ppref_probability = recovar_ppref_probability_all[
        recovar_rotation_rows, native_to_recovar_translation
    ]

    n_keep = int(np.count_nonzero(native_significant))
    native_ppref_top = np.zeros(native_probability.size, dtype=bool)
    recovar_ppref_top = np.zeros(native_probability.size, dtype=bool)
    native_ppref_top[np.argsort(native_ppref_probability, kind="stable")[-n_keep:]] = True
    recovar_ppref_top[np.argsort(recovar_ppref_probability, kind="stable")[-n_keep:]] = True
    native_direction_ids = native_parent_rotations // 48

    def direction(probability: np.ndarray, support: np.ndarray) -> np.ndarray:
        result = np.zeros(768, dtype=np.float64)
        np.add.at(result, native_direction_ids, probability * support)
        return result

    native_direction = direction(native_probability, native_significant)
    native_ppref_direction = direction(native_ppref_probability, native_ppref_top)
    recovar_ppref_direction = direction(recovar_ppref_probability, recovar_ppref_top)

    _require(selected_native_ppref_projection is not None, "selected operand rotation was not evaluated")
    fine_operand = load_fine_operand_capture(fine_operand_path)
    pixels = fine_operand.pixels.reshape(1, fine_operand.image_size)[0]
    native_reference = (
        pixels["reference_real"].astype(np.float32)
        + 1j * pixels["reference_imag"].astype(np.float32)
    ).astype(np.complex64)
    supported_full = np.flatnonzero(full_to_compact >= 0)
    supported_compact = full_to_compact[supported_full]
    native_reference_dense = -native_reference[supported_full] * np.float32(128**2)

    native_ppref_score_stats = _centered_stats(mapped_native_ppref_raw, native_raw)
    recovar_ppref_score_stats = _centered_stats(mapped_recovar_ppref_raw, native_raw)
    if output_npz is not None:
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_npz,
            native_ppref_raw=native_ppref_raw,
            recovar_ppref_raw=recovar_ppref_raw,
            native_raw=native_raw,
            native_rotation_rows=native_rotation_rows,
            recovar_rotation_rows=recovar_rotation_rows,
            recovar_translation_rows=native_to_recovar_translation,
            native_ppref_probability=native_ppref_probability,
            recovar_ppref_probability=recovar_ppref_probability,
            native_probability=native_probability,
        )
    return {
        "schema": "recovar.em.k1_exact_ppref_fine_boundary.v1",
        "status": "complete",
        "identity": {
            "original_index_zero_based": int(np.asarray(recovar["original_index"]).item()),
            "stack_index_one_based": int(round(_scalar(native_directory, "pass1_acc_stack_index.bin"))),
            "rotation_count": n_rotations,
            "translation_count": int(translations.shape[0]),
            "native_candidate_count": int(native_raw.size),
        },
        "ppref": ppref_metadata,
        "selected_projection": {
            "rotation_row": fine_operand_rotation_row,
            "native_ppref_texture_vs_native_relion": _projection_stats(
                selected_native_ppref_projection[supported_compact], native_reference_dense
            ),
            "native_ppref_texture_vs_recovar_ppref_texture": _projection_stats(
                selected_native_ppref_projection, selected_recovar_ppref_projection
            ),
        },
        "raw_score_centered_vs_native_relion": {
            "native_ppref": native_ppref_score_stats,
            "recovar_ppref": recovar_ppref_score_stats,
            "rms_improvement_ratio_recovar_over_native_ppref": float(
                recovar_ppref_score_stats["rms"] / max(native_ppref_score_stats["rms"], np.finfo(float).tiny)
            ),
            "largest_native_ppref_residuals": _largest_score_residuals(
                mapped_native_ppref_raw,
                native_raw,
                native_rotation_rows=native_rotation_rows,
                recovar_rotation_rows=recovar_rotation_rows,
                recovar_translation_rows=native_to_recovar_translation,
                recovar_global_rotations=np.asarray(recovar["oversampled_rot_indices"], dtype=np.int64),
                translations=translations,
            ),
        },
        "posterior_vs_native_relion": {
            "native_ppref": _posterior_stats(native_ppref_probability, native_probability),
            "recovar_ppref": _posterior_stats(recovar_ppref_probability, native_probability),
        },
        "fixed_native_significant_count_support": {
            "count": n_keep,
            "native_ppref_mismatch_count": int(np.count_nonzero(native_ppref_top != native_significant)),
            "recovar_ppref_mismatch_count": int(np.count_nonzero(recovar_ppref_top != native_significant)),
        },
        "direction_sufficient_statistic": {
            "native_ppref_l1_vs_native": float(np.sum(np.abs(native_ppref_direction - native_direction))),
            "recovar_ppref_l1_vs_native": float(np.sum(np.abs(recovar_ppref_direction - native_direction))),
        },
        "native_prior_centered_max_abs": float(
            np.max(np.abs((combined_prior[recovar_rotation_rows, native_to_recovar_translation] - np.max(combined_prior[recovar_rotation_rows, native_to_recovar_translation])) - (native_prior - np.max(native_prior))))
        ),
        "artifacts": {
            "ppref": str(ppref_path.resolve()),
            "ppref_sha256": _sha256(ppref_path),
            "recovar": str(recovar_path.resolve()),
            "recovar_sha256": _sha256(recovar_path),
            "native_directory": str(native_directory.resolve()),
            "native_fine_score": (
                None if native_fine_score_path is None else str(native_fine_score_path.resolve())
            ),
            "native_fine_score_sha256": (
                None if native_fine_score_path is None else _sha256(native_fine_score_path)
            ),
            "fine_operand": str(fine_operand_path.resolve()),
            "fine_operand_sha256": _sha256(fine_operand_path),
            "raw_dump": None if output_npz is None else str(output_npz.resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppref", type=Path, required=True)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--fine-operand", type=Path, required=True)
    parser.add_argument("--native-fine-score", type=Path)
    parser.add_argument("--fine-operand-rotation-row", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        ppref_path=args.ppref,
        recovar_path=args.recovar,
        native_directory=args.native_directory,
        fine_operand_path=args.fine_operand,
        fine_operand_rotation_row=args.fine_operand_rotation_row,
        chunk_size=args.chunk_size,
        output_npz=args.output_npz,
        native_fine_score_path=args.native_fine_score,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
