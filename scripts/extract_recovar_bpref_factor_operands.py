#!/usr/bin/env python3
"""Extract production-f32 and genuine-f64 RECOVAR BPref factor operands."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar.core.ctf import _compute_spa_ctf
from recovar.cuda_backproject import relion_preprocess_real_f32
from recovar.data_io.image_backends import _centered_rfft2_jax
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _half_translation_phase_table_for_indices,
)
from recovar.em.dense_single_volume.local_backprojection import compute_local_mstep_sums
from scripts.recompute_bpref_high_precision import (
    _ctf_float64,
    _half_lattice,
    _particle_processed_fft,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, object]:
    left = np.asarray(lhs)
    right = np.asarray(rhs)
    _require(left.shape == right.shape, "factor metric shape mismatch")
    delta = right.astype(np.complex128) - left.astype(np.complex128)
    denominator = max(float(np.linalg.norm(left.astype(np.complex128))), np.finfo(np.float64).tiny)
    return {
        "shape": list(left.shape),
        "lhs_dtype": str(left.dtype),
        "rhs_dtype": str(right.dtype),
        "exact_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "relative_l2_over_lhs": float(np.linalg.norm(delta) / denominator),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
    }


def _compact_indices(values: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    image_shape = tuple(int(value) for value in values["image_shape"])
    height, width = image_shape
    half_width = width // 2 + 1
    window = np.asarray(values["window_indices"], dtype=np.int32)
    fftw_rows = window // half_width
    centered = (((fftw_rows - height // 2) % height) * half_width + window % half_width).astype(
        np.int32, copy=False
    )
    return window, centered


def _load_selection(path: Path) -> tuple[list[int], dict[int, int]]:
    payload = json.loads(path.read_text())
    _require(payload.get("schema") == "bpref-factor-stratification-v1", "selection schema changed")
    records = payload.get("selected")
    _require(isinstance(records, list) and len(records) == 32, "expected exactly 32 selected particles")
    stacks = [int(record["stack_index_1based"]) for record in records]
    rotations = {
        int(record["stack_index_1based"]): int(record["recovar_global_rotation_index"])
        for record in records
    }
    _require(len(set(stacks)) == len(stacks) and 111721 not in stacks, "invalid selected stack set")
    return stacks, rotations


def _extract_shard_f32(
    values: dict[str, np.ndarray], selected: list[tuple[int, int, int]]
) -> dict[int, dict[str, np.ndarray | int | float]]:
    _require(jax.default_backend() == "gpu", "production factor extraction requires a JAX GPU")
    image_shape = tuple(int(value) for value in values["image_shape"])
    window, compact_np = _compact_indices(values)
    compact = jnp.asarray(compact_np, dtype=jnp.int32)
    raw = jnp.asarray(values["raw_real_images"], dtype=jnp.float32)
    normalization = jnp.asarray(values["relion_preprocess_normalization_factors"], dtype=jnp.float32)
    integer_shifts = jnp.asarray(values["integer_pre_shifts"], dtype=jnp.int32)
    _, preprocessed = relion_preprocess_real_f32(raw, normalization, integer_shifts, 1.0, 1.0, False)
    processed = _centered_rfft2_jax(preprocessed).reshape(raw.shape[0], -1).astype(jnp.complex64)
    ctf = _compute_spa_ctf(
        jnp.asarray(values["ctf_params"], dtype=jnp.float32),
        image_shape,
        float(np.asarray(values["voxel_size"]).item()),
        half_image=True,
    ).astype(jnp.float32)
    noise = jnp.asarray(values["noise_variance_half"], dtype=jnp.float32)
    scale = jnp.asarray(values["scale_corrections"], dtype=jnp.float32)
    translations = jnp.asarray(values["fine_translations"], dtype=jnp.float32)
    phases = _half_translation_phase_table_for_indices(translations, image_shape, compact)
    probs = jnp.asarray(values["reconstruction_probs"], dtype=jnp.float32)
    weighted = processed[:, compact] * ctf[:, compact] / noise[compact]
    weighted = weighted * scale[:, None]
    shifted = weighted[:, None, :] * phases[None, :, :]
    ctf_weight = ctf[:, compact] ** 2 / noise[compact]
    ctf_weight = ctf_weight * scale[:, None] ** 2
    numerator, denominator = compute_local_mstep_sums(
        probs,
        shifted,
        ctf_weight,
        relion_x_half=True,
        sequential_translation_reduction=False,
    )

    host = {
        "processed": np.asarray(processed[:, compact]),
        "ctf": np.asarray(ctf[:, compact]),
        "noise": np.asarray(noise[compact]),
        "scale": np.asarray(scale),
        "phases": np.asarray(phases),
        "probs": np.asarray(probs),
        "numerator": np.asarray(numerator),
        "denominator": np.asarray(denominator),
    }
    results: dict[int, dict[str, np.ndarray | int | float]] = {}
    active_particles = np.asarray(values["active_particle_rows"], dtype=np.int32)
    active_global = np.asarray(values["active_global_rotation_indices"], dtype=np.int64)
    active_summed = np.asarray(values["active_summed"], dtype=np.complex64)
    active_weight = np.asarray(values["active_ctf_probs"], dtype=np.float32)
    rotation_indices = np.asarray(values["oversampled_rotation_indices"], dtype=np.int64)
    for stack, particle, global_rotation in selected:
        rotation_match = np.flatnonzero(rotation_indices[particle] == global_rotation)
        _require(rotation_match.size == 1, f"stack {stack}: rotation identity is not unique")
        rotation = int(rotation_match[0])
        nonzero = np.argwhere(host["probs"][particle] != 0)
        _require(nonzero.shape == (1, 2), f"stack {stack}: expected one nonzero reconstruction hypothesis")
        _require(int(nonzero[0, 0]) == rotation, f"stack {stack}: winner rotation changed")
        translation = int(nonzero[0, 1])
        probability = np.float32(host["probs"][particle, rotation, translation])
        _require(probability == np.float32(1), f"stack {stack}: winner probability is not exactly one")
        active_match = np.flatnonzero(
            (active_particles == particle) & (active_global == global_rotation)
        )
        _require(active_match.size == 1, f"stack {stack}: active winner identity is not unique")
        active = int(active_match[0])
        processed_row = host["processed"][particle]
        ctf_row = host["ctf"][particle]
        phase_row = host["phases"][translation]
        shifted_image = (processed_row * phase_row).astype(np.complex64)
        weighted_ctf = (
            probability * ctf_row / host["noise"] * host["scale"][particle]
        ).astype(np.float32)
        term = (shifted_image * weighted_ctf).astype(np.complex64)
        weight_term = (
            probability * ctf_row**2 / host["noise"] * host["scale"][particle] ** 2
        ).astype(np.float32)
        results[stack] = {
            "window_indices": window,
            "rotation": rotation,
            "translation": translation,
            "translation_vector": np.asarray(values["fine_translations"][translation], dtype=np.float32),
            "probability": probability,
            "processed_fft": processed_row,
            "ctf": ctf_row,
            "noise": host["noise"],
            "scale": np.float32(host["scale"][particle]),
            "phase": phase_row,
            "shifted_image": shifted_image,
            "weighted_ctf": weighted_ctf,
            "term": term,
            "weight_term": weight_term,
            "numerator": host["numerator"][particle, rotation],
            "denominator": host["denominator"][particle, rotation],
            "captured_numerator": active_summed[active],
            "captured_denominator": active_weight[active],
        }
    return results


def _extract_shard_f64(
    values: dict[str, np.ndarray], selected: list[tuple[int, int, int]]
) -> dict[int, dict[str, np.ndarray | float]]:
    image_shape = tuple(int(value) for value in values["image_shape"])
    window, compact = _compact_indices(values)
    lattice = _half_lattice(image_shape, 1.0)
    translations = np.asarray(values["fine_translations"], dtype=np.float64)
    phases = np.exp(-2j * np.pi * (translations @ lattice.T))[:, compact]
    noise = np.asarray(values["noise_variance_half"], dtype=np.float64)[window]
    results: dict[int, dict[str, np.ndarray | float]] = {}
    for stack, particle, _global_rotation in selected:
        probs = np.asarray(values["reconstruction_probs"][particle], dtype=np.float64)
        nonzero = np.argwhere(probs != 0)
        _require(nonzero.shape == (1, 2), f"stack {stack}: f64 hypothesis support changed")
        rotation, translation = (int(value) for value in nonzero[0])
        probability = np.float64(probs[rotation, translation])
        processed = _particle_processed_fft(values, particle)[compact]
        ctf = _ctf_float64(
            values["ctf_params"][particle],
            image_shape,
            float(np.asarray(values["voxel_size"]).item()),
        )[compact]
        scale = np.float64(values["scale_corrections"][particle])
        phase = phases[translation]
        shifted_image = processed * phase
        weighted_ctf = probability * ctf / noise * scale
        term = shifted_image * weighted_ctf
        weight_term = probability * ctf**2 / noise * scale**2
        results[stack] = {
            "translation_vector": translations[translation],
            "probability": probability,
            "processed_fft": processed,
            "ctf": ctf,
            "noise": noise,
            "scale": scale,
            "phase": phase,
            "shifted_image": shifted_image,
            "weighted_ctf": weighted_ctf,
            "term": term,
            "weight_term": weight_term,
            "numerator": term,
            "denominator": weight_term,
        }
    return results


def extract(contribution_directory: Path, selection_json: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    stacks, selected_rotations = _load_selection(selection_json)
    locations: dict[int, tuple[Path, int]] = {}
    shard_hashes: dict[str, str] = {}
    for path in sorted(contribution_directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            shard_stacks = np.asarray(archive["stack_indices_1based"], dtype=np.int64)
        for stack in stacks:
            match = np.flatnonzero(shard_stacks == stack)
            if match.size:
                _require(match.size == 1 and stack not in locations, f"stack {stack}: duplicate contribution")
                locations[stack] = (path, int(match[0]))
        if any(location[0] == path for location in locations.values()):
            shard_hashes[path.name] = _sha256(path)
    _require(set(locations) == set(stacks), "selected contribution identities are incomplete")

    first: dict[int, dict[str, np.ndarray | int | float]] = {}
    repeat: dict[int, dict[str, np.ndarray | int | float]] = {}
    high: dict[int, dict[str, np.ndarray | float]] = {}
    for path in sorted({location[0] for location in locations.values()}):
        selected = [
            (stack, particle, selected_rotations[stack])
            for stack, (candidate, particle) in locations.items()
            if candidate == path
        ]
        with np.load(path, allow_pickle=False) as archive:
            values = {name: archive[name] for name in archive.files}
        first.update(_extract_shard_f32(values, selected))
        repeat.update(_extract_shard_f32(values, selected))
        high.update(_extract_shard_f64(values, selected))

    factor_names = (
        "processed_fft",
        "ctf",
        "noise",
        "phase",
        "shifted_image",
        "weighted_ctf",
        "term",
        "weight_term",
        "numerator",
        "denominator",
    )
    arrays: dict[str, np.ndarray] = {
        "stack_indices_1based": np.asarray(stacks, dtype=np.int64),
        "window_indices": np.stack([np.asarray(first[stack]["window_indices"]) for stack in stacks]),
        "rotation_rows": np.asarray([first[stack]["rotation"] for stack in stacks], dtype=np.int32),
        "translation_rows": np.asarray([first[stack]["translation"] for stack in stacks], dtype=np.int32),
        "translation_vectors_f32": np.stack(
            [np.asarray(first[stack]["translation_vector"]) for stack in stacks]
        ).astype(np.float32),
        "translation_vectors_f64": np.stack(
            [np.asarray(high[stack]["translation_vector"]) for stack in stacks]
        ).astype(np.float64),
        "probability_f32": np.asarray([first[stack]["probability"] for stack in stacks], dtype=np.float32),
        "probability_f64": np.asarray([high[stack]["probability"] for stack in stacks], dtype=np.float64),
        "scale_f32": np.asarray([first[stack]["scale"] for stack in stacks], dtype=np.float32),
        "scale_f64": np.asarray([high[stack]["scale"] for stack in stacks], dtype=np.float64),
    }
    for name in factor_names:
        arrays[f"{name}_f32"] = np.stack([np.asarray(first[stack][name]) for stack in stacks])
        arrays[f"{name}_f64"] = np.stack([np.asarray(high[stack][name]) for stack in stacks])

    repeat_metrics = {
        name: _metrics(
            np.stack([np.asarray(first[stack][name]) for stack in stacks]),
            np.stack([np.asarray(repeat[stack][name]) for stack in stacks]),
        )
        for name in factor_names
    }
    capture_closure = {
        "numerator": _metrics(
            arrays["numerator_f32"],
            np.stack([np.asarray(first[stack]["captured_numerator"]) for stack in stacks]),
        ),
        "denominator": _metrics(
            arrays["denominator_f32"],
            np.stack([np.asarray(first[stack]["captured_denominator"]) for stack in stacks]),
        ),
    }
    _require(all(bool(record["exact_equal"]) for record in repeat_metrics.values()), "production factor repeat changed")
    _require(all(bool(record["exact_equal"]) for record in capture_closure.values()), "production factor extraction did not close captured RECOVAR operands")
    report = {
        "schema": "recovar-bpref-factor-operands-v1",
        "metric_policy": "exact/array metrics for intermediate operands; no correlation",
        "production_backend": str(jax.devices()[0]),
        "particle_count": len(stacks),
        "contribution_shard_count": len(shard_hashes),
        "selection_json": str(selection_json.resolve()),
        "selection_sha256": _sha256(selection_json),
        "contribution_sha256": shard_hashes,
        "production_f32_repeat": repeat_metrics,
        "production_f32_capture_closure": capture_closure,
        "genuine_f64_operand_recomputation": True,
        "factor_ready": True,
    }
    return report, arrays


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contribution_directory", type=Path)
    parser.add_argument("--selection-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_npz.exists() or args.output_json.exists():
        raise FileExistsError("refusing to overwrite a factor operand artifact")
    report, arrays = extract(args.contribution_directory, args.selection_json)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **arrays)
    report["output_npz"] = str(args.output_npz.resolve())
    report["output_npz_sha256"] = _sha256(args.output_npz)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
