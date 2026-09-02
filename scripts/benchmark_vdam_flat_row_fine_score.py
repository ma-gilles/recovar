#!/usr/bin/env python3
"""GPU microbenchmark for call-neutral packed exact-local fine scoring."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject
from recovar.em.dense_single_volume.helpers.flat_local_rows import (
    build_pool_flat_local_row_plan,
    scatter_flat_local_rows,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_diff2_to_scores,
    _relion_f32_fine_posterior,
)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _pool_padded_rows(counts: np.ndarray, dense_rotation_count: int, radix: int) -> int:
    plan = build_pool_flat_local_row_plan(
        counts,
        dense_rotation_count,
        pool_size=3,
        exact_local_bucket_radix=radix,
    )
    return int(plan.packed_row_count)


def _case_layout(metadata_path: Path) -> dict:
    metadata = json.loads(metadata_path.read_text())
    profile = metadata["halfset_0_profile_summary"]
    counts = np.asarray(profile["local_rotation_counts"], dtype=np.int32)
    chunk_sizes = [int(size) for size in profile["chunk_sizes"]]
    radix = int(metadata["effective_exact_local_bucket_radix"])
    dense_rotation_count = int(profile["sum_padded_rows"]) // int(counts.size)
    chunks = []
    start = 0
    for size in chunk_sizes:
        stop = start + size
        chunk_counts = counts[start:stop]
        chunks.append(
            {
                "start": start,
                "stop": stop,
                "counts": chunk_counts,
                "packed_rows": _pool_padded_rows(
                    chunk_counts,
                    dense_rotation_count,
                    radix,
                ),
            },
        )
        start = stop
    if start != int(counts.size):
        raise ValueError(f"{metadata_path}: chunk sizes do not cover the selected stream")
    packed_row_count = max(int(chunk["packed_rows"]) for chunk in chunks)
    logical_packed_rows = sum(int(chunk["packed_rows"]) for chunk in chunks)
    selected_chunk = min(chunks, key=lambda chunk: int(chunk["packed_rows"]))
    dense_batch_size = max(chunk_sizes)
    plan = build_pool_flat_local_row_plan(
        selected_chunk["counts"],
        dense_rotation_count,
        pool_size=3,
        exact_local_bucket_radix=radix,
        packed_row_count=packed_row_count,
        dense_batch_size=dense_batch_size,
    )
    padded_tail_rows = packed_row_count - int(selected_chunk["packed_rows"])
    if padded_tail_rows < 1:
        raise ValueError(f"{metadata_path}: exactness probe requires a static padded tail")
    if len(chunks) != len(chunk_sizes):
        raise AssertionError(f"{metadata_path}: packed layout changed outer-call cardinality")
    return {
        "metadata": metadata,
        "profile": profile,
        "radix": radix,
        "dense_rotation_count": dense_rotation_count,
        "dense_batch_size": dense_batch_size,
        "packed_row_count": packed_row_count,
        "logical_packed_rows": logical_packed_rows,
        "padded_tail_rows": padded_tail_rows,
        "selected_chunk": selected_chunk,
        "plan": plan,
        "chunk_packed_rows": [int(chunk["packed_rows"]) for chunk in chunks],
    }


def _make_operands(case: dict, seed: int) -> dict:
    metadata = case["metadata"]
    profile = case["profile"]
    plan = case["plan"]
    batch = int(case["dense_batch_size"])
    rotations = int(case["dense_rotation_count"])
    pixels = int(profile["n_windowed"])
    translations = int(metadata["n_translations"])
    current_size = int(metadata["current_size"])
    rng = np.random.default_rng(seed)

    reference = (
        rng.standard_normal((batch, rotations, pixels), dtype=np.float32)
        + 1j * rng.standard_normal((batch, rotations, pixels), dtype=np.float32)
    ).astype(np.complex64)
    image = (
        rng.standard_normal((batch, pixels), dtype=np.float32)
        + 1j * rng.standard_normal((batch, pixels), dtype=np.float32)
    ).astype(np.complex64)
    weight = np.abs(rng.standard_normal((batch, pixels), dtype=np.float32)) + np.float32(0.125)
    translation_angles = rng.uniform(-0.08, 0.08, size=(translations, 2)).astype(np.float32)
    initial_diff2 = rng.uniform(0.0, 5.0, size=batch).astype(np.float32)
    rotation_log_prior = rng.uniform(-4.0, 0.0, size=(batch, rotations)).astype(np.float32)
    translation_log_prior = rng.uniform(-3.0, 0.0, size=(batch, translations)).astype(np.float32)

    lookup = np.full(current_size * (current_size // 2 + 1), -1, dtype=np.int32)
    if pixels > lookup.size:
        raise ValueError(f"compact pixels {pixels} exceed current-size rows {lookup.size}")
    lookup[:pixels] = np.arange(pixels, dtype=np.int32)

    physical_counts = np.asarray(case["selected_chunk"]["counts"], dtype=np.int32)
    rotation_mask = np.zeros((batch, rotations), dtype=bool)
    for image_index, count in enumerate(physical_counts):
        rotation_mask[image_index, : int(count)] = True
    valid_image_mask = np.arange(batch) < int(physical_counts.size)
    candidate_mask = rotation_mask[:, :, None] & valid_image_mask[:, None, None]

    flat_reference = reference[plan.image_indices, plan.rotation_rows]
    flat_reference[~plan.present_mask] = np.complex64(np.nan + 1j * np.nan)
    return {
        "reference": jnp.asarray(reference),
        "flat_reference": jnp.asarray(flat_reference),
        "row_image_ids": jnp.asarray(plan.image_indices),
        "row_rotation_rows": jnp.asarray(plan.rotation_rows),
        "row_present_mask": jnp.asarray(plan.present_mask),
        "padded_tail_rows": int(case["padded_tail_rows"]),
        "image": jnp.asarray(image),
        "translation_angles": jnp.asarray(translation_angles),
        "weight": jnp.asarray(weight),
        "initial_diff2": jnp.asarray(initial_diff2),
        "lookup": jnp.asarray(lookup),
        "rotation_log_prior": jnp.asarray(rotation_log_prior),
        "translation_log_prior": jnp.asarray(translation_log_prior),
        "candidate_mask": jnp.asarray(candidate_mask),
        "current_size": current_size,
        "batch": batch,
        "rotations": rotations,
        "translations": translations,
    }


def _build_calls(operands: dict):
    current_size = int(operands["current_size"])
    batch = int(operands["batch"])
    rotations = int(operands["rotations"])

    @jax.jit
    def dense_raw(reference, image, angles, weight, initial, lookup):
        return cuda_backproject.relion_fine_diff2_fused_translate_rectangular_f32(
            reference,
            image,
            angles,
            weight,
            lookup,
            initial,
            current_size=current_size,
        )

    @jax.jit
    def flat_raw(reference, row_image_ids, image, angles, weight, initial, lookup):
        return cuda_backproject.relion_fine_diff2_fused_translate_flat_rows_f32(
            reference,
            row_image_ids,
            image,
            angles,
            weight,
            lookup,
            initial,
            current_size=current_size,
        )

    @jax.jit
    def dense_score_posterior(
        reference,
        image,
        angles,
        weight,
        initial,
        lookup,
        rotation_prior,
        translation_prior,
        candidate_mask,
    ):
        raw = dense_raw(reference, image, angles, weight, initial, lookup)
        scores = _relion_cuda_fine_diff2_to_scores(
            raw,
            rotation_prior[:, :, None],
            translation_prior[:, None, :],
            candidate_mask,
        )
        posterior = _relion_f32_fine_posterior(
            scores,
            adaptive_fraction=0.999,
        )
        return raw, scores, posterior

    @jax.jit
    def flat_score_posterior(
        reference,
        row_image_ids,
        row_rotation_rows,
        row_present_mask,
        image,
        angles,
        weight,
        initial,
        lookup,
        rotation_prior,
        translation_prior,
        candidate_mask,
    ):
        raw_flat = flat_raw(
            reference,
            row_image_ids,
            image,
            angles,
            weight,
            initial,
            lookup,
        )
        raw = scatter_flat_local_rows(
            raw_flat,
            row_image_ids,
            row_rotation_rows,
            row_present_mask,
            batch_size=batch,
            dense_rotation_count=rotations,
            fill_value=jnp.inf,
        )
        scores = _relion_cuda_fine_diff2_to_scores(
            raw,
            rotation_prior[:, :, None],
            translation_prior[:, None, :],
            candidate_mask,
        )
        posterior = _relion_f32_fine_posterior(
            scores,
            adaptive_fraction=0.999,
        )
        return raw, scores, posterior

    return dense_raw, flat_raw, dense_score_posterior, flat_score_posterior


def _block(value):
    return jax.block_until_ready(value)


def _time_call(fn, args) -> float:
    start = time.perf_counter()
    _block(fn(*args))
    return time.perf_counter() - start


def _measure_abba(control_fn, control_args, candidate_fn, candidate_args, repeats: int) -> tuple[list, list]:
    control_times = []
    candidate_times = []
    for _ in range(repeats):
        control_times.append(_time_call(control_fn, control_args))
        candidate_times.append(_time_call(candidate_fn, candidate_args))
        candidate_times.append(_time_call(candidate_fn, candidate_args))
        control_times.append(_time_call(control_fn, control_args))
    return control_times, candidate_times


def _array_equal(left, right) -> bool:
    return bool(np.asarray(jax.device_get(jnp.all(jnp.asarray(left) == jnp.asarray(right)))))


def _audit_outputs(dense_output, flat_output, candidate_mask, padded_tail_rows: int) -> dict:
    dense_raw, dense_scores, dense_posterior = dense_output
    flat_raw, flat_scores, flat_posterior = flat_output
    active = jnp.broadcast_to(candidate_mask, dense_raw.shape)
    raw_active_equal = _array_equal(
        jnp.where(active, dense_raw, jnp.float32(0.0)),
        jnp.where(active, flat_raw, jnp.float32(0.0)),
    )
    score_equal = _array_equal(dense_scores, flat_scores)
    posterior_equal = [
        _array_equal(control, candidate) for control, candidate in zip(dense_posterior, flat_posterior, strict=True)
    ]
    padded_tail_poison_noop = bool(
        padded_tail_rows > 0 and not np.asarray(jax.device_get(jnp.any(jnp.isnan(flat_raw)))).item()
    )
    return {
        "raw_active_bitwise_equal": raw_active_equal,
        "scores_bitwise_equal": score_equal,
        "posterior_outputs_bitwise_equal": posterior_equal,
        "padded_tail_rows": int(padded_tail_rows),
        "padded_tail_poison_noop": padded_tail_poison_noop,
        "all_bitwise_equal": bool(
            raw_active_equal and score_equal and all(posterior_equal) and padded_tail_poison_noop
        ),
    }


def _summary(times: list[float]) -> dict:
    values = np.asarray(times, dtype=np.float64)
    return {
        "samples_s": values.tolist(),
        "median_s": float(np.median(values)),
        "mean_s": float(np.mean(values)),
        "min_s": float(np.min(values)),
        "max_s": float(np.max(values)),
    }


def benchmark_case(
    metadata_path: Path,
    *,
    repeats: int,
    warmups: int,
    seed: int,
    exactness_only: bool,
) -> dict:
    case = _case_layout(metadata_path)
    operands = _make_operands(case, seed)
    dense_raw, flat_raw, dense_pipeline, flat_pipeline = _build_calls(operands)

    dense_raw_args = (
        operands["reference"],
        operands["image"],
        operands["translation_angles"],
        operands["weight"],
        operands["initial_diff2"],
        operands["lookup"],
    )
    flat_raw_args = (
        operands["flat_reference"],
        operands["row_image_ids"],
        operands["image"],
        operands["translation_angles"],
        operands["weight"],
        operands["initial_diff2"],
        operands["lookup"],
    )
    dense_pipeline_args = dense_raw_args + (
        operands["rotation_log_prior"],
        operands["translation_log_prior"],
        operands["candidate_mask"],
    )
    flat_pipeline_args = (
        operands["flat_reference"],
        operands["row_image_ids"],
        operands["row_rotation_rows"],
        operands["row_present_mask"],
        operands["image"],
        operands["translation_angles"],
        operands["weight"],
        operands["initial_diff2"],
        operands["lookup"],
        operands["rotation_log_prior"],
        operands["translation_log_prior"],
        operands["candidate_mask"],
    )

    for _ in range(warmups):
        _block(dense_raw(*dense_raw_args))
        _block(flat_raw(*flat_raw_args))
        _block(dense_pipeline(*dense_pipeline_args))
        _block(flat_pipeline(*flat_pipeline_args))

    dense_output = _block(dense_pipeline(*dense_pipeline_args))
    flat_output = _block(flat_pipeline(*flat_pipeline_args))
    exactness = _audit_outputs(
        dense_output,
        flat_output,
        operands["candidate_mask"],
        operands["padded_tail_rows"],
    )
    if not exactness["all_bitwise_equal"]:
        raise RuntimeError(f"flat-row exactness failed for {metadata_path}: {exactness}")

    metadata = case["metadata"]
    profile = case["profile"]
    outer_call_count = len(profile["chunk_sizes"])
    if outer_call_count != len(case["chunk_packed_rows"]):
        raise AssertionError(f"{metadata_path}: packed layout changed outer-call cardinality")
    if exactness_only:
        result = {
            "iteration": int(metadata_path.stem.split("_it", 1)[1].split("_", 1)[0]),
            "metadata_path": str(metadata_path),
            "current_size": int(metadata["current_size"]),
            "outer_call_count_control": outer_call_count,
            "outer_call_count_candidate": outer_call_count,
            "outer_call_cardinality_equal": True,
            "padded_tail_rows": int(case["padded_tail_rows"]),
            "exactness": exactness,
        }
        del dense_output, flat_output, operands
        gc.collect()
        jax.clear_caches()
        return result

    dense_raw_times, flat_raw_times = _measure_abba(
        dense_raw,
        dense_raw_args,
        flat_raw,
        flat_raw_args,
        repeats,
    )
    dense_pipeline_times, flat_pipeline_times = _measure_abba(
        dense_pipeline,
        dense_pipeline_args,
        flat_pipeline,
        flat_pipeline_args,
        repeats,
    )
    dense_raw_summary = _summary(dense_raw_times)
    flat_raw_summary = _summary(flat_raw_times)
    dense_pipeline_summary = _summary(dense_pipeline_times)
    flat_pipeline_summary = _summary(flat_pipeline_times)
    pipeline_delta = 100.0 * (flat_pipeline_summary["median_s"] / dense_pipeline_summary["median_s"] - 1.0)
    raw_delta = 100.0 * (flat_raw_summary["median_s"] / dense_raw_summary["median_s"] - 1.0)

    result = {
        "iteration": int(metadata_path.stem.split("_it", 1)[1].split("_", 1)[0]),
        "metadata_path": str(metadata_path),
        "current_size": int(metadata["current_size"]),
        "pixels": int(profile["n_windowed"]),
        "translations": int(metadata["n_translations"]),
        "outer_call_count_control": outer_call_count,
        "outer_call_count_candidate": outer_call_count,
        "outer_call_cardinality_equal": True,
        "dense_batch_size": int(case["dense_batch_size"]),
        "dense_rotation_count": int(case["dense_rotation_count"]),
        "dense_rows_per_call": int(case["dense_batch_size"] * case["dense_rotation_count"]),
        "packed_rows_per_call": int(case["packed_row_count"]),
        "chunk_packed_rows": case["chunk_packed_rows"],
        "padded_tail_rows": int(case["padded_tail_rows"]),
        "logical_packed_rows_across_calls": int(case["logical_packed_rows"]),
        "static_dense_rows_across_calls": int(
            len(profile["chunk_sizes"]) * case["dense_batch_size"] * case["dense_rotation_count"]
        ),
        "static_packed_rows_across_calls": int(len(profile["chunk_sizes"]) * case["packed_row_count"]),
        "static_row_reduction_percent": float(
            100.0 * (1.0 - case["packed_row_count"] / (case["dense_batch_size"] * case["dense_rotation_count"]))
        ),
        "exactness": exactness,
        "raw_control": dense_raw_summary,
        "raw_candidate": flat_raw_summary,
        "raw_candidate_minus_control_percent": raw_delta,
        "score_posterior_control": dense_pipeline_summary,
        "score_posterior_candidate": flat_pipeline_summary,
        "score_posterior_candidate_minus_control_percent": pipeline_delta,
        "estimated_score_posterior_iteration_control_s": float(
            dense_pipeline_summary["median_s"] * len(profile["chunk_sizes"])
        ),
        "estimated_score_posterior_iteration_candidate_s": float(
            flat_pipeline_summary["median_s"] * len(profile["chunk_sizes"])
        ),
        "passes_two_percent_call_gate": bool(pipeline_delta <= -2.0),
    }
    del dense_output, flat_output, operands
    gc.collect()
    jax.clear_caches()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, nargs="+", default=(20, 40, 60, 80))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--exactness-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 1:
        raise ValueError("--repeats and --warmups must be positive")
    if jax.default_backend() != "gpu":
        raise RuntimeError("flat-row scorer benchmark requires a GPU backend")
    if not cuda_backproject.custom_cuda_requested():
        raise RuntimeError("flat-row scorer benchmark requires RECOVAR custom CUDA")

    payload = {
        "schema": "recovar.vdam_flat_row_fine_score_microbenchmark.v1",
        "git_head": _git_head(),
        "device": str(jax.devices("gpu")[0]),
        "result": "completed",
        "mode": "exactness_only" if args.exactness_only else "timed",
        "cases": [
            benchmark_case(
                args.metadata_root / f"run_it{iteration:03d}_recovar_meta.json",
                repeats=args.repeats,
                warmups=args.warmups,
                seed=args.seed + iteration,
                exactness_only=args.exactness_only,
            )
            for iteration in args.iterations
        ],
    }
    payload["all_bitwise_equal"] = all(case["exactness"]["all_bitwise_equal"] for case in payload["cases"])
    if args.exactness_only:
        payload["all_pass_two_percent_call_gate"] = None
        payload["promotion_decision"] = "not_evaluated"
    else:
        payload["all_pass_two_percent_call_gate"] = all(
            case["passes_two_percent_call_gate"] for case in payload["cases"]
        )
        payload["promotion_decision"] = (
            "accept" if payload["all_bitwise_equal"] and payload["all_pass_two_percent_call_gate"] else "reject"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
