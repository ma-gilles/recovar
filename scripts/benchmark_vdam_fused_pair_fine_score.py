"""Benchmark selected-pair RELION fine scoring against dense flat rows."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _timed(call) -> tuple[jax.Array, float]:
    started = time.perf_counter()
    result = call()
    result.block_until_ready()
    return result, time.perf_counter() - started


def _complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    real = rng.standard_normal(shape, dtype=np.float32)
    imag = rng.standard_normal(shape, dtype=np.float32)
    return (real + np.complex64(1j) * imag).astype(np.complex64, copy=False)


def _pair_indices(
    *,
    batch_size: int,
    rotation_count: int,
    translation_count: int,
    pair_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_count = rotation_count * translation_count
    if pair_count > candidate_count:
        raise ValueError(
            f"pair_count {pair_count} exceeds dense candidates {candidate_count}"
        )
    # 197 is coprime to the representative 256 * 196 candidate rectangle.
    # The batch-dependent phase avoids identical launch addresses while
    # retaining unique, source-order-independent selections within each row.
    base = np.arange(pair_count, dtype=np.int64)
    pair_reference_rows = np.empty((batch_size, pair_count), dtype=np.int32)
    pair_translation_ids = np.empty_like(pair_reference_rows)
    for batch in range(batch_size):
        dense_ids = (base * 197 + batch * 101) % candidate_count
        pair_reference_rows[batch] = (
            batch * rotation_count + dense_ids // translation_count
        ).astype(np.int32)
        pair_translation_ids[batch] = (dense_ids % translation_count).astype(
            np.int32
        )
    return pair_reference_rows, pair_translation_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-size", type=int, default=88)
    parser.add_argument("--logical-size", type=int, default=84)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rotation-count", type=int, default=256)
    parser.add_argument("--translation-count", type=int, default=196)
    parser.add_argument("--pair-counts", type=int, nargs="+", default=(256, 1024, 4096))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if jax.default_backend() != "gpu":
        raise RuntimeError("fused-pair fine-score benchmark requires a GPU")
    if not cuda_backproject.cuda_available():
        raise RuntimeError(cuda_backproject.cuda_unavailable_error())
    if args.logical_size > args.physical_size:
        raise ValueError("logical size cannot exceed physical size")
    if args.logical_size <= 0 or args.logical_size % 2:
        raise ValueError("logical size must be a positive even integer")

    rng = np.random.default_rng(20260903)
    physical_pixels = args.physical_size * (args.physical_size // 2 + 1)
    logical_pixels = args.logical_size * (args.logical_size // 2 + 1)
    flat_row_count = args.batch_size * args.rotation_count
    reference = _complex_normal(rng, (flat_row_count, physical_pixels))
    image = _complex_normal(rng, (args.batch_size, physical_pixels))
    translation_angles = rng.normal(
        0.0,
        0.2,
        (args.translation_count, 2),
    ).astype(np.float32)
    weight = rng.uniform(
        0.0,
        150_000.0,
        (args.batch_size, physical_pixels),
    ).astype(np.float32)
    initial_diff2 = rng.uniform(0.0, 0.1, args.batch_size).astype(np.float32)
    row_image_ids = np.repeat(
        np.arange(args.batch_size, dtype=np.int32),
        args.rotation_count,
    )
    lookup = np.pad(
        np.arange(logical_pixels, dtype=np.int32),
        (0, physical_pixels - logical_pixels),
        constant_values=0,
    )

    device_operands = tuple(
        jnp.asarray(value)
        for value in (
            reference,
            row_image_ids,
            image,
            translation_angles,
            weight,
            lookup,
            initial_diff2,
        )
    )
    logical_size = jnp.asarray(args.logical_size, dtype=jnp.int32)

    def dense_call():
        return cuda_backproject.relion_fine_diff2_fused_translate_runtime_flat_rows_f32(
            *device_operands[:-1],
            logical_size,
            device_operands[-1],
        )

    for _ in range(args.warmups):
        dense_call().block_until_ready()
    dense_reference = dense_call()
    dense_reference.block_until_ready()

    cases = []
    for pair_count in args.pair_counts:
        pair_reference_rows, pair_translation_ids = _pair_indices(
            batch_size=args.batch_size,
            rotation_count=args.rotation_count,
            translation_count=args.translation_count,
            pair_count=pair_count,
        )
        pair_reference_rows_device = jnp.asarray(pair_reference_rows)
        pair_translation_ids_device = jnp.asarray(pair_translation_ids)

        def pair_call():
            return cuda_backproject.relion_fine_diff2_fused_translate_runtime_pairs_f32(
                device_operands[0],
                device_operands[2],
                device_operands[3],
                device_operands[4],
                pair_reference_rows_device,
                pair_translation_ids_device,
                device_operands[5],
                logical_size,
                device_operands[6],
            )

        for _ in range(args.warmups):
            pair_call().block_until_ready()
        pair_reference = pair_call()
        pair_reference.block_until_ready()
        expected = np.asarray(dense_reference)[
            pair_reference_rows,
            pair_translation_ids,
        ]
        actual = np.asarray(pair_reference)
        bitwise_equal = bool(
            np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
        )
        if not bitwise_equal:
            raise AssertionError(f"pair_count={pair_count} is not bitwise exact")

        dense_times = []
        pair_times = []
        for _ in range(args.repeats):
            _, elapsed = _timed(dense_call)
            dense_times.append(elapsed)
            _, elapsed = _timed(pair_call)
            pair_times.append(elapsed)
            _, elapsed = _timed(pair_call)
            pair_times.append(elapsed)
            _, elapsed = _timed(dense_call)
            dense_times.append(elapsed)
        dense_median = statistics.median(dense_times)
        pair_median = statistics.median(pair_times)
        cases.append(
            {
                "pair_count": pair_count,
                "candidate_fraction": pair_count
                / (args.rotation_count * args.translation_count),
                "bitwise_equal": bitwise_equal,
                "dense_times_s": dense_times,
                "pair_times_s": pair_times,
                "dense_median_s": dense_median,
                "pair_median_s": pair_median,
                "speedup": dense_median / pair_median,
            }
        )

    payload = {
        "schema": "recovar.vdam_fused_pair_fine_score_microbenchmark.v1",
        "git_head": _git_head(),
        "backend": jax.default_backend(),
        "device": str(jax.devices("gpu")[0]),
        "physical_size": args.physical_size,
        "logical_size": args.logical_size,
        "batch_size": args.batch_size,
        "rotation_count": args.rotation_count,
        "translation_count": args.translation_count,
        "flat_row_count": flat_row_count,
        "physical_pixels": physical_pixels,
        "logical_pixels": logical_pixels,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "all_bitwise_equal": all(case["bitwise_equal"] for case in cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
