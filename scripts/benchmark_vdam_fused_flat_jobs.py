"""Benchmark globally packed RELION fine jobs against per-image pair padding."""

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
from recovar.em.scoring.compact_candidates import build_compact_fine_job_plan, build_compact_pair_index_arrays


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    real = rng.standard_normal(shape, dtype=np.float32)
    imag = rng.standard_normal(shape, dtype=np.float32)
    return (real + np.complex64(1j) * imag).astype(np.complex64, copy=False)


def _elapsed(call) -> float:
    started = time.perf_counter()
    call().block_until_ready()
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-size", type=int, default=88)
    parser.add_argument("--logical-size", type=int, default=84)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    if jax.default_backend() != "gpu":
        raise RuntimeError("flat fine-job benchmark requires a GPU")
    if not cuda_backproject.cuda_available():
        raise RuntimeError(cuda_backproject.cuda_unavailable_error())
    if args.logical_size > args.physical_size or args.logical_size % 2:
        raise ValueError("logical size must be even and no larger than physical size")

    # This skew matches the observed production failure mode: roughly 99k
    # selected hypotheses, but one outlier image forces every image to carry a
    # 50,176-slot row in the old B x P ABI.
    counts = np.asarray(
        [
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            50176,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
        ],
        dtype=np.int32,
    )
    batch_size = int(counts.size)
    rotation_count = 256
    translation_count = 196
    dense_count = rotation_count * translation_count
    candidate_mask = np.zeros(
        (batch_size, rotation_count, translation_count),
        dtype=bool,
    )
    for image_row, count in enumerate(counts):
        candidate_mask[image_row].reshape(-1)[: int(count)] = True

    reference_lookup = np.arange(
        batch_size * rotation_count,
        dtype=np.int32,
    ).reshape(batch_size, rotation_count)
    pair = build_compact_pair_index_arrays(candidate_mask)
    flat = build_compact_fine_job_plan(candidate_mask, reference_lookup)
    pair_mask = np.asarray(pair["pair_mask"], dtype=bool)
    pair_reference_rows = np.where(
        pair_mask,
        reference_lookup[
            np.arange(batch_size, dtype=np.int32)[:, None],
            np.maximum(pair["local_rotation_row"], 0),
        ],
        -1,
    ).astype(np.int32, copy=False)

    rng = np.random.default_rng(20260903)
    physical_pixels = args.physical_size * (args.physical_size // 2 + 1)
    logical_pixels = args.logical_size * (args.logical_size // 2 + 1)
    reference = _complex_normal(
        rng,
        (batch_size * rotation_count, physical_pixels),
    )
    image = _complex_normal(rng, (batch_size, physical_pixels))
    translation_angles = rng.normal(
        0.0,
        0.2,
        (translation_count, 2),
    ).astype(np.float32)
    weight = rng.uniform(
        0.0,
        150_000.0,
        (batch_size, physical_pixels),
    ).astype(np.float32)
    initial_diff2 = rng.uniform(0.0, 0.1, batch_size).astype(np.float32)
    lookup = np.pad(
        np.arange(logical_pixels, dtype=np.int32),
        (0, physical_pixels - logical_pixels),
        constant_values=0,
    )

    operands = tuple(
        jnp.asarray(value)
        for value in (
            reference,
            image,
            translation_angles,
            weight,
            lookup,
            initial_diff2,
        )
    )
    pair_reference_device = jnp.asarray(pair_reference_rows)
    pair_translation_device = jnp.asarray(pair["translation_idx"])
    job_plan_device = jnp.asarray(flat["job_plan"])
    logical_size_device = jnp.asarray(args.logical_size, dtype=jnp.int32)

    def pair_call():
        return cuda_backproject.relion_fine_diff2_fused_translate_runtime_pairs_f32(
            operands[0],
            operands[1],
            operands[2],
            operands[3],
            pair_reference_device,
            pair_translation_device,
            operands[4],
            logical_size_device,
            operands[5],
        )

    def job_call():
        return cuda_backproject.relion_fine_diff2_fused_translate_runtime_jobs_f32(
            operands[0],
            operands[1],
            operands[2],
            operands[3],
            job_plan_device,
            operands[4],
            logical_size_device,
            operands[5],
        )

    for _ in range(args.warmups):
        pair_call().block_until_ready()
        job_call().block_until_ready()
    pair_result = np.asarray(pair_call())
    job_result = np.asarray(job_call())
    valid_job_count = int(flat["valid_job_count"])
    bitwise_equal = np.array_equal(
        pair_result[pair_mask].view(np.uint32),
        job_result[:valid_job_count].view(np.uint32),
    )
    inert_tail = bool(np.all(np.isposinf(job_result[valid_job_count:])))
    if not bitwise_equal or not inert_tail:
        raise AssertionError("global compact jobs changed score bits or padding")

    pair_times: list[float] = []
    job_times: list[float] = []
    for _ in range(args.repeats):
        pair_times.append(_elapsed(pair_call))
        job_times.append(_elapsed(job_call))
        job_times.append(_elapsed(job_call))
        pair_times.append(_elapsed(pair_call))
    pair_median = statistics.median(pair_times)
    job_median = statistics.median(job_times)
    pair_slots = int(batch_size * pair["pair_bucket_size"])
    job_slots = int(flat["job_bucket_size"])
    payload = {
        "schema": "recovar.vdam_fused_flat_jobs_microbenchmark.v1",
        "git_head": _git_head(),
        "backend": jax.default_backend(),
        "device": str(jax.devices("gpu")[0]),
        "logical_size": args.logical_size,
        "physical_size": args.physical_size,
        "batch_size": batch_size,
        "rotation_count": rotation_count,
        "translation_count": translation_count,
        "dense_candidates_per_image": dense_count,
        "per_image_valid_counts": counts.tolist(),
        "valid_job_count": valid_job_count,
        "pair_slots": pair_slots,
        "job_slots": job_slots,
        "pair_blocks": (pair_slots + 3) // 4,
        "job_blocks": (job_slots + 3) // 4,
        "slot_reduction": pair_slots / job_slots,
        "bitwise_equal": bool(bitwise_equal),
        "inert_tail": inert_tail,
        "pair_times_s": pair_times,
        "job_times_s": job_times,
        "pair_median_s": pair_median,
        "job_median_s": job_median,
        "speedup": pair_median / job_median,
        "warmups": args.warmups,
        "repeats": args.repeats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
