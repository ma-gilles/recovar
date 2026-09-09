#!/usr/bin/env python
"""Focused exactness/compile-count gate for stable VDAM fine-score shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject
from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_stable_fourier_window_shape_plan,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_full_to_compact_lookup,
)


def _sha256(array) -> str:
    return hashlib.sha256(np.asarray(array).tobytes(order="C")).hexdigest()


def _timed_call(function, operands):
    start = time.perf_counter()
    result = jax.block_until_ready(function(*operands))
    return result, time.perf_counter() - start


def _make_operands(*, logical_size, physical_size, base, image_shape, translations, initial):
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    plan = make_stable_fourier_window_shape_plan(
        image_shape,
        logical_size,
        n_half,
        enabled=True,
        quantum=8,
    )
    if plan.physical_current_size != physical_size:
        raise AssertionError((logical_size, plan.physical_current_size, physical_size))

    logical_indices = plan.logical_spec.score_indices_np
    packed_indices = plan.packed_indices_np("score")
    logical_lookup = _relion_cuda_fine_full_to_compact_lookup(
        image_shape,
        logical_size,
        logical_indices,
    )
    rectangle_tail = plan.physical_rectangle_pixels - plan.logical_rectangle_pixels
    # A valid lookup in the physical-only tail makes an accidental physical
    # loop bound observable; the runtime kernel must never issue it.
    packed_lookup = np.pad(logical_lookup, (0, rectangle_tail), constant_values=0)

    reference_full, image_full, weight_full = base
    static = (
        jnp.asarray(reference_full[..., logical_indices]),
        jnp.asarray(image_full[..., logical_indices]),
        translations,
        jnp.asarray(weight_full[..., logical_indices]),
        jnp.asarray(logical_lookup),
        initial,
    )
    runtime_reference = reference_full[..., packed_indices].copy()
    runtime_image = image_full[..., packed_indices].copy()
    runtime_weight = weight_full[..., packed_indices].copy()
    logical_pixels = plan.logical_score_pixels
    runtime_reference[..., logical_pixels:] = np.complex64(7.0 + 3.0j)
    runtime_image[..., logical_pixels:] = np.complex64(5.0 + 2.0j)
    runtime_weight[..., logical_pixels:] = np.float32(1.25e5)
    runtime = (
        jnp.asarray(runtime_reference),
        jnp.asarray(runtime_image),
        translations,
        jnp.asarray(runtime_weight),
        jnp.asarray(packed_lookup),
        jnp.asarray(logical_size, dtype=jnp.int32),
        initial,
    )
    return static, runtime, plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("runtime-first", "static-first"), required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if jax.default_backend() != "gpu":
        raise RuntimeError("stable-window benchmark requires a GPU")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    jax.block_until_ready(jnp.arange(1024, dtype=jnp.float32) + 1.0)

    image_shape = (128, 128)
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    logical_sizes = (68, 70, 72)
    physical_size = 72
    batch_size = 2
    rotation_count = 128
    rng = np.random.default_rng(20260831)
    base = (
        (
            rng.normal(0, 0.02, (batch_size, rotation_count, n_half))
            + 1j * rng.normal(0, 0.02, (batch_size, rotation_count, n_half))
        ).astype(np.complex64),
        (
            rng.normal(0, 0.02, (batch_size, n_half))
            + 1j * rng.normal(0, 0.02, (batch_size, n_half))
        ).astype(np.complex64),
        rng.uniform(0, 150_000, (batch_size, n_half)).astype(np.float32),
    )
    translations = jnp.asarray(rng.normal(0, 0.2, (4, 2)).astype(np.float32))
    initial = jnp.asarray(rng.normal(0, 0.1, batch_size).astype(np.float32))
    operands = {
        size: _make_operands(
            logical_size=size,
            physical_size=physical_size,
            base=base,
            image_shape=image_shape,
            translations=translations,
            initial=initial,
        )
        for size in logical_sizes
    }
    jax.block_until_ready(
        tuple(value for static, runtime, _plan in operands.values() for value in (*static, *runtime))
    )

    static_function = cuda_backproject.relion_fine_diff2_fused_translate_rectangular_f32
    runtime_function = (
        cuda_backproject.relion_fine_diff2_fused_translate_runtime_rectangular_f32
    )
    warm_static, warm_runtime, _warm_plan = _make_operands(
        logical_size=64,
        physical_size=64,
        base=base,
        image_shape=image_shape,
        translations=translations,
        initial=initial,
    )
    warm_static_output = jax.block_until_ready(
        static_function(*warm_static, current_size=64)
    )
    warm_runtime_output = jax.block_until_ready(runtime_function(*warm_runtime))
    warmup_bitwise_equal = np.array_equal(
        np.asarray(warm_static_output).view(np.uint32),
        np.asarray(warm_runtime_output).view(np.uint32),
    )
    if not warmup_bitwise_equal:
        raise AssertionError("target warmup differs at current_size=64")
    # Keep FFI registration and the CUDA module resident, but force fresh JAX
    # executable-cache accounting for the measured physical class.
    jax.clear_caches()
    if static_function._cache_size() or runtime_function._cache_size():
        raise AssertionError("benchmark requires cold fine-score JIT caches")

    outputs = {"static": {}, "runtime": {}}
    first_call_seconds = {"static": [], "runtime": []}
    cache_counts = {"static": [], "runtime": []}
    mode_order = ("runtime", "static") if args.order == "runtime-first" else ("static", "runtime")
    for mode in mode_order:
        for logical_size in logical_sizes:
            static, runtime, _plan = operands[logical_size]
            if mode == "static":
                def function(*values, size=logical_size):
                    return static_function(*values, current_size=size)

                active_operands = static
                cache_function = static_function
            else:
                function = runtime_function
                active_operands = runtime
                cache_function = runtime_function
            result, elapsed = _timed_call(function, active_operands)
            outputs[mode][logical_size] = result
            first_call_seconds[mode].append(elapsed)
            cache_counts[mode].append(cache_function._cache_size())

    hashes = {}
    bitwise_equal = {}
    for logical_size in logical_sizes:
        static_output = np.asarray(outputs["static"][logical_size])
        runtime_output = np.asarray(outputs["runtime"][logical_size])
        equal = np.array_equal(static_output.view(np.uint32), runtime_output.view(np.uint32))
        bitwise_equal[str(logical_size)] = bool(equal)
        hashes[str(logical_size)] = {
            "static": _sha256(static_output),
            "runtime": _sha256(runtime_output),
        }
        if not equal:
            raise AssertionError(f"runtime cutoff differs at current_size={logical_size}")

    steady_seconds = {"static": 0.0, "runtime": 0.0}
    steady_size = logical_sizes[1]
    static, runtime, _plan = operands[steady_size]
    for mode in reversed(mode_order):
        start = time.perf_counter()
        for _ in range(args.repeats):
            if mode == "static":
                result = static_function(*static, current_size=steady_size)
            else:
                result = runtime_function(*runtime)
            jax.block_until_ready(result)
        steady_seconds[mode] = time.perf_counter() - start

    static_compile_calls = int(cache_counts["static"][-1])
    runtime_compile_calls = int(cache_counts["runtime"][-1])
    static_first_total = float(sum(first_call_seconds["static"]))
    runtime_first_total = float(sum(first_call_seconds["runtime"]))
    report = {
        "order": args.order,
        "logical_sizes": logical_sizes,
        "physical_size": physical_size,
        "target_warmup_size": 64,
        "target_warmup_bitwise_equal": bool(warmup_bitwise_equal),
        "physical_score_pixels": [operands[size][2].physical_score_pixels for size in logical_sizes],
        "logical_score_pixels": [operands[size][2].logical_score_pixels for size in logical_sizes],
        "bitwise_equal": bitwise_equal,
        "output_sha256": hashes,
        "invocation_count_per_mode": len(logical_sizes),
        "jit_cache_counts_after_each_call": cache_counts,
        "unique_compile_calls": {
            "static": static_compile_calls,
            "runtime": runtime_compile_calls,
        },
        "first_call_seconds": first_call_seconds,
        "first_call_total_seconds": {
            "static": static_first_total,
            "runtime": runtime_first_total,
        },
        "first_call_total_gain_fraction": (
            (static_first_total - runtime_first_total) / static_first_total
            if static_first_total
            else 0.0
        ),
        "steady_repeats": args.repeats,
        "steady_total_seconds": steady_seconds,
        "steady_runtime_over_static": (
            steady_seconds["runtime"] / steady_seconds["static"]
            if steady_seconds["static"]
            else None
        ),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
