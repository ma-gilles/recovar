#!/usr/bin/env python3
"""Qualify selected and fail-closed K=1 hybrid batches on one Slurm H100."""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from scripts.qualify_vdam_gf46_certificate_state_h100 import (
    GF46_GEOMETRY,
    _assert_exact_geometry,
    _block_tree,
    _device_memory_stats,
    _repository_provenance,
    _runtime_provenance,
    _seal_output,
    _sha256_bytes,
    _sha256_file,
    _utc_now,
    _write_artifact_manifest,
    _write_json,
)

SOURCE_FILES = (
    "recovar/cuda/Makefile",
    "recovar/cuda/cuda_backproject.cu",
    "recovar/cuda_backproject.py",
    "recovar/em/scoring/coarse_gemm_hybrid.py",
    "recovar/em/scoring/scoring.py",
    "recovar/em/scoring/significance.py",
    "scripts/qualify_vdam_gf46_certificate_state_h100.py",
    "scripts/qualify_vdam_gf46_hybrid_batch_h100.py",
    "scripts/run_vdam_gf46_hybrid_batch_h100.sbatch",
    "scripts/vdam_gpu_selection.sh",
    "tests/unit/test_coarse_gemm_certificate_scoring.py",
    "tests/unit/test_coarse_gemm_hybrid.py",
    "tests/unit/test_coarse_gemm_hybrid_significance.py",
    "tests/unit/test_coarse_gaussian_gemm_macro.py",
    "tests/unit/initial_model/test_vdam_gf46_hybrid_batch_h100.py",
    "pixi.lock",
    "pixi.toml",
    "pyproject.toml",
)
SELECTED_SOURCE_BLOCKS = 2
SELECTED_CAPACITY = 8
FALLBACK_CAPACITY = 1
FALLBACK_BATCH_SIZE = 8
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _source_manifest(repo_root: Path) -> tuple[bytes, list[dict[str, Any]]]:
    entries = []
    lines = []
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"hybrid qualification source is missing: {relative}")
        digest = _sha256_file(path)
        entries.append(
            {
                "path": relative,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            },
        )
        lines.append(f"{digest}  {relative}\n")
    return "".join(lines).encode(), entries


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expected-repo-head")
    parser.add_argument("--expected-repo-tree")
    parser.add_argument("--expected-source-manifest-sha256")
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument("--print-source-manifest-sha256", action="store_true")
    return parser.parse_args(argv)


def _synchronize_result(result) -> None:
    result.scores.block_until_ready()
    result.raw_score_max.block_until_ready()


def _selected_summary(result) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    g = GF46_GEOMETRY
    selected_rotations = SELECTED_SOURCE_BLOCKS * g.source_rotation_block
    scores = result.scores
    finite_count, best_pose, finite_minimum, finite_maximum = _block_tree(
        (
            jnp.sum(jnp.isfinite(scores), dtype=jnp.int64),
            jnp.argmax(scores.reshape(g.batch_size, -1), axis=1),
            jnp.min(jnp.where(jnp.isfinite(scores), scores, jnp.inf)),
            jnp.max(jnp.where(jnp.isfinite(scores), scores, -jnp.inf)),
        ),
    )
    selected_values = np.asarray(
        jax.device_get(scores[:, :selected_rotations, :]),
        dtype=np.float32,
    )
    raw_max = np.asarray(jax.device_get(result.raw_score_max), dtype=np.float32)
    block_ids = np.asarray(result.selection.block_ids, dtype=np.int32)
    block_count = np.asarray(result.selection.block_count, dtype=np.int32)
    expected_finite = g.batch_size * selected_rotations * g.translation_count
    checks = {
        "used_selected_rescore": bool(result.used_selected_rescore),
        "no_fallback_reason": result.fallback_reason is None,
        "scores_include_priors": bool(result.scores_include_priors),
        "selected_block_count_exact": bool(np.all(block_count == SELECTED_SOURCE_BLOCKS)),
        "selected_block_ids_exact": bool(np.all(block_ids[:, :SELECTED_SOURCE_BLOCKS] == (0, 1))),
        "selected_block_padding_exact": bool(np.all(block_ids[:, SELECTED_SOURCE_BLOCKS:] == -1)),
        "finite_candidate_count_exact": int(np.asarray(finite_count)) == expected_finite,
        "selected_scores_exact": bool(
            np.all(selected_values == np.float32(-0.125)),
        ),
        "raw_score_max_exact": bool(np.all(raw_max == np.float32(-0.125))),
        "best_pose_exact": bool(np.all(np.asarray(best_pose) == 0)),
        "finite_minimum_exact": float(np.asarray(finite_minimum)) == -0.125,
        "finite_maximum_exact": float(np.asarray(finite_maximum)) == -0.125,
    }
    if not all(checks.values()):
        raise RuntimeError(f"selected hybrid semantic checks failed: {checks}")
    combined = hashlib.sha256()
    combined.update(selected_values.tobytes(order="C"))
    combined.update(raw_max.tobytes(order="C"))
    combined.update(block_ids.tobytes(order="C"))
    combined.update(block_count.tobytes(order="C"))
    return {
        "schema": "recovar.vdam.gf46_hybrid_selected_summary.v1",
        "checks": checks,
        "combined_sha256": combined.hexdigest(),
        "finite_candidate_count": int(np.asarray(finite_count)),
        "full_candidate_count": g.batch_size * g.total_rotations * g.translation_count,
        "selected_candidate_fraction": expected_finite
        / (g.batch_size * g.total_rotations * g.translation_count),
        "selected_block_count_minimum": int(block_count.min()),
        "selected_block_count_maximum": int(block_count.max()),
    }


def _posterior_summary(result) -> tuple[dict[str, Any], float]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from recovar.em.helpers.oversampling import relion_cuda_f32_coarse_posterior

    g = GF46_GEOMETRY
    started = time.perf_counter_ns()
    posterior = relion_cuda_f32_coarse_posterior(
        result.scores.reshape(g.batch_size, -1),
        adaptive_fraction=0.999,
        max_significants=2_000,
        tie_score_ulps=0,
        min_diff2_offsets=-result.raw_score_max,
    )
    _block_tree(posterior)
    seconds = (time.perf_counter_ns() - started) / 1e9
    weights, mask, n_significant, cutoff_count, sum_weight, threshold = posterior
    expected_count = SELECTED_SOURCE_BLOCKS * g.source_rotation_block * g.translation_count
    fields = tuple(
        np.asarray(jax.device_get(value))
        for value in (
            jnp.sum(mask, axis=1),
            n_significant,
            cutoff_count,
            sum_weight,
            threshold,
            jnp.max(weights, axis=1),
            jnp.sum(weights, axis=1, dtype=jnp.float32),
        )
    )
    mask_count, n_sig, cutoff, total, cutoff_weight, max_probability, probability_sum = fields
    checks = {
        "mask_count_exact": bool(np.all(mask_count == expected_count)),
        "n_significant_exact": bool(np.all(n_sig == expected_count)),
        "cutoff_count_exact": bool(np.all(cutoff == expected_count)),
        "sum_weight_finite_positive_and_row_identical": bool(
            np.all(np.isfinite(total))
            and np.all(total > 0.0)
            and np.all(total.view(np.uint32) == total[:1].view(np.uint32))
        ),
        "cutoff_weight_finite_positive_and_row_identical": bool(
            np.all(np.isfinite(cutoff_weight))
            and np.all(cutoff_weight > 0.0)
            and np.all(
                cutoff_weight.view(np.uint32)
                == cutoff_weight[:1].view(np.uint32)
            )
        ),
        "maximum_probability_finite_positive_and_row_identical": bool(
            np.all(np.isfinite(max_probability))
            and np.all(max_probability > 0.0)
            and np.all(
                max_probability.view(np.uint32)
                == max_probability[:1].view(np.uint32)
            )
        ),
        "probability_sum_near_one": bool(
            np.allclose(probability_sum, np.float32(1.0), rtol=2e-5, atol=2e-5),
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"mature posterior semantic checks failed: {checks}")
    del posterior, weights, mask
    gc.collect()
    return {
        "schema": "recovar.vdam.gf46_hybrid_posterior_summary.v1",
        "checks": checks,
        "expected_equal_weight_support": expected_count,
    }, seconds


def _fallback_summary(result) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    g = GF46_GEOMETRY
    finite_count, raw_max = _block_tree(
        (
            jnp.sum(jnp.isfinite(result.scores), dtype=jnp.int64),
            result.raw_score_max,
        ),
    )
    raw_max_np = np.asarray(jax.device_get(raw_max), dtype=np.float32)
    expected_finite = FALLBACK_BATCH_SIZE * g.total_rotations * g.translation_count
    checks = {
        "selected_rescore_disabled": not bool(result.used_selected_rescore),
        "raw_scores_have_no_priors": not bool(result.scores_include_priors),
        "capacity_overflow_reason": result.fallback_reason == "block_capacity_overflow",
        "finite_candidate_count_exact": int(np.asarray(finite_count)) == expected_finite,
        "raw_score_max_exact": bool(np.all(raw_max_np == np.float32(-0.125))),
    }
    if not all(checks.values()):
        raise RuntimeError(f"full-direct fallback semantic checks failed: {checks}")
    return {
        "schema": "recovar.vdam.gf46_hybrid_fallback_summary.v1",
        "checks": checks,
        "finite_candidate_count": int(np.asarray(finite_count)),
    }


def _execute(timed_runs: int) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from recovar import cuda_backproject
    from recovar.em.scoring import significance
    from recovar.em.scoring.coarse_gemm_hybrid import plan_coarse_gemm_certificate_topology

    _assert_exact_geometry()
    if not cuda_backproject.cuda_available():
        raise RuntimeError("custom CUDA library is unavailable")
    g = GF46_GEOMETRY
    if g.total_rotations % g.source_rotation_block:
        raise RuntimeError("GF46 rotation count is not source16 aligned")
    topology = plan_coarse_gemm_certificate_topology(
        np.arange(g.pixel_count, dtype=np.int32),
        compact_pixel_count=g.pixel_count,
        translation_count=g.translation_count,
    )

    allocation_started = time.perf_counter_ns()
    near = np.complex64(0.125 + 0.0625j)
    far = np.complex64(1.0 - 0.5j)
    cache = jnp.full(
        (1, g.total_rotations, g.pixel_count),
        far,
        dtype=jnp.complex64,
    )
    cache = cache.at[
        0,
        : SELECTED_SOURCE_BLOCKS * g.source_rotation_block,
        :,
    ].set(near)
    shifted = jnp.full(
        (g.batch_size, g.translation_count, g.pixel_count),
        near,
        dtype=jnp.complex64,
    )
    weight = jnp.full(
        (g.batch_size, g.pixel_count),
        np.float32(0.75),
        dtype=jnp.float32,
    )
    initial = jnp.full(
        (g.batch_size,),
        np.float32(0.125),
        dtype=jnp.float32,
    )
    _block_tree((cache, shifted, weight, initial))
    allocation_seconds = (time.perf_counter_ns() - allocation_started) / 1e9

    def selected_call():
        return significance._compute_coarse_gaussian_gemm_hybrid_batch(
            cache,
            shifted,
            weight,
            initial,
            topology=topology,
            actual_image_count=g.batch_size,
            class_log_prior=np.float32(0.0),
            certificate_chunk_rows=g.rotation_block,
            block_capacity=SELECTED_CAPACITY,
        )

    warm_started = time.perf_counter_ns()
    selected_result = selected_call()
    _synchronize_result(selected_result)
    warm_seconds = (time.perf_counter_ns() - warm_started) / 1e9
    first_summary = _selected_summary(selected_result)
    del selected_result
    gc.collect()

    timed_seconds = []
    final_summary = None
    selected_result = None
    for _ in range(timed_runs):
        started = time.perf_counter_ns()
        selected_result = selected_call()
        _synchronize_result(selected_result)
        timed_seconds.append((time.perf_counter_ns() - started) / 1e9)
        final_summary = _selected_summary(selected_result)
    if final_summary is None or selected_result is None:
        raise RuntimeError("selected timing did not produce an output")
    if first_summary["combined_sha256"] != final_summary["combined_sha256"]:
        raise RuntimeError("repeated selected hybrid summaries are not byte-identical")

    posterior, posterior_seconds = _posterior_summary(selected_result)
    del selected_result
    gc.collect()

    fallback_started = time.perf_counter_ns()
    fallback = significance._compute_coarse_gaussian_gemm_hybrid_batch(
        cache,
        shifted[:FALLBACK_BATCH_SIZE],
        weight[:FALLBACK_BATCH_SIZE],
        initial[:FALLBACK_BATCH_SIZE],
        topology=topology,
        actual_image_count=FALLBACK_BATCH_SIZE,
        class_log_prior=np.float32(0.0),
        certificate_chunk_rows=g.rotation_block,
        block_capacity=FALLBACK_CAPACITY,
    )
    _synchronize_result(fallback)
    fallback_seconds = (time.perf_counter_ns() - fallback_started) / 1e9
    fallback_summary = _fallback_summary(fallback)
    del fallback
    gc.collect()

    return {
        "schema": "recovar.vdam.gf46_hybrid_batch_h100_execution.v1",
        "geometry": GF46_GEOMETRY._asdict(),
        "selected_capacity": SELECTED_CAPACITY,
        "fallback_capacity": FALLBACK_CAPACITY,
        "fallback_batch_size": FALLBACK_BATCH_SIZE,
        "allocation_seconds": allocation_seconds,
        "selected_warm_compile_and_execute_seconds": warm_seconds,
        "selected_warm_timed_seconds": timed_seconds,
        "selected_warm_timed_summary_seconds": {
            "minimum": min(timed_seconds),
            "median": statistics.median(timed_seconds),
            "mean": statistics.fmean(timed_seconds),
            "maximum": max(timed_seconds),
        },
        "selected_repeat_summary_byte_identical": True,
        "selected": final_summary,
        "mature_relion_f32_posterior": posterior,
        "mature_relion_f32_posterior_seconds": posterior_seconds,
        "forced_full_direct_fallback": fallback_summary,
        "forced_full_direct_fallback_seconds": fallback_seconds,
        "device_memory_stats_after_execution": _device_memory_stats(
            jax.devices("gpu")[0],
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    if args.print_source_manifest_sha256:
        manifest, _entries = _source_manifest(repo_root)
        print(_sha256_bytes(manifest))
        return 0
    required = (
        args.output_root,
        args.expected_repo_head,
        args.expected_repo_tree,
        args.expected_source_manifest_sha256,
        args.expected_gpu_uuid,
    )
    if any(value is None for value in required):
        raise SystemExit("output, repository, and GPU qualification arguments are required")
    if args.timed_runs < 2:
        raise SystemExit("hybrid qualification requires at least two timed runs")
    if not _SHA256_RE.fullmatch(args.expected_source_manifest_sha256):
        raise SystemExit("expected source manifest must be a lowercase SHA-256")
    if Path.cwd().resolve() != repo_root:
        raise SystemExit(f"run from the repository root: {repo_root}")
    output_root = args.output_root.resolve()
    if not args.output_root.is_absolute() or output_root.exists():
        raise SystemExit(f"output root must be a new absolute path: {args.output_root}")
    if not output_root.parent.is_dir() or output_root.is_relative_to(repo_root):
        raise SystemExit(f"invalid output root: {output_root}")

    output_root.mkdir()
    (output_root / "provenance").mkdir()
    (output_root / "results").mkdir()
    (output_root / "SAFE_TO_DELETE").write_text(
        "Disposable GF46 H100 hybrid-batch qualification artifact.\n",
    )
    failure = None
    stage = "repository_and_runtime"
    report: dict[str, Any] = {
        "schema": "recovar.vdam.gf46_hybrid_batch_h100_qualification.v1",
        "classification": "incomplete",
        "geometry": GF46_GEOMETRY._asdict(),
    }
    try:
        repository = _repository_provenance(
            repo_root,
            expected_head=args.expected_repo_head,
            expected_tree=args.expected_repo_tree,
        )
        initial_manifest, source_entries = _source_manifest(repo_root)
        source_digest = _sha256_bytes(initial_manifest)
        if source_digest != args.expected_source_manifest_sha256:
            raise RuntimeError(
                "source-manifest mismatch: "
                f"expected={args.expected_source_manifest_sha256}, observed={source_digest}",
            )
        runtime = _runtime_provenance(repo_root, args.expected_gpu_uuid)
        cuda_library = Path(os.environ["RECOVAR_CUDA_LIB"]).resolve()
        if not cuda_library.is_file():
            raise RuntimeError(f"custom CUDA library is missing: {cuda_library}")
        _write_json(output_root / "provenance" / "repository.json", repository)
        _write_json(output_root / "provenance" / "runtime.json", runtime)
        _write_json(
            output_root / "provenance" / "source_manifest.json",
            {
                "sha256": source_digest,
                "entries": source_entries,
            },
        )
        (output_root / "provenance" / "source_manifest.sha256").write_bytes(
            initial_manifest,
        )
        _write_json(
            output_root / "provenance" / "cuda_library.json",
            {
                "path": str(cuda_library),
                "size_bytes": cuda_library.stat().st_size,
                "sha256": _sha256_file(cuda_library),
            },
        )

        stage = "selected_and_fallback_execution"
        execution = _execute(args.timed_runs)
        _write_json(output_root / "results" / "execution.json", execution)
        report.update(
            {
                "classification": "qualified_selected_and_fail_closed_h100",
                "completed_utc": _utc_now(),
                "repository": repository,
                "runtime": runtime,
                "source_manifest_sha256": source_digest,
                "execution": execution,
            },
        )

        stage = "final_source_verification"
        final_manifest, _entries = _source_manifest(repo_root)
        if final_manifest != initial_manifest:
            raise RuntimeError("hybrid qualification source changed during execution")
        if _repository_provenance(
            repo_root,
            expected_head=args.expected_repo_head,
            expected_tree=args.expected_repo_tree,
        ) != repository:
            raise RuntimeError("repository provenance changed during execution")
        _write_json(output_root / "results" / "qualification.json", report)
        (output_root / "COMPLETED").write_text(
            "GF46 H100 selected/fallback hybrid qualification completed.\n",
        )
    except BaseException as error:
        failure = error
        report["classification"] = "qualification_failed"
        report["failure"] = {
            "stage": stage,
            "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
            "traceback": traceback.format_exc(),
            "timestamp_utc": _utc_now(),
        }
        _write_json(output_root / "results" / "qualification.json", report)
        (output_root / "FAILED").write_text(
            f"{stage}: {type(error).__name__}: {error}\n",
        )
    finally:
        _write_artifact_manifest(output_root)
        _seal_output(output_root)
    if failure is not None:
        print(f"hybrid H100 qualification failed at {stage}: {failure}", file=sys.stderr)
        return 1
    print(f"hybrid H100 qualification complete: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
