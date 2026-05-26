#!/usr/bin/env python
"""Run PPCA refinement quality/performance regression guards.

This is intentionally fixture-driven.  It protects the dense PPCA speedups
added during the K-class/PPCA integration work:

* sparse pass-2 must skip most pass-2 rotation blocks on the easy GT-pose
  ribosome fixture,
* compiled PPCA block boundaries must keep steady-state iteration time low,
* pose recovery and fixed-statistics M-step objective diagnostics must remain
  sane.

Run through pixi from the repository root, for example:

    pixi run python scripts/run_ppca_refinement_regression_guard.py

Add ``--run-kclass-benchmark`` when you also want the slower apples-to-apples
PPCA-vs-K-class timing comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


DEFAULT_FIXTURE_ROOT = Path("/scratch/gpfs/GILLES/mg6942/tmp/ppca_ribosome_k4_gtpose_20260506")
DEFAULT_G128_WINDOW_FIXTURE_ROOT = Path("/scratch/gpfs/GILLES/mg6942/tmp/ppca_g128_window_smoke_20260506")


@dataclass(frozen=True)
class GuardThresholds:
    max_first_iter_s: float = 45.0
    max_steady_iter_s: float = 20.0
    min_final_rotation_exact: float = 0.999
    min_final_translation_exact: float = 0.999
    min_pmax_mean: float = 0.999
    max_nsig_mean: float = 1.05
    min_sparse_skip_fraction: float = 0.85
    min_mstep_solved_delta_per_image: float = -1.0e-4
    max_ppca_benchmark_s: float = 30.0
    max_ppca_over_kclass_adjusted: float = 0.5


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _fixture_paths(fixture_root: Path) -> dict[str, Path]:
    return {
        "data_star": fixture_root / "dataset_g64_n5000_noise1e-6_seed20260506/test_dataset/particles.star",
        "simulation_info": fixture_root / "dataset_g64_n5000_noise1e-6_seed20260506/test_dataset/simulation_info.pkl",
        "init_npz": fixture_root / "init_gt_mean_randomW_q3_seed20260506/ppca_init.npz",
        "prior_init_npz": fixture_root / "init_gt_scaled_k4_q3_recovar_dft/ppca_init.npz",
    }


def _g128_window_fixture_paths(fixture_root: Path) -> dict[str, Path]:
    return {
        "data_star": fixture_root / "dataset_g128_n5000_noise1e-6_seed20260506/test_dataset/particles.star",
        "simulation_info": fixture_root / "dataset_g128_n5000_noise1e-6_seed20260506/test_dataset/simulation_info.pkl",
        "init_npz": fixture_root / "init_assets_k3_q2_g128_from_simulator_n5000/ppca_init.npz",
    }


def require_fixture(paths: dict[str, Path], *, label: str = "PPCA regression") -> None:
    missing = [f"{name}={path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {label} fixture files:\n  "
            + "\n  ".join(missing)
            + "\nRecreate or copy the fixture before running this guard."
        )


def validate_ppca_summary(
    summary: dict[str, Any],
    thresholds: GuardThresholds,
    *,
    require_monotone_objective: bool = True,
    require_fourier_window: bool = False,
    expected_image_shape: tuple[int, int] | None = None,
) -> list[str]:
    failures: list[str] = []
    if not bool(summary.get("passed", False)):
        failures.append("summary['passed'] is false")
    if expected_image_shape is not None and tuple(summary.get("image_shape", ())) != tuple(expected_image_shape):
        failures.append(f"image_shape {summary.get('image_shape')} != {list(expected_image_shape)}")
    iterations = list(summary.get("iterations", []))
    if len(iterations) < 2:
        failures.append(f"expected at least 2 PPCA iterations, got {len(iterations)}")
        return failures

    first_iter_s = float(iterations[0].get("elapsed_s", float("inf")))
    if first_iter_s > thresholds.max_first_iter_s:
        failures.append(f"first iteration {first_iter_s:.3f}s > {thresholds.max_first_iter_s:.3f}s")

    steady_iter_s = [float(item.get("elapsed_s", float("inf"))) for item in iterations[1:]]
    worst_steady = max(steady_iter_s)
    if worst_steady > thresholds.max_steady_iter_s:
        failures.append(f"worst steady iteration {worst_steady:.3f}s > {thresholds.max_steady_iter_s:.3f}s")

    if require_monotone_objective:
        log_likelihoods = [float(item["diagnostics"]["log_likelihood"]) for item in iterations]
        if any(next_ll < prev_ll for prev_ll, next_ll in zip(log_likelihoods, log_likelihoods[1:])):
            failures.append(f"log_likelihood is not monotone nondecreasing: {log_likelihoods}")

    for idx, item in enumerate(iterations, start=1):
        diag = item["diagnostics"]
        pmax_mean = float(diag.get("pmax_mean", 0.0))
        nsig_mean = float(diag.get("nsig_mean", float("inf")))
        skip_fraction = float(diag.get("sparse_pass2_skipped_fraction", 0.0))
        if "mstep_objective_solved_delta_per_image" in diag:
            mstep_delta = float(diag["mstep_objective_solved_delta_per_image"])
            if mstep_delta < thresholds.min_mstep_solved_delta_per_image:
                failures.append(
                    f"iteration {idx} M-step solved objective delta/image {mstep_delta:.6g} < "
                    f"{thresholds.min_mstep_solved_delta_per_image:.6g}"
                )
        elif require_monotone_objective:
            failures.append(f"iteration {idx} is missing mstep_objective_solved_delta_per_image")
        if require_fourier_window and not bool(diag.get("uses_fourier_window", False)):
            failures.append(f"iteration {idx} did not use the Fourier window")
        if require_fourier_window:
            score_size = int(diag.get("score_fourier_size", 0))
            full_size = int(diag.get("full_half_fourier_size", 0))
            if score_size <= 0 or full_size <= 0 or score_size >= full_size:
                failures.append(
                    f"iteration {idx} score Fourier size {score_size} is not smaller than full half size {full_size}"
                )
        if pmax_mean < thresholds.min_pmax_mean:
            failures.append(f"iteration {idx} pmax_mean {pmax_mean:.6f} < {thresholds.min_pmax_mean:.6f}")
        if nsig_mean > thresholds.max_nsig_mean:
            failures.append(f"iteration {idx} nsig_mean {nsig_mean:.6f} > {thresholds.max_nsig_mean:.6f}")
        if skip_fraction < thresholds.min_sparse_skip_fraction:
            failures.append(
                f"iteration {idx} sparse skip fraction {skip_fraction:.6f} < "
                f"{thresholds.min_sparse_skip_fraction:.6f}"
            )

    final_pose = iterations[-1].get("gt_pose_diagnostics", {})
    rot_exact = float(final_pose.get("rotation_exact_fraction", 0.0))
    trans_exact = float(final_pose.get("translation_exact_fraction", 0.0))
    if rot_exact < thresholds.min_final_rotation_exact:
        failures.append(f"final rotation exact fraction {rot_exact:.6f} < {thresholds.min_final_rotation_exact:.6f}")
    if trans_exact < thresholds.min_final_translation_exact:
        failures.append(
            f"final translation exact fraction {trans_exact:.6f} < "
            f"{thresholds.min_final_translation_exact:.6f}"
        )
    return failures


def validate_benchmark_summary(summary: dict[str, Any], thresholds: GuardThresholds) -> list[str]:
    failures: list[str] = []
    if not bool(summary.get("passed", False)):
        failures.append("benchmark summary['passed'] is false")
    timing = summary.get("timing", {})
    ppca_s = float(timing.get("ppca", {}).get("median_s", float("inf")))
    adjusted = float(timing.get("ppca_over_kclass_q2_half_adjusted", float("inf")))
    if ppca_s > thresholds.max_ppca_benchmark_s:
        failures.append(f"PPCA benchmark median {ppca_s:.3f}s > {thresholds.max_ppca_benchmark_s:.3f}s")
    if adjusted > thresholds.max_ppca_over_kclass_adjusted:
        failures.append(
            f"q^2/2 adjusted PPCA/K-class ratio {adjusted:.3f} > "
            f"{thresholds.max_ppca_over_kclass_adjusted:.3f}"
        )
    return failures


def run_g128_window_guard(args: argparse.Namespace, thresholds: GuardThresholds, output_root: Path) -> dict[str, Any]:
    fixture_root = Path(args.g128_fixture_root)
    paths = _g128_window_fixture_paths(fixture_root)
    require_fixture(paths, label="PPCA g128 window")

    run_dir = output_root / (
        f"ppca_g128_window_n{args.g128_n_images}_it{args.g128_n_iters}"
        f"_cs{args.current_size}_rb{args.rotation_block_size}"
    )
    command = [
        sys.executable,
        "scripts/run_ppca_dense_from_init_npz.py",
        "--data-star",
        str(paths["data_star"]),
        "--simulation-info",
        str(paths["simulation_info"]),
        "--init-npz",
        str(paths["init_npz"]),
        "--output-dir",
        str(run_dir),
        "--q",
        "2",
        "--n-iters",
        str(args.g128_n_iters),
        "--n-images",
        str(args.g128_n_images),
        "--rotation-source",
        "simulation-info",
        "--translation-source",
        "simulation-info-unique",
        "--current-size",
        str(args.current_size),
        "--image-batch-size",
        str(args.image_batch_size),
        "--rotation-block-size",
        str(args.rotation_block_size),
        "--mstep-chunk-size",
        str(args.mstep_chunk_size),
        "--postprocess-strategy",
        "none",
        "--sparse-pass2",
    ]
    t0 = time.time()
    _run(command)
    ppca_summary = json.loads((run_dir / "summary.json").read_text())
    failures = validate_ppca_summary(
        ppca_summary,
        thresholds,
        require_monotone_objective=False,
        require_fourier_window=True,
        expected_image_shape=(128, 128),
    )
    return {
        "passed": not failures,
        "failures": failures,
        "fixture_root": fixture_root,
        "ppca_run_dir": run_dir,
        "ppca_elapsed_wall_s": time.time() - t0,
        "ppca_summary": ppca_summary,
    }


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def run_guard(args: argparse.Namespace) -> dict[str, Any]:
    fixture_root = Path(args.fixture_root)
    paths = _fixture_paths(fixture_root)
    require_fixture(paths, label="PPCA ribosome")

    thresholds = GuardThresholds(
        max_first_iter_s=float(args.max_first_iter_s),
        max_steady_iter_s=float(args.max_steady_iter_s),
        min_final_rotation_exact=float(args.min_final_rotation_exact),
        min_final_translation_exact=float(args.min_final_translation_exact),
        min_pmax_mean=float(args.min_pmax_mean),
        max_nsig_mean=float(args.max_nsig_mean),
        min_sparse_skip_fraction=float(args.min_sparse_skip_fraction),
        min_mstep_solved_delta_per_image=float(args.min_mstep_solved_delta_per_image),
        max_ppca_benchmark_s=float(args.max_ppca_benchmark_s),
        max_ppca_over_kclass_adjusted=float(args.max_ppca_over_kclass_adjusted),
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_dir = output_root / f"ppca_dense_n{args.n_images}_it{args.n_iters}_cs{args.current_size}_rb{args.rotation_block_size}"
    command = [
        sys.executable,
        "scripts/run_ppca_dense_from_init_npz.py",
        "--data-star",
        str(paths["data_star"]),
        "--simulation-info",
        str(paths["simulation_info"]),
        "--init-npz",
        str(paths["init_npz"]),
        "--prior-init-npz",
        str(paths["prior_init_npz"]),
        "--output-dir",
        str(run_dir),
        "--q",
        "3",
        "--n-iters",
        str(args.n_iters),
        "--n-images",
        str(args.n_images),
        "--rotation-source",
        "simulation-info",
        "--translation-source",
        "simulation-info-unique",
        "--current-size",
        str(args.current_size),
        "--image-batch-size",
        str(args.image_batch_size),
        "--rotation-block-size",
        str(args.rotation_block_size),
        "--mstep-chunk-size",
        str(args.mstep_chunk_size),
        "--prior-from-init",
        "gt-row-norm",
        "--gt-prior-box-power",
        "0",
        "--gt-w-prior-scale",
        "1",
        "--gt-mean-prior-scale",
        "1",
        "--postprocess-strategy",
        "none",
        "--sparse-pass2",
    ]
    t0 = time.time()
    _run(command)
    ppca_summary = json.loads((run_dir / "summary.json").read_text())
    failures = validate_ppca_summary(ppca_summary, thresholds)
    guard_summary: dict[str, Any] = {
        "passed": not failures,
        "failures": failures,
        "thresholds": asdict(thresholds),
        "fixture_root": fixture_root,
        "output_root": output_root,
        "ppca_run_dir": run_dir,
        "ppca_elapsed_wall_s": time.time() - t0,
        "ppca_summary": ppca_summary,
    }

    if args.run_kclass_benchmark:
        benchmark_path = output_root / f"ppca_vs_kclass_n{args.n_images}_cs{args.current_size}_rb{args.rotation_block_size}.json"
        benchmark_command = [
            sys.executable,
            "scripts/benchmark_ppca_vs_kclass_timing.py",
            "--data-star",
            str(paths["data_star"]),
            "--simulation-info",
            str(paths["simulation_info"]),
            "--init-npz",
            str(paths["init_npz"]),
            "--prior-init-npz",
            str(paths["prior_init_npz"]),
            "--output",
            str(benchmark_path),
            "--q",
            "3",
            "--k-classes",
            "4",
            "--n-images",
            str(args.n_images),
            "--rotation-source",
            "simulation-info",
            "--translation-source",
            "simulation-info-unique",
            "--current-size",
            str(args.current_size),
            "--image-batch-size",
            str(args.image_batch_size),
            "--rotation-block-size",
            str(args.rotation_block_size),
            "--warmups",
            "1",
            "--repeats",
            "1",
            "--ppca-prior",
            "gt-row-norm",
            "--gt-prior-box-power",
            "0",
            "--postprocess-strategy",
            "none",
            "--ppca-sparse-pass2",
            "--kclass-sparse-pass2",
        ]
        _run(benchmark_command)
        benchmark_summary = json.loads(benchmark_path.read_text())
        benchmark_failures = validate_benchmark_summary(benchmark_summary, thresholds)
        guard_summary["benchmark_path"] = benchmark_path
        guard_summary["benchmark_summary"] = benchmark_summary
        guard_summary["failures"].extend(benchmark_failures)
        guard_summary["passed"] = not guard_summary["failures"]

    if args.run_g128_window_guard:
        g128_summary = run_g128_window_guard(args, thresholds, output_root)
        guard_summary["g128_window_summary"] = g128_summary
        guard_summary["failures"].extend(g128_summary["failures"])
        guard_summary["passed"] = not guard_summary["failures"]

    guard_path = output_root / "ppca_regression_guard_summary.json"
    guard_path.write_text(json.dumps(_jsonable(guard_summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_jsonable(guard_summary), indent=2, sort_keys=True), flush=True)
    return guard_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", default=str(DEFAULT_FIXTURE_ROOT))
    parser.add_argument("--g128-fixture-root", default=str(DEFAULT_G128_WINDOW_FIXTURE_ROOT))
    parser.add_argument(
        "--output-root",
        default=f"/scratch/gpfs/GILLES/mg6942/tmp/ppca_regression_guard_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--n-images", type=int, default=5000)
    parser.add_argument("--n-iters", type=int, default=3)
    parser.add_argument("--current-size", type=int, default=64)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--rotation-block-size", type=int, default=512)
    parser.add_argument("--mstep-chunk-size", type=int, default=65536)
    parser.add_argument("--run-kclass-benchmark", action="store_true")
    parser.add_argument(
        "--run-g128-window-guard",
        action="store_true",
        help="Also run the 128-box, 5k-image Fourier-window PPCA throughput/pose guard.",
    )
    parser.add_argument("--g128-n-images", type=int, default=5000)
    parser.add_argument("--g128-n-iters", type=int, default=3)
    parser.add_argument("--max-first-iter-s", type=float, default=GuardThresholds.max_first_iter_s)
    parser.add_argument("--max-steady-iter-s", type=float, default=GuardThresholds.max_steady_iter_s)
    parser.add_argument(
        "--min-final-rotation-exact",
        type=float,
        default=GuardThresholds.min_final_rotation_exact,
    )
    parser.add_argument(
        "--min-final-translation-exact",
        type=float,
        default=GuardThresholds.min_final_translation_exact,
    )
    parser.add_argument("--min-pmax-mean", type=float, default=GuardThresholds.min_pmax_mean)
    parser.add_argument("--max-nsig-mean", type=float, default=GuardThresholds.max_nsig_mean)
    parser.add_argument(
        "--min-sparse-skip-fraction",
        type=float,
        default=GuardThresholds.min_sparse_skip_fraction,
    )
    parser.add_argument(
        "--min-mstep-solved-delta-per-image",
        type=float,
        default=GuardThresholds.min_mstep_solved_delta_per_image,
    )
    parser.add_argument("--max-ppca-benchmark-s", type=float, default=GuardThresholds.max_ppca_benchmark_s)
    parser.add_argument(
        "--max-ppca-over-kclass-adjusted",
        type=float,
        default=GuardThresholds.max_ppca_over_kclass_adjusted,
    )
    return parser.parse_args()


def main() -> None:
    summary = run_guard(_parse_args())
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
