import json
from pathlib import Path

import pytest

from scripts.analyze_vdam_late_profile_pair import analyze


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _profile_root(tmp_path: Path) -> Path:
    root = tmp_path / "profile"
    root.mkdir()
    (root / "COMPLETED").touch()
    _write(
        root / "provenance" / "recovar_execution_contract.json",
        {
            "mode": "all_optimized_q32",
            "cold": {"profile_checked": True, "profile_exact": True},
            "warm": {"profile_checked": True, "profile_exact": True},
        },
    )
    _write(root / "provenance" / "run.json", {"host_attribution": False})
    _write(
        root / "recovar_profiled" / "profile_summary.json",
        {
            "profiled_iteration": 48,
            "cold": {"wall_s": 25.0},
            "warm": {
                "wall_s": 5.0,
                "iteration_profile": {
                    "pre_artifact_time_s": 1.2,
                    "expectation_time_s": 0.8,
                },
                "sparse_pass2_profile": {
                    "pass1_time_s": 0.1,
                    "pass2_time_s": 0.6,
                    "mean_significant_samples": 4.0,
                },
                "halfset_profiles": {
                    "halfset_0_profile_summary": {
                        "em_time_s": 0.5,
                        "sum_padded_rows": 100,
                    }
                },
            },
        },
    )
    (root / "recovar_profiled.stderr").write_text(
        "Finished XLA compilation of jit(add) in 2.0 sec\n"
        "Finished XLA compilation of jit(add) in 1.0 seconds\n"
        "Finished XLA compilation of jit(large) in 4.0 sec\n"
    )
    common_kernel = {
        "name": "kernel",
        "count": 2,
        "total_ns": 100_000_000,
        "mean_ns": 50_000_000.0,
        "max_ns": 60_000_000,
    }
    common_api = {
        "name": "cudaLaunchKernel",
        "count": 2,
        "total_ns": 20_000_000,
        "mean_ns": 10_000_000.0,
        "max_ns": 12_000_000,
    }
    _write(
        root / "nsight" / "native_summary.json",
        {
            "capture_span_ns": 1_000_000_000,
            "devices": {
                "0": {
                    "gpu_busy_ns": 200_000_000,
                    "gpu_idle_ns_within_capture": 800_000_000,
                    "kernel_count": 10,
                    "gpu_busy_fraction_within_capture": 0.2,
                }
            },
            "kernels": [common_kernel],
            "cuda_apis": [common_api],
        },
    )
    _write(
        root / "nsight" / "recovar_summary.json",
        {
            "capture_span_ns": 2_000_000_000,
            "devices": {
                "0": {
                    "gpu_busy_ns": 300_000_000,
                    "gpu_idle_ns_within_capture": 1_700_000_000,
                    "kernel_count": 15,
                    "gpu_busy_fraction_within_capture": 0.15,
                }
            },
            "kernels": [common_kernel],
            "cuda_apis": [common_api],
        },
    )
    return root


def test_late_profile_pair_analyzer_reports_phase_kernel_and_api_gap(tmp_path):
    root = _profile_root(tmp_path)

    report = analyze(root, top=1)

    assert report["contract_mode"] == "all_optimized_q32"
    assert report["capture"] == {
        "native_s": 1.0,
        "recovar_s": 2.0,
        "recovar_over_native": 2.0,
        "recovar_minus_native_s": 1.0,
    }
    assert report["gpu"]["delta"] == {
        "busy_s": pytest.approx(0.1),
        "idle_s": pytest.approx(0.9),
        "kernel_count": 5,
    }
    assert report["recovar_phases"]["capture_minus_profiled_iteration_s"] == pytest.approx(0.8)
    assert report["kernels"]["recovar_top"][0]["total_s"] == pytest.approx(0.1)
    assert report["cuda_apis"]["native_top"][0]["total_s"] == pytest.approx(0.02)
    assert report["cold_compilation"] == {
        "log": str(root / "recovar_profiled.stderr"),
        "compile_count": 3,
        "total_s": 7.0,
        "unique_module_count": 2,
        "top_modules": [
            {
                "name": "jit(large)",
                "count": 1,
                "total_s": 4.0,
                "mean_s": 4.0,
                "max_s": 4.0,
            }
        ],
        "cold_wall_s": 25.0,
        "warm_wall_s": 5.0,
        "cold_warm_delta_s": 20.0,
        "fraction_of_cold_warm_delta": 0.35,
        "cold_warm_delta_minus_compile_s": 13.0,
    }


def test_late_profile_pair_analyzer_rejects_host_attribution_as_timing(tmp_path):
    root = _profile_root(tmp_path)
    _write(root / "provenance" / "run.json", {"host_attribution": True})

    with pytest.raises(RuntimeError, match="not timing truth"):
        analyze(root)
