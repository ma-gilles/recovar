#!/usr/bin/env python3
"""Build a machine-readable RECOVAR/native late-iteration profile comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON mapping: {path}")
    return value


def _one_device(summary: dict[str, Any], *, label: str) -> dict[str, Any]:
    devices = summary.get("devices")
    if not isinstance(devices, dict) or len(devices) != 1:
        raise RuntimeError(f"{label} must contain exactly one profiled GPU")
    device = next(iter(devices.values()))
    if not isinstance(device, dict):
        raise RuntimeError(f"{label} GPU summary is not a mapping")
    return device


def _seconds(value: int | float) -> float:
    return float(value) / 1e9


def _ranked_seconds(rows: object, *, top: int) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise RuntimeError("Nsight ranked rows are not a list")
    result = []
    for raw in rows[:top]:
        if not isinstance(raw, dict):
            raise RuntimeError("Nsight ranked row is not a mapping")
        row = dict(raw)
        for source, target in (
            ("total_ns", "total_s"),
            ("mean_ns", "mean_s"),
            ("max_ns", "max_s"),
        ):
            if source in row:
                row[target] = _seconds(row[source])
        result.append(row)
    return result


def _phase_seconds(profile: object, *, label: str) -> dict[str, float]:
    if not isinstance(profile, dict):
        raise RuntimeError(f"{label} phase profile is not a mapping")
    phases = {
        str(name): float(value)
        for name, value in profile.items()
        if str(name).endswith(("_s", "_time_s"))
        and isinstance(value, (int, float))
    }
    invalid = [name for name, value in phases.items() if value < 0.0]
    if invalid:
        raise RuntimeError(f"{label} contains negative phases: {invalid}")
    return dict(sorted(phases.items(), key=lambda item: (-item[1], item[0])))


def analyze(root: Path, *, top: int = 20) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not (root / "COMPLETED").is_file():
        raise RuntimeError(f"profile root has no COMPLETED marker: {root}")
    if top <= 0:
        raise ValueError("top must be positive")

    contract = _load(root / "provenance" / "recovar_execution_contract.json")
    for arm in ("cold", "warm"):
        arm_contract = contract.get(arm)
        if not isinstance(arm_contract, dict):
            raise RuntimeError(f"execution contract omitted {arm}")
        if not arm_contract.get("profile_checked") or not arm_contract.get(
            "profile_exact"
        ):
            raise RuntimeError(f"execution contract did not accept {arm}")

    profile = _load(root / "recovar_profiled" / "profile_summary.json")
    run = _load(root / "provenance" / "run.json")
    native = _load(root / "nsight" / "native_summary.json")
    recovar = _load(root / "nsight" / "recovar_summary.json")
    if bool(run.get("host_attribution", False)):
        raise RuntimeError("host-attribution traces are not timing truth")

    native_device = _one_device(native, label="native")
    recovar_device = _one_device(recovar, label="recovar")
    native_span = _seconds(native["capture_span_ns"])
    recovar_span = _seconds(recovar["capture_span_ns"])
    native_busy = _seconds(native_device["gpu_busy_ns"])
    recovar_busy = _seconds(recovar_device["gpu_busy_ns"])
    native_idle = _seconds(native_device["gpu_idle_ns_within_capture"])
    recovar_idle = _seconds(recovar_device["gpu_idle_ns_within_capture"])
    if min(native_span, recovar_span) <= 0.0:
        raise RuntimeError("profile capture spans must be positive")

    warm = profile.get("warm")
    if not isinstance(warm, dict):
        raise RuntimeError("RECOVAR profile omitted the warm arm")
    iteration_phases = _phase_seconds(
        warm.get("iteration_profile"),
        label="iteration",
    )
    sparse_phases = _phase_seconds(
        warm.get("sparse_pass2_profile"),
        label="sparse pass",
    )
    halfsets = warm.get("halfset_profiles")
    if not isinstance(halfsets, dict) or not halfsets:
        raise RuntimeError("RECOVAR profile omitted halfset stage timings")
    local_phases = {
        name: _phase_seconds(value, label=name)
        for name, value in sorted(halfsets.items())
    }
    profiled_iteration = iteration_phases.get("pre_artifact_time_s")
    if profiled_iteration is None:
        raise RuntimeError("RECOVAR profile omitted pre_artifact_time_s")

    return {
        "schema": "recovar.vdam_late_profile_pair.v1",
        "classification": "diagnostic_performance_only",
        "root": str(root),
        "contract_mode": contract.get("mode"),
        "profiled_iteration": profile.get("profiled_iteration"),
        "timing_truth": True,
        "capture": {
            "native_s": native_span,
            "recovar_s": recovar_span,
            "recovar_over_native": recovar_span / native_span,
            "recovar_minus_native_s": recovar_span - native_span,
        },
        "gpu": {
            "native": {
                "busy_s": native_busy,
                "idle_s": native_idle,
                "kernel_count": int(native_device["kernel_count"]),
                "busy_fraction": float(native_device["gpu_busy_fraction_within_capture"]),
            },
            "recovar": {
                "busy_s": recovar_busy,
                "idle_s": recovar_idle,
                "kernel_count": int(recovar_device["kernel_count"]),
                "busy_fraction": float(recovar_device["gpu_busy_fraction_within_capture"]),
            },
            "delta": {
                "busy_s": recovar_busy - native_busy,
                "idle_s": recovar_idle - native_idle,
                "kernel_count": int(recovar_device["kernel_count"])
                - int(native_device["kernel_count"]),
            },
        },
        "recovar_phases": {
            "iteration": iteration_phases,
            "sparse_pass": sparse_phases,
            "local_halfsets": local_phases,
            "capture_minus_profiled_iteration_s": recovar_span
            - profiled_iteration,
        },
        "kernels": {
            "native_top": _ranked_seconds(native.get("kernels"), top=top),
            "recovar_top": _ranked_seconds(recovar.get("kernels"), top=top),
        },
        "cuda_apis": {
            "native_top": _ranked_seconds(native.get("cuda_apis"), top=top),
            "recovar_top": _ranked_seconds(recovar.get("cuda_apis"), top=top),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)
    report = analyze(args.root, top=args.top)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"capture": report["capture"], "gpu": report["gpu"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
