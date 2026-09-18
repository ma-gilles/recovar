#!/usr/bin/env python3
"""Summarize Python, GIL, and OS-runtime evidence from a VDAM host trace."""

from __future__ import annotations

import argparse
import base64
import html
import json
import sqlite3
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON mapping: {path}")
    return value


def _clipped_ns(start: int, end: int, lower: int, upper: int) -> int:
    return max(0, min(end, upper) - max(start, lower))


def _frame(function: str, file: str, line: int) -> dict[str, object]:
    return {"function": function, "file": file, "line": int(line)}


def _is_project_file(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return "/recovar/" in normalized and "/.pixi/" not in normalized


def _rank_frames(
    counts: Counter[tuple[str, str, int]], *, total: int, top: int
) -> list[dict[str, object]]:
    rows = []
    for (function, file, line), count in counts.most_common(top):
        rows.append(
            {
                **_frame(function, file, line),
                "count": count,
                "fraction": count / total if total else 0.0,
            }
        )
    return rows


def _python_samples(
    json_export: Path,
    *,
    lower: int,
    upper: int,
    phase_windows: list[tuple[str, int, int]],
    top: int,
) -> tuple[dict[str, Any], int]:
    samples: list[tuple[int, int, int, list[tuple[str, str, int]]]] = []
    tids: Counter[int] = Counter()
    with json_export.open() as stream:
        header = json.loads(next(stream))
        strings = header.get("data")
        if not isinstance(strings, list):
            raise RuntimeError("Nsight JSON export omitted the StringIds header")
        strings = [html.unescape(str(value)) for value in strings]
        for raw_line in stream:
            value = json.loads(raw_line)
            if value.get("Type") != 34:
                continue
            event = value.get("NvtxEvent")
            if not isinstance(event, dict):
                continue
            timestamp = int(event["Timestamp"])
            if not lower <= timestamp < upper:
                continue
            payload = event.get("Payload", {}).get("BinaryData", [])
            if not payload:
                continue
            binary = base64.b64decode(payload[0]["Data"])
            if len(binary) < 16:
                raise RuntimeError("truncated Nsight Python sample payload")
            _, state, frame_count = struct.unpack_from("<QII", binary)
            expected = 16 + 24 * frame_count
            if len(binary) != expected:
                raise RuntimeError(
                    f"invalid Nsight Python sample payload: {len(binary)} != {expected}"
                )
            frames = []
            for index in range(frame_count):
                function_id, file_id, line = struct.unpack_from(
                    "<QQQ", binary, 16 + 24 * index
                )
                frames.append(
                    (strings[function_id], strings[file_id], int(line))
                )
            tid = int(event["GlobalTid"])
            tids[tid] += 1
            samples.append((timestamp, tid, int(state), frames))

    if not samples:
        raise RuntimeError(f"no Python samples in capture window: {json_export}")
    main_tid, main_count = tids.most_common(1)[0]
    state_counts: Counter[int] = Counter()
    leaf_counts: Counter[tuple[str, str, int]] = Counter()
    project_counts: Counter[tuple[str, str, int]] = Counter()
    stack_counts: Counter[tuple[tuple[str, str, int], ...]] = Counter()
    for _, tid, state, frames in samples:
        if tid != main_tid:
            continue
        state_counts[state] += 1
        if not frames:
            continue
        leaf_counts[frames[0]] += 1
        project = next((frame for frame in frames if _is_project_file(frame[1])), None)
        if project is not None:
            project_counts[project] += 1
        stack_counts[tuple(frames[:4])] += 1

    state_names = {
        0: "Unknown",
        1: "Running",
        2: "Interruptible",
        3: "Uninterruptible",
        4: "Stopped",
        5: "Terminated",
        6: "Unscheduled",
        7: "Waiting",
        8: "OSRuntime",
        9: "Initialized",
        10: "Transition",
    }
    top_stacks = []
    for stack, count in stack_counts.most_common(top):
        top_stacks.append(
            {
                "count": count,
                "fraction": count / main_count,
                "frames_leaf_first": [
                    _frame(function, file, line) for function, file, line in stack
                ],
            }
        )

    phases = []
    for name, phase_start, phase_end in phase_windows:
        all_phase_samples = [
            sample
            for sample in samples
            if phase_start <= sample[0] < phase_end
        ]
        main_phase_samples = [
            sample for sample in all_phase_samples if sample[1] == main_tid
        ]
        phase_states: Counter[int] = Counter(
            sample[2] for sample in all_phase_samples
        )
        main_phase_states: Counter[int] = Counter(
            sample[2] for sample in main_phase_samples
        )
        phase_leaf: Counter[tuple[str, str, int]] = Counter(
            sample[3][0] for sample in all_phase_samples if sample[3]
        )
        phase_project: Counter[tuple[str, str, int]] = Counter()
        for _, _, _, frames in all_phase_samples:
            project = next(
                (frame for frame in frames if _is_project_file(frame[1])), None
            )
            if project is not None:
                phase_project[project] += 1
        phase_count = len(all_phase_samples)
        main_phase_count = len(main_phase_samples)
        phases.append(
            {
                "name": name,
                "start_ns": phase_start,
                "end_ns": phase_end,
                "duration_s": (phase_end - phase_start) / 1e9,
                "sample_count": phase_count,
                "main_sample_count": main_phase_count,
                "states": [
                    {
                        "id": state,
                        "name": state_names.get(state, f"state_{state}"),
                        "count": count,
                        "fraction": count / phase_count if phase_count else 0.0,
                    }
                    for state, count in phase_states.most_common()
                ],
                "main_states": [
                    {
                        "id": state,
                        "name": state_names.get(state, f"state_{state}"),
                        "count": count,
                        "fraction": (
                            count / main_phase_count if main_phase_count else 0.0
                        ),
                    }
                    for state, count in main_phase_states.most_common()
                ],
                "top_leaf_frames": _rank_frames(
                    phase_leaf, total=phase_count, top=top
                ),
                "top_project_frames": _rank_frames(
                    phase_project, total=phase_count, top=top
                ),
            }
        )
    return (
        {
            "sample_count": len(samples),
            "thread_sample_counts": [
                {"global_tid": tid, "count": count}
                for tid, count in tids.most_common()
            ],
            "main_global_tid": main_tid,
            "main_sample_count": main_count,
            "main_sample_fraction": main_count / len(samples),
            "main_thread_states": [
                {
                    "id": state,
                    "name": state_names.get(state, f"state_{state}"),
                    "count": count,
                    "fraction": count / main_count,
                }
                for state, count in state_counts.most_common()
            ],
            "main_top_leaf_frames": _rank_frames(
                leaf_counts, total=main_count, top=top
            ),
            "main_top_project_frames": _rank_frames(
                project_counts, total=main_count, top=top
            ),
            "main_top_stacks": top_stacks,
            "main_phases": phases,
        },
        main_tid,
    )


def _string_map(connection: sqlite3.Connection) -> dict[int, str]:
    return {
        int(identifier): str(value)
        for identifier, value in connection.execute("SELECT id, value FROM StringIds")
    }


def _profiling_window(
    connection: sqlite3.Connection, *, fallback_start: int, fallback_end: int
) -> tuple[int, int, str]:
    try:
        rows = list(
            connection.execute(
                """
                SELECT timestamp, text FROM DIAGNOSTIC_EVENT
                WHERE text LIKE 'Profiling started due to cudaProfilerStart%'
                   OR text = 'Profiling has stopped.'
                ORDER BY timestamp
                """
            )
        )
    except sqlite3.OperationalError:
        rows = []
    starts = [int(timestamp) for timestamp, text in rows if str(text).startswith("Profiling started")]
    stops = [int(timestamp) for timestamp, text in rows if text == "Profiling has stopped."]
    if starts and stops:
        start = starts[0]
        stop = next((value for value in stops if value > start), None)
        if stop is not None:
            return start, stop, "nsight_diagnostic_events"
    return fallback_start, fallback_end, "cuda_api_activity_fallback"


def _phase_windows(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    *,
    lower: int,
    upper: int,
) -> list[tuple[str, int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    for start, end, text, text_id in connection.execute(
        """
        SELECT start, end, text, textId FROM NVTX_EVENTS
        WHERE domainId = 9 AND end IS NOT NULL AND start < ? AND end > ?
        """,
        (upper, lower),
    ):
        name = str(text) if text is not None else strings.get(int(text_id), "<unknown>")
        if name in {
            "kclass.adaptive.pass1_significance",
            "local.run_local_em_exact",
        }:
            ranges[name] = (max(lower, int(start)), min(upper, int(end)))
    pass1 = ranges.get("kclass.adaptive.pass1_significance")
    local = ranges.get("local.run_local_em_exact")
    if pass1 is None or local is None or not (
        lower <= pass1[0] < pass1[1] <= local[0] < local[1] <= upper
    ):
        return [("whole_capture", lower, upper)]
    return [
        ("pre_pass1", lower, pass1[0]),
        ("pass1_significance", pass1[0], pass1[1]),
        ("between_pass1_and_local", pass1[1], local[0]),
        ("local_em", local[0], local[1]),
        ("post_local", local[1], upper),
    ]


def _gil_summary(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    *,
    main_tid: int,
    lower: int,
    upper: int,
) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    for start, end, tid, text, text_id in connection.execute(
        """
        SELECT start, end, globalTid, text, textId
        FROM NVTX_EVENTS
        WHERE domainId = 2 AND start < ? AND COALESCE(end, start) > ?
        """,
        (upper, lower),
    ):
        if end is None:
            continue
        name = str(text) if text is not None else strings.get(int(text_id), "<unknown>")
        key = (int(tid), name)
        grouped[key][0] += 1
        grouped[key][1] += _clipped_ns(int(start), int(end), lower, upper)

    main = []
    for (tid, name), (count, duration_ns) in grouped.items():
        if tid == main_tid:
            main.append(
                {
                    "name": name,
                    "count": count,
                    "duration_s": duration_ns / 1e9,
                    "capture_fraction": duration_ns / (upper - lower),
                }
            )
    main.sort(key=lambda row: (-float(row["duration_s"]), str(row["name"])))
    sampler_wait_ns = sum(
        duration_ns
        for (tid, name), (_, duration_ns) in grouped.items()
        if tid != main_tid and name == "Waiting for GIL"
    )
    return {
        "main_thread": main,
        "other_threads_waiting_for_gil_s": sampler_wait_ns / 1e9,
        "event_count": sum(count for count, _ in grouped.values()),
    }


def _callchain_frames(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    callchain_id: int,
    *,
    limit: int = 12,
) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT symbol, module, stackDepth
        FROM OSRT_CALLCHAINS WHERE id = ? ORDER BY stackDepth LIMIT ?
        """,
        (callchain_id, limit),
    )
    return [
        {
            "symbol": strings.get(int(symbol), "<unknown>"),
            "module": strings.get(int(module), "<unknown>"),
            "stack_depth": int(depth),
        }
        for symbol, module, depth in rows
    ]


def _osrt_summary(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    *,
    main_tid: int,
    lower: int,
    upper: int,
    top: int,
) -> dict[str, Any]:
    by_name: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    by_chain: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0, 0])
    for start, end, name_id, callchain_id in connection.execute(
        """
        SELECT start, end, nameId, callchainId FROM OSRT_API
        WHERE globalTid = ? AND start < ? AND end > ?
        """,
        (main_tid, upper, lower),
    ):
        duration = _clipped_ns(int(start), int(end), lower, upper)
        full_duration = int(end) - int(start)
        name = strings.get(int(name_id), "<unknown>")
        for bucket in (by_name[name], by_chain[(name, int(callchain_id))]):
            bucket[0] += 1
            bucket[1] += duration
            bucket[2] = max(bucket[2], full_duration)

    top_calls = [
        {
            "name": name,
            "count": values[0],
            "duration_s": values[1] / 1e9,
            "capture_fraction": values[1] / (upper - lower),
            "max_s": values[2] / 1e9,
        }
        for name, values in by_name.items()
    ]
    top_calls.sort(key=lambda row: (-float(row["duration_s"]), str(row["name"])))
    top_chains = []
    for (name, callchain_id), values in sorted(
        by_chain.items(), key=lambda item: (-item[1][1], item[0])
    )[:top]:
        top_chains.append(
            {
                "name": name,
                "callchain_id": callchain_id,
                "count": values[0],
                "duration_s": values[1] / 1e9,
                "max_s": values[2] / 1e9,
                "frames_leaf_first": _callchain_frames(
                    connection, strings, callchain_id
                ),
            }
        )
    return {
        "main_global_tid": main_tid,
        "note": "durations are observed call time and may overlap through nesting",
        "top_calls": top_calls[:top],
        "top_callchains": top_chains,
    }


def _high_level_ranges(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    *,
    lower: int,
    upper: int,
    top: int,
) -> list[dict[str, Any]]:
    domain_names = {
        int(domain_id): str(text)
        for domain_id, text in connection.execute(
            "SELECT domainId, text FROM NVTX_EVENTS WHERE eventType = 75"
        )
        if domain_id is not None and text is not None
    }
    excluded = {"Python Periodic Sampling", "GIL Trace", "TSL", "CCCL"}
    grouped: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for start, end, domain_id, text, text_id in connection.execute(
        """
        SELECT start, end, domainId, text, textId FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND start < ? AND end > ?
        """,
        (upper, lower),
    ):
        domain = domain_names.get(int(domain_id or 0), f"domain_{domain_id}")
        if domain in excluded:
            continue
        name = str(text) if text is not None else strings.get(int(text_id), "<unknown>")
        values = grouped[(domain, name)]
        duration = _clipped_ns(int(start), int(end), lower, upper)
        values[0] += 1
        values[1] += duration
        values[2] = max(values[2], int(end) - int(start))
    rows = [
        {
            "domain": domain,
            "name": name,
            "count": values[0],
            "duration_s": values[1] / 1e9,
            "max_s": values[2] / 1e9,
        }
        for (domain, name), values in grouped.items()
    ]
    rows.sort(key=lambda row: (-float(row["duration_s"]), str(row["name"])))
    return rows[:top]


def analyze(
    root: Path,
    *,
    json_export: Path | None = None,
    sqlite_path: Path | None = None,
    top: int = 20,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if top <= 0:
        raise ValueError("top must be positive")
    if not (root / "COMPLETED").is_file():
        raise RuntimeError(f"host profile root has no COMPLETED marker: {root}")
    run = _load_mapping(root / "provenance" / "run.json")
    if run.get("host_attribution") is not True:
        raise RuntimeError("run is not marked as host attribution")
    if run.get("timing_truth_allowed") is not False:
        raise RuntimeError("host-attribution run must be excluded from timing truth")

    summary_path = root / "nsight" / "recovar_summary.json"
    summary = _load_mapping(summary_path)
    cuda_lower = int(summary["capture_start_ns"])
    cuda_upper = int(summary["capture_end_ns"])
    if cuda_upper <= cuda_lower:
        raise RuntimeError("invalid Nsight capture window")
    json_export = (json_export or root / "nsight" / "recovar_host.json").resolve()
    sqlite_path = (sqlite_path or root / "nsight" / "recovar.sqlite").resolve()
    if not json_export.is_file():
        raise FileNotFoundError(json_export)
    if not sqlite_path.is_file():
        raise FileNotFoundError(sqlite_path)

    with sqlite3.connect(sqlite_path) as connection:
        strings = _string_map(connection)
        lower, upper, window_source = _profiling_window(
            connection, fallback_start=cuda_lower, fallback_end=cuda_upper
        )
        phases = _phase_windows(
            connection, strings, lower=lower, upper=upper
        )
        python, main_tid = _python_samples(
            json_export,
            lower=lower,
            upper=upper,
            phase_windows=phases,
            top=top,
        )
        gil = _gil_summary(
            connection,
            strings,
            main_tid=main_tid,
            lower=lower,
            upper=upper,
        )
        osrt = _osrt_summary(
            connection,
            strings,
            main_tid=main_tid,
            lower=lower,
            upper=upper,
            top=top,
        )
        ranges = _high_level_ranges(
            connection, strings, lower=lower, upper=upper, top=top
        )

    devices = summary.get("devices")
    if not isinstance(devices, dict) or len(devices) != 1:
        raise RuntimeError("RECOVAR summary must contain exactly one GPU")
    device = next(iter(devices.values()))
    if not isinstance(device, dict):
        raise RuntimeError("RECOVAR GPU summary is not a mapping")
    return {
        "schema": "recovar.vdam_host_attribution.v1",
        "classification": "diagnostic_performance_only",
        "timing_truth": False,
        "root": str(root),
        "job_id": run.get("job_id"),
        "git_head": run.get("git_head"),
        "profiled_iteration": run.get("profiled_iteration"),
        "sources": {
            "summary": str(summary_path),
            "sqlite": str(sqlite_path),
            "json_export": str(json_export),
        },
        "capture": {
            "start_ns": lower,
            "end_ns": upper,
            "span_s": (upper - lower) / 1e9,
            "window_source": window_source,
            "cuda_activity_start_ns": cuda_lower,
            "cuda_activity_end_ns": cuda_upper,
            "cuda_activity_span_s": (cuda_upper - cuda_lower) / 1e9,
            "gpu_busy_s": int(device["gpu_busy_ns"]) / 1e9,
            "gpu_busy_fraction": int(device["gpu_busy_ns"]) / (upper - lower),
            "kernel_count": int(device["kernel_count"]),
        },
        "python_sampling": python,
        "gil": gil,
        "os_runtime": osrt,
        "high_level_nvtx_ranges": ranges,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json-export", type=Path)
    parser.add_argument("--sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)
    report = analyze(
        args.root,
        json_export=args.json_export,
        sqlite_path=args.sqlite,
        top=args.top,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "capture": report["capture"],
                "python_sampling": {
                    "main_sample_count": report["python_sampling"][
                        "main_sample_count"
                    ],
                    "top_project_frames": report["python_sampling"][
                        "main_top_project_frames"
                    ][:10],
                },
                "gil": report["gil"],
                "os_runtime_top": report["os_runtime"]["top_calls"][:10],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
