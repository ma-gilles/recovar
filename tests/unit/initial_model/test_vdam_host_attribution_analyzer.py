import base64
import json
import sqlite3
import struct
from pathlib import Path

import pytest

from scripts.analyze_vdam_host_attribution import analyze


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _python_sample(
    *, timestamp: int, tid: int, state: int, frames: list[tuple[int, int, int]]
) -> dict:
    binary = struct.pack("<QII", 1, state, len(frames)) + b"".join(
        struct.pack("<QQQ", *frame) for frame in frames
    )
    return {
        "Type": 34,
        "NvtxEvent": {
            "Type": 34,
            "Timestamp": str(timestamp),
            "GlobalTid": str(tid),
            "DomainId": "1",
            "Payload": {
                "BinaryData": [
                    {"SchemaId": "1", "Data": base64.b64encode(binary).decode()}
                ]
            },
        },
    }


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "profile"
    root.mkdir()
    (root / "COMPLETED").touch()
    _write(
        root / "provenance" / "run.json",
        {
            "host_attribution": True,
            "timing_truth_allowed": False,
            "job_id": "7",
            "git_head": "abc",
            "profiled_iteration": 48,
        },
    )
    _write(
        root / "nsight" / "recovar_summary.json",
        {
            "capture_start_ns": 100,
            "capture_end_ns": 1_000_000_100,
            "devices": {
                "0": {
                    "gpu_busy_ns": 100_000_000,
                    "gpu_busy_fraction_within_capture": 0.1,
                    "kernel_count": 12,
                }
            },
        },
    )
    strings = [
        "leaf",
        "/work/recovar/em/example.py",
        "caller",
        "/work/scripts/driver.py",
        "Holding GIL",
        "Waiting for GIL",
        "fread",
        "_IO_fread",
        "/lib/libc.so",
        "recovar_em",
        "local.run",
    ]
    events = [
        _python_sample(timestamp=200, tid=123, state=1, frames=[(0, 1, 4), (2, 3, 8)]),
        _python_sample(timestamp=300, tid=123, state=3, frames=[(0, 1, 4)]),
        _python_sample(timestamp=400, tid=456, state=1, frames=[(2, 3, 8)]),
    ]
    export = root / "nsight" / "recovar_host.json"
    with export.open("w") as stream:
        stream.write(json.dumps({"data": strings}) + "\n")
        for event in events:
            stream.write(json.dumps(event) + "\n")

    database = root / "nsight" / "recovar.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS (
              start INTEGER, end INTEGER, eventType INTEGER, text TEXT,
              globalTid INTEGER, textId INTEGER, domainId INTEGER
            );
            CREATE TABLE OSRT_API (
              start INTEGER, end INTEGER, globalTid INTEGER,
              nameId INTEGER, callchainId INTEGER
            );
            CREATE TABLE OSRT_CALLCHAINS (
              id INTEGER, symbol INTEGER, module INTEGER, stackDepth INTEGER
            );
            CREATE TABLE DIAGNOSTIC_EVENT (timestamp INTEGER, text TEXT);
            """
        )
        connection.executemany(
            "INSERT INTO StringIds VALUES (?, ?)", enumerate(strings)
        )
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (-10, None, 75, "recovar_em", 123, None, 9),
                (100, 400_000_100, 59, None, 123, 4, 2),
                (500_000_100, 600_000_100, 59, None, 123, 5, 2),
                (200, 700_000_100, 59, None, 123, 10, 9),
            ],
        )
        connection.execute(
            "INSERT INTO OSRT_API VALUES (?, ?, ?, ?, ?)",
            (300, 50_000_300, 123, 6, 9),
        )
        connection.execute(
            "INSERT INTO OSRT_CALLCHAINS VALUES (?, ?, ?, ?)", (9, 7, 8, 0)
        )
        connection.executemany(
            "INSERT INTO DIAGNOSTIC_EVENT VALUES (?, ?)",
            [
                (100, "Profiling started due to cudaProfilerStart API invoked in the application."),
                (1_000_000_100, "Profiling has stopped."),
            ],
        )
    return root


def test_host_analyzer_reports_python_gil_osrt_and_ranges(tmp_path):
    root = _root(tmp_path)

    report = analyze(root, top=2)

    assert report["timing_truth"] is False
    assert report["capture"] == {
        "start_ns": 100,
        "end_ns": 1_000_000_100,
        "span_s": 1.0,
        "window_source": "nsight_diagnostic_events",
        "cuda_activity_start_ns": 100,
        "cuda_activity_end_ns": 1_000_000_100,
        "cuda_activity_span_s": 1.0,
        "gpu_busy_s": 0.1,
        "gpu_busy_fraction": 0.1,
        "kernel_count": 12,
    }
    python = report["python_sampling"]
    assert python["sample_count"] == 3
    assert python["main_global_tid"] == 123
    assert python["main_thread_states"][:2] == [
        {"id": 1, "name": "Running", "count": 1, "fraction": 0.5},
        {"id": 3, "name": "Uninterruptible", "count": 1, "fraction": 0.5},
    ]
    assert python["main_top_project_frames"][0]["function"] == "leaf"
    assert python["main_top_project_frames"][0]["count"] == 2
    assert report["gil"]["main_thread"][0]["name"] == "Holding GIL"
    assert report["gil"]["main_thread"][0]["duration_s"] == pytest.approx(0.4)
    assert report["os_runtime"]["top_calls"][0]["name"] == "fread"
    assert report["os_runtime"]["top_calls"][0]["duration_s"] == pytest.approx(0.05)
    assert report["os_runtime"]["top_callchains"][0]["frames_leaf_first"][0] == {
        "symbol": "_IO_fread",
        "module": "/lib/libc.so",
        "stack_depth": 0,
    }
    assert report["high_level_nvtx_ranges"][0]["name"] == "local.run"
    assert report["high_level_nvtx_ranges"][0]["duration_s"] == pytest.approx(0.7)


def test_host_analyzer_rejects_timing_truth_run(tmp_path):
    root = _root(tmp_path)
    _write(
        root / "provenance" / "run.json",
        {"host_attribution": False, "timing_truth_allowed": True},
    )

    with pytest.raises(RuntimeError, match="not marked"):
        analyze(root)
