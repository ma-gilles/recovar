import json
from pathlib import Path

import pytest

from scripts.analyze_vdam_cold_compile_attribution import analyze


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "profile"
    root.mkdir()
    (root / "COMPLETED").touch()
    _write(
        root / "provenance" / "run.json",
        {
            "cold_compile_attribution": True,
            "timing_truth_allowed": False,
            "job_id": "9",
            "git_head": "abc",
            "profiled_iteration": 48,
        },
    )
    records = [
        {
            "sequence": 0,
            "module": "jit_a",
            "cache_status": "miss",
            "elapsed_s": 2.0,
            "callsite": {
                "file": "recovar/em/a.py",
                "line": 10,
                "function": "f",
                "source": "a()",
            },
        },
        {
            "sequence": 1,
            "module": "jit_a",
            "cache_status": "miss",
            "elapsed_s": 1.0,
            "callsite": {
                "file": "recovar/em/a.py",
                "line": 10,
                "function": "f",
                "source": "a()",
            },
        },
        {
            "sequence": 2,
            "module": "jit_b",
            "cache_status": "hit",
            "elapsed_s": 0.25,
            "callsite": {
                "file": "recovar/em/b.py",
                "line": 20,
                "function": "g",
                "source": "b()",
            },
        },
    ]
    records_path = root / "provenance" / "cold_compile_calls.jsonl"
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    (root / "recovar_profiled.stderr").write_text(
        "Finished XLA compilation of jit_a in 2.5 sec\n"
        "Finished XLA compilation of jit_b in 0.2 seconds\n"
    )
    return root


def test_cold_compile_analyzer_ranks_callsite_file_module_and_status(tmp_path):
    root = _root(tmp_path)

    report = analyze(root, top=2)

    assert report["timing_truth"] is False
    assert report["records"]["count"] == 3
    assert report["records"]["total_s"] == pytest.approx(3.25)
    assert report["records"]["by_cache_status"] == [
        {
            "status": "miss",
            "count": 2,
            "total_s": 3.0,
            "mean_s": 1.5,
            "max_s": 2.0,
        },
        {
            "status": "hit",
            "count": 1,
            "total_s": 0.25,
            "mean_s": 0.25,
            "max_s": 0.25,
        },
    ]
    assert report["top_callsites"][0]["file"] == "recovar/em/a.py"
    assert report["top_callsites"][0]["count"] == 2
    assert report["top_files"][0]["total_s"] == pytest.approx(3.0)
    assert report["top_modules"][0]["module"] == "jit_a"
    assert report["stderr_xla_compilation"]["compile_count"] == 2
    assert report["stderr_xla_compilation"]["total_s"] == pytest.approx(2.7)


def test_cold_compile_analyzer_rejects_non_attribution_run(tmp_path):
    root = _root(tmp_path)
    _write(
        root / "provenance" / "run.json",
        {"cold_compile_attribution": False, "timing_truth_allowed": True},
    )

    with pytest.raises(RuntimeError, match="not marked"):
        analyze(root)
