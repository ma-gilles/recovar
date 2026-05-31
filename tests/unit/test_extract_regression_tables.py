import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "extract_regression_tables.py"
_SPEC = importlib.util.spec_from_file_location("extract_regression_tables", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
extract_regression_tables = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(extract_regression_tables)


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_perf_table_uses_current_gpu_baseline(tmp_path):
    baseline = {
        "NVIDIA H100 80GB HBM3": {
            "stages": {
                "pipeline": {
                    "wall_seconds": 100.0,
                    "peak_gpu_memory_gb": 10.0,
                    "peak_cpu_memory_gb": 10.0,
                }
            }
        },
        "NVIDIA A100-SXM4-80GB": {
            "stages": {
                "pipeline": {
                    "wall_seconds": 200.0,
                    "peak_gpu_memory_gb": 10.0,
                    "peak_cpu_memory_gb": 10.0,
                }
            }
        },
    }
    current = {
        "hardware": {"gpu_name": "NVIDIA A100-SXM4-80GB"},
        "stages": {
            "pipeline": {
                "wall_seconds": 205.0,
                "peak_gpu_memory_gb": 10.0,
                "peak_cpu_memory_gb": 10.0,
            }
        },
    }
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write_json(baseline_path, baseline)
    _write_json(current_path, current)

    table = extract_regression_tables.perf_table(str(baseline_path), str(current_path), "SPA")

    assert "| pipeline | wall_time | 200.0s | 205.0s | +2.5% (worse) | OK |" in table
    assert "**REGRESSED**" not in table


def test_perf_table_falls_back_when_current_gpu_is_unknown(tmp_path):
    baseline = {
        "NVIDIA H100 80GB HBM3": {
            "stages": {
                "pipeline": {
                    "wall_seconds": 100.0,
                    "peak_gpu_memory_gb": 10.0,
                    "peak_cpu_memory_gb": 10.0,
                }
            }
        }
    }
    current = {
        "hardware": {"gpu_name": "Unknown GPU"},
        "stages": {
            "pipeline": {
                "wall_seconds": 105.0,
                "peak_gpu_memory_gb": 10.0,
                "peak_cpu_memory_gb": 10.0,
            }
        },
    }
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write_json(baseline_path, baseline)
    _write_json(current_path, current)

    table = extract_regression_tables.perf_table(str(baseline_path), str(current_path), "SPA")

    assert "| pipeline | wall_time | 100.0s | 105.0s | +5.0% (worse) | OK |" in table


def test_perf_table_does_not_flag_tiny_gpu_memory_baseline(tmp_path):
    baseline = {
        "stages": {
            "metrics": {
                "wall_seconds": 10.0,
                "peak_gpu_memory_gb": 0.08,
                "peak_cpu_memory_gb": 1.0,
            }
        }
    }
    current = {
        "stages": {
            "metrics": {
                "wall_seconds": 10.0,
                "peak_gpu_memory_gb": 0.14,
                "peak_cpu_memory_gb": 1.0,
            }
        },
    }
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write_json(baseline_path, baseline)
    _write_json(current_path, current)

    table = extract_regression_tables.perf_table(str(baseline_path), str(current_path), "SPA")

    assert "| metrics | GPU_memory | 0.1GB | 0.1GB | +75.0% (worse) | OK |" in table
