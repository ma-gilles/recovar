import json
import sqlite3
from pathlib import Path

import pytest

from scripts.run_vdam_late_iteration_profile import _profile_metadata
from scripts.summarize_vdam_nsys_sqlite import summarize

ROOT = Path(__file__).resolve().parents[3]


def test_late_profile_metadata_requires_exactly_one_diagnostic_iteration(tmp_path):
    prefix = tmp_path / "warm" / "run"
    prefix.parent.mkdir()
    meta_path = Path(f"{prefix}_it181_recovar_meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "current_size": 128,
                "healpix_order": 3,
                "n_rotations": 294912,
                "n_translations": 116,
                "subset_size": 1000,
                "random_perturbation": 0.1,
                "vdam_iteration_profile_summary": {"expectation_time_s": 1.25},
            }
        )
    )
    Path(f"{prefix}_diagnostic_continuation.json").write_text(
        json.dumps(
            {
                "classification": "diagnostic_performance_only",
                "iteration": 180,
            }
        )
    )

    report = _profile_metadata(prefix, 181)

    assert report["schedule"]["n_rotations"] == 294912
    assert report["iteration_profile"]["expectation_time_s"] == 1.25
    Path(f"{prefix}_it182_recovar_meta.json").write_text("{}")
    with pytest.raises(RuntimeError, match="exactly one iteration"):
        _profile_metadata(prefix, 181)


def test_nsys_sqlite_summary_reports_invocations_shapes_and_busy_fraction(tmp_path):
    sqlite_path = tmp_path / "trace.sqlite"
    connection = sqlite3.connect(sqlite_path)
    connection.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO StringIds VALUES (1, 'coarse_kernel'), (2, 'cudaLaunchKernel');
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER,
            end INTEGER,
            deviceId INTEGER,
            shortName INTEGER,
            gridX INTEGER,
            gridY INTEGER,
            gridZ INTEGER,
            blockX INTEGER,
            blockY INTEGER,
            blockZ INTEGER
        );
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
            (10, 30, 0, 1, 8, 1, 1, 256, 1, 1),
            (20, 40, 0, 1, 8, 1, 1, 256, 1, 1);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER,
            end INTEGER,
            nameId INTEGER
        );
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0, 100, 2);
        """
    )
    connection.commit()
    connection.close()

    report = summarize(sqlite_path)

    assert report["capture_span_source"] == "cuda_api"
    assert report["capture_span_ns"] == 100
    assert report["devices"]["0"]["kernel_count"] == 2
    assert report["devices"]["0"]["gpu_busy_ns"] == 30
    assert report["devices"]["0"]["gpu_idle_ns_within_capture"] == 70
    assert report["devices"]["0"]["gpu_busy_fraction_within_capture"] == pytest.approx(0.3)
    assert report["kernels"][0]["name"] == "coarse_kernel"
    assert report["kernels"][0]["count"] == 2
    assert report["kernel_signatures"][0]["shape"] == {
        "gridX": 8,
        "gridY": 1,
        "gridZ": 1,
        "blockX": 256,
        "blockY": 1,
        "blockZ": 1,
    }


def test_late_profile_slurm_gate_is_one_iteration_and_fail_closed():
    launcher = (ROOT / "scripts" / "run_vdam_late_iteration_profile.sbatch").read_text()
    gdb_commands = (ROOT / "scripts" / "vdam_relion_one_iteration.gdb").read_text()

    assert "#SBATCH --constraint=h100" in launcher
    assert "--capture-range=cudaProfilerApi" in launcher
    assert "--capture-range-end=stop" in launcher
    assert "EXPECTED_RELION_SHA256" in launcher
    assert "EXPECTED_RELION_BIND_SHA256" in launcher
    assert "status --porcelain=v1 --untracked-files=no" in launcher
    assert "test ! -e" in launcher
    assert 'test ! -e "${NATIVE_PROFILE}/run_it' in launcher
    assert "--diagnostic_continue_optimiser" in (ROOT / "scripts" / "run_vdam_late_iteration_profile.py").read_text()
    assert "VDAM_GDB_FIRST_EXPECTATION" in gdb_commands
    assert "VDAM_GDB_SECOND_EXPECTATION" in gdb_commands
    assert "cudaProfilerStart" in gdb_commands
    assert "cudaProfilerStop" in gdb_commands
