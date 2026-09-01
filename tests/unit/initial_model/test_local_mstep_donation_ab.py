"""Focused contracts for the same-source late-VDAM donation A/B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile

from scripts import analyze_local_mstep_donation_ab as analyzer
from scripts import run_local_mstep_donation_ab as runner

pytestmark = pytest.mark.unit


def _write_products(prefix: Path, *, iteration: int = 181, delta: float = 0.0) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    map_path = Path(f"{prefix}_it{iteration:03d}_class001.mrc")
    volume = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    volume[0, 0, 0] += np.float32(delta)
    with mrcfile.new(map_path, overwrite=False) as handle:
        handle.set_data(volume)
        handle.voxel_size = 1.5
        handle.header.origin.x = 0.25
        handle.header.origin.y = -0.5
        handle.header.origin.z = 0.75

    tables = {
        "general": {
            "rlnOutputRootName": str(prefix),
            "rlnCurrentIteration": iteration,
            "rlnCurrentResolution": 12.5,
        },
        "particles": pd.DataFrame(
            {
                "rlnImageId": [1, 2],
                "rlnAngleRot": [10.0, 20.0],
                "rlnReferenceImage": [
                    f"1@{prefix}_it{iteration:03d}_class001.mrc",
                    f"1@{prefix}_it{iteration:03d}_class001.mrc",
                ],
            }
        ),
    }
    for suffix in ("data", "model"):
        starfile.write(tables, f"{prefix}_it{iteration:03d}_{suffix}.star")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _science_outputs(prefix: Path, iteration: int = 181) -> dict[str, dict[str, object]]:
    paths = {
        "class_map": Path(f"{prefix}_it{iteration:03d}_class001.mrc"),
        "data_star": Path(f"{prefix}_it{iteration:03d}_data.star"),
        "model_star": Path(f"{prefix}_it{iteration:03d}_model.star"),
    }
    outputs = {
        name: {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
    }
    outputs["output_prefix"] = {"path": str(prefix.resolve())}
    return outputs


def _arm_report(prefix: Path, arm: str, *, git_head: str, gpu_uuid: str) -> dict:
    alias_bytes = 96 if arm == "donated" else 0
    speed = 9.0 if arm == "donated" else 10.0
    program = {
        "program_key": "program-1",
        "first_phase": "cold",
        "signature": {"dynamic_tree": "tree", "dynamic_leaves": [], "static": {}},
        "accumulator_bytes": {"Ft_y": 64, "Ft_ctf": 32, "total": 96},
        "adjoints_enabled": True,
        "mstep_relion_x_half": True,
        "memory_analysis": {
            "generated_code_size_in_bytes": 1,
            "argument_size_in_bytes": 200,
            "output_size_in_bytes": 100,
            "alias_size_in_bytes": alias_bytes,
            "temp_size_in_bytes": 20,
            "host_generated_code_size_in_bytes": 0,
            "host_argument_size_in_bytes": 0,
            "host_output_size_in_bytes": 0,
            "host_alias_size_in_bytes": 0,
            "host_temp_size_in_bytes": 0,
        },
    }
    halfset = {
        "big_jit_bucket_s": speed / 2,
        "local_backproject_y_s": 1.0,
        "local_final_accumulator_s": 0.25,
        "big_jit_bucket_count": 14,
        "chunk_sizes": [75, 25],
        "class_posterior_sums_full": [1000.0],
    }
    return {
        "schema": runner.SCHEMA,
        "classification": "diagnostic_performance_only",
        "passed": True,
        "arm": arm,
        "git_head": git_head,
        "gpu_uuid": gpu_uuid,
        "warm": {
            "wall_s": speed,
            "timed_start_unix_ns": 100,
            "timed_end_unix_ns": 200,
            "jax_memory_after": {"peak_bytes_in_use": 900 if arm == "donated" else 1000},
            "jax_memory_scope": "whole_process_cumulative_not_warm_isolated",
            "science_outputs": _science_outputs(prefix),
            "iteration_profile": {
                "expectation_time_s": speed,
                "mstep_time_s": speed / 5,
            },
            "schedule": {
                "current_size": 128,
                "n_rotations": 294_912,
                "subset_size": 1000,
            },
            "halfset_profiles": {
                "halfset_0_profile_summary": halfset,
                "halfset_1_profile_summary": halfset,
            },
            "sparse_pass2_profile": {"logical_rows": 1000, "wall_s": speed},
        },
        "compiled_local_programs": {
            "contract": {
                "same_wrapped_numeric_source": True,
                "numeric_source_sha256": "source-sha",
            },
            "programs": [program],
        },
    }


def _analysis_tree(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "gate"
    git_head = "a" * 40
    gpu_uuid = "GPU-test"
    for repeat in range(1, 4):
        for arm in ("control", "donated"):
            run_dir = root / "runs" / f"repeat-{repeat:02d}" / arm
            prefix = run_dir / "result" / "warm" / "run"
            _write_products(prefix)
            report = _arm_report(prefix, arm, git_head=git_head, gpu_uuid=gpu_uuid)
            summary = run_dir / "result" / "donation_arm_summary.json"
            summary.write_text(json.dumps(report, indent=2) + "\n")
            (run_dir / "nvidia_smi.tsv").write_text(
                "unix_ns\tmemory_mib\n150\t900\n" if arm == "donated" else "unix_ns\tmemory_mib\n150\t1000\n"
            )
    return root, git_head, gpu_uuid


def test_control_and_donated_wrappers_share_one_sealed_numeric_source():
    control = runner.LocalMstepDonationMonitor("control")
    donated = runner.LocalMstepDonationMonitor("donated")

    assert control.selected.__wrapped__ is donated.selected.__wrapped__
    assert control.contract()["selected_donate_argnums"] == []
    assert donated.contract()["selected_donate_argnums"] == [7, 8]
    assert donated.contract()["selected_donate_argnames"] == ["Ft_y", "Ft_ctf"]
    assert tuple(donated.contract()["static_argnames"]) == runner.SEALED_STATIC_ARGNAMES


def test_science_snapshot_normalizes_only_output_prefix_and_keeps_exact_values(tmp_path):
    left = tmp_path / "left" / "warm" / "run"
    right = tmp_path / "right" / "warm" / "run"
    _write_products(left)
    _write_products(right)
    left_report = {"warm": {"science_outputs": _science_outputs(left)}}
    right_report = {"warm": {"science_outputs": _science_outputs(right)}}

    assert analyzer.science_snapshot(left_report) == analyzer.science_snapshot(right_report)

    _write_products(tmp_path / "changed" / "warm" / "run", delta=1.0)
    changed = tmp_path / "changed" / "warm" / "run"
    changed_report = {"warm": {"science_outputs": _science_outputs(changed)}}
    assert analyzer.science_snapshot(left_report) != analyzer.science_snapshot(changed_report)


def test_aggregate_requires_exact_science_and_qualifies_expected_alias(tmp_path):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path)
    payload = analyzer.analyze(
        root,
        expected_repo_head=git_head,
        expected_gpu_uuid=gpu_uuid,
    )

    assert payload["passed"] is True
    assert payload["same_source_exact_science_outputs"] is True
    assert payload["donation_specific_exact_science_contract"] is True
    assert payload["arithmetic_changed"] is False
    assert payload["production_xhalf_adjoint_program_observed"] is True
    assert payload["alias_contract_passed"] is True
    assert payload["qualifying_programs"][0]["accumulator_bytes"] == 96
    assert payload["donated_over_control_ratios"]["e2e_wall_s"] == pytest.approx(0.9)
    policy = payload["broader_optimized_arithmetic_policy"]
    assert policy["stable_repeat_bounded_noise_allowed"] is True
    assert policy["directional_bias_allowed"] is False
    assert policy["iteration_amplified_drift_allowed"] is False
    assert policy["material_end_to_end_runtime_gain_required"] is True
    assert payload["default_promotion_allowed"] is False


def test_aggregate_rejects_nonexact_map_voxels(tmp_path):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path)
    changed_prefix = root / "runs" / "repeat-03" / "donated" / "result" / "warm" / "run"
    map_path = Path(f"{changed_prefix}_it181_class001.mrc")
    with mrcfile.open(map_path, mode="r+") as handle:
        handle.data[0, 0, 0] = np.float32(1.0)

    with pytest.raises(RuntimeError, match="science outputs are not exact"):
        analyzer.analyze(
            root,
            expected_repo_head=git_head,
            expected_gpu_uuid=gpu_uuid,
        )


def test_slurm_runner_is_crossed_fresh_process_and_fail_closed():
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "scripts/run_local_mstep_donation_ab.sbatch").read_text()

    required = (
        ': "${EXPECTED_REPO_HEAD:',
        ': "${EXPECTED_SOURCE_MANIFEST_SHA256:',
        ': "${EXPECTED_NODE_NAME:',
        ': "${TARGET_GPU_UUID:',
        'test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"',
        "run_arm 1 01 donated",
        "run_arm 2 01 control",
        "run_arm 3 02 control",
        "run_arm 4 02 donated",
        "run_arm 5 03 donated",
        "run_arm 6 03 control",
        "--query-gpu=memory.used",
        'assert payload["same_source_exact_science_outputs"] is True',
        'assert payload["production_xhalf_adjoint_program_observed"] is True',
        'assert payload["alias_contract_passed"] is True',
        'cmp "${PROVENANCE}/source_manifest.sha256"',
        'touch "${ROOT}/COMPLETED"',
    )
    for fragment in required:
        assert fragment in source
    harness = (repo_root / "scripts/run_local_mstep_donation_ab.py").read_text()
    assert "clear_memory_stats" not in harness
    assert "whole_process_cumulative_not_warm_isolated" in harness
    assert source.rindex('touch "${ROOT}/COMPLETED"') > source.rindex('cmp "${PROVENANCE}/source_manifest.sha256"')
