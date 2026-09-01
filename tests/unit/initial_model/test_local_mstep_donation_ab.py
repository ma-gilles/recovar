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


INPUT_MANIFEST_SHA256 = "b" * 64


def _write_products(
    prefix: Path,
    *,
    iteration: int = 181,
    delta: float = 0.0,
    meta_delta: float = 0.0,
    timing_s: float = 1.0,
) -> None:
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
    with mrcfile.new(prefix.parent / "initial_model.mrc", overwrite=False) as handle:
        handle.set_data(volume.copy())
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
    meta = {
        "class_posterior_sums": [1000.0 + meta_delta],
        "class_posterior_sums_full": [1000.0],
        "class_direction_posterior_sums": [[500.0, 500.0]],
        "class_reconstruction_support_sums": [812.0],
        "class_assignments": [0, 0],
        "pose_assignments": [10, 20],
        "noise_sumw": 999.5,
        "wsum_img_power": [1.0, 2.0],
        "wsum_sigma2_noise": [3.0, 4.0],
        "halfset_0_profile_summary": {
            "big_jit_bucket_s": timing_s,
            "chunk_sizes": [75, 25],
            "class_posterior_sums_full": [1000.0],
        },
        "sparse_pass2_profile_summary": {
            "pass1_time_s": timing_s,
            "pass2_time_s": timing_s / 2,
            "mean_significant_samples": 17.0,
        },
        "vdam_iteration_profile_summary": {
            "expectation_time_s": timing_s,
            "mstep_time_s": timing_s / 5,
        },
        # The suffix alone must never make a field excludable.
        "science_rate_s": 7.0,
    }
    Path(f"{prefix}_it{iteration:03d}_recovar_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _science_outputs(prefix: Path, iteration: int = 181) -> dict[str, dict[str, object]]:
    paths = {
        "class_map": Path(f"{prefix}_it{iteration:03d}_class001.mrc"),
        "initial_model_map": prefix.parent / "initial_model.mrc",
        "data_star": Path(f"{prefix}_it{iteration:03d}_data.star"),
        "model_star": Path(f"{prefix}_it{iteration:03d}_model.star"),
        "recovar_meta": Path(f"{prefix}_it{iteration:03d}_recovar_meta.json"),
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


def _phase_report(prefix: Path, speed: float) -> dict:
    halfset = {
        "big_jit_bucket_s": speed / 2,
        "local_backproject_y_s": 1.0,
        "local_final_accumulator_s": 0.25,
        "big_jit_bucket_count": 14,
        "chunk_sizes": [75, 25],
        "class_posterior_sums_full": [1000.0],
    }
    return {
        "wall_s": speed,
        "normalized_argv": runner.SEALED_NORMALIZED_RECOVAR_ARGV,
        "timed_start_unix_ns": 100,
        "timed_end_unix_ns": 200,
        "jax_memory_after": {"peak_bytes_in_use": 1000},
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
    }


def _arm_report(
    cold_prefix: Path,
    warm_prefix: Path,
    arm: str,
    *,
    git_head: str,
    gpu_uuid: str,
    run_dir: Path,
) -> dict:
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
    return {
        "schema": runner.SCHEMA,
        "classification": "diagnostic_performance_only",
        "passed": True,
        "arm": arm,
        "git_head": git_head,
        "gpu_uuid": gpu_uuid,
        "checkpoint_iteration": runner.GF46_CHECKPOINT_ITERATION,
        "profiled_iteration": runner.GF46_PROFILED_ITERATION,
        "nr_iter_schedule": runner.GF46_NR_ITER_SCHEDULE,
        "normalized_options": runner.SEALED_NORMALIZED_OPTIONS,
        "normalized_recovar_argv": runner.SEALED_NORMALIZED_RECOVAR_ARGV,
        "input_manifest": {
            "path": "/sealed/input_manifest.json",
            "sha256": INPUT_MANIFEST_SHA256,
            "schema": runner.INPUT_MANIFEST_SCHEMA,
            "entries": [
                {
                    "relative_name": "checkpoint/optimiser.star",
                    "source_name": "run_it180_optimiser.star",
                    "size_bytes": 1,
                    "sha256": "c" * 64,
                },
                {
                    "relative_name": "input/run_it180_data.star",
                    "source_name": "run_it180_data.star",
                    "size_bytes": 1,
                    "sha256": "e" * 64,
                },
                {
                    "relative_name": "particles/particles.128.mrcs",
                    "source_name": "particles.128.mrcs",
                    "size_bytes": 1,
                    "sha256": "f" * 64,
                },
            ],
            "hashes": {
                "checkpoint/optimiser.star": "c" * 64,
                "input/run_it180_data.star": "e" * 64,
                "particles/particles.128.mrcs": "f" * 64,
            },
        },
        "input_hashes": {
            "checkpoint/optimiser.star": "c" * 64,
            "input/run_it180_data.star": "e" * 64,
            "particles/particles.128.mrcs": "f" * 64,
        },
        "checkpoint_optimiser_sha256": "c" * 64,
        "input_star": "/sealed/run_it180_data.star",
        "input_star_sha256": "e" * 64,
        "particle_stack": "/sealed/particles.128.mrcs",
        "particle_stack_sha256": "f" * 64,
        "runtime_provenance": {
            "repo_root": str(Path.cwd().resolve()),
            "pixi_env": str((Path.cwd() / ".pixi/envs/default").resolve()),
            "python_executable": str((Path.cwd() / ".pixi/envs/default/bin/python").resolve()),
            "python_prefix": str((Path.cwd() / ".pixi/envs/default").resolve()),
            "jax_path": str((Path.cwd() / ".pixi/envs/default/lib/python/site-packages/jax/__init__.py").resolve()),
            "recovar_path": str((Path.cwd() / "recovar/__init__.py").resolve()),
            "parity_ancestors_verified": True,
            "required_parity_ancestors": ["ancestor"],
        },
        "jax_persistent_cache": {
            "path": str((run_dir / "jax_cache").resolve()),
            "initial_empty": True,
            "initial_entry_count": 0,
            "final": {"entries": [], "manifest_sha256": "d" * 64},
        },
        "cold": _phase_report(cold_prefix, speed),
        "warm": _phase_report(warm_prefix, speed),
        "compiled_local_programs": {
            "contract": {
                "same_wrapped_numeric_source": True,
                "numeric_source_sha256": "source-sha",
            },
            "programs": [program],
        },
    }


def _analysis_tree(
    tmp_path: Path,
    *,
    cold_delta: float = 0.0,
    warm_delta: float = 0.0,
) -> tuple[Path, str, str]:
    root = tmp_path / "gate"
    git_head = "a" * 40
    gpu_uuid = "GPU-test"
    ordinal_by_run = {
        (1, "donated"): 1,
        (1, "control"): 2,
        (2, "control"): 3,
        (2, "donated"): 4,
        (3, "donated"): 5,
        (3, "control"): 6,
    }
    for repeat in range(1, 4):
        for arm in ("control", "donated"):
            ordinal = ordinal_by_run[(repeat, arm)]
            run_dir = root / "runs" / f"repeat-{repeat:02d}" / arm
            cold_prefix = run_dir / "result" / "cold" / "run"
            warm_prefix = run_dir / "result" / "warm" / "run"
            _write_products(cold_prefix, delta=cold_delta, timing_s=11.0)
            _write_products(warm_prefix, delta=warm_delta, timing_s=9.0)
            (run_dir / "jax_cache").mkdir()
            (run_dir / "launch_contract.json").write_text(
                json.dumps(
                    {
                        "schema": "recovar.local_mstep_donation_launch.v1",
                        "ordinal": ordinal,
                        "repeat": f"{repeat:02d}",
                        "arm": arm,
                        "jax_cache_dir": str((run_dir / "jax_cache").resolve()),
                        "jax_cache_initial_empty": True,
                        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
                        "checkpoint_iteration": 180,
                        "profiled_iteration": 181,
                        "nr_iter_schedule": 200,
                    },
                    indent=2,
                )
                + "\n"
            )
            report = _arm_report(
                cold_prefix,
                warm_prefix,
                arm,
                git_head=git_head,
                gpu_uuid=gpu_uuid,
                run_dir=run_dir,
            )
            report["warm"]["jax_memory_after"]["peak_bytes_in_use"] = 900 if arm == "donated" else 1000
            summary = run_dir / "result" / "donation_arm_summary.json"
            summary.write_text(json.dumps(report, indent=2) + "\n")
            (run_dir / "nvidia_smi.tsv").write_text(
                "unix_ns\tmemory_mib\n150\t900\n" if arm == "donated" else "unix_ns\tmemory_mib\n150\t1000\n"
            )
    return root, git_head, gpu_uuid


def _numeric_sample(value: float, *, discrete_value: int = 0) -> dict:
    return {
        "snapshot": {"value": float(value)},
        "discrete": {"choice": discrete_value},
        "continuous": {"value": np.asarray([value], dtype=np.float64)},
    }


def _numeric_gate(control: list[float], donated: list[float]) -> dict:
    return analyzer._phase_numeric_gate(
        {
            "control": [_numeric_sample(value) for value in control],
            "donated": [_numeric_sample(value) for value in donated],
        },
        "warm",
    )


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

    left_snapshot = analyzer.science_snapshot(left_report, "warm")
    right_snapshot = analyzer.science_snapshot(right_report, "warm")
    # mrcfile.py embeds a wall-clock creation label in raw container bytes.
    # Preserve those hashes for forensics but compare the complete parsed
    # science contract independently of that non-science label.
    for snapshot in (left_snapshot, right_snapshot):
        for name in ("class_map", "initial_model_map"):
            assert len(snapshot[name].pop("file_sha256")) == 64
    assert left_snapshot == right_snapshot

    _write_products(tmp_path / "changed" / "warm" / "run", delta=1.0)
    changed = tmp_path / "changed" / "warm" / "run"
    changed_report = {"warm": {"science_outputs": _science_outputs(changed)}}
    changed_snapshot = analyzer.science_snapshot(changed_report, "warm")
    for name in ("class_map", "initial_model_map"):
        changed_snapshot[name].pop("file_sha256")
    assert left_snapshot != changed_snapshot


def test_aggregate_requires_exact_science_and_qualifies_expected_alias(tmp_path):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path)
    payload = analyzer.analyze(
        root,
        expected_repo_head=git_head,
        expected_gpu_uuid=gpu_uuid,
        expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
    )

    assert payload["passed"] is True
    assert payload["same_source_exact_science_outputs"] is True
    assert payload["phase_matched_complete_science_outputs_exact"] == {
        "cold": True,
        "warm": True,
    }
    assert payload["numeric_qualification_status"] == "exact_pass"
    assert payload["numeric_qualification_allowed"] is True
    assert payload["unique_fresh_jax_cache_count"] == 6
    assert payload["strict_exactness_is_strong_evidence_not_universal_requirement"] is True
    assert payload["recovar_meta_exclusion_policy"]["explicit_timing_key_count"] == 49
    assert payload["recovar_meta_exclusion_policy"]["suffix_or_substring_exclusion_rules_used"] is False
    assert payload["arithmetic_changed"] is False
    assert payload["production_xhalf_adjoint_program_observed"] is True
    assert payload["alias_contract_passed"] is True
    assert payload["qualifying_programs"][0]["accumulator_bytes"] == 96
    assert payload["donated_over_control_ratios"]["e2e_wall_s"] == pytest.approx(0.9)
    assert payload["speed_claim_allowed"] is True
    policy = payload["broader_optimized_arithmetic_policy"]
    assert policy["stable_repeat_bounded_noise_allowed"] is True
    assert policy["directional_bias_allowed"] is False
    assert policy["iteration_amplified_drift_allowed"] is False
    assert policy["material_end_to_end_runtime_gain_required"] is True
    assert payload["default_promotion_allowed"] is False


def test_nonexact_map_noise_is_explicitly_inconclusive_with_only_3x3(tmp_path):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path)
    changed_prefix = root / "runs" / "repeat-03" / "donated" / "result" / "warm" / "run"
    map_path = Path(f"{changed_prefix}_it181_class001.mrc")
    with mrcfile.open(map_path, mode="r+") as handle:
        handle.data[0, 0, 0] = np.float32(1.0)

    payload = analyzer.analyze(
        root,
        expected_repo_head=git_head,
        expected_gpu_uuid=gpu_uuid,
        expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
    )

    assert payload["passed"] is True
    assert payload["phase_numeric_gates"]["cold"]["status"] == "exact_pass"
    assert payload["phase_numeric_gates"]["warm"]["status"] == "inconclusive_3x3"
    assert payload["numeric_qualification_status"] == "inconclusive_3x3"
    assert payload["numeric_qualification_allowed"] is False
    assert payload["speed_claim_allowed"] is False


def test_cold_only_noise_does_not_contaminate_the_warm_cohort(tmp_path):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path)
    changed = root / "runs/repeat-02/control/result/cold/initial_model.mrc"
    with mrcfile.open(changed, mode="r+") as handle:
        handle.data[1, 1, 1] += np.float32(1.0)

    payload = analyzer.analyze(
        root,
        expected_repo_head=git_head,
        expected_gpu_uuid=gpu_uuid,
        expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
    )

    assert payload["phase_numeric_gates"]["cold"]["status"] == "inconclusive_3x3"
    assert payload["phase_numeric_gates"]["warm"]["status"] == "exact_pass"
    assert payload["numeric_qualification_status"] == "inconclusive_3x3"


def test_non_timing_recovar_meta_noise_is_measured_not_hidden(tmp_path):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path)
    meta_path = root / "runs/repeat-02/donated/result/warm/run_it181_recovar_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["class_direction_posterior_sums"][0][1] += 1.0
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    payload = analyzer.analyze(
        root,
        expected_repo_head=git_head,
        expected_gpu_uuid=gpu_uuid,
        expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
    )

    warm = payload["phase_numeric_gates"]["warm"]
    assert warm["status"] == "inconclusive_3x3"
    assert warm["exact_continuous_science"] is False
    assert "meta/class_direction_posterior_sums" in warm["continuous_distance"]["field_names"]
    assert payload["speed_claim_allowed"] is False


def test_cold_and_warm_may_differ_when_each_phase_matched_cohort_is_exact(tmp_path):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path, cold_delta=2.0)
    payload = analyzer.analyze(
        root,
        expected_repo_head=git_head,
        expected_gpu_uuid=gpu_uuid,
        expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
    )

    assert payload["numeric_qualification_status"] == "exact_pass"
    assert payload["phase_numeric_gates"]["cold"]["status"] == "exact_pass"
    assert payload["phase_numeric_gates"]["warm"]["status"] == "exact_pass"
    snapshots = payload["runs"][0]["parsed_science_snapshots"]
    assert snapshots["cold"] != snapshots["warm"]


def test_control_repeat_variation_calibrates_nonexact_3x3_as_inconclusive():
    gate = _numeric_gate(
        control=[-1.0, 0.0, 1.0],
        donated=[-0.8, 0.2, 0.8],
    )

    assert gate["status"] == "inconclusive_3x3"
    assert max(gate["control_control_distances"]) > 0.0
    assert max(gate["donated_donated_distances"]) > 0.0
    assert gate["nondirectional_distribution_failure"] is False
    assert gate["variance_inflation_failure"] is False
    assert gate["hard_failure"] is False


def test_phase_gate_rejects_a_consistent_donated_shift():
    gate = _numeric_gate(
        control=[-0.1, 0.0, 0.1],
        donated=[9.9, 10.0, 10.1],
    )

    assert gate["status"] == "fail"
    assert gate["complete_cross_within_separation"] is True
    assert gate["energy_exact_permutation_p"] == pytest.approx(0.1)
    assert gate["nondirectional_distribution_failure"] is True


def test_phase_gate_rejects_donated_variance_inflation():
    gate = _numeric_gate(
        control=[0.0, 0.0, 0.0],
        donated=[-10.0, 10.0, 10.0],
    )

    assert gate["status"] == "fail"
    assert gate["variance_delta"] > 0.0
    assert gate["variance_inflation_exact_permutation_p"] == pytest.approx(0.05)
    assert gate["variance_inflation_failure"] is True


def test_continuous_distance_keeps_complex_imaginary_differences():
    samples = [
        {"continuous": {"complex_star_value": np.asarray([1.0 + 1.0j])}},
        {"continuous": {"complex_star_value": np.asarray([1.0 + 2.0j])}},
    ]

    matrix, report = analyzer._continuous_distance_matrix(samples)

    assert report["field_names"] == ["complex_star_value"]
    assert matrix[0, 1] > 0.0


def test_phase_gate_requires_exact_discrete_choices():
    controls = [_numeric_sample(0.0) for _ in range(3)]
    donated = [_numeric_sample(0.0) for _ in range(3)]
    donated[-1] = _numeric_sample(0.0, discrete_value=1)

    with pytest.raises(RuntimeError, match="discrete choices/identity/topology are not exact"):
        analyzer._phase_numeric_gate(
            {"control": controls, "donated": donated},
            "cold",
        )


def test_meta_exclusion_allowlist_is_narrow_and_timing_only(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    base = {
        "class_assignments": [0, 0],
        "science_rate_s": 7.0,
        "vdam_iteration_profile_summary": {"expectation_time_s": 10.0},
        "halfset_0_profile_summary": {
            "big_jit_bucket_s": 4.0,
            "class_posterior_sums_full": [1000.0],
        },
    }
    changed_timing = json.loads(json.dumps(base))
    changed_timing["vdam_iteration_profile_summary"]["expectation_time_s"] = 5.0
    changed_timing["halfset_0_profile_summary"]["big_jit_bucket_s"] = 2.0
    left.write_text(json.dumps(base))
    right.write_text(json.dumps(changed_timing))
    assert analyzer.recovar_meta_science_snapshot(left) == analyzer.recovar_meta_science_snapshot(right)

    changed_science = json.loads(json.dumps(changed_timing))
    changed_science["science_rate_s"] = 8.0
    right.write_text(json.dumps(changed_science))
    assert analyzer.recovar_meta_science_snapshot(left) != analyzer.recovar_meta_science_snapshot(right)


def test_aggregate_rejects_shared_or_nonempty_jax_cache_provenance(tmp_path):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path)
    report_path = root / "runs/repeat-03/donated/result/donation_arm_summary.json"
    report = json.loads(report_path.read_text())
    report["jax_persistent_cache"]["initial_empty"] = False
    report["jax_persistent_cache"]["initial_entry_count"] = 1
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    with pytest.raises(RuntimeError, match="cache was not fresh"):
        analyzer.analyze(
            root,
            expected_repo_head=git_head,
            expected_gpu_uuid=gpu_uuid,
            expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
        )


def test_runner_rejects_a_nonempty_per_run_jax_cache(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    output_root = run_dir / "result"
    cache_dir = run_dir / "jax_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "old-cache-entry").write_text("stale")
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(cache_dir))

    with pytest.raises(RuntimeError, match="not empty"):
        runner._assert_fresh_jax_cache(output_root, cache_dir)


def test_aggregate_rejects_input_manifest_or_schedule_drift(tmp_path):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path)
    report_path = root / "runs/repeat-01/control/result/donation_arm_summary.json"
    report = json.loads(report_path.read_text())
    report["input_manifest"]["sha256"] = "e" * 64
    report["checkpoint_iteration"] = 179
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    with pytest.raises(RuntimeError, match="checkpoint schedule drifted"):
        analyzer.analyze(
            root,
            expected_repo_head=git_head,
            expected_gpu_uuid=gpu_uuid,
            expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
        )


def test_aggregate_rejects_input_manifest_drift_with_sealed_schedule(tmp_path):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path)
    report_path = root / "runs/repeat-01/control/result/donation_arm_summary.json"
    report = json.loads(report_path.read_text())
    report["input_manifest"]["sha256"] = "0" * 64
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    with pytest.raises(RuntimeError, match="input-manifest SHA drifted"):
        analyzer.analyze(
            root,
            expected_repo_head=git_head,
            expected_gpu_uuid=gpu_uuid,
            expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
        )


def test_runner_rejects_any_override_of_the_reviewed_gf46_schedule():
    args = runner._parse_args(
        [
            "--arm",
            "control",
            "--checkpoint-optimiser",
            "run_it180_optimiser.star",
            "--input-star",
            "run_it180_data.star",
            "--data-dir",
            "data",
            "--particle-stack",
            "data/particles.128.mrcs",
            "--output-root",
            "out",
            "--input-manifest",
            "manifest.json",
            "--expected-input-manifest-sha256",
            INPUT_MANIFEST_SHA256,
            "--expected-jax-cache-dir",
            "cache",
            "--expected-repo-head",
            "a" * 40,
            "--expected-gpu-uuid",
            "GPU-test",
            "--checkpoint-iteration",
            "179",
        ]
    )
    with pytest.raises(RuntimeError, match="reviewed GF46 iteration-181 schedule"):
        runner._validate_sealed_gf46_options(args)


def test_gf46_input_manifest_is_root_independent_and_content_sensitive(tmp_path):
    def make_fixture(root: Path) -> tuple[Path, Path, Path]:
        checkpoint = root / "checkpoint/run_it180_optimiser.star"
        checkpoint.parent.mkdir(parents=True)
        prefix = Path(str(checkpoint)[: -len("_optimiser.star")])
        for suffix in (
            "optimiser.star",
            "model.star",
            "data.star",
            "sampling.star",
            "class001.mrc",
            "1moment001.mrc",
            "1moment002.mrc",
            "2moment001.mrc",
        ):
            Path(f"{prefix}_{suffix}").write_bytes(f"fixture:{suffix}".encode())
        input_star = root / "input/run_it180_data.star"
        particle_stack = root / "particles/particles.128.mrcs"
        input_star.parent.mkdir()
        particle_stack.parent.mkdir()
        input_star.write_text("input-star")
        particle_stack.write_bytes(b"particle-stack")
        return checkpoint, input_star, particle_stack

    left = make_fixture(tmp_path / "left")
    right = make_fixture(tmp_path / "right")
    left_payload = runner.gf46_input_manifest_payload(*left)
    right_payload = runner.gf46_input_manifest_payload(*right)
    assert left_payload == right_payload
    assert all("/left/" not in json.dumps(entry) for entry in left_payload["entries"])

    right[2].write_bytes(b"particle-stack-drift")
    assert runner.gf46_input_manifest_payload(*right) != left_payload


def test_aggregate_rejects_missing_runtime_ancestry(tmp_path):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path)
    report_path = root / "runs/repeat-01/donated/result/donation_arm_summary.json"
    report = json.loads(report_path.read_text())
    report["runtime_provenance"]["parity_ancestors_verified"] = False
    report["runtime_provenance"]["jax_path"] = "/tmp/other-env/jax/__init__.py"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    with pytest.raises(RuntimeError, match="required parity ancestors"):
        analyzer.analyze(
            root,
            expected_repo_head=git_head,
            expected_gpu_uuid=gpu_uuid,
            expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
        )


def test_aggregate_rejects_jax_outside_exact_worktree_pixi_env(tmp_path):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path)
    report_path = root / "runs/repeat-01/donated/result/donation_arm_summary.json"
    report = json.loads(report_path.read_text())
    report["runtime_provenance"]["jax_path"] = "/tmp/other-env/jax/__init__.py"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    with pytest.raises(RuntimeError, match="jax_path is outside"):
        analyzer.analyze(
            root,
            expected_repo_head=git_head,
            expected_gpu_uuid=gpu_uuid,
            expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
        )


def test_slurm_runner_is_crossed_fresh_process_and_fail_closed():
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "scripts/run_local_mstep_donation_ab.sbatch").read_text()

    required = (
        ': "${EXPECTED_REPO_HEAD:',
        ': "${EXPECTED_SOURCE_MANIFEST_SHA256:',
        ': "${EXPECTED_INPUT_MANIFEST_SHA256:',
        ': "${EXPECTED_NODE_NAME:',
        ': "${TARGET_GPU_UUID:',
        'test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"',
        "run_arm 1 01 donated",
        "run_arm 2 01 control",
        "run_arm 3 02 control",
        "run_arm 4 02 donated",
        "run_arm 5 03 donated",
        "run_arm 6 03 control",
        "local cache_dir=${run_dir}/jax_cache",
        'test ! -e "${cache_dir}"',
        'test -z "$(find "${cache_dir}" -mindepth 1 -print -quit)"',
        '--expected-jax-cache-dir "${cache_dir}"',
        '--expected-input-manifest-sha256 "${input_manifest_sha256}"',
        "--query-gpu=memory.used",
        'assert payload["cold_and_warm_are_separate_phase_matched_cohorts"] is True',
        'assert payload["numeric_qualification_status"] in {"exact_pass", "inconclusive_3x3"}',
        'assert payload["numeric_qualification_allowed"] is False',
        'assert payload["speed_claim_allowed"] is False',
        "assert_parity_ancestors()",
        "pathlib.Path(jax.__file__).resolve().is_relative_to(pixi_env)",
        'assert payload["production_xhalf_adjoint_program_observed"] is True',
        'assert payload["alias_contract_passed"] is True',
        'cmp "${PROVENANCE}/source_manifest.sha256"',
        'cmp "${PROVENANCE}/input_manifest.json"',
        'touch "${ROOT}/COMPLETED"',
    )
    for fragment in required:
        assert fragment in source
    harness = (repo_root / "scripts/run_local_mstep_donation_ab.py").read_text()
    assert "clear_memory_stats" not in harness
    assert "whole_process_cumulative_not_warm_isolated" in harness
    assert source.rindex('touch "${ROOT}/COMPLETED"') > source.rindex('cmp "${PROVENANCE}/source_manifest.sha256"')
