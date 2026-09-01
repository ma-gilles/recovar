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
LAUNCH_MANIFEST_SHA256 = "1" * 64


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


def _sealed_environment(run_dir: Path, gpu_uuid: str) -> dict:
    observed = {name: "sealed-test-value" for name in runner.SEALED_ARM_ENVIRONMENT_NAMES}
    observed.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            "JAX_COMPILATION_CACHE_DIR": str((run_dir / "jax_cache").resolve()),
            "RECOVAR_EXPECTED_REPO_ROOT": str(Path.cwd().resolve()),
        }
    )
    return {
        "schema": runner.SEALED_ARM_ENVIRONMENT_SCHEMA,
        "exact_allowlist": True,
        "allowed_names": sorted(runner.SEALED_ARM_ENVIRONMENT_NAMES),
        "observed": observed,
        "unexpected_names": [],
        "iref_replay_forbidden": True,
    }


def _sealed_analysis_inputs(root: Path, *, git_head: str, gpu_uuid: str) -> dict:
    sealed = root / "sealed"
    data_dir = sealed / "particles"
    provenance = root / "provenance"
    data_dir.mkdir(parents=True)
    provenance.mkdir()
    checkpoint_prefix = sealed / "run_it180"
    checkpoint_optimiser = Path(f"{checkpoint_prefix}_optimiser.star")
    input_star = Path(f"{checkpoint_prefix}_data.star")
    particle_stack = data_dir / "particles.128.mrcs"
    named_paths = [
        ("checkpoint/optimiser.star", checkpoint_optimiser),
        ("checkpoint/model.star", Path(f"{checkpoint_prefix}_model.star")),
        ("checkpoint/data.star", input_star),
        ("checkpoint/sampling.star", Path(f"{checkpoint_prefix}_sampling.star")),
        ("checkpoint/class001.mrc", Path(f"{checkpoint_prefix}_class001.mrc")),
        ("checkpoint/1moment001.mrc", Path(f"{checkpoint_prefix}_1moment001.mrc")),
        ("checkpoint/1moment002.mrc", Path(f"{checkpoint_prefix}_1moment002.mrc")),
        ("checkpoint/2moment001.mrc", Path(f"{checkpoint_prefix}_2moment001.mrc")),
        (f"input/{input_star.name}", input_star),
        (f"particles/000/{particle_stack.name}", particle_stack),
    ]
    for index, (_role, path) in enumerate(named_paths):
        if not path.exists():
            path.write_bytes(f"sealed-analysis-input-{index}\n".encode())
    entries = [
        {
            "relative_name": role,
            "source_name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for role, path in named_paths
    ]
    hashes = {entry["relative_name"]: entry["sha256"] for entry in entries}
    resolved_inputs = {
        "schema": runner.RESOLVED_INPUT_CONTRACT_SCHEMA,
        "data_dir": str(data_dir.resolve()),
        "consumed": [
            {"role": role, "path": str(path.resolve())}
            for role, path in named_paths
        ],
        "particle_stacks": [str(particle_stack.resolve())],
    }
    input_manifest_path = provenance / "input_manifest.json"
    input_manifest_path.write_text("{}\n")
    input_manifest = {
        "path": str(input_manifest_path.resolve()),
        "sha256": INPUT_MANIFEST_SHA256,
        "schema": runner.INPUT_MANIFEST_SCHEMA,
        "entries": entries,
        "resolved_inputs": resolved_inputs,
        "hashes": hashes,
    }
    launch_payload = {
        "schema": runner.LAUNCH_MANIFEST_SCHEMA,
        "output_root": str(root.resolve()),
        "repo_root": str(Path.cwd().resolve()),
        "checkpoint_optimiser": str(checkpoint_optimiser.resolve()),
        "input_star": str(input_star.resolve()),
        "data_dir": str(data_dir.resolve()),
        "particle_stacks": [str(particle_stack.resolve())],
        "expected_repo_head": git_head,
        "expected_source_manifest_sha256": "2" * 64,
        "expected_input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "expected_node_name": "test-node",
        "target_gpu_uuid": gpu_uuid,
        "relion_bind_binary": "/bin/true",
        "expected_relion_bind_sha256": "3" * 64,
        "expected_focused_test_count": 1,
    }
    launch_manifest_path = provenance / "launch_manifest.json"
    launch_manifest_path.write_text(json.dumps(launch_payload, indent=2, sort_keys=True) + "\n")
    return {
        "checkpoint_optimiser": str(checkpoint_optimiser.resolve()),
        "input_star": str(input_star.resolve()),
        "data_dir": str(data_dir.resolve()),
        "particle_stack": str(particle_stack.resolve()),
        "input_manifest": input_manifest,
        "launch_manifest": {
            "path": str(launch_manifest_path.resolve()),
            "sha256": LAUNCH_MANIFEST_SHA256,
            "schema": runner.LAUNCH_MANIFEST_SCHEMA,
            "payload": launch_payload,
        },
    }


def _arm_report(
    cold_prefix: Path,
    warm_prefix: Path,
    arm: str,
    *,
    git_head: str,
    gpu_uuid: str,
    run_dir: Path,
    sealed_inputs: dict,
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
        "launch_manifest": sealed_inputs["launch_manifest"],
        "input_manifest": sealed_inputs["input_manifest"],
        "input_hashes": sealed_inputs["input_manifest"]["hashes"],
        "checkpoint_optimiser": sealed_inputs["checkpoint_optimiser"],
        "checkpoint_optimiser_sha256": sealed_inputs["input_manifest"]["hashes"][
            "checkpoint/optimiser.star"
        ],
        "input_star": sealed_inputs["input_star"],
        "input_star_sha256": sealed_inputs["input_manifest"]["hashes"][
            "input/run_it180_data.star"
        ],
        "data_dir": sealed_inputs["data_dir"],
        "particle_stacks": [
            {
                "path": sealed_inputs["particle_stack"],
                "sha256": sealed_inputs["input_manifest"]["hashes"][
                    "particles/000/particles.128.mrcs"
                ],
            }
        ],
        "runtime_provenance": {
            "repo_root": str(Path.cwd().resolve()),
            "pixi_env": str((Path.cwd() / ".pixi/envs/default").resolve()),
            "python_executable": str((Path.cwd() / ".pixi/envs/default/bin/python").resolve()),
            "python_prefix": str((Path.cwd() / ".pixi/envs/default").resolve()),
            "jax_path": str((Path.cwd() / ".pixi/envs/default/lib/python/site-packages/jax/__init__.py").resolve()),
            "recovar_path": str((Path.cwd() / "recovar/__init__.py").resolve()),
            "parity_ancestors_verified": True,
            "required_parity_ancestors": ["ancestor"],
            "installed_runtime_content_hash_complete": False,
            "reproducible_runtime_speed_claim_allowed": False,
            "runtime_claim_scope": analyzer.RUNTIME_CLAIM_SCOPE,
        },
        "speed_claim_allowed": False,
        "same_job_preliminary_speed_signal_allowed": False,
        "runtime_claim_scope": analyzer.RUNTIME_CLAIM_SCOPE,
        "memory_claim_allowed": False,
        "default_promotion_allowed": False,
        "sealed_environment": _sealed_environment(run_dir, gpu_uuid),
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
    root.mkdir(parents=True)
    git_head = "a" * 40
    gpu_uuid = "GPU-test"
    sealed_inputs = _sealed_analysis_inputs(root, git_head=git_head, gpu_uuid=gpu_uuid)
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
                        "launch_manifest_sha256": LAUNCH_MANIFEST_SHA256,
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
                sealed_inputs=sealed_inputs,
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


def test_compiled_executable_key_preserves_bound_process_function_identity():
    from recovar.core.configs import ForwardModelConfig
    from recovar.core.ctf import CTFEvaluator

    class Dataset:
        def process(self, value):
            return value

    def config(dataset):
        return ForwardModelConfig(
            image_shape=(8, 8),
            volume_shape=(8, 8, 8),
            grid_size=8,
            voxel_size=1.0,
            padding=0,
            disc_type="linear_interp",
            ctf=CTFEvaluator(),
            process_fn=dataset.process,
        )

    first = runner._executable_variant_key("stable-program", (config(Dataset()),))
    second = runner._executable_variant_key("stable-program", (config(Dataset()),))

    assert first[0] == second[0]
    assert first != second


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
        expected_launch_manifest_sha256=LAUNCH_MANIFEST_SHA256,
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
    assert payload["donated_over_control_ratio_scope"] == "unpaired_arm_medians_diagnostic_only"
    assert payload["paired_e2e_ratios"] == pytest.approx([0.9, 0.9, 0.9])
    assert payload["paired_e2e_median_ratio"] == pytest.approx(0.9)
    assert payload["paired_e2e_max_ratio"] == pytest.approx(0.9)
    assert payload["paired_e2e_spread_factor"] == pytest.approx(1.0)
    assert payload["paired_runtime_consistency_gate_passed"] is True
    assert payload["material_runtime_win"] is True
    assert payload["speed_claim_allowed"] is False
    assert payload["same_job_preliminary_speed_signal_allowed"] is True
    assert payload["reproducible_runtime_speed_claim_allowed"] is False
    assert payload["runtime_claim_scope"] == analyzer.RUNTIME_CLAIM_SCOPE
    assert {run["sampled_memory"]["sample_count"] for run in payload["runs"]} == {1}
    assert payload["sampled_memory_diagnostic_below_0_95"] is True
    assert payload["sampled_memory_acceptance_use"] == "diagnostic_only_not_used_for_acceptance"
    assert payload["material_memory_win"] is False
    assert payload["memory_claim_allowed"] is False
    policy = payload["broader_optimized_arithmetic_policy"]
    assert policy["stable_repeat_bounded_noise_allowed"] is True
    assert policy["directional_bias_allowed"] is False
    assert policy["iteration_amplified_drift_allowed"] is False
    assert policy["material_end_to_end_runtime_gain_required"] is True
    assert payload["default_promotion_allowed"] is False


def test_paired_runtime_gate_rejects_hidden_single_repeat_regression(tmp_path):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path)
    report_path = root / "runs/repeat-03/donated/result/donation_arm_summary.json"
    report = json.loads(report_path.read_text())
    report["warm"]["wall_s"] = 12.0
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    payload = analyzer.analyze(
        root,
        expected_repo_head=git_head,
        expected_gpu_uuid=gpu_uuid,
        expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
        expected_launch_manifest_sha256=LAUNCH_MANIFEST_SHA256,
    )

    # The unpaired arm medians still show an apparent 10% win, but repeat 03
    # regresses by 20% and makes the paired signal too dispersed to qualify.
    assert payload["donated_over_control_ratios"]["e2e_wall_s"] == pytest.approx(0.9)
    assert payload["paired_e2e_ratios"] == pytest.approx([0.9, 0.9, 1.2])
    assert payload["paired_e2e_median_ratio"] == pytest.approx(0.9)
    assert payload["paired_median_material_runtime_win"] is True
    assert payload["paired_e2e_no_material_regression"] is False
    assert payload["paired_e2e_spread_factor"] == pytest.approx(4.0 / 3.0)
    assert payload["paired_e2e_spread_guard_passed"] is False
    assert payload["paired_runtime_consistency_gate_passed"] is False
    assert payload["material_runtime_win"] is False
    assert payload["passed"] is False
    assert payload["speed_claim_allowed"] is False
    assert payload["same_job_preliminary_speed_signal_allowed"] is False


@pytest.mark.parametrize("bad_runtime", [0.0, float("nan")], ids=("zero", "nan"))
def test_paired_runtime_gate_rejects_nonpositive_or_nonfinite_values(tmp_path, bad_runtime):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path)
    report_path = root / "runs/repeat-02/donated/result/donation_arm_summary.json"
    report = json.loads(report_path.read_text())
    report["warm"]["wall_s"] = bad_runtime
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    with pytest.raises(RuntimeError, match="runtime ratio operands must be finite and positive"):
        analyzer.analyze(
            root,
            expected_repo_head=git_head,
            expected_gpu_uuid=gpu_uuid,
            expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
            expected_launch_manifest_sha256=LAUNCH_MANIFEST_SHA256,
        )


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
        expected_launch_manifest_sha256=LAUNCH_MANIFEST_SHA256,
    )

    assert payload["passed"] is True
    assert payload["phase_numeric_gates"]["cold"]["status"] == "exact_pass"
    assert payload["phase_numeric_gates"]["warm"]["status"] == "inconclusive_3x3"
    assert payload["numeric_qualification_status"] == "inconclusive_3x3"
    assert payload["numeric_qualification_allowed"] is False
    assert payload["speed_claim_allowed"] is False
    assert payload["same_job_preliminary_speed_signal_allowed"] is False


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
        expected_launch_manifest_sha256=LAUNCH_MANIFEST_SHA256,
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
        expected_launch_manifest_sha256=LAUNCH_MANIFEST_SHA256,
    )

    warm = payload["phase_numeric_gates"]["warm"]
    assert warm["status"] == "inconclusive_3x3"
    assert warm["exact_continuous_science"] is False
    assert "meta/class_direction_posterior_sums" in warm["continuous_distance"]["field_names"]
    assert payload["speed_claim_allowed"] is False
    assert payload["same_job_preliminary_speed_signal_allowed"] is False


def test_cold_and_warm_may_differ_when_each_phase_matched_cohort_is_exact(tmp_path):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path, cold_delta=2.0)
    payload = analyzer.analyze(
        root,
        expected_repo_head=git_head,
        expected_gpu_uuid=gpu_uuid,
        expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
        expected_launch_manifest_sha256=LAUNCH_MANIFEST_SHA256,
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


@pytest.mark.parametrize("delta", [1e-200, np.nextafter(0.0, 1.0)])
def test_exactness_does_not_use_underflowed_l2_distance(delta: float):
    gate = _numeric_gate(
        control=[0.0, 0.0, 0.0],
        donated=[delta, delta, delta],
    )

    assert gate["continuous_distance"]["matrix"] == [[0.0] * 6 for _ in range(6)]
    assert gate["exact_continuous_science"] is False
    assert gate["exact_parsed_science"] is False
    assert gate["status"] == "inconclusive_3x3"


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


def test_common_iref_replay_path_across_every_arm_is_forbidden(tmp_path):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path)
    for report_path in root.glob("runs/repeat-*/**/donation_arm_summary.json"):
        report = json.loads(report_path.read_text())
        for phase in ("cold", "warm"):
            meta_path = Path(report[phase]["science_outputs"]["recovar_meta"]["path"])
            meta = json.loads(meta_path.read_text())
            meta["diagnostic_iref_replay_paths"] = ["/path/to/unpinned_replay.mrc"]
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    with pytest.raises(RuntimeError, match="IREF replay is forbidden"):
        analyzer.analyze(
            root,
            expected_repo_head=git_head,
            expected_gpu_uuid=gpu_uuid,
            expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
            expected_launch_manifest_sha256=LAUNCH_MANIFEST_SHA256,
        )


def test_runner_rejects_inherited_iref_replay_environment(tmp_path):
    repo_root = tmp_path / "repo"
    cache_dir = tmp_path / "run/jax_cache"
    cuda_lib = tmp_path / "runtime/libcuda_backproject.so"
    relion_bind_dir = tmp_path / "runtime/relion_bind"
    cusparse = tmp_path / "runtime/libcusparse.so.12"
    for directory in (
        repo_root,
        cache_dir,
        relion_bind_dir,
        tmp_path / "home",
        tmp_path / "pixi_home",
        tmp_path / "rattler_cache",
        tmp_path / "tmp",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    cuda_lib.parent.mkdir(parents=True, exist_ok=True)
    cuda_lib.write_bytes(b"cuda")
    cusparse.write_bytes(b"cusparse")
    environment = {name: "sealed-test-value" for name in runner.SEALED_ARM_ENVIRONMENT_NAMES}
    environment.update(runner._SEALED_ARM_STATIC_ENVIRONMENT)
    environment.update(
        {
            "HOME": str(tmp_path / "home"),
            "PIXI_HOME": str(tmp_path / "pixi_home"),
            "RATTLER_CACHE_DIR": str(tmp_path / "rattler_cache"),
            "TMPDIR": str(tmp_path / "tmp"),
            "PATH": "/usr/bin:/bin",
            "LD_LIBRARY_PATH": "/sealed/lib",
            "LD_PRELOAD": str(cusparse.resolve()),
            "CUDA_VISIBLE_DEVICES": "GPU-test",
            "JAX_COMPILATION_CACHE_DIR": str(cache_dir.resolve()),
            "RECOVAR_CUDA_LIB": str(cuda_lib.resolve()),
            "RECOVAR_EXPECTED_REPO_ROOT": str(repo_root.resolve()),
            "RECOVAR_RELION_BIND_BUILD_DIR": str(relion_bind_dir.resolve()),
        }
    )
    contract = runner._assert_sealed_arm_environment(
        repo_root=repo_root,
        cache_dir=cache_dir,
        cuda_lib=cuda_lib,
        relion_bind_dir=relion_bind_dir,
        cusparse_library=cusparse,
        gpu_uuid="GPU-test",
        environment=environment,
    )
    assert contract["exact_allowlist"] is True

    environment["RECOVAR_INITIALMODEL_IREF_REPLAY_TEMPLATE"] = "/path/to/unpinned_replay.mrc"
    with pytest.raises(RuntimeError, match="unexpected=.*RECOVAR_INITIALMODEL_IREF_REPLAY_TEMPLATE"):
        runner._assert_sealed_arm_environment(
            repo_root=repo_root,
            cache_dir=cache_dir,
            cuda_lib=cuda_lib,
            relion_bind_dir=relion_bind_dir,
            cusparse_library=cusparse,
            gpu_uuid="GPU-test",
            environment=environment,
        )


def test_aggregate_cannot_qualify_a_reported_replay_environment(tmp_path):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path)
    report_path = root / "runs/repeat-01/donated/result/donation_arm_summary.json"
    report = json.loads(report_path.read_text())
    replay_name = "RECOVAR_INITIALMODEL_IREF_REPLAY_TEMPLATE"
    report["sealed_environment"]["observed"][replay_name] = "/path/to/unpinned_replay.mrc"
    report["sealed_environment"]["unexpected_names"] = [replay_name]
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    with pytest.raises(RuntimeError, match="unexpected variables"):
        analyzer.analyze(
            root,
            expected_repo_head=git_head,
            expected_gpu_uuid=gpu_uuid,
            expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
            expected_launch_manifest_sha256=LAUNCH_MANIFEST_SHA256,
        )


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
            expected_launch_manifest_sha256=LAUNCH_MANIFEST_SHA256,
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
            expected_launch_manifest_sha256=LAUNCH_MANIFEST_SHA256,
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
            expected_launch_manifest_sha256=LAUNCH_MANIFEST_SHA256,
        )


def test_aggregate_rejects_incomplete_transitive_particle_topology(tmp_path):
    root, git_head, gpu_uuid = _analysis_tree(tmp_path)
    report_path = root / "runs/repeat-01/control/result/donation_arm_summary.json"
    report = json.loads(report_path.read_text())
    report["input_manifest"]["entries"].pop()
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    with pytest.raises(RuntimeError, match="transitive input topology drifted"):
        analyzer.analyze(
            root,
            expected_repo_head=git_head,
            expected_gpu_uuid=gpu_uuid,
            expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
            expected_launch_manifest_sha256=LAUNCH_MANIFEST_SHA256,
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
            "--expected-particle-stack",
            "data/particles.128.mrcs",
            "--output-root",
            "out",
            "--input-manifest",
            "manifest.json",
            "--expected-input-manifest-sha256",
            INPUT_MANIFEST_SHA256,
            "--launch-manifest",
            "launch.json",
            "--expected-launch-manifest-sha256",
            LAUNCH_MANIFEST_SHA256,
            "--expected-jax-cache-dir",
            "cache",
            "--expected-cuda-lib",
            "libcuda_backproject.so",
            "--expected-relion-bind-dir",
            "relion_bind",
            "--expected-cusparse-library",
            "libcusparse.so.12",
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
            expected_launch_manifest_sha256=LAUNCH_MANIFEST_SHA256,
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
            expected_launch_manifest_sha256=LAUNCH_MANIFEST_SHA256,
        )


def test_slurm_runner_is_crossed_fresh_process_and_fail_closed():
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "scripts/run_local_mstep_donation_ab.sbatch").read_text()

    required = (
        "#!/bin/bash",
        "#SBATCH --export=NIL",
        "if (( $# != 2 ))",
        "readonly LAUNCH_MANIFEST_ARG=$1",
        "readonly EXPECTED_LAUNCH_MANIFEST_SHA256=$2",
        'test "$(/usr/bin/sha256sum "${LAUNCH_MANIFEST}"',
        'test -z "$(/usr/bin/git -C "${REPO_ROOT}" status --porcelain=v1)"',
        "run_arm 1 01 donated",
        "run_arm 2 01 control",
        "run_arm 3 02 control",
        "run_arm 4 02 donated",
        "run_arm 5 03 donated",
        "run_arm 6 03 control",
        "local cache_dir=${run_dir}/jax_cache",
        'test ! -e "${cache_dir}"',
        'test -z "$(/usr/bin/find "${cache_dir}" -mindepth 1 -print -quit)"',
        '/usr/bin/env -i "${SEALED_COMMON_ENV[@]}"',
        "/usr/bin/env -i HOME=\"${SEALED_BUILD_HOME}\" PATH=/usr/bin:/bin",
        "/usr/bin/make -rR -B",
        "RECOVAR_INITIAL_MODEL_PROFILE=1",
        '"${particle_args[@]}"',
        '--expected-particle-stack "${particle}"',
        '--launch-manifest "${PROVENANCE}/launch_manifest.json"',
        '--expected-launch-manifest-sha256 "${EXPECTED_LAUNCH_MANIFEST_SHA256}"',
        '--expected-jax-cache-dir "${cache_dir}"',
        '--expected-cuda-lib "${CUDA_BINARY}"',
        '--expected-relion-bind-dir "${RELION_BIND_DIR}"',
        '--expected-cusparse-library "${CUSPARSE_LIBRARY}"',
        '--expected-input-manifest-sha256 "${input_manifest_sha256}"',
        "--query-gpu=memory.used",
        'assert payload["cold_and_warm_are_separate_phase_matched_cohorts"] is True',
        'assert payload["numeric_qualification_status"] in {"exact_pass", "inconclusive_3x3"}',
        'assert payload["numeric_qualification_allowed"] is False',
        'assert payload["speed_claim_allowed"] is False',
        'assert payload["same_job_preliminary_speed_signal_allowed"] is False',
        "assert_parity_ancestors()",
        "pathlib.Path(jax.__file__).resolve().is_relative_to(pixi_env)",
        'assert payload["production_xhalf_adjoint_program_observed"] is True',
        'assert payload["alias_contract_passed"] is True',
        'assert payload["memory_claim_allowed"] is False',
        '/usr/bin/cmp "${PROVENANCE}/source_manifest.sha256"',
        '/usr/bin/cmp "${PROVENANCE}/input_manifest.json"',
        '/usr/bin/cmp "${PROVENANCE}/resolved_inputs.json"',
        '/usr/bin/touch "${ROOT}/COMPLETED"',
    )
    for fragment in required:
        assert fragment in source
    harness = (repo_root / "scripts/run_local_mstep_donation_ab.py").read_text()
    assert "clear_memory_stats" not in harness
    assert "whole_process_cumulative_not_warm_isolated" in harness
    assert source.rindex('/usr/bin/touch "${ROOT}/COMPLETED"') > source.rindex(
        '/usr/bin/cmp "${PROVENANCE}/source_manifest.sha256"'
    )


def test_slurm_runner_rejects_pre_arm_loader_and_toolchain_overrides():
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "scripts/run_local_mstep_donation_ab.sbatch").read_text()
    assert source.startswith("#!/bin/bash\n")
    assert "#SBATCH --export=NIL" in source
    assert '${LOCAL_MSTEP_DONATION_ROOT' not in source
    assert '${VDAM_LATE_PROFILE_' not in source
    assert '${EXPECTED_REPO_HEAD:-' not in source
    assert ': "${EXPECTED_REPO_HEAD' not in source
    assert '${CUDA_BUILD_TOOLKIT:-' not in source
    assert '${RELION_RUNTIME:-' not in source
    assert '${MPI_ROOT:-' not in source
    assert '${CUSPARSE_LIBRARY:-' not in source
    assert 'readonly CUSPARSE_LIBRARY=$(/usr/bin/readlink -f --' in source
    assert "MAKEFILES=" not in source
    assert "NVCC_CCBIN=" not in source
    assert "/usr/bin/env -i HOME=\"${SEALED_BUILD_HOME}\" PATH=/usr/bin:/bin" in source
    assert 'NVCC="${CUDA_BUILD_TOOLKIT}/bin/nvcc"' in source
    assert '/usr/bin/git -C "${REPO_ROOT}"' in source
    assert '/usr/bin/sha256sum "${LAUNCH_MANIFEST}"' in source
    assert '/usr/bin/cp --reflink=auto "${LAUNCH_MANIFEST}"' in source
    assert '"schema": "recovar.local_mstep_donation_ab_slurm.v4"' in source
    assert '"speed_claim_allowed": False' in source
