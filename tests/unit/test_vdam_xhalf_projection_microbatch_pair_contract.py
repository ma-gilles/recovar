import json
from pathlib import Path

import pytest

from scripts.summarize_vdam_xhalf_projection_microbatch_pair import (
    _benchmark_topology_evidence,
    _causal_speedup_magnitude_valid,
    _performance_rung_evidence_pass,
    _profile_absence,
    _profile_metadata_contract,
    _target_schedules_match,
    derive_it80_topology,
)

ROOT = Path(__file__).resolve().parents[2]


def _coarse_selector_audit() -> dict:
    return {
        "score_mode": "gaussian",
        "translation_count": 21,
        "requested_fused": True,
        "effective_fused": True,
        "requested_workers": 0,
        "effective_workers": 0,
        "requested_atomic": False,
        "effective_atomic": False,
        "requested_prehalf": False,
        "effective_prehalf": False,
        "wrapper": "relion_coarse_diff2_projector_f32",
        "target": "cuda_relion_coarse_diff2_projector_f32",
        "counts": {
            "fused_calls": 2,
            "actual_rows": 360,
            "multistream_calls": 0,
            "native_atomic_selected_calls": 0,
            "prehalf_selected_calls": 0,
        },
    }


def _route_only_metadata() -> dict:
    return {
        "current_size": 128,
        "halfset_0_profile_summary": {
            "coarse_selector_audit": _coarse_selector_audit(),
        },
    }


def test_profile_unset_contract_accepts_only_always_emitted_route_audit():
    row = _profile_metadata_contract(_route_only_metadata(), 80)

    assert row["profile_keys"] == ["halfset_0_profile_summary"]
    assert row["route_only_profile_container_present"] is True
    assert row["profile_contract_violations"] == []
    assert row["profile_unset_contract_pass"] is True


def test_profile_unset_contract_rejects_missing_scored_route_audit():
    row = _profile_metadata_contract({"current_size": 128}, 80)

    assert row["profile_unset_contract_pass"] is False
    assert row["profile_contract_violations"] == [
        "missing required route-only profile container: halfset_0_profile_summary"
    ]


def test_profile_unset_contract_requires_route_audit_for_warmups(tmp_path):
    labels = (
        "warmup-control",
        "warmup-candidate",
        "scored-control-1",
        "scored-control-2",
        "scored-candidate-1",
        "scored-candidate-2",
    )
    for label in labels:
        prefix = tmp_path / label / "recovar_prefix"
        prefix.mkdir(parents=True)
        (prefix / "run_it001_recovar_meta.json").write_text(
            json.dumps(_route_only_metadata())
        )
    (tmp_path / "warmup-control" / "recovar_prefix" / "run_it001_recovar_meta.json").write_text(
        json.dumps({"current_size": 38})
    )

    report = _profile_absence(tmp_path, 1)

    assert report["warmup-control"]["profile_unset_contract_pass"] is False
    assert report["warmup-candidate"]["profile_unset_contract_pass"] is True


@pytest.mark.parametrize(
    ("mutate", "expected_violation"),
    [
        (
            lambda meta: meta.update(
                {"vdam_iteration_profile_summary": {"total_time_s": 1.0}}
            ),
            "unexpected profile metadata key: vdam_iteration_profile_summary",
        ),
        (
            lambda meta: meta["halfset_0_profile_summary"].update(
                {"local_total_time_s": 1.0}
            ),
            "halfset_0_profile_summary keys differ from the route-only contract",
        ),
        (
            lambda meta: meta["halfset_0_profile_summary"][
                "coarse_selector_audit"
            ].update({"stage_time_s": 1.0}),
            "coarse_selector_audit keys differ from the route-only contract",
        ),
        (
            lambda meta: meta["halfset_0_profile_summary"][
                "coarse_selector_audit"
            ].pop("requested_prehalf"),
            "coarse_selector_audit keys differ from the route-only contract",
        ),
        (
            lambda meta: meta["halfset_0_profile_summary"][
                "coarse_selector_audit"
            ]["counts"].update({"timed_calls": 1}),
            "coarse_selector_audit counts keys differ from the route-only contract",
        ),
    ],
)
def test_profile_unset_contract_rejects_stage_timing_or_extra_nested_keys(
    mutate,
    expected_violation,
):
    meta = _route_only_metadata()
    mutate(meta)

    row = _profile_metadata_contract(meta, 80)

    assert row["profile_unset_contract_pass"] is False
    assert any(
        violation.startswith(expected_violation)
        for violation in row["profile_contract_violations"]
    )


@pytest.mark.parametrize(
    ("mutate", "expected_violation"),
    [
        (
            lambda meta: meta.update({"halfset_0_profile_summary": []}),
            "halfset_0_profile_summary must be a JSON object",
        ),
        (
            lambda meta: meta["halfset_0_profile_summary"].update(
                {"coarse_selector_audit": []}
            ),
            "coarse_selector_audit must be a JSON object",
        ),
        (
            lambda meta: meta["halfset_0_profile_summary"][
                "coarse_selector_audit"
            ].update({"counts": []}),
            "invalid coarse_selector_audit: coarse selector audit counts must be a dict",
        ),
        (
            lambda meta: meta.update(
                {"halfset_1_profile_summary": meta.pop("halfset_0_profile_summary")}
            ),
            "unexpected profile metadata key: halfset_1_profile_summary",
        ),
        (
            lambda meta: meta.update(
                {"Halfset_0_Profile_Summary": meta.pop("halfset_0_profile_summary")}
            ),
            "unexpected profile metadata key: Halfset_0_Profile_Summary",
        ),
    ],
)
def test_profile_unset_contract_rejects_malformed_or_alternate_route_containers(
    mutate,
    expected_violation,
):
    meta = _route_only_metadata()
    mutate(meta)

    row = _profile_metadata_contract(meta, 80)

    assert row["profile_unset_contract_pass"] is False
    assert expected_violation in row["profile_contract_violations"]


def test_profile_unset_contract_rejects_semantically_invalid_route_audit():
    meta = _route_only_metadata()
    meta["halfset_0_profile_summary"]["coarse_selector_audit"][
        "effective_fused"
    ] = False

    row = _profile_metadata_contract(meta, 80)

    assert row["profile_unset_contract_pass"] is False
    assert any(
        violation.startswith("invalid coarse_selector_audit")
        for violation in row["profile_contract_violations"]
    )


def test_performance_rung_evidence_remains_distinct_from_science_promotion():
    assert _performance_rung_evidence_pass(
        benchmark_topology_valid=True,
        profile_unset=True,
        repeatable_gain=True,
    )
    assert not _performance_rung_evidence_pass(
        benchmark_topology_valid=True,
        profile_unset=False,
        repeatable_gain=True,
    )


def _topology_evidence_inputs():
    arms = {"control": 80_000_000, "candidate": 160_000_000}
    execution = [
        {
            "label": "scored-control-1",
            "arm": "control",
            "row_pixels": 80_000_000,
            "provenance_row_pixels": 80_000_000,
        },
        {
            "label": "scored-candidate-1",
            "arm": "candidate",
            "row_pixels": 160_000_000,
            "provenance_row_pixels": 160_000_000,
        },
    ]
    topology = {
        "control": {"predicted_big_jit_bucket_count": 3},
        "candidate": {"predicted_big_jit_bucket_count": 2},
    }
    return arms, execution, topology


def test_benchmark_topology_requires_declared_observed_and_derived_evidence():
    arms, execution, topology = _topology_evidence_inputs()

    evidence = _benchmark_topology_evidence(arms, execution, topology)

    assert evidence == {
        "declared_caps_positive": True,
        "candidate_cap_gt_control": True,
        "execution_caps_match_declaration": True,
        "per_run_provenance_caps_match_declaration": True,
        "candidate_bucket_count_lt_control": True,
        "pass": True,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda arms, execution, topology: arms.update(control=0),
        lambda arms, execution, topology: arms.update(candidate=40_000_000),
        lambda arms, execution, topology: execution[0].update(row_pixels=40_000_000),
        lambda arms, execution, topology: execution[1].update(
            provenance_row_pixels=80_000_000
        ),
        lambda arms, execution, topology: topology["candidate"].update(
            predicted_big_jit_bucket_count=3
        ),
    ],
)
def test_benchmark_topology_fails_closed_when_any_evidence_differs(mutate):
    arms, execution, topology = _topology_evidence_inputs()
    mutate(arms, execution, topology)

    evidence = _benchmark_topology_evidence(arms, execution, topology)

    assert evidence["pass"] is False


def test_target_schedule_match_exposes_translation_workload_mismatch():
    schedule = {
        "current_size": 128,
        "healpix_order": 3,
        "n_rotations": 294_912,
        "n_translations": 84,
        "subset_size": 360,
        "oversampling": 1,
        "max_significants": 100,
        "effective_image_batch_size": 500,
        "pass2_engine": "local",
    }
    per_run = {
        "scored-control-1": dict(schedule),
        "scored-candidate-1": dict(schedule),
    }
    assert _target_schedules_match(per_run)

    per_run["scored-control-1"]["n_translations"] = 196
    assert not _target_schedules_match(per_run)


def test_causal_speedup_requires_full_schedule_and_cross_arm_state_equality():
    assert _causal_speedup_magnitude_valid(
        performance_rung_evidence_pass=True,
        all_iteration_schedules_matched=True,
        cross_arm_state_discrete_equal=True,
    )
    for missing_proof in (
        "performance_rung_evidence_pass",
        "all_iteration_schedules_matched",
        "cross_arm_state_discrete_equal",
    ):
        evidence = {
            "performance_rung_evidence_pass": True,
            "all_iteration_schedules_matched": True,
            "cross_arm_state_discrete_equal": True,
        }
        evidence[missing_proof] = False
        assert not _causal_speedup_magnitude_valid(**evidence)


def test_vdam_xhalf_projection_pair_is_same_gpu_warm_balanced_and_pinned():
    runner = (
        ROOT / "scripts/run_vdam_xhalf_projection_microbatch_pair.sbatch"
    ).read_text()

    required = [
        'ACTUAL_REPO_HEAD=$(git -C "${REPO_ROOT}" rev-parse HEAD)',
        ': "${RECOVAR_CUDA_LIB_OVERRIDE:?',
        ': "${LAYOUT_EVIDENCE_META:?',
        "#SBATCH --exclusive",
        'vdam_select_target_gpu "${TARGET_GPU_UUID}" 0',
        "exclusive_gpu_local_index=",
        "mapfile -t exclusive_gpu_rows",
        "--query-gpu=index,uuid",
        "export CUDA_VISIBLE_DEVICES=${exclusive_gpu_local_index}",
        "selected_gpu_local_index.txt",
        "selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}",
        "cache_control=${OUTPUT_ROOT}_jax_cache_control",
        "cache_candidate=${OUTPUT_ROOT}_jax_cache_candidate",
        "warmup-control warmup-candidate",
        "scored-control-1 scored-candidate-1 scored-candidate-2 scored-control-2",
        'VDAM_JAX_COMPILATION_CACHE_DIR="${cache_dir}"',
        "export RECOVAR_EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS=${row_pixels}",
        "unset RECOVAR_INITIAL_MODEL_PROFILE",
        'vdam_verify_selected_gpu "${selected_gpu_uuid}"',
        "CAPTURE_NATIVE_REPLAY=0",
        'start_ns=$(date +%s%N)',
        'end_ns=$(date +%s%N)',
        "gpu_memory.tsv",
        '--query-gpu=memory.used,memory.total',
        "summarize_vdam_xhalf_projection_microbatch_pair",
    ]
    for text in required:
        assert text in runner

    assert "RECOVAR_CUDA_LIB_OVERRIDE) ;;" in runner
    assert "RECOVAR_*) unset" in runner
    for variable in (
        "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE",
        "RECOVAR_K1_COARSE_FUSED_PROJECTOR",
        "RECOVAR_K1_COARSE_GAUSSIAN_SKIP_PADDED_IMAGES",
        "RECOVAR_K1_COARSE_GAUSSIAN_FFI",
        "RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF",
        "RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS",
        "RECOVAR_K1_RELION_F32_COARSE_SUPPORT",
        "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION",
    ):
        assert f"unset {variable}" in runner


def test_gf46_it80_layout_derives_2x_3x_and_5x_bucket_counts(monkeypatch):
    monkeypatch.delenv("RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS", raising=False)
    monkeypatch.delenv("RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB", raising=False)
    evidence = {
        "n_translations": 84,
        "halfset_0_profile_summary": {
            "chunk_sizes": [75, 75, 75, 75, 60],
            "chunk_padded_rotations": [4800, 4800, 4800, 4800, 3840],
            "n_projection_windowed": 8320,
        },
    }

    expected = {
        40_000_000: [75, 75, 75, 75, 60],
        80_000_000: [150, 150, 60],
        120_000_000: [225, 135],
        200_000_000: [266, 94],
    }
    for row_pixels, chunk_sizes in expected.items():
        topology = derive_it80_topology(evidence, row_pixels)
        assert topology["predicted_chunk_sizes"] == chunk_sizes
        assert topology["predicted_big_jit_bucket_count"] == len(chunk_sizes)
        assert sum(topology["predicted_chunk_padded_rotations"]) == 23_040
