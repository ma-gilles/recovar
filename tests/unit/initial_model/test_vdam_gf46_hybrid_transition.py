from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/analyze_vdam_gf46_hybrid_transition.py"
RUNNER = ROOT / "scripts/run_vdam_gf46_hybrid_transition.sbatch"
SPEC = importlib.util.spec_from_file_location("analyze_vdam_gf46_hybrid_transition", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _support_audit() -> dict:
    counts = [[3] * 1_000]
    row_hashes = [[hashlib.sha256(f"row-{index}".encode()).hexdigest() for index in range(1_000)]]
    counts_le = np.ascontiguousarray(np.asarray(counts, dtype="<i8"))
    return {
        "schema": MODULE.SUPPORT_SCHEMA,
        "classification": "diagnostic_only",
        "canonical_encoding": MODULE.EXPECTED_SUPPORT_ENCODING,
        "n_classes": 1,
        "n_images": 1_000,
        "samples_per_class": 36_864 * 29,
        "selected_count_sum": 3_000,
        "selected_count_min": 3,
        "selected_count_max": 3,
        "per_class_image_selected_counts": counts,
        "per_class_image_selected_counts_sha256": hashlib.sha256(
            counts_le.tobytes(order="C"),
        ).hexdigest(),
        "per_class_image_support_sha256": row_hashes,
        "aggregate_support_sha256": hashlib.sha256(b"aggregate").hexdigest(),
    }


def _change_support_row(audit: dict, index: int, count: int, tag: bytes) -> None:
    audit["per_class_image_selected_counts"][0][index] = count
    audit["per_class_image_support_sha256"][0][index] = hashlib.sha256(tag).hexdigest()
    counts = np.ascontiguousarray(
        np.asarray(audit["per_class_image_selected_counts"], dtype="<i8"),
    )
    audit["selected_count_sum"] = int(counts.sum())
    audit["selected_count_min"] = int(counts.min())
    audit["selected_count_max"] = int(counts.max())
    audit["per_class_image_selected_counts_sha256"] = hashlib.sha256(
        counts.tobytes(order="C"),
    ).hexdigest()
    audit["aggregate_support_sha256"] = hashlib.sha256(
        repr(
            (
                audit["per_class_image_selected_counts"],
                audit["per_class_image_support_sha256"],
            )
        ).encode(),
    ).hexdigest()


def _hybrid_stats() -> dict:
    return {
        "enabled": True,
        "default_enabled": False,
        "published_score_source": "exact_relion_source16_or_full_rectangular",
        "expanded_gemm_scores_published": False,
        "whole_batch_fail_closed_fallback": True,
        "batch_count": 6,
        "selected_rescore_batch_count": 6,
        "fallback_batch_count": 0,
        "selected_rescore_image_count": 1_000,
        "fallback_image_count": 0,
        "selected_source16_block_count": 4_000,
        "selected_exact_candidate_count": 1_856_000,
        "full_candidate_count_for_selected_images": 1_069_547_520,
        "selected_exact_candidate_fraction": 0.001735,
        "max_selected_blocks_per_image": 4,
        "selected_block_capacity": 64,
        "certificate_chunk_rows": 4_608,
        "certificate_chunk_count_per_batch": 8,
        "topology_full_to_compact_sha256": hashlib.sha256(b"topology").hexdigest(),
        "fallback_reasons": {},
    }


def _audit_metadata(*, hybrid: bool) -> dict:
    support = _support_audit()
    profile = {"coarse_significance_support_audit": support}
    if hybrid:
        profile["coarse_gaussian_gemm_hybrid"] = _hybrid_stats()
    return {
        "halfset_0_profile_summary": profile,
        "significant_counts": list(support["per_class_image_selected_counts"][0]),
    }


@pytest.mark.unit
def test_source_manifest_covers_real_transition_and_high_risk_dispatch() -> None:
    required = {
        "recovar/cuda/cuda_backproject.cu",
        "recovar/em/dense_single_volume/helpers/coarse_gemm_hybrid.py",
        "recovar/em/dense_single_volume/helpers/scoring.py",
        "recovar/em/dense_single_volume/helpers/significance.py",
        "recovar/em/dense_single_volume/k_class.py",
        "recovar/em/initial_model/dense_adapter.py",
        "scripts/analyze_vdam_gf46_hybrid_transition.py",
        "scripts/run_vdam_gf46_hybrid_transition.sbatch",
        "scripts/run_vdam_late_iteration_profile.py",
        "tests/unit/initial_model/test_vdam_gf46_hybrid_transition.py",
    }
    assert required.issubset(set(MODULE.SOURCE_FILES))
    manifest, entries = MODULE._source_manifest(ROOT)
    assert len(entries) == len(MODULE.SOURCE_FILES)
    assert hashlib.sha256(manifest).hexdigest() == hashlib.sha256(MODULE._source_manifest(ROOT)[0]).hexdigest()


@pytest.mark.unit
def test_runner_separates_support_audit_from_clean_abba_timing() -> None:
    source = RUNNER.read_text()
    for required in (
        "RUN_LABELS=(audit_direct audit_hybrid direct_1 hybrid_1 hybrid_2 direct_2)",
        "RUN_HYBRID=(0 1 0 1 1 0)",
        "RUN_AUDIT=(1 1 0 0 0 0)",
        "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT=1",
        "-u RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY=64",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB=4",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB=2",
        '"CUDA_ARCH=${CUDA_ARCH_FLAGS}"',
        "RelionCoarseDiff2RectangularF32",
        "RelionCoarseDiff2RotationBlocksF32",
        "EXPECTED_CUSPARSE_SHA256=58ffc54edb1d007f56a1718aaadcb30f45bbf662f43515920ea8ff094304bdbf",
        "mapfile -t initially_visible_gpu_uuids",
        '[[ "${gpu_name}" == *H100* ]]',
        "--image-batch-size 500",
        "--checkpoint-iteration 180",
        "JAX_COMPILATION_CACHE_DIR=",
    ):
        assert required in source
    assert "export JAX_ENABLE_X64" not in source
    assert "#SBATCH --nodelist" not in source
    assert "sbatch " not in source
    command_start = source.index("  command=(")
    diagnostic_unset = source.index(
        "-u RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR",
        command_start,
    )
    conditional_audit = source.index('"${audit_env[@]}"', command_start)
    assert diagnostic_unset < conditional_audit


@pytest.mark.unit
def test_support_gate_requires_hybrid_rows_from_direct_repeat_outcomes() -> None:
    arms = {
        "audit_direct": {
            "cold": {"metadata": _audit_metadata(hybrid=False)},
            "warm": {"metadata": _audit_metadata(hybrid=False)},
        },
        "audit_hybrid": {
            "cold": {"metadata": _audit_metadata(hybrid=True)},
            "warm": {"metadata": _audit_metadata(hybrid=True)},
        },
    }
    result = MODULE._support_gate(arms)
    assert result["pass"]
    assert result["all_direct_hybrid_cold_warm_support_exact"]
    assert result["hybrid_novel_support_row_count"] == 0

    changed = copy.deepcopy(arms)
    changed_audit = changed["audit_hybrid"]["warm"]["metadata"]["halfset_0_profile_summary"][
        "coarse_significance_support_audit"
    ]
    _change_support_row(changed_audit, 73, 4, b"changed")
    assert not MODULE._support_gate(changed)["pass"]


@pytest.mark.unit
def test_support_gate_accepts_direct_inclusive_tie_without_a_novel_hybrid_state() -> None:
    arms = {
        "audit_direct": {
            "cold": {"metadata": _audit_metadata(hybrid=False)},
            "warm": {"metadata": _audit_metadata(hybrid=False)},
        },
        "audit_hybrid": {
            "cold": {"metadata": _audit_metadata(hybrid=True)},
            "warm": {"metadata": _audit_metadata(hybrid=True)},
        },
    }
    direct_warm = arms["audit_direct"]["warm"]["metadata"]["halfset_0_profile_summary"][
        "coarse_significance_support_audit"
    ]
    _change_support_row(direct_warm, 73, 4, b"direct-inclusive-tie")

    result = MODULE._support_gate(arms)

    assert result["pass"]
    assert not result["all_direct_hybrid_cold_warm_support_exact"]
    assert result["direct_repeat_changed_rows"] == [73]
    assert result["direct_repeat_count_absolute_delta"] == 1
    assert result["persisted_cutoff_counts_exact"]
    assert result["hybrid_support_within_observed_direct_outcomes"]


@pytest.mark.unit
def test_support_audit_distinguishes_cutoff_rank_from_inclusive_ties() -> None:
    metadata = _audit_metadata(hybrid=False)
    metadata["significant_counts"][500] = 2
    support = MODULE._validated_support(metadata, "test")
    assert support["inclusive_tie_surplus_sum"] == 1
    assert support["inclusive_tie_surplus_row_count"] == 1
    assert support["inclusive_tie_surplus_max"] == 1

    metadata["significant_counts"][500] = 4
    with pytest.raises(MODULE.HybridTransitionSetupError, match="smaller"):
        MODULE._validated_support(metadata, "test")


@pytest.mark.unit
def test_hybrid_telemetry_is_exact_score_fail_closed_and_accounted() -> None:
    metadata = _audit_metadata(hybrid=True)
    stats = MODULE._validated_hybrid(metadata, "test")
    assert stats["selected_rescore_image_count"] == 1_000
    assert stats["expanded_gemm_scores_published"] is False

    bad = copy.deepcopy(metadata)
    bad_stats = bad["halfset_0_profile_summary"]["coarse_gaussian_gemm_hybrid"]
    bad_stats["fallback_batch_count"] = 1
    bad_stats["fallback_image_count"] = 500
    with pytest.raises(MODULE.HybridTransitionSetupError, match="batch accounting"):
        MODULE._validated_hybrid(bad, "test")


@pytest.mark.unit
def test_material_runtime_gate_is_ten_percent_and_bounded_scope() -> None:
    assert MODULE.MATERIAL_WALL_RATIO == 0.90
    assert MODULE.CONTROL_REPEAT_ENVELOPE_MULTIPLIER == 2.0
    source = SCRIPT.read_text()
    assert '"default_enablement_allowed": False' in source
    assert '"long_trajectory_no_growth_evaluated": False' in source
    assert "diagnostic_one_transition_only" in source


@pytest.mark.unit
def test_numeric_gate_pools_six_direct_executions_and_rejects_escape() -> None:
    base = np.linspace(1.0, 2.0, 32, dtype=np.float64)
    noise = np.linspace(0.25, 1.0, 32, dtype=np.float64) * 1.0e-6
    arms: dict[str, dict] = {}

    def add(label: str, cold_scale: float, warm_scale: float) -> None:
        arms[label] = {
            "cold": {"map": base + cold_scale * noise},
            "warm": {"map": base + warm_scale * noise},
        }

    add("audit_direct", 0.0, 1.0)
    add("direct_1", -1.0, 0.5)
    add("direct_2", -0.5, 0.25)
    add("audit_hybrid", 0.1, 0.9)
    add("hybrid_1", -0.8, 0.4)
    add("hybrid_2", -0.4, 0.2)

    result = MODULE._numeric_panel(arms, "map")
    assert result["pass"]
    assert result["direct_execution_count"] == 6
    assert result["direct_repeat_pair_count"] == 15
    assert result["hybrid_repeat_pair_count"] == 15
    assert result["crossed_pair_count"] == 36

    escaped = copy.deepcopy(arms)
    escaped["hybrid_2"]["warm"]["map"] = base + 10.0 * noise
    assert not MODULE._numeric_panel(escaped, "map")["pass"]
