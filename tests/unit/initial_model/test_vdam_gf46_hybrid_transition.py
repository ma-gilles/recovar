from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

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
        "per_class_image_selected_counts_sha256": hashlib.sha256(b"counts").hexdigest(),
        "per_class_image_support_sha256": row_hashes,
        "aggregate_support_sha256": hashlib.sha256(b"aggregate").hexdigest(),
    }


def _hybrid_stats() -> dict:
    return {
        "enabled": True,
        "default_enabled": False,
        "published_score_source": "exact_relion_source16_or_full_rectangular",
        "expanded_gemm_scores_published": False,
        "whole_batch_fail_closed_fallback": True,
        "batch_count": 2,
        "selected_rescore_batch_count": 2,
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
    assert hashlib.sha256(manifest).hexdigest() == hashlib.sha256(
        MODULE._source_manifest(ROOT)[0]
    ).hexdigest()


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
    ):
        assert required in source
    assert "export JAX_ENABLE_X64" not in source
    assert "#SBATCH --nodelist" not in source
    assert "sbatch " not in source


@pytest.mark.unit
def test_support_gate_requires_exact_localizable_direct_hybrid_identity() -> None:
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

    changed = copy.deepcopy(arms)
    changed_audit = changed["audit_hybrid"]["warm"]["metadata"][
        "halfset_0_profile_summary"
    ]["coarse_significance_support_audit"]
    changed_audit["per_class_image_support_sha256"][0][73] = hashlib.sha256(
        b"changed"
    ).hexdigest()
    assert not MODULE._support_gate(changed)["pass"]


@pytest.mark.unit
def test_support_audit_must_match_persisted_significant_counts() -> None:
    metadata = _audit_metadata(hybrid=False)
    MODULE._validated_support(metadata, "test")
    metadata["significant_counts"][500] = 4
    with pytest.raises(MODULE.HybridTransitionSetupError, match="significant_counts"):
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
    source = SCRIPT.read_text()
    assert '"default_enablement_allowed": False' in source
    assert '"long_trajectory_no_growth_evaluated": False' in source
    assert "diagnostic_one_transition_only" in source
