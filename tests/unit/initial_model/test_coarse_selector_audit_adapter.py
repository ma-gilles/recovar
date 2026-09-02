"""InitialModel propagation contracts for the shared coarse-selector audit."""

from __future__ import annotations

from inspect import getsource
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

from recovar.em.initial_model import dense_adapter

pytestmark = pytest.mark.unit


class _ProfileResult(NamedTuple):
    profile_summary: dict | None


class _CoarseResult(NamedTuple):
    profile_summary: dict | None
    significant_counts: np.ndarray | None
    pose_assignments: np.ndarray


def _active_audit(*, workers: int, atomic: bool) -> dict:
    multistream = workers == 8
    return {
        "score_mode": "gaussian",
        "translation_count": 29,
        "requested_fused": True,
        "effective_fused": True,
        "requested_workers": workers,
        "effective_workers": workers,
        "requested_atomic": atomic,
        "effective_atomic": atomic,
        "wrapper": (
            "relion_coarse_diff2_projector_multistream_f32" if multistream else "relion_coarse_diff2_projector_f32"
        ),
        "target": (
            "cuda_relion_coarse_diff2_projector_multistream_f32"
            if multistream
            else "cuda_relion_coarse_diff2_projector_f32"
        ),
        "counts": {
            "fused_calls": 3,
            "actual_rows": 17,
            "multistream_calls": 3 if multistream else 0,
            "native_atomic_selected_calls": 3 if atomic else 0,
        },
    }


@pytest.mark.parametrize("workers", [0, 8], ids=["serial", "multistream"])
@pytest.mark.parametrize("atomic", [False, True], ids=["canonical", "atomic"])
def test_sparse_adapter_propagates_each_selector_topology(workers, atomic):
    audit = _active_audit(workers=workers, atomic=atomic)
    result = _ProfileResult(profile_summary={"pass2_time_s": 1.25})

    validated = dense_adapter._coarse_selector_audit_from_full_stats({"coarse_selector_audit": audit})
    sealed = dense_adapter._with_coarse_selector_audit(result, validated)
    meta = dense_adapter._estep_meta({0: SimpleNamespace(profile_summary=sealed.profile_summary)})

    assert sealed.profile_summary["pass2_time_s"] == 1.25
    assert sealed.profile_summary["coarse_selector_audit"] == audit
    assert meta["halfset_0_profile_summary"]["coarse_selector_audit"] == audit


@pytest.mark.parametrize(
    ("workers", "atomic", "corrupt"),
    [
        (0, False, lambda audit: audit.update(wrapper="wrong")),
        (8, False, lambda audit: audit["counts"].update(multistream_calls=0)),
        (0, True, lambda audit: audit["counts"].update(native_atomic_selected_calls=0)),
        (8, True, lambda audit: audit.update(effective_atomic=False)),
    ],
    ids=[
        "serial-wrong-wrapper",
        "multistream-missing-calls",
        "serial-atomic-missing-calls",
        "multistream-atomic-mode-mismatch",
    ],
)
def test_sparse_adapter_fails_closed_on_invalid_selector_topology(
    workers,
    atomic,
    corrupt,
):
    audit = _active_audit(workers=workers, atomic=atomic)
    corrupt(audit)

    with pytest.raises(RuntimeError, match="invalid coarse selector audit"):
        dense_adapter._coarse_selector_audit_from_full_stats({"coarse_selector_audit": audit})


def test_sparse_adapter_fails_closed_when_selector_audit_is_missing():
    with pytest.raises(RuntimeError, match="did not return.*execution audit"):
        dense_adapter._coarse_selector_audit_from_full_stats({})


def test_sparse_adapter_propagates_real_coarse_support_hybrid_and_counts():
    audit = _active_audit(workers=0, atomic=False)
    support = {
        "schema": "recovar.coarse_significance_support_audit.v2",
        "aggregate_support_sha256": "a" * 64,
    }
    hybrid = {
        "enabled": True,
        "published_score_source": "exact_relion_source16_or_full_rectangular",
    }
    result = _CoarseResult(
        profile_summary={"pass2_time_s": 1.25},
        significant_counts=None,
        pose_assignments=np.asarray([3, 7], dtype=np.int32),
    )

    sealed = dense_adapter._with_initial_model_coarse_diagnostics(
        result,
        full_stats={
            "significant_cutoff_counts": np.asarray([17, 23], dtype=np.int32),
            "coarse_significance_support_audit": support,
            "coarse_gaussian_gemm_hybrid": hybrid,
        },
        selector_audit=audit,
    )

    np.testing.assert_array_equal(sealed.significant_counts, [17, 23])
    assert sealed.profile_summary == {
        "pass2_time_s": 1.25,
        "coarse_selector_audit": audit,
        "coarse_significance_support_audit": support,
        "coarse_gaussian_gemm_hybrid": hybrid,
    }


def test_sparse_adapter_rejects_coarse_count_shape_drift():
    result = _CoarseResult(
        profile_summary=None,
        significant_counts=None,
        pose_assignments=np.asarray([3, 7], dtype=np.int32),
    )

    with pytest.raises(RuntimeError, match="significant counts.*pass-2 images"):
        dense_adapter._with_initial_model_coarse_diagnostics(
            result,
            full_stats={"significant_cutoff_counts": np.asarray([17], dtype=np.int32)},
            selector_audit=None,
        )


def test_sparse_adapter_extracts_before_pass2_and_seals_before_meta():
    source = getsource(dense_adapter._run_sparse_pass2_initial_model_estep)

    extraction = source.index("coarse_selector_audit = _coarse_selector_audit_from_full_stats(")
    pass2 = min(
        source.index("result = _run_sparse_k_class_adaptive_pass2("),
        source.index("result = run_local_k_class_em("),
    )
    sealing = source.index("result = _with_initial_model_coarse_diagnostics(")
    result_storage = source.index("halfset_results[int(halfset_idx)] = result")

    assert extraction < pass2
    assert sealing < result_storage
