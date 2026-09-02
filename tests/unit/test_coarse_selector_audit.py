from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

pytest.importorskip("jax")

from recovar.em.dense_single_volume.helpers import significance
from recovar.em.dense_single_volume.k_class import (
    _coarse_selector_audit_from_full_stats,
    _with_coarse_selector_audit,
)
from recovar.em.initial_model.dense_adapter import _estep_meta


def _control_audit() -> dict:
    return {
        "score_mode": "gaussian",
        "translation_count": 29,
        "requested_fused": False,
        "effective_fused": False,
        "requested_workers": 0,
        "effective_workers": 0,
        "requested_atomic": False,
        "effective_atomic": False,
        "wrapper": None,
        "target": None,
        "counts": {
            "fused_calls": 0,
            "actual_rows": 0,
            "multistream_calls": 0,
            "native_atomic_selected_calls": 0,
        },
    }


def _active_audit(*, workers: int = 0, atomic: bool = False) -> dict:
    multistream = workers > 0
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
            "relion_coarse_diff2_projector_multistream_f32"
            if multistream
            else "relion_coarse_diff2_projector_f32"
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


def _prehalf_audit(*, workers: int = 8) -> dict:
    audit = _active_audit(workers=workers, atomic=True)
    audit.update(requested_prehalf=True, effective_prehalf=True)
    audit["counts"] = dict(audit["counts"], prehalf_selected_calls=3)
    return audit


def test_default_selector_control_remains_inactive(monkeypatch):
    monkeypatch.delenv("RECOVAR_K1_COARSE_FUSED_PROJECTOR", raising=False)
    monkeypatch.delenv("RECOVAR_K1_COARSE_MULTISTREAM_WORKERS", raising=False)
    monkeypatch.delenv(
        "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION",
        raising=False,
    )
    monkeypatch.delenv("RECOVAR_K1_COARSE_PREHALF_WEIGHT", raising=False)

    assert not significance._k1_coarse_fused_projector_enabled()
    assert significance._k1_coarse_multistream_worker_count() == 0
    assert not significance._k1_coarse_native_atomic_reduction_enabled()
    assert not significance._k1_coarse_prehalf_weight_enabled()
    assert significance._validate_coarse_selector_audit(_control_audit()) == (
        _control_audit()
    )


def test_active_multistream_native_atomic_audit_is_accepted():
    audit = _active_audit(workers=8, atomic=True)

    assert significance._validate_coarse_selector_audit(audit) == audit


def test_active_prehalf_audit_is_accepted_and_requires_atomic_execution():
    audit = _prehalf_audit()
    assert significance._validate_coarse_selector_audit(audit) == audit

    no_atomic = _prehalf_audit()
    no_atomic["effective_atomic"] = False
    no_atomic["counts"]["native_atomic_selected_calls"] = 0
    with pytest.raises(ValueError, match="prehalf weight requires native-atomic"):
        significance._validate_coarse_selector_audit(no_atomic)


def test_active_prehalf_audit_rejects_a_noop_call_count():
    audit = _prehalf_audit()
    audit["counts"]["prehalf_selected_calls"] = 0
    with pytest.raises(ValueError, match="prehalf call count"):
        significance._validate_coarse_selector_audit(audit)


@pytest.mark.parametrize("missing_field", ["requested_prehalf", "effective_prehalf"])
def test_prehalf_audit_requires_paired_selector_fields(missing_field):
    audit = _prehalf_audit()
    del audit[missing_field]

    with pytest.raises(ValueError, match="requested/effective prehalf together"):
        significance._validate_coarse_selector_audit(audit)


def test_prehalf_audit_requires_the_execution_counter():
    audit = _prehalf_audit()
    del audit["counts"]["prehalf_selected_calls"]

    with pytest.raises(ValueError, match="prehalf_selected_calls"):
        significance._validate_coarse_selector_audit(audit)


def test_effective_selector_rejects_noop_zero_call_count():
    audit = _active_audit()
    audit["counts"] = dict(audit["counts"], fused_calls=0)

    with pytest.raises(ValueError, match="zero calls"):
        significance._validate_coarse_selector_audit(audit)


def test_effective_selector_rejects_zero_actual_row_count():
    audit = _active_audit()
    audit["counts"] = dict(audit["counts"], actual_rows=0)

    with pytest.raises(ValueError, match="zero actual rows"):
        significance._validate_coarse_selector_audit(audit)


def test_effective_selector_rejects_wrong_target():
    audit = _active_audit(workers=8)
    audit["target"] = "cuda_relion_coarse_diff2_projector_f32"

    with pytest.raises(ValueError, match="wrong wrapper/target"):
        significance._validate_coarse_selector_audit(audit)


def test_profile_boundary_rejects_missing_audit():
    with pytest.raises(RuntimeError, match="did not return.*execution audit"):
        _coarse_selector_audit_from_full_stats({})


def test_profile_boundary_rejects_invalid_execution_counts():
    audit = _active_audit(atomic=True)
    audit["counts"] = dict(audit["counts"], native_atomic_selected_calls=0)

    with pytest.raises(RuntimeError, match="invalid coarse selector audit"):
        _coarse_selector_audit_from_full_stats(
            {"coarse_selector_audit": audit}
        )


class _ProfileResult(NamedTuple):
    profile_summary: dict | None


def test_audit_propagates_to_initial_model_meta_without_losing_profile_fields():
    audit = _active_audit(workers=8, atomic=True)
    result = _ProfileResult(profile_summary={"pass2_s": 1.25})

    result = _with_coarse_selector_audit(result, audit)
    meta = _estep_meta({0: SimpleNamespace(profile_summary=result.profile_summary)})

    assert result.profile_summary["pass2_s"] == 1.25
    assert result.profile_summary["coarse_selector_audit"] == audit
    assert meta["halfset_0_profile_summary"]["coarse_selector_audit"] == audit


def test_absent_audit_preserves_result_identity():
    result = _ProfileResult(profile_summary={"control": True})

    assert _with_coarse_selector_audit(result, None) is result


def test_host_counters_are_adjacent_to_the_selected_wrapper_invocation():
    source = Path(significance.__file__).read_text()
    call = source.index("diff2 = coarse_projector(")
    start = source.rindex(
        'selected_wrapper = getattr(coarse_projector, "__name__", None)',
        0,
        call,
    )
    invocation_audit = source[start:call]

    assert "_TARGET_RELION_COARSE_DIFF2_PROJECTOR_MULTISTREAM_F32" in invocation_audit
    assert "_TARGET_RELION_COARSE_DIFF2_PROJECTOR_F32" in invocation_audit
    assert 'coarse_selector_execution["fused_calls"] += 1' in invocation_audit
    assert 'coarse_selector_execution["actual_rows"] += int(actual_batch_size)' in invocation_audit
    assert 'coarse_selector_execution["multistream_calls"] += 1' in invocation_audit
    assert (
        'coarse_selector_execution["native_atomic_selected_calls"] += 1'
        in invocation_audit
    )
    assert (
        'coarse_selector_execution["prehalf_selected_calls"] += 1'
        in invocation_audit
    )
    assert (
        'coarse_projector_kwargs["prehalf_weight"] = coarse_prehalf_weight_enabled'
        in invocation_audit
    )
    assert '"coarse_selector_audit": coarse_selector_audit' in source
