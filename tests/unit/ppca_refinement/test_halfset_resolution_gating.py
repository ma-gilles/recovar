import numpy as np
import pytest

from recovar.em.ppca_refinement.schedule import PPCARefinementScheduleState, evaluate_halfset_resolution_gate


pytestmark = pytest.mark.unit


def test_pmax_alone_does_not_increase_resolution():
    state = PPCARefinementScheduleState(
        current_size=32,
        healpix_order=2,
        best_pose_indices=np.array([0, 0], dtype=np.int32),
        previous_best_pose_indices=np.array([0, 0], dtype=np.int32),
        pmax_mean=0.999,
        kclass_schedule_allows=False,
        halfset_means_aligned=False,
        halfset_resolution_supports=False,
    )
    decision = evaluate_halfset_resolution_gate(state)
    assert not decision.allow_increase
    assert "kclass schedule blocked" in decision.reasons
    assert "halfset means not aligned" in decision.reasons


def test_halfset_mean_fsc_without_pose_stability_does_not_increase_resolution():
    state = PPCARefinementScheduleState(
        current_size=32,
        healpix_order=2,
        best_pose_indices=np.array([0, 1, 2], dtype=np.int32),
        previous_best_pose_indices=np.array([0, 9, 2], dtype=np.int32),
        halfset_mean_fsc=np.ones(16, dtype=np.float32),
        halfset_means_aligned=True,
        halfset_resolution_supports=True,
        no_halfset_drift=True,
        kclass_schedule_allows=True,
        pmax_mean=0.95,
    )
    decision = evaluate_halfset_resolution_gate(state)
    assert not decision.allow_increase
    assert decision.pose_change_fraction == pytest.approx(1.0 / 3.0)


def test_halfset_drift_blocks_even_when_pose_and_fsc_pass():
    state = PPCARefinementScheduleState(
        current_size=32,
        healpix_order=2,
        best_pose_indices=np.array([4, 4, 4], dtype=np.int32),
        previous_best_pose_indices=np.array([4, 4, 4], dtype=np.int32),
        halfset_means_aligned=True,
        halfset_resolution_supports=True,
        no_halfset_drift=False,
        kclass_schedule_allows=True,
        pmax_mean=0.95,
    )
    decision = evaluate_halfset_resolution_gate(state)
    assert not decision.allow_increase
    assert "halfset drift or frame mismatch detected" in decision.reasons
