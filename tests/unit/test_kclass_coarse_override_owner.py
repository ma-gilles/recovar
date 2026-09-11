"""Both adaptive K-class pass-2 paths replace class assignments with the coarse winner through one owner."""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

import recovar.em.dense_single_volume.k_class as k_class

pytestmark = pytest.mark.unit


class _Result:
    def __init__(self, per_class_hard):
        self.per_class_hard_assignments = per_class_hard
        self.replaced = None

    def _replace(self, **kw):
        out = _Result(self.per_class_hard_assignments)
        out.replaced = kw
        return out


def test_override_takes_the_winning_class_pose_and_decodes_details(monkeypatch):
    per_class_hard = jnp.asarray([[5, 6, 7], [8, 9, 10]], dtype=jnp.int32)
    winners = np.asarray([1, 0, 1])
    monkeypatch.setattr(k_class, "_decode_dense_best_pose_details", lambda hard, rots, trans: (("R", hard.tolist()), ("T", trans.shape), np.asarray(hard)))
    out = k_class._override_class_assignments_with_coarse_winner(
        _Result(per_class_hard), winners, return_best_pose_details=True, fine_rotations_np=np.zeros((11, 3, 3)), fine_translations_np=np.zeros((11, 2)),
    )
    kw = out.replaced
    assert kw["class_assignments"].dtype == jnp.int32 and kw["class_assignments"].tolist() == [1, 0, 1]
    assert kw["pose_assignments"].tolist() == [8, 6, 10]
    assert kw["best_pose_rotations"] == ("R", [8, 6, 10]) and kw["best_pose_translations"] == ("T", (11, 2))
    assert kw["best_pose_rotation_ids"].tolist() == [8, 6, 10]
    assert kw["best_pose_eulers_deg"] is None and kw["per_class_best_pose_eulers_deg"] is None


def test_override_without_details_replaces_only_assignments():
    per_class_hard = jnp.asarray([[5, 6], [8, 9]], dtype=jnp.int32)
    out = k_class._override_class_assignments_with_coarse_winner(
        _Result(per_class_hard), np.asarray([0, 1]), return_best_pose_details=False, fine_rotations_np=np.zeros((10, 3, 3)), fine_translations_np=np.zeros((10, 2)),
    )
    assert set(out.replaced) == {"class_assignments", "pose_assignments"} and out.replaced["pose_assignments"].tolist() == [5, 9]


def test_both_pass2_paths_use_the_owner():
    source = inspect.getsource(k_class.run_dense_k_class_em_adaptive)
    assert source.count("_override_class_assignments_with_coarse_winner(") == 2
    assert "per_class_hard[coarse_assn, image_indices]" not in source
