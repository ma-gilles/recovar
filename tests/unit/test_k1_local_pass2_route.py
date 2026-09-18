"""The local K=1 pass-2 route lays out exactly the compact engine's candidate set."""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.classification.k1_local_pass2 import (
    K1_PASS2_ENGINE_ENV,
    align_layout_to_pass2_grids,
    coarse_prior_from_pass2_prior,
    k1_local_pass2_engine_selected,
)
from recovar.em.helpers.oversampling import build_adaptive_pass2_grids
from recovar.em.local.local_layout import build_pass2_hypothesis_layout
from recovar.em.sampling import rotation_grid_size
from recovar.em.scoring.compact_candidates import _candidate_mask_to_dense
from recovar.em.scoring.sparse_bucket_arrays import _prepare_per_image_pass2_inputs

pytestmark = pytest.mark.unit

ORDER = 1
OVERSAMPLING = 1
CHILDREN = 8**OVERSAMPLING
N_ROT = rotation_grid_size(ORDER)
BASE = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
N_TRANS = int(BASE.shape[0])


def _grids(perturbation: float):
    rot = np.repeat(np.eye(3, dtype=np.float32)[None], N_ROT, axis=0)
    return build_adaptive_pass2_grids(
        rot, BASE.astype(np.float32), BASE, ORDER, OVERSAMPLING, 1.0, perturbation,
        return_mstep_rotations=True,
    )


def _significant_samples(seed: int):
    rng = np.random.default_rng(seed)
    samples = []
    for count in (5, 1, 17, 0):
        if count == 0:
            samples.append(None)  # full support
        else:
            samples.append(np.sort(rng.choice(N_ROT * N_TRANS, size=count, replace=False)).astype(np.int32))
    return samples


def test_engine_selection_env():
    import os

    old = os.environ.pop(K1_PASS2_ENGINE_ENV, None)
    try:
        assert not k1_local_pass2_engine_selected()
        os.environ[K1_PASS2_ENGINE_ENV] = "local"
        assert k1_local_pass2_engine_selected()
        os.environ[K1_PASS2_ENGINE_ENV] = "dense"
        with pytest.raises(ValueError):
            k1_local_pass2_engine_selected()
    finally:
        if old is None:
            os.environ.pop(K1_PASS2_ENGINE_ENV, None)
        else:
            os.environ[K1_PASS2_ENGINE_ENV] = old


def test_coarse_prior_recovered_from_parent_major_fine_prior():
    coarse = np.linspace(-3.0, 2.0, N_ROT, dtype=np.float32)
    parent_map = np.repeat(np.arange(N_ROT), CHILDREN)
    fine = coarse[parent_map]
    assert np.array_equal(coarse_prior_from_pass2_prior(fine, n_rot_coarse=N_ROT, children_per_parent=CHILDREN), coarse)
    assert np.array_equal(coarse_prior_from_pass2_prior(coarse, n_rot_coarse=N_ROT, children_per_parent=CHILDREN), coarse)
    assert coarse_prior_from_pass2_prior(None, n_rot_coarse=N_ROT, children_per_parent=CHILDREN) is None
    with pytest.raises(ValueError):
        coarse_prior_from_pass2_prior(coarse[:-1], n_rot_coarse=N_ROT, children_per_parent=CHILDREN)


@pytest.mark.parametrize("perturbation", [0.0, 0.25])
def test_layout_matches_compact_engine_candidates(perturbation):
    (
        coarse_rot, coarse_trans, fine_rot, fine_trans, rot_parent, trans_parent, fine_mstep_rot,
    ) = _grids(perturbation)
    samples = _significant_samples(seed=3)
    coarse_prior = np.linspace(-1.0, 1.0, N_ROT, dtype=np.float32)
    fine_prior = coarse_prior[rot_parent]
    layout = build_pass2_hypothesis_layout(
        samples, N_ROT, N_TRANS, ORDER, coarse_trans,
        oversampling_order=OVERSAMPLING, translation_step=1.0,
        rotation_log_prior=coarse_prior_from_pass2_prior(fine_prior, n_rot_coarse=N_ROT, children_per_parent=CHILDREN),
        random_perturbation=perturbation, allow_empty=True, dtype=np.float32,
    )
    layout = align_layout_to_pass2_grids(
        layout,
        fine_rotations=fine_rot, fine_mstep_rotations=fine_mstep_rot, fine_translations=fine_trans,
        fine_source_eulers=None, children_per_parent=CHILDREN, n_rot_coarse=N_ROT,
    )
    compact = _prepare_per_image_pass2_inputs(
        samples, N_ROT, N_TRANS, ORDER, OVERSAMPLING, int(fine_trans.shape[0]), trans_parent,
        fine_prior, perturbation,
        fine_rotations_override=fine_rot, fine_mstep_rotations_override=fine_mstep_rot,
        fine_rotation_parent_override=rot_parent, dtype=np.float32,
    )
    assert np.array_equal(layout.translation_grid, fine_trans.astype(np.float32))
    for image, (start, stop) in enumerate(zip(layout.rotation_offsets[:-1], layout.rotation_offsets[1:])):
        ids = layout.rotation_ids_flat[start:stop]
        assert np.array_equal(ids, compact["oversampled_rot_indices"][image])
        assert np.array_equal(layout.rotations_flat[start:stop], compact["oversampled_rots"][image])
        assert np.array_equal(layout.mstep_rotations_flat[start:stop], compact["oversampled_mstep_rots"][image])
        assert np.array_equal(layout.rotation_posterior_ids_flat[start:stop], ids // CHILDREN)
        # Every child inherits its coarse parent's prior (RELION pushback semantics).
        # The compact engine indexes its fine-length prior with coarse ids, so its
        # rows only agree for a uniform prior; the route keeps the parent's value.
        assert np.array_equal(layout.rotation_log_priors_flat[start:stop], coarse_prior[ids // CHILDREN])
        assert np.array_equal(compact["log_prior"][image], fine_prior[ids // CHILDREN])
        assert np.array_equal(
            layout.sample_mask_rows(start, stop), _candidate_mask_to_dense(compact["candidate_mask"][image])
        )


def test_alignment_fails_closed_on_a_foreign_rotation_grid():
    grids = _grids(0.0)
    layout = build_pass2_hypothesis_layout(
        _significant_samples(seed=5), N_ROT, N_TRANS, ORDER, grids[1],
        oversampling_order=OVERSAMPLING, translation_step=1.0, random_perturbation=0.0,
        allow_empty=True, dtype=np.float32,
    )
    foreign = _grids(0.25)[2]
    with pytest.raises(RuntimeError):
        align_layout_to_pass2_grids(
            layout, fine_rotations=foreign, fine_mstep_rotations=None, fine_translations=grids[3],
            fine_source_eulers=None, children_per_parent=CHILDREN, n_rot_coarse=N_ROT,
        )
