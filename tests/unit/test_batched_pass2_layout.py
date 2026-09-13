"""Exact batch invariance for shared pass-2 orientation construction."""

import numpy as np
import pytest

from recovar.em import sampling
from recovar.em.local import local_layout

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("index_order", ["recovar", "relion", "relion_hidden"])
@pytest.mark.parametrize("perturbation", [0.0, -0.43747416138648987])
def test_batched_pass2_matches_independent_image_layouts(order, index_order, perturbation):
    n_rotations = sampling.rotation_grid_size(0)
    samples = [
        np.array([2 * 3, 2 * 3 + 1, 5 * 3 + 2, 2 * 3]),
        np.array([1 * 3 + 2, 5 * 3, (n_rotations - 1) * 3]),
        np.array([], dtype=np.int64),
        None,
    ]
    kwargs = dict(
        n_coarse_rotations=n_rotations,
        n_coarse_translations=3,
        nside_level=0,
        translations=np.array([[-1, 0], [0, 0], [1, 0]], dtype=np.float32),
        translation_step=1.0,
        rotation_log_prior=np.linspace(-2, -1, n_rotations, dtype=np.float32),
        translation_log_prior=np.array([-3, -2, -1], dtype=np.float32),
        oversampling_order=order,
        random_perturbation=perturbation,
        rotation_index_order=index_order,
        allow_empty=True,
    )
    batched = local_layout.build_pass2_hypothesis_layout(samples, **kwargs)
    for image, sample in enumerate(samples):
        single = local_layout.build_pass2_hypothesis_layout([sample], **kwargs)
        start, stop = batched.rotation_offsets[image : image + 2]
        assert batched.rotation_counts[image] == single.rotation_counts[0]
        for field in (
            "rotations_flat", "rotation_ids_flat", "rotation_posterior_ids_flat",
            "rotation_log_priors_flat", "sample_mask_flat",
        ):
            np.testing.assert_array_equal(getattr(batched, field)[start:stop], getattr(single, field))
        np.testing.assert_array_equal(batched.translation_grid, single.translation_grid)
        # Compare matrices directly to the mature per-image sampler as well.
        parents = (np.arange(n_rotations) if sample is None else
                   np.unique(sample // 3) if sample.size else np.array([0]))
        rotations, _, ids = sampling.get_oversampled_rotation_grid_from_samples(
            parents, 0, order, random_perturbation=perturbation,
            return_rotation_indices=True, rotation_index_order=index_order,
        )
        np.testing.assert_array_equal(batched.rotations_flat[start:stop], rotations)
        np.testing.assert_array_equal(batched.rotation_ids_flat[start:stop], ids)


def test_pass2_generates_only_requested_parent_union_once(monkeypatch):
    calls = []
    original = local_layout.get_oversampled_rotation_grid_from_samples

    def record(parents, *args, **kwargs):
        calls.append(np.asarray(parents).copy())
        return original(parents, *args, **kwargs)

    monkeypatch.setattr(local_layout, "get_oversampled_rotation_grid_from_samples", record)
    local_layout.build_pass2_hypothesis_layout(
        [np.array([3, 9]), np.array([9, 15]), np.array([3])],
        n_coarse_rotations=2**40, n_coarse_translations=3, nside_level=0,
        translations=np.array([[-1, 0], [0, 0], [1, 0]], dtype=np.float32),
        translation_step=1.0, oversampling_order=1,
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [1, 3, 5])
