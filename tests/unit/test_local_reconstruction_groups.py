"""Particle group admission preserves host IDs and fails before execution."""

import numpy as np
import pytest

from recovar.em.local.local_batch_planning import prepare_reconstruction_groups

pytestmark = pytest.mark.unit


def test_group_ids_are_retained_without_a_copy():
    ids = np.array([0, 1, 0], dtype=np.int32)
    result, count = prepare_reconstruction_groups(
        ids, 2, n_images=3, score_only=False, source_faithful_bpref=True,
    )
    assert result is ids
    assert count == 2


def test_ungrouped_reconstruction_needs_no_source_faithful_mode():
    assert prepare_reconstruction_groups(
        None, None, n_images=0, score_only=True, source_faithful_bpref=False,
    ) == (None, 1)


@pytest.mark.parametrize("ids,count,n_images,score_only,faithful,error,match", [
    (np.array([0], dtype=np.int64), 1, 1, False, True, TypeError, "must be int32"),
    (np.array([0], dtype=np.int32), 1, 2, False, True, ValueError, "image axis"),
    (np.array([0], dtype=np.int32), None, 1, False, True, ValueError, "count is required"),
    (np.array([0], dtype=np.int32), 0, 1, False, True, ValueError, "must be positive"),
    (np.array([-1], dtype=np.int32), 1, 1, False, True, ValueError, "out-of-range"),
    (np.array([1], dtype=np.int32), 1, 1, False, True, ValueError, "out-of-range"),
    (np.array([0], dtype=np.int32), 1, 1, True, True, ValueError, "score-only"),
    (np.array([0], dtype=np.int32), 1, 1, False, False, ValueError, "source-faithful"),
    (None, 2, 1, False, True, ValueError, "ids is required"),
])
def test_invalid_group_contract_rejected(ids, count, n_images, score_only, faithful, error, match):
    with pytest.raises(error, match=match):
        prepare_reconstruction_groups(
            ids, count, n_images=n_images, score_only=score_only, source_faithful_bpref=faithful,
        )
