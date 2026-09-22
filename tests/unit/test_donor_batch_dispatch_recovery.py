"""Donor memory policy and metadata-only batch shape regressions."""
import numpy as np
import pytest
from recovar.em.sparse_pass2 import sparse_pass2_budget as budget
from recovar.em.helpers.shape_buckets import pad_batch_data_ctf_and_valid_mask
from recovar.em.scoring.significance import _pad_significance_preprocess_inputs


def test_fused_kclass_score_gather_fraction_env(monkeypatch):
    monkeypatch.delenv(budget._FUSED_KCLASS_SCORE_GATHER_FRACTION_ENV, raising=False)
    assert budget._fused_kclass_score_gather_device_fraction() == 0.100
    kwargs = dict(score_only=False, use_window=True, has_external_normalization=False,
                  conservative_dump_execution=False, fused_k_class=True, fused_k_class_count=4,
                  n_score_pixels=596, device_memory_bytes=85899345920)
    base = budget._max_hypotheses_per_microbatch_for_pass(**kwargs)
    monkeypatch.setenv(budget._FUSED_KCLASS_SCORE_GATHER_FRACTION_ENV, "0.2")
    doubled = budget._max_hypotheses_per_microbatch_for_pass(**kwargs)
    assert abs(doubled / base - 2.0) < 0.01
    for value in ["0", "-0.1", "0.9", "nan", "inf"]:
        monkeypatch.setenv(budget._FUSED_KCLASS_SCORE_GATHER_FRACTION_ENV, value)
        with pytest.raises(ValueError, match="must be in"):
            budget._max_hypotheses_per_microbatch_for_pass(**kwargs)


class ShapeOnlyBatch:
    shape = (3, 8, 8)
    def __array__(self, *args, **kwargs):
        raise AssertionError("Shape inspection copied the image batch")


def test_unpadded_batch_shape_does_not_copy_to_host():
    batch = ShapeOnlyBatch()
    ctf = np.ones((3, 9))
    out = pad_batch_data_ctf_and_valid_mask(batch, ctf, 3)
    assert out[0] is batch and out[1] is ctf
    np.testing.assert_array_equal(out[2], [True, True, True])
    assert out[3:] == (3, 3)
    out = _pad_significance_preprocess_inputs(batch, ctf, None, None, None, None, target_size=3)
    assert out[0] is batch and out[1] is ctf
