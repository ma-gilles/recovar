"""Bounded transfers must preserve the producing bucket and update order."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.classification.k_class_results import DeferredHostUpdates

pytestmark = pytest.mark.unit


def test_deferred_updates_keep_bucket_widths_and_per_image_tables(monkeypatch):
    get = jax.device_get
    transfers = []
    updates = []

    def capture(values):
        transfers.append(len(values))
        return get(values)

    def apply(*, batch, distances, posterior):
        assert posterior.shape == (batch, 2)
        updates.append(float(np.sum(np.asarray(posterior, np.float64) * distances)))

    monkeypatch.setattr(jax, "device_get", capture)
    queue = DeferredHostUpdates(max_records=4)
    expected = []
    for index, batch in enumerate((3, 1, 5, 2, 1, 4)):
        values = np.full((batch, 2), index + 0.25, dtype=np.float32)
        distances = np.arange(batch * 2).reshape(batch, 2) + index
        expected.append(float(np.sum(values.astype(np.float64) * distances)))
        queue.append(apply, host=dict(batch=batch, distances=distances), device=dict(posterior=jnp.asarray(values)))
    assert transfers == [4]
    assert updates == expected[:4]
    queue.flush()
    queue.flush()
    assert transfers == [4, 2]
    assert updates == expected
    assert queue.pending_bytes == 0
    assert not queue.records


def test_byte_limit_flushes_before_overflow_and_isolates_large_records(monkeypatch):
    transfers = []
    updates = []

    def capture(records):
        transfers.append([record["values"].nbytes for record in records])
        return records

    monkeypatch.setattr(jax, "device_get", capture)
    queue = DeferredHostUpdates(max_records=8, max_bytes=20)
    for index, length in enumerate((3, 3, 9, 1)):
        queue.append(
            lambda *, index, values: updates.append((index, len(values))),
            host=dict(index=index),
            device=dict(values=np.ones(length, np.float32)),
        )
    queue.flush()
    assert transfers == [[12], [12], [36], [4]]
    assert updates == [(0, 3), (1, 3), (2, 9), (3, 1)]


def test_duplicate_input_names_fail_before_enqueueing():
    queue = DeferredHostUpdates()
    with pytest.raises(ValueError, match="disjoint"):
        queue.append(None, host=dict(values=1), device=dict(values=np.ones(1)))
    assert not queue.records


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_noise_offset_replay_uses_each_buckets_distance_table(dtype):
    from recovar.em.classification.k_class_results import SparseKClassNoiseStatistics

    target = np.zeros(2, np.float64)
    noise = SparseKClassNoiseStatistics(
        class_posterior_sums_mstep=np.zeros(2),
        noise_img_power_total=[np.zeros(5) for _ in range(2)],
        noise_norm_correction_total=[np.zeros(13) for _ in range(2)],
        noise_sumw_total=np.zeros(2),
        noise_sigma2_offset_total=target,
        noise_scale_correction_xa_total=np.zeros((2, 4)),
        noise_scale_correction_aa_total=np.zeros((2, 4)),
        noise_wsum_total=[np.zeros(5) for _ in range(2)],
    )
    queue = DeferredHostUpdates()
    expected = np.zeros(2, np.float64)
    for batch in (12, 13):
        for class_index in range(2):
            distances = np.arange(batch * 3).reshape(batch, 3) / (batch + 0.5)
            values = np.full((batch, 3), 1 / (class_index + 3), dtype=dtype)
            expected[class_index] += float(np.sum(values.astype(np.float64) * distances, dtype=np.float64))
            queue.append(
                noise.offset,
                host=dict(class_index=class_index, translation_sqdist_ang=distances),
                device=dict(translation_posterior_jax=jnp.asarray(values)),
            )
    queue.flush()
    np.testing.assert_array_equal(target.view(np.uint8), expected.view(np.uint8))
