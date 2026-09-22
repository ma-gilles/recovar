"""Recovered group-static row planner and engine regressions."""
import numpy as np
from test_compact_real_rows_integration import _fused_kclass_multibucket_fixture, _fused_kclass_result_arrays, _assert_fused_arrays_identical
from recovar.em.scoring import sparse_bucket_arrays as planning

def test_padded_active_row_count_matches_the_row_builder(monkeypatch):
    """The planner's padded-count predictor and the row builder agree for every count, and
    ``pad_to`` pads with repeated first indices under a zero mask, never past the slots."""
    from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_DEVICE_ACTIVE_ROW_INDICES", "0")
    rows = 64
    for counts in ([3, 0, 5], [64, 64, 64], [1], [17, 40, 2, 9], [0, 0]):
        for pad_multiple in (1, 8, 1024):
            idx, mask, total = bucketed_mod._real_flat_row_indices_from_actual_counts(
                counts, rows, pad_multiple=pad_multiple)
            predicted = planning._padded_active_row_count(sum(min(c, rows) for c in counts), len(counts) * rows, pad_multiple)
            assert idx.shape[0] == predicted, (counts, pad_multiple, idx.shape, predicted)
            assert total == sum(min(c, rows) for c in counts)
    idx, mask, total = bucketed_mod._real_flat_row_indices_from_actual_counts([3, 0, 5], rows, pad_multiple=1)
    assert idx.shape[0] == 8 and total == 8
    idx2, mask2, total2 = bucketed_mod._real_flat_row_indices_from_actual_counts([3, 0, 5], rows, pad_multiple=1, pad_to=20)
    assert idx2.shape[0] == 20 and total2 == 8
    np.testing.assert_array_equal(idx2[:8], idx)
    assert np.all(idx2[8:] == idx[0]) and np.all(mask2[:8] == 1.0) and np.all(mask2[8:] == 0.0)
    # pad_to below the multiple padding changes nothing; pad_to past the slots is clipped
    idx3, _, _ = bucketed_mod._real_flat_row_indices_from_actual_counts([3, 0, 5], rows, pad_multiple=8, pad_to=4)
    assert idx3.shape[0] == 8
    idx4, _, _ = bucketed_mod._real_flat_row_indices_from_actual_counts([3, 0, 5], rows, pad_multiple=1, pad_to=10**6)
    assert idx4.shape[0] == 3 * rows

def test_group_static_active_row_targets_take_the_group_maximum():
    """Chunks with the same (execution width, class bucket sizes, image capacity) share one
    target, the maximum of their own padded counts; other groups keep their own."""
    from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed_mod

    rots_a = [np.zeros((n, 3, 3)) for n in (5, 9, 2, 7, 1, 12)]
    rots_b = [np.zeros((n, 3, 3)) for n in (4, 4, 4, 4, 4, 4)]
    per_image_inputs_by_class = ({"oversampled_rots": rots_a}, {"oversampled_rots": rots_b})
    def bucket(images, width, class_sizes):
        return {"image_indices": np.asarray(images), "_execution_mode": "compact_pair",
                "_execution_bucket_size": width, "class_bucket_sizes": class_sizes,
                "image_capacity_budget": None}
    buckets = [bucket([0, 1], 4096, (16, 8)), bucket([2, 3], 4096, (16, 8)),
               bucket([4, 5], 4096, (16, 8)), bucket([0, 1, 2, 3], 8192, (16, 8)),
               {"image_indices": np.asarray([4]), "_execution_mode": "rectangular"}]
    stats = planning._attach_group_static_active_row_targets(
        buckets, per_image_inputs_by_class, pad_multiple=1, rotation_block_size_for_quantization=5000)
    # class a totals per chunk: 14, 9, 13 -> target 14; class b: 8 each -> 8
    assert buckets[0]["_active_row_pad_targets"] == (14, 8)
    assert buckets[1]["_active_row_pad_targets"] == (14, 8)
    assert buckets[2]["_active_row_pad_targets"] == (14, 8)
    assert buckets[3]["_active_row_pad_targets"] == (23, 16)
    assert "_active_row_pad_targets" not in buckets[4]
    assert stats["groups"] == 2 and stats["chunks"] == 4
    assert stats["real_rows"] == (14 + 9 + 13 + 23) + (8 * 3 + 16)
    assert stats["group_static_rows"] == 3 * (14 + 8) + (23 + 16)

def test_group_static_active_rows_change_no_result(monkeypatch):
    """With the flag on, every chunk is padded to a target the planner attached, and every
    output of the fused engine is unchanged."""
    from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed_mod

    pad_to_seen = []
    original = bucketed_mod._real_flat_row_indices_from_actual_counts

    def spy(*a, **kw):
        pad_to_seen.append(kw.get("pad_to"))
        return original(*a, **kw)

    monkeypatch.setattr(bucketed_mod, "_real_flat_row_indices_from_actual_counts", spy)

    def run(flag):
        monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH", "4")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS", "0")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_ADJOINT_REAL_ROWS", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_ACTIVE_ROW_PAD_MULTIPLE", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_GROUP_STATIC_ACTIVE_ROWS", flag)
        pad_to_seen.clear()
        kwargs = _fused_kclass_multibucket_fixture(n_images=13)
        kwargs["accumulate_noise"] = True
        kwargs["relion_f32_fine_posterior"] = True
        return _fused_kclass_result_arrays(bucketed_mod.compute_k_class_pass2_stats_sparse_fused(**kwargs))

    base = run("0")
    assert pad_to_seen and all(v is None for v in pad_to_seen)
    on = run("1")
    assert pad_to_seen and all(isinstance(v, int) for v in pad_to_seen), pad_to_seen[:5]
    # Everything except the two adjoint volumes is bit-identical. The padded rows scatter
    # exact zeros, but on GPU the XLA scatter-add's float32 accumulation order over the
    # enlarged row set is not fixed, so the volumes are bounded like the other engine
    # comparisons in this file (rtol/atol 1e-5) and the measured error is reported.
    volumes = {k for k in base if k.startswith(("Ft_y", "Ft_ctf"))}
    _assert_fused_arrays_identical(
        {k: v for k, v in base.items() if k not in volumes},
        {k: v for k, v in on.items() if k not in volumes},
        "group-static active rows",
    )
    for k in sorted(volumes):
        x = np.asarray(base[k]); y = np.asarray(on[k])
        assert x.shape == y.shape and x.dtype == y.dtype
        scale = float(np.max(np.abs(x))) or 1.0
        err = float(np.max(np.abs(y - x)))
        np.testing.assert_allclose(y, x, rtol=1e-5, atol=1e-5,
                                   err_msg=f"{k}: max|diff|={err:.3e}, max|ref|={scale:.3e}, rel={err/scale:.3e}")
