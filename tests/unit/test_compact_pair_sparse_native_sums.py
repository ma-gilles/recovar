"""Pair-sparse native weighted sums for compact K-class buckets.

``dual_weighted_sums_f32`` loops over every (row, translation) slot of a dense
``(B, R, T)`` probability table and skips zeros; compact pairs populate ~1 % of
that table and most rows are padding. ``dual_weighted_sums_pairs_f32`` iterates
each row's own sorted pair range with the same fma order, so it must be
bit-identical. The CSR sorter is pinned on CPU; the kernel pin needs a GPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as spb


def _random_unique_pairs(rng, batch, n_rows, n_trans, n_pairs, *, fill=0.6):
    """Source-ordered pairs as build_compact_pair_index_arrays emits them.

    Rotation-major, translation-minor valid prefix (C order of the (R, T) mask),
    then -1 padding. The CSR helper relies on this order instead of sorting.
    """
    rows = np.full((batch, n_pairs), -1, dtype=np.int32)
    trans = np.full((batch, n_pairs), -1, dtype=np.int32)
    for b in range(batch):
        n_valid = int(rng.integers(1, max(2, int(fill * n_pairs))))
        keys = np.sort(rng.choice(n_rows * n_trans, size=min(n_valid, n_rows * n_trans), replace=False))
        rows[b, : keys.size] = keys // n_trans
        trans[b, : keys.size] = keys % n_trans
    mask = rows >= 0
    probs = np.where(mask, rng.random((batch, n_pairs)), 0.0).astype(np.float32)
    return rows, trans, mask, probs


def test_sorted_csr_matches_dense_probabilities_in_ascending_translation_order():
    rng = np.random.default_rng(20260912)
    batch, n_rows, n_trans, n_pairs = 4, 9, 6, 20
    rows, trans, mask, probs = _random_unique_pairs(rng, batch, n_rows, n_trans, n_pairs)
    probs[1, 0] = np.inf  # non-finite pairs are dropped exactly like the dense path
    sorted_probs, sorted_trans, offsets = spb._compact_pair_sorted_csr(
        jnp.asarray(probs), jnp.asarray(rows), jnp.asarray(trans), jnp.asarray(mask),
        n_rotation_rows=n_rows, n_trans=n_trans,
    )
    dense = np.asarray(spb._compact_pair_dense_probs_and_reductions(
        jnp.asarray(probs), jnp.asarray(rows), jnp.asarray(trans), jnp.asarray(mask),
        n_rotation_rows=n_rows, n_trans=n_trans,
    ))
    sorted_probs, sorted_trans, offsets = map(np.asarray, (sorted_probs, sorted_trans, offsets))
    assert offsets.dtype == np.int32 and offsets.shape == (batch, n_rows + 1)
    assert sorted_probs.dtype == np.float32 and sorted_trans.dtype == np.int32
    for b in range(batch):
        assert offsets[b, 0] == 0 and np.all(np.diff(offsets[b]) >= 0)
        for r in range(n_rows):
            seg = slice(int(offsets[b, r]), int(offsets[b, r + 1]))
            t = sorted_trans[b, seg]
            w = sorted_probs[b, seg]
            assert np.all(np.diff(t) > 0), "translations must be strictly ascending within a row"
            # a zero-weight slot (the non-finite pair) stays in its row; the kernel skips it
            np.testing.assert_array_equal(t[w != 0], np.flatnonzero(dense[b, r]))
            np.testing.assert_array_equal(w[w != 0], dense[b, r, t[w != 0]])
        # padding sits after the last row; the non-finite pair keeps its slot with zero weight
        assert np.all(sorted_probs[b, int(offsets[b, -1]):] == 0.0)
    assert sorted_probs[1, 0] == 0.0 and offsets[1, rows[1, 0]] <= 0 < offsets[1, rows[1, 0] + 1]


def test_builder_emits_source_order_the_csr_helper_relies_on():
    from recovar.em.dense_single_volume.helpers.compact_candidates import (
        SparseCandidateMask, build_compact_pair_index_arrays,
    )
    rng = np.random.default_rng(3)
    n_rows, n_trans, c_rot, c_trans = 15, 8, 5, 4
    ftp = np.repeat(np.arange(c_trans), 2).astype(np.int32)
    masks = [
        SparseCandidateMask(mode="coarse", n_rows=n_rows, n_fine_trans=n_trans,
                            parent_map=rng.integers(0, c_rot, n_rows), coarse_valid=rng.random((c_rot, c_trans)) < 0.5,
                            fine_translation_parent=ftp)
        for _ in range(3)
    ] + [SparseCandidateMask(mode="full", n_rows=n_rows, n_fine_trans=n_trans)]
    arrays = build_compact_pair_index_arrays(masks, pair_bucket_size=n_rows * n_trans)
    for b in range(len(masks)):
        count = int(arrays["pair_counts"][b])
        keys = arrays["local_rotation_row"][b, :count].astype(np.int64) * n_trans + arrays["translation_idx"][b, :count]
        assert np.all(np.diff(keys) > 0), "valid prefix must be rotation-major, translation-minor"
        assert np.all(arrays["local_rotation_row"][b, count:] == -1) and not arrays["pair_mask"][b, count:].any()


@pytest.mark.gpu
def test_pair_sparse_native_sums_match_dense_native_bitwise(monkeypatch, custom_cuda_lib, gpu_device):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(7)
    batch, n_rows, n_trans, n_pairs, n_recon, n_img = 3, 40, 11, 150, 37, 29
    rows, trans, mask, probs = _random_unique_pairs(rng, batch, n_rows, n_trans, n_pairs)
    recon = (rng.normal(0, 1, (batch, n_trans, n_recon)) + 1j * rng.normal(0, 1, (batch, n_trans, n_recon))).astype(np.complex64)
    image = (rng.normal(0, 1, (batch, n_trans, n_img)) + 1j * rng.normal(0, 1, (batch, n_trans, n_img))).astype(np.complex64)
    ctf2 = rng.uniform(0.5, 2.0, (batch, n_recon)).astype(np.float32)
    args = (jnp.asarray(probs), jnp.asarray(rows), jnp.asarray(trans), jnp.asarray(mask), jnp.asarray(recon), jnp.asarray(image), jnp.asarray(ctf2))
    with jax.default_device(gpu_device):
        monkeypatch.setenv(spb._SPARSE_KCLASS_NATIVE_PAIR_SPARSE_SUMS_ENV, "0")
        dense = spb._compact_pair_weighted_rotation_and_image_sums_native(*args, n_rotation_rows=n_rows)
        monkeypatch.setenv(spb._SPARSE_KCLASS_NATIVE_PAIR_SPARSE_SUMS_ENV, "1")
        spb._compact_pair_weighted_rotation_and_image_sums_native.clear_cache()
        sparse = spb._compact_pair_weighted_rotation_and_image_sums_native(*args, n_rotation_rows=n_rows)
        dense, sparse = jax.block_until_ready((dense, sparse))
    names = ("summed", "summed_image", "ctf_probs", "probs_sum_t", "translation_posterior")
    for name, a, b in zip(names, dense, sparse):
        a = np.asarray(a); b = np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype, name
        if name in ("summed", "summed_image"):
            # The claim: the pair-sparse kernel reproduces the dense kernel exactly.
            np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8), err_msg=name)
        else:
            # These come from the same dense-table reductions in both variants, but
            # XLA fuses them differently once the consumer graph changes, which on
            # H100 moved ctf_probs by 1-2 float32 ULP (job 13806345). Bound, not pin.
            np.testing.assert_allclose(b, a, rtol=8 * np.finfo(np.float32).eps, atol=0.0, err_msg=name)
    assert np.asarray(sparse[0]).shape == (batch, n_rows, n_recon)
