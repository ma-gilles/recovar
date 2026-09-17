"""Image-axis ladder round 2: seven pass-2 helpers must be unchanged for real images and inert for padded rows."""

import numpy as np
import pytest

import jax.numpy as jnp

from recovar.em.helpers.image_axis_ladder import image_axis_ladder_size, pad_image_axis
from recovar.em.sparse_pass2 import sparse_pass2_posterior as post
from recovar.em.sparse_pass2 import sparse_pass2_scoring as scoring
from recovar.em.sparse_pass2 import sparse_pass2_wavg as wavg

# n=5/13/22 pad; n=4/8 exact rung (wrapper must be a pure pass-through)
IMAGE_COUNTS = [1, 4, 5, 8, 13, 22]


def _rng(seed):
    return np.random.default_rng(seed)


def _scores(n, r, t, seed, sparse=True):
    rng = _rng(seed)
    s = jnp.asarray(rng.standard_normal((n, r, t)) * 30.0 - 1000.0, dtype=jnp.float32)
    if sparse:
        s = jnp.where(jnp.asarray(rng.random((n, r, t)) < 0.6), s, -jnp.inf)
    return s


def _equal(a, b):
    a = np.asarray(a); b = np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    np.testing.assert_array_equal(a, b)


def _poison(a, n, padded, fill):
    """Pad with the wrapper's neutral fill, then overwrite the padded rows with garbage of the same dtype."""
    p = pad_image_axis(a, padded, fill)
    garbage = jnp.asarray(7.0, dtype=a.dtype) if a.dtype != bool else True
    return p.at[n:].set(garbage)


@pytest.mark.parametrize("n", IMAGE_COUNTS)
def test_logsumexp_and_normalize(n):
    s = _scores(n, 16, 7, seed=n)
    lz_w = post._logsumexp_pass2_bucket_score_only(s)
    lz_c = post._logsumexp_pass2_bucket_score_only_core(s)
    _equal(lz_w, lz_c)
    out_w = post._normalize_pass2_bucket_with_log_z(s, lz_c)
    out_c = post._normalize_pass2_bucket_with_log_z_core(s, lz_c)
    assert len(out_w) == len(out_c) == 5
    for a, b in zip(out_w, out_c, strict=True):
        _equal(a, b)


def test_logsumexp_padded_rows_inert():
    n = 5; padded = image_axis_ladder_size(n); assert padded > n
    s = _scores(n, 16, 7, seed=3)
    base = post._logsumexp_pass2_bucket_score_only_core(pad_image_axis(s, padded, -jnp.inf))[:n]
    poisoned = post._logsumexp_pass2_bucket_score_only_core(_poison(s, n, padded, -jnp.inf))[:n]
    _equal(base, poisoned)
    lz = post._logsumexp_pass2_bucket_score_only_core(s)
    for a, b in zip(
        post._normalize_pass2_bucket_with_log_z_core(pad_image_axis(s, padded, -jnp.inf), pad_image_axis(lz, padded, -jnp.inf)),
        post._normalize_pass2_bucket_with_log_z_core(_poison(s, n, padded, -jnp.inf), _poison(lz, n, padded, -jnp.inf)),
        strict=True,
    ):
        _equal(np.asarray(a)[:n], np.asarray(b)[:n])


@pytest.mark.parametrize("n", IMAGE_COUNTS)
@pytest.mark.parametrize("keep_all", [False, True])
@pytest.mark.parametrize("with_sum_weight", [False, True])
def test_f32_fine_posterior(n, keep_all, with_sum_weight):
    s = _scores(n, 8, 5, seed=100 + n)
    sw = jnp.asarray(_rng(n).random(n) + 0.5, dtype=jnp.float32) if with_sum_weight else None
    out_w = post._relion_f32_fine_posterior(s, adaptive_fraction=0.999, normalization_sum_weight=sw, keep_all=keep_all)
    out_c = post._relion_f32_fine_posterior_core(s, adaptive_fraction=0.999, normalization_sum_weight=sw, keep_all=keep_all)
    assert len(out_w) == len(out_c) == 6
    for a, b in zip(out_w, out_c, strict=True):
        _equal(a, b)


def test_f32_fine_posterior_padded_rows_inert():
    n = 5; padded = image_axis_ladder_size(n)
    s = _scores(n, 8, 5, seed=9)
    kw = dict(adaptive_fraction=0.999, keep_all=False)
    base = post._relion_f32_fine_posterior_core(pad_image_axis(s, padded, -jnp.inf), **kw)
    # garbage in padded rows: large finite scores, which would dominate any cross-image reduction
    poisoned = post._relion_f32_fine_posterior_core(pad_image_axis(s, padded, -jnp.inf).at[n:].set(jnp.float32(5.0)), **kw)
    for a, b in zip(base, poisoned, strict=True):
        _equal(np.asarray(a)[:n], np.asarray(b)[:n])


@pytest.mark.parametrize("n", IMAGE_COUNTS)
@pytest.mark.parametrize("prior_form", ["per_image", "broadcast"])
@pytest.mark.parametrize("external_min", [False, True])
def test_diff2_min_and_to_scores(n, prior_form, external_min):
    rng = _rng(200 + n)
    r, t = 16, 7
    diff2 = jnp.asarray(rng.random((n, r, t)) * 2000.0 + 500.0, dtype=jnp.float32)
    mask = jnp.asarray(rng.random((n, r, t)) < 0.7)
    if prior_form == "per_image":
        rot = jnp.asarray(rng.standard_normal((n, r, 1)), dtype=jnp.float32)
        tr = jnp.asarray(rng.standard_normal((n, 1, t)), dtype=jnp.float32)
    else:
        rot = jnp.asarray(rng.standard_normal((1, r, 1)), dtype=jnp.float32)
        tr = jnp.asarray(rng.standard_normal((1, 1, t)), dtype=jnp.float32)
    mn_w = scoring._relion_cuda_fine_diff2_min(diff2, mask)
    mn_c = scoring._relion_cuda_fine_diff2_min_core(diff2, mask)
    _equal(mn_w, mn_c)
    ext = mn_c if external_min else None
    sc_w = scoring._relion_cuda_fine_diff2_to_scores(diff2, rot, tr, mask, min_diff2=ext)
    sc_c = scoring._relion_cuda_fine_diff2_to_scores_core(diff2, rot, tr, mask, min_diff2=ext)
    _equal(sc_w, sc_c)


def test_diff2_padded_rows_inert():
    n = 5; padded = image_axis_ladder_size(n); rng = _rng(4); r, t = 16, 7
    diff2 = jnp.asarray(rng.random((n, r, t)) * 2000.0 + 500.0, dtype=jnp.float32)
    mask = jnp.asarray(rng.random((n, r, t)) < 0.7)
    rot = jnp.asarray(rng.standard_normal((n, r, 1)), dtype=jnp.float32)
    tr = jnp.asarray(rng.standard_normal((n, 1, t)), dtype=jnp.float32)
    args_clean = (pad_image_axis(diff2, padded, jnp.inf), pad_image_axis(rot, padded), pad_image_axis(tr, padded), pad_image_axis(mask, padded, False))
    args_poison = (_poison(diff2, n, padded, jnp.inf), _poison(rot, n, padded, 0.0), _poison(tr, n, padded, 0.0), _poison(mask, n, padded, False))
    _equal(scoring._relion_cuda_fine_diff2_min_core(args_clean[0], args_clean[3])[:n], scoring._relion_cuda_fine_diff2_min_core(args_poison[0], args_poison[3])[:n])
    _equal(scoring._relion_cuda_fine_diff2_to_scores_core(*args_clean)[:n], scoring._relion_cuda_fine_diff2_to_scores_core(*args_poison)[:n])


def _wavg_operands(n, r, t, p, seed):
    rng = _rng(seed)
    c = lambda *s: jnp.asarray(rng.standard_normal(s) + 1j * rng.standard_normal(s), dtype=jnp.complex64)
    f = lambda *s: jnp.asarray(rng.standard_normal(s), dtype=jnp.float32)
    proj = c(n, r, p); raw_ctf = jnp.abs(f(n, p)); scale = jnp.asarray(rng.random(n) + 0.5, dtype=jnp.float32)
    shifted = c(n, t, p); posterior = jnp.abs(f(n, r, t))
    exact_terms = f(n, r, p, 3)
    p_rect = p + 4
    shifted_rect = c(n, t, p_rect)
    positions = jnp.asarray(np.sort(rng.choice(p_rect, size=p, replace=False)).astype(np.int32))
    return proj, raw_ctf, scale, shifted, posterior, exact_terms, shifted_rect, positions


@pytest.mark.parametrize("n", IMAGE_COUNTS)
def test_wavg_terms(n):
    proj, raw_ctf, scale, shifted, posterior, exact_terms, shifted_rect, positions = _wavg_operands(n, 8, 3, 11, seed=300 + n)
    _equal(
        wavg._relion_wavg_sequential_triplet_terms_jax(proj, raw_ctf, scale, shifted, posterior),
        wavg._relion_wavg_sequential_triplet_terms_jax_core(proj, raw_ctf, scale, shifted, posterior),
    )
    _equal(
        wavg._relion_wavg_rectangle_triplet_terms(exact_terms, shifted_rect, posterior, positions),
        wavg._relion_wavg_rectangle_triplet_terms_core(exact_terms, shifted_rect, posterior, positions),
    )


def test_wavg_padded_rows_inert():
    n = 5; padded = image_axis_ladder_size(n)
    proj, raw_ctf, scale, shifted, posterior, exact_terms, shifted_rect, positions = _wavg_operands(n, 8, 3, 11, seed=8)
    clean = (pad_image_axis(proj, padded), pad_image_axis(raw_ctf, padded), pad_image_axis(scale, padded, 1.0), pad_image_axis(shifted, padded), pad_image_axis(posterior, padded))
    poison = (_poison(proj, n, padded, 0), _poison(raw_ctf, n, padded, 0.0), _poison(scale, n, padded, 1.0), _poison(shifted, n, padded, 0), _poison(posterior, n, padded, 0.0))
    _equal(wavg._relion_wavg_sequential_triplet_terms_jax_core(*clean)[:n], wavg._relion_wavg_sequential_triplet_terms_jax_core(*poison)[:n])
    clean_r = (pad_image_axis(exact_terms, padded), pad_image_axis(shifted_rect, padded), pad_image_axis(posterior, padded), positions)
    poison_r = (_poison(exact_terms, n, padded, 0.0), _poison(shifted_rect, n, padded, 0), _poison(posterior, n, padded, 0.0), positions)
    _equal(wavg._relion_wavg_rectangle_triplet_terms_core(*clean_r)[:n], wavg._relion_wavg_rectangle_triplet_terms_core(*poison_r)[:n])
