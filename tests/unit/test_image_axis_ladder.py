"""Image-axis ladder: padded evaluation must not change any real image's value."""

import numpy as np
import pytest

import jax.numpy as jnp

from recovar.em.helpers import projection


@pytest.mark.parametrize("n_images", [1, 5, 12, 17, 32, 33, 40])
def test_ladder_rounds_up_and_never_shrinks(n_images):
    padded = projection.image_axis_ladder_size(n_images)
    assert padded >= n_images
    assert padded % 4 == 0 or padded == 1 << (padded.bit_length() - 1)


def test_ladder_collapses_many_counts_onto_few_rungs():
    counts = list(range(1, 41)) + [126, 194, 220]
    rungs = {projection.image_axis_ladder_size(n) for n in counts}
    assert len(rungs) < len(counts) / 3          # 43 counts -> few rungs
    assert max(rungs) <= 256


def _operands(n_images, rot, px, seed):
    rng = np.random.default_rng(seed)
    c = lambda: jnp.asarray(rng.standard_normal((n_images, rot, px)) + 1j * rng.standard_normal((n_images, rot, px)), dtype=jnp.complex64)
    f = lambda: jnp.asarray(rng.standard_normal((n_images, rot, px)), dtype=jnp.float32)
    return c(), f(), c(), f(), jnp.asarray(rng.random(px) + 0.5, dtype=jnp.float32)


@pytest.mark.parametrize("n_images", [1, 5, 17, 33])
def test_norm_residual_padded_equals_unpadded(n_images):
    """Padded evaluation returns exactly the unpadded per-image values."""
    proj, abs2, summed, ctf, noise = _operands(n_images, 8, 13, seed=20260916)
    got = projection.compute_norm_residual_per_image(proj, abs2, summed, ctf, noise)
    want = projection._compute_norm_residual_per_image_core(proj, abs2, summed, ctf, noise)
    assert got.shape == want.shape == (n_images,)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_padded_rows_cannot_reach_real_images():
    """Changing the padded region's content leaves every real image unchanged.

    This is the inertness property the padding relies on: the core reduces over
    axes (1, 2) only, so no padded row can contribute to a real image's value.
    """
    n = 17
    proj, abs2, summed, ctf, noise = _operands(n, 8, 13, seed=7)
    padded = projection.image_axis_ladder_size(n)
    assert padded > n
    base = projection.compute_norm_residual_per_image(proj, abs2, summed, ctf, noise)
    # pad manually with NON-zero garbage in the padded rows
    def poison(a):
        p = projection.pad_image_axis(a, padded)
        return p.at[n:].set(jnp.asarray(1e6, dtype=a.dtype))
    poisoned = projection._compute_norm_residual_per_image_core(
        poison(proj), poison(abs2), poison(summed), poison(ctf), noise
    )[:n]
    np.testing.assert_array_equal(np.asarray(base), np.asarray(poisoned))


def _scale_operands(n_images, rot, px, seed):
    proj, abs2, summed, ctf, noise = _operands(n_images, rot, px, seed)
    rng = np.random.default_rng(seed + 1)
    per_image_scale = jnp.asarray(rng.random(n_images) + 0.5, dtype=jnp.float32)
    mask = jnp.asarray(rng.random(px) > 0.3)
    return proj, abs2, summed, ctf, noise, per_image_scale, mask


_SCALE_FORMS = {
    "per_image": lambda s: s,
    "broadcast_1": lambda s: s[:1],
    "python_float": lambda s: 1.25,
    "zero_dim": lambda s: jnp.asarray(0.8, dtype=jnp.float32),
}


@pytest.mark.parametrize("n_images", [1, 5, 17, 33])
@pytest.mark.parametrize("scale_form", sorted(_SCALE_FORMS))
@pytest.mark.parametrize("with_mask", [False, True])
def test_scale_terms_padded_equal_unpadded(n_images, scale_form, with_mask):
    """Scale XA/AA on the ladder equal the unpadded core for every scale form."""
    proj, abs2, summed, ctf, noise, scale, mask = _scale_operands(n_images, 8, 13, seed=11)
    scale = _SCALE_FORMS[scale_form](scale)
    mask_arg = mask if with_mask else None
    got = projection.compute_scale_correction_terms_per_image(proj, abs2, summed, ctf, noise, scale, mask_arg)
    want = projection._compute_scale_correction_terms_per_image_core(proj, abs2, summed, ctf, noise, scale, mask_arg)
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape == (n_images,)
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


def test_scale_terms_pixel_count_equal_to_image_count_never_pads_pixel_axis():
    """Regression: pixels == images must not pad noise or mask (lead review of 7db7054a2)."""
    n = rot = 5
    proj, abs2, summed, ctf, noise, scale, mask = _scale_operands(n, 2, n, seed=3)
    assert noise.shape == (n,) and mask.shape == (n,)
    got = projection.compute_scale_correction_terms_per_image(proj, abs2, summed, ctf, noise, scale, mask)
    want = projection._compute_scale_correction_terms_per_image_core(proj, abs2, summed, ctf, noise, scale, mask)
    for g, w in zip(got, want, strict=True):
        assert g.shape == (n,)
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))
    got_norm = projection.compute_norm_residual_per_image(proj, abs2, summed, ctf, noise)
    want_norm = projection._compute_norm_residual_per_image_core(proj, abs2, summed, ctf, noise)
    np.testing.assert_array_equal(np.asarray(got_norm), np.asarray(want_norm))


def test_scale_terms_all_keyword_call():
    proj, abs2, summed, ctf, noise, scale, mask = _scale_operands(5, 2, 5, seed=5)
    kwargs = dict(proj_half=proj, proj_abs2_half=abs2, summed_masked=summed, ctf_probs=ctf,
                  noise_variance_half=noise, old_scale=scale, scale_correction_pixel_mask=mask)
    got = projection.compute_scale_correction_terms_per_image(**kwargs)
    want = projection._compute_scale_correction_terms_per_image_core(**kwargs)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))
    got_norm = projection.compute_norm_residual_per_image(
        proj_half=proj, proj_abs2_half=abs2, summed_masked=summed, ctf_probs=ctf, noise_variance_half=noise)
    np.testing.assert_array_equal(np.asarray(got_norm), np.asarray(
        projection._compute_norm_residual_per_image_core(proj, abs2, summed, ctf, noise)))


@pytest.mark.parametrize("n_images", [5, 17])
def test_broadcast_ctf_probs_is_not_padded(n_images):
    """A shared (1, R, P) ctf_probs broadcasts against the padded axis; padding it would zero real rows."""
    proj, abs2, summed, ctf, noise, scale, mask = _scale_operands(n_images, 4, 9, seed=9)
    ctf1 = ctf[:1]
    got = projection.compute_norm_residual_per_image(proj, abs2, summed, ctf1, noise)
    want = projection._compute_norm_residual_per_image_core(proj, abs2, summed, ctf1, noise)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    got_s = projection.compute_scale_correction_terms_per_image(proj, abs2, summed, ctf1, noise, scale, mask)
    want_s = projection._compute_scale_correction_terms_per_image_core(proj, abs2, summed, ctf1, noise, scale, mask)
    for g, w in zip(got_s, want_s, strict=True):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


def test_scale_padded_rows_cannot_reach_real_images():
    n = 17
    proj, abs2, summed, ctf, noise, scale, mask = _scale_operands(n, 8, 13, seed=17)
    padded = projection.image_axis_ladder_size(n)
    assert padded > n
    base = projection.compute_scale_correction_terms_per_image(proj, abs2, summed, ctf, noise, scale, mask)

    def poison(a):
        return projection.pad_image_axis(a, padded).at[n:].set(jnp.asarray(1e6, dtype=a.dtype))

    poisoned = projection._compute_scale_correction_terms_per_image_core(
        poison(proj), poison(abs2), poison(summed), poison(ctf), noise, poison(scale), mask
    )
    for b, p in zip(base, poisoned, strict=True):
        np.testing.assert_array_equal(np.asarray(b), np.asarray(p[:n]))
