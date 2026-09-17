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
