"""E5: Posterior weight parity — RELION vs recovar.

Tests that convert_squared_differences_to_weights produces
exactly the same posteriors as recovar's softmax + prior computation.

Exact parity required: rel_err < 1e-12.
"""

import numpy as np
from recovar.relion_bind._relion_bind_core import (
    convert_squared_differences_to_weights,
)


def _recovar_posteriors(diff2, orient_prior, offset_prior, min_diff2):
    """Pure numpy reimplementation of recovar's posterior computation.

    Matches RELION's formula: weight = pdf_orient * pdf_offset * exp(-(d2 - min_d2))
    then normalize to sum=1.
    """
    n_orient, n_trans = diff2.shape

    orient_mean = orient_prior.mean()
    offset_mean = offset_prior.mean()

    pdf_o = orient_prior / orient_mean if orient_mean > 0 else np.ones_like(orient_prior)
    pdf_t = offset_prior / offset_mean if offset_mean > 0 else np.ones_like(offset_prior)

    d = diff2 - min_diff2
    weights = np.zeros_like(diff2)
    for i in range(n_orient):
        for j in range(n_trans):
            w = pdf_o[i] * pdf_t[j]
            if d[i, j] > 700.0:
                w = 0.0
            else:
                w *= np.exp(-d[i, j])
            weights[i, j] = w

    s = weights.sum()
    if s > 0:
        weights /= s
    return weights


class TestE5BasicParity:
    """Exact parity between RELION binding and numpy reference."""

    def test_uniform_priors(self):
        rng = np.random.default_rng(42)
        n_orient, n_trans = 100, 29
        diff2 = rng.uniform(0, 50, (n_orient, n_trans))
        min_diff2 = diff2.min()
        orient_prior = np.ones(n_orient)
        offset_prior = np.ones(n_trans)

        relion = convert_squared_differences_to_weights(diff2, orient_prior, offset_prior, min_diff2)
        reference = _recovar_posteriors(diff2, orient_prior, offset_prior, min_diff2)

        max_diff = np.max(np.abs(relion - reference))
        assert max_diff < 1e-14, f"Uniform priors: max_diff={max_diff}"

    def test_nonuniform_priors(self):
        rng = np.random.default_rng(99)
        n_orient, n_trans = 50, 15
        diff2 = rng.uniform(0, 30, (n_orient, n_trans))
        min_diff2 = diff2.min()
        orient_prior = rng.exponential(1.0, n_orient)
        offset_prior = rng.exponential(1.0, n_trans)

        relion = convert_squared_differences_to_weights(diff2, orient_prior, offset_prior, min_diff2)
        reference = _recovar_posteriors(diff2, orient_prior, offset_prior, min_diff2)

        max_diff = np.max(np.abs(relion - reference))
        assert max_diff < 1e-14, f"Nonuniform priors: max_diff={max_diff}"

    def test_overflow_guard(self):
        """diff2 - min_diff2 > 700 should give weight=0."""
        n_orient, n_trans = 10, 5
        diff2 = np.full((n_orient, n_trans), 800.0)
        diff2[0, 0] = 0.0
        min_diff2 = 0.0
        orient_prior = np.ones(n_orient)
        offset_prior = np.ones(n_trans)

        relion = convert_squared_differences_to_weights(diff2, orient_prior, offset_prior, min_diff2)

        assert relion[0, 0] > 0.99, f"Best entry should dominate: {relion[0, 0]}"
        assert np.all(relion[1:, :] == 0.0), "Overflow entries should be exactly 0"

    def test_normalization(self):
        """Posteriors must sum to exactly 1."""
        rng = np.random.default_rng(77)
        diff2 = rng.uniform(0, 20, (200, 50))
        min_diff2 = diff2.min()
        orient_prior = rng.exponential(2.0, 200)
        offset_prior = rng.exponential(1.0, 50)

        relion = convert_squared_differences_to_weights(diff2, orient_prior, offset_prior, min_diff2)

        total = relion.sum()
        assert abs(total - 1.0) < 1e-14, f"Sum of posteriors: {total}"

    def test_large_grid(self):
        """Realistic grid size: 4608 orientations × 29 translations."""
        rng = np.random.default_rng(55)
        n_orient, n_trans = 4608, 29
        diff2 = rng.uniform(0, 100, (n_orient, n_trans))
        min_diff2 = diff2.min()
        orient_prior = rng.exponential(1.0, n_orient)
        offset_prior = np.exp(-rng.uniform(0, 2, n_trans))

        relion = convert_squared_differences_to_weights(diff2, orient_prior, offset_prior, min_diff2)
        reference = _recovar_posteriors(diff2, orient_prior, offset_prior, min_diff2)

        max_diff = np.max(np.abs(relion - reference))
        rel_err = max_diff / (np.max(np.abs(reference)) + 1e-30)
        assert max_diff < 1e-12, f"Large grid: max_diff={max_diff}, rel_err={rel_err}"
