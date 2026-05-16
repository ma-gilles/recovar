"""Regression tests for `recovar.em.initial_model.layout`.

These pin behavior that's load-bearing for InitialModel/VDAM RELION parity:
- `run_em_output_to_bpref` clamps near-denormal weight noise to 0 so RELION's
  `BackProjector::updateSSNRarrays` invariant (sigma2 must be either exactly
  0 or > 1e-20) holds.
- The round-trip with `bpref_to_run_em_output` is lossless on values RELION
  would actually emit (always either 0 or > 1e-15 in practice).
"""

from __future__ import annotations

import numpy as np

from recovar.em.initial_model.layout import (
    bpref_to_run_em_output,
    run_em_output_to_bpref,
)


def _make_full_with_centered_slab(ori_size: int, r_max: int, slab: np.ndarray) -> np.ndarray:
    """Embed a centered cropped half-slab into a full (N,N,N) Fourier volume.

    Mirrors the inverse of run_em_output_to_bpref's centered-cropped path so
    we can craft known weight slabs as input.
    """
    full = np.zeros((ori_size, ori_size, ori_size), dtype=np.complex128)
    c = ori_size // 2
    half_ps = r_max + 1
    full[c - half_ps : c + half_ps + 1, c - half_ps : c + half_ps + 1, c : c + half_ps + 1] = slab
    return full


def test_run_em_output_to_bpref_clamps_denormal_weight_to_zero():
    """Denormal-range weights (|w| < 1e-15) must become exactly 0.

    Without this clamp, RELION's `BackProjector::updateSSNRarrays` aborts
    with "unexpectedly small, yet non-zero sigma2 value" — observed at
    iter-7 of K=4 nr_iter=10 where the iter-by-iter drift compounded
    high-frequency shells into the denormal band.
    """
    ori_size = 8
    r_max = 2  # cropped slab shape for ori_size=8: (7, 7, 4)

    half_ps = r_max + 1
    slab_shape = (2 * half_ps + 1, 2 * half_ps + 1, half_ps + 1)

    # Mix of magnitudes: exact 0 / typical (1.0) / denormal-positive / denormal-negative.
    weight_slab = np.zeros(slab_shape, dtype=np.complex128)
    weight_slab[..., 0] = 0.0
    weight_slab[..., 1] = 1.0
    weight_slab[..., 2] = 1e-25  # below threshold → must clamp
    weight_slab[..., 3] = -1e-30  # below threshold → must clamp

    Fy = np.zeros((ori_size, ori_size, ori_size), dtype=np.complex128)
    Fc = _make_full_with_centered_slab(ori_size, r_max, weight_slab)

    _bp_data, bp_weight = run_em_output_to_bpref(Fy, Fc, ori_size, r_max)

    assert bp_weight.dtype == np.float64
    assert bp_weight.shape == slab_shape
    # exact 0 stays 0
    assert (bp_weight[..., 0] == 0.0).all()
    # 1.0 passes through
    assert (bp_weight[..., 1] == 1.0).all()
    # 1e-25 clamped to 0
    assert (bp_weight[..., 2] == 0.0).all(), "1e-25 must clamp to 0"
    # -1e-30 clamped to 0
    assert (bp_weight[..., 3] == 0.0).all(), "-1e-30 must clamp to 0"


def test_run_em_output_to_bpref_preserves_typical_weights():
    """Weights of physical magnitude (≥ 1e-12) must pass through unchanged.

    Guards against an over-aggressive clamp that would drop real signal.
    """
    ori_size = 8
    r_max = 2
    half_ps = r_max + 1
    slab_shape = (2 * half_ps + 1, 2 * half_ps + 1, half_ps + 1)

    rng = np.random.default_rng(42)
    raw_real = rng.uniform(0, 1.0, size=slab_shape) * 1e6 + 1e-12
    weight_slab = raw_real.astype(np.complex128)
    Fy = np.zeros((ori_size, ori_size, ori_size), dtype=np.complex128)
    Fc = _make_full_with_centered_slab(ori_size, r_max, weight_slab)

    _bp_data, bp_weight = run_em_output_to_bpref(Fy, Fc, ori_size, r_max)

    np.testing.assert_array_equal(bp_weight, raw_real.astype(np.float64))


def test_run_em_output_to_bpref_clamp_threshold_boundary():
    """The clamp threshold is 1e-15. Values just above pass; just below clamp."""
    ori_size = 8
    r_max = 2
    half_ps = r_max + 1
    slab_shape = (2 * half_ps + 1, 2 * half_ps + 1, half_ps + 1)

    weight_slab = np.zeros(slab_shape, dtype=np.complex128)
    weight_slab[..., 0] = 1.5e-15  # above threshold → preserved
    weight_slab[..., 1] = 5e-16  # below threshold → clamped
    weight_slab[..., 2] = 0.0  # exact 0 → stays 0
    weight_slab[..., 3] = 1e-15  # exactly at threshold → clamped (strict |w| < 1e-15 is False here, so preserved)
    Fy = np.zeros((ori_size, ori_size, ori_size), dtype=np.complex128)
    Fc = _make_full_with_centered_slab(ori_size, r_max, weight_slab)

    _bp_data, bp_weight = run_em_output_to_bpref(Fy, Fc, ori_size, r_max)

    assert (bp_weight[..., 0] == 1.5e-15).all()
    assert (bp_weight[..., 1] == 0.0).all()
    assert (bp_weight[..., 2] == 0.0).all()
    # exactly 1e-15 sits at the boundary (`<` strict), so preserved
    assert (bp_weight[..., 3] == 1e-15).all()


def test_run_em_output_to_bpref_relion_invariant_holds():
    """End-to-end invariant: emitted weight values are either exactly 0 OR
    strictly > 1e-20 — the precondition RELION's BackProjector asserts.

    Earlier crashes (iter-7 K=4 nr_iter=10) hit this with a 2.99e-29 value.
    """
    ori_size = 8
    r_max = 2
    half_ps = r_max + 1
    slab_shape = (2 * half_ps + 1, 2 * half_ps + 1, half_ps + 1)

    # Adversarial mix including the exact magnitude that crashed RELION (2.99e-29).
    rng = np.random.default_rng(13)
    weight_slab = rng.choice(
        np.array([0.0, 1e-25, 2.99e-29, 1e-18, 1.0, 1e6, -1e-30, 5e-21]),
        size=slab_shape,
    ).astype(np.complex128)
    Fy = np.zeros((ori_size, ori_size, ori_size), dtype=np.complex128)
    Fc = _make_full_with_centered_slab(ori_size, r_max, weight_slab)

    _bp_data, bp_weight = run_em_output_to_bpref(Fy, Fc, ori_size, r_max)

    nonzero = bp_weight != 0.0
    if nonzero.any():
        # Every non-zero weight must satisfy RELION's > 1e-20 invariant.
        bad = nonzero & (np.abs(bp_weight) <= 1e-20)
        assert not bad.any(), (
            f"weight contains values in (0, 1e-20] band that would crash RELION: {np.sort(np.abs(bp_weight[bad]))[:5]}"
        )


def test_run_em_output_to_bpref_round_trip_on_realistic_data():
    """Round-trip via bpref_to_run_em_output must be exact for realistic
    values (everything > 1e-15). This pins the clamp's correctness on
    typical RELION-emitted data — no real RELION dump has denormal weights.
    """
    ori_size = 8
    r_max = 2
    half_ps = r_max + 1
    slab_shape = (2 * half_ps + 1, 2 * half_ps + 1, half_ps + 1)

    rng = np.random.default_rng(7)
    bp_data = (rng.standard_normal(slab_shape) + 1j * rng.standard_normal(slab_shape)).astype(np.complex128)
    bp_weight = rng.uniform(1e-3, 100.0, size=slab_shape).astype(np.float64)

    Ft_y, Ft_ctf = bpref_to_run_em_output(bp_data, bp_weight, ori_size, r_max)
    bp_data_rt, bp_weight_rt = run_em_output_to_bpref(Ft_y, Ft_ctf, ori_size, r_max)

    np.testing.assert_array_equal(bp_data, bp_data_rt, "data round-trip lossy")
    np.testing.assert_array_equal(bp_weight, bp_weight_rt, "weight round-trip lossy")
