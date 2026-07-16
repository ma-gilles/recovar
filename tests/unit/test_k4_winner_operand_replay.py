import numpy as np

from recovar.em.k4_winner_operand_replay import (
    _canonical_identities_from_recovar,
    _canonical_reorder,
    _relion_cc_contributions,
    controls_close,
    float32_ulp_distance,
    score_control_metrics,
)


def _window_indices(image_size: int, current_size: int) -> np.ndarray:
    rows = np.arange(current_size, dtype=np.int64)
    rows = np.where(rows <= current_size // 2, rows, rows - current_size)
    columns = np.arange(current_size // 2 + 1, dtype=np.int64)
    return ((rows[:, None] + image_size // 2) * (image_size // 2 + 1) + columns).reshape(-1)


def test_float32_ulp_distance_handles_both_signs():
    left = np.asarray([1.0, -1.0, 0.0], dtype=np.float32)
    right = np.asarray(
        [
            np.nextafter(left[0], np.float32(np.inf)),
            np.nextafter(left[1], np.float32(-np.inf)),
            left[2],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(float32_ulp_distance(left, right), [1, 1, 0])


def test_score_controls_gate_recovar_ulp_and_relion_residuals():
    production = np.asarray([0.5, 0.25, 0.125, 0.0625], dtype=np.float32)
    recovar_replay = production.copy()
    recovar_replay[0] = np.nextafter(recovar_replay[0], np.float32(np.inf))
    relion_replay = production + np.asarray([2.0e-7, 3.0e-7, 2.5e-7, 2.0e-7], dtype=np.float32)

    closed, policy = controls_close(
        score_control_metrics(production, recovar_replay),
        score_control_metrics(production, relion_replay),
    )

    assert closed
    assert all(policy["checks"].values())

    failed, policy = controls_close(
        score_control_metrics(production, recovar_replay),
        score_control_metrics(production, production + np.float32(2.0e-6)),
    )
    assert not failed
    assert not policy["checks"]["relion_captured_float32_absolute"]


def test_canonical_identities_reconcile_compact_and_fftw_orders():
    image_size = 128
    for current_size in (8, 14, 40):
        relion_order = _window_indices(image_size, current_size)
        recovar_order = relion_order[np.random.default_rng(current_size).permutation(relion_order.size)]

        recovar_identities, relion_identities = _canonical_identities_from_recovar(
            recovar_order,
            image_size=image_size,
        )
        recovar_values = recovar_identities.astype(np.float64) * 0.25
        relion_values = relion_identities.astype(np.float64) * 0.25

        np.testing.assert_array_equal(
            _canonical_reorder(recovar_values, recovar_identities),
            _canonical_reorder(relion_values, relion_identities),
        )


def test_relion_cc_contributions_weight_both_numerator_and_norm():
    projection = np.asarray([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex64)
    shifted = np.asarray([5.0 - 6.0j, 7.0 + 8.0j], dtype=np.complex64)
    corr = np.asarray([0.25, 2.0], dtype=np.float32)
    half_weights = np.ones(2, dtype=np.float32)

    contributions = _relion_cc_contributions(projection, shifted, corr, half_weights)

    expected_numerator = (projection.real * shifted.real + projection.imag * shifted.imag) * corr
    expected_norm = (projection.real**2 + projection.imag**2) * corr
    np.testing.assert_array_equal(contributions.numerator, expected_numerator)
    np.testing.assert_array_equal(contributions.norm, expected_norm)
