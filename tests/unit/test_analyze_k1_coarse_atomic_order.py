from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts import analyze_k1_coarse_atomic_order as analyzer


def test_captured_lane_partials_uses_relion_thread_mapping():
    header = [0] * 32
    header[14] = 29
    header[17] = 128
    raw = np.arange(2 * 128, dtype=np.float32).reshape(2, 128)
    capture = SimpleNamespace(header=tuple(header), lane_partials=raw)

    actual = analyzer._captured_lane_partials(capture)

    assert actual.shape == (2, 29, 4)
    np.testing.assert_array_equal(actual[0, 13], raw[0, [13, 42, 71, 100]])
    np.testing.assert_array_equal(actual[1, 28], raw[1, [28, 57, 86, 115]])


def test_rotation_key_to_recovar_transposes_direction_psi_order():
    assert analyzer._rotation_key_to_recovar(25632, 768, 48) == 534
    assert analyzer._rotation_key_to_recovar(19619, 768, 48) == 27288


def test_summed_scores_enumerates_every_four_lane_order():
    partials = np.asarray([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
    orders, scores = analyzer._summed_scores(partials, np.float32(0.5))

    assert len(orders) == 24
    assert scores.shape == (24, 1, 1)
    np.testing.assert_array_equal(scores, np.full((24, 1, 1), 10.5, dtype=np.float32))


def test_pair_order_audit_distinguishes_independent_from_shared_order() -> None:
    permutations = [(0, 1), (1, 0)]
    scores = np.asarray(
        [
            [[10.0, 8.0]],
            [[10.25, 8.25]],
        ],
        dtype=np.float32,
    )
    native = np.asarray([[10.0, 8.25]], dtype=np.float32)
    recovar = np.asarray([[10.25, 8.0]], dtype=np.float32)

    report = analyzer._pair_order_audit(
        permutations=permutations,
        scores=scores,
        native=native,
        recovar=recovar,
        target=(0, 0),
        winner=(0, 1),
    )

    assert report["native_both_scores_independently_attainable"] is True
    assert report["native_relative_independent_order_pair_count"] == 1
    assert report["native_relative_independent_lane_order_pairs"] == [
        [[0, 1], [1, 0]]
    ]
    assert report["native_relative_same_order_count"] == 0
    assert report["recovar_relative_independent_lane_order_pairs"] == [
        [[1, 0], [0, 1]]
    ]
