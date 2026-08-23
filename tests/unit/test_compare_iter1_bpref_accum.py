from __future__ import annotations

import numpy as np

from scripts.compare_iter1_bpref_accum import _apply_recovar_frame, coordinate_stats, scan_coordinate_mappings


def test_apply_recovar_frame_converts_native_accumulators_to_relion_units():
    avg = np.asarray([16.0 + 32.0j], dtype=np.complex128)
    weight = np.asarray([1.5], dtype=np.float64)

    converted_avg, converted_weight = _apply_recovar_frame(
        avg,
        weight,
        grid_size=4,
        recovar_frame="relion",
    )

    np.testing.assert_allclose(converted_avg, avg / 16.0)
    np.testing.assert_allclose(converted_weight, weight * 256.0)


def test_coordinate_stats_uses_relion_kij_to_recovar_ikj_axis_order():
    avg = np.zeros((5, 5, 3), dtype=np.complex128)
    weight = np.zeros((5, 5, 3), dtype=np.float64)
    down_radius = 2
    relion_dump = {
        "k": np.asarray([1], dtype=np.int64),
        "i": np.asarray([-1], dtype=np.int64),
        "j": np.asarray([2], dtype=np.int64),
        "real": np.asarray([7.0], dtype=np.float64),
        "imag": np.asarray([-3.0], dtype=np.float64),
        "weight": np.asarray([11.0], dtype=np.float64),
    }
    avg[relion_dump["i"][0] + down_radius, relion_dump["k"][0] + down_radius, relion_dump["j"][0]] = 7.0 - 3.0j
    weight[relion_dump["i"][0] + down_radius, relion_dump["k"][0] + down_radius, relion_dump["j"][0]] = 11.0

    stats = coordinate_stats(avg, weight, down_radius, relion_dump)

    assert stats["valid"] == 1
    assert stats["total"] == 1
    assert stats["avg_sign"] == 1
    np.testing.assert_allclose(stats["avg_err"], np.asarray([0.0]))
    np.testing.assert_allclose(stats["weight_err"], np.asarray([0.0]))


def test_coordinate_mapping_scan_ranks_expected_kij_to_ikj_mapping_first():
    avg = np.zeros((5, 5, 3), dtype=np.complex128)
    weight = np.zeros((5, 5, 3), dtype=np.float64)
    down_radius = 2
    relion_dump = {
        "k": np.asarray([1, 0], dtype=np.int64),
        "i": np.asarray([-1, 2], dtype=np.int64),
        "j": np.asarray([2, 1], dtype=np.int64),
        "real": np.asarray([7.0, -5.0], dtype=np.float64),
        "imag": np.asarray([-3.0, 2.5], dtype=np.float64),
        "weight": np.asarray([11.0, 13.0], dtype=np.float64),
    }
    for row in range(2):
        z = relion_dump["i"][row] + down_radius
        y = relion_dump["k"][row] + down_radius
        x = relion_dump["j"][row]
        avg[z, y, x] = relion_dump["real"][row] + 1j * relion_dump["imag"][row]
        weight[z, y, x] = relion_dump["weight"][row]

    best = scan_coordinate_mappings(avg, weight, down_radius, relion_dump, top_n=1)[0]

    assert best["perm"] == (1, 0, 2)
    assert best["signs"] == (1, 1, 1)
    assert best["avg_sign"] == 1
    assert best["avg_median"] == 0.0
    assert best["weight_median"] == 0.0
