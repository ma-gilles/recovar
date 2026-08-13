import numpy as np

from scripts import analyze_k1_norm_residual_bpref_boundary as analyzer
from scripts.validate_relion_bpref_prescatter import ROTATION_DTYPE, ROW_DTYPE


def test_dense_native_operands_align_rotations_pixels_and_units() -> None:
    recovar_rotations = np.stack(
        (
            np.eye(3, dtype=np.float32),
            np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
        )
    )
    native_rotations = np.zeros(2, dtype=ROTATION_DTYPE)
    native_rotations["orientation_local"] = np.arange(2, dtype=np.uint32)
    native_rotations["matrix"] = recovar_rotations.transpose(0, 2, 1).reshape(2, 9)
    rows = np.zeros(2, dtype=ROW_DTYPE)
    rows["orientation_local"] = [1, 0]
    rows["x"] = [1, 2]
    rows["y"] = [-1, 0]
    rows["source_re"] = [64.0, -128.0]
    rows["source_im"] = [32.0, 0.0]
    rows["source_weight"] = [4096.0, 8192.0]

    data, weight, identity = analyzer._dense_native_operands(
        rows=rows,
        native_rotations=native_rotations,
        recovar_rotations=recovar_rotations,
        recon_window_indices=np.asarray([16, 22], dtype=np.int32),
        physical_image_size=8,
    )

    expected_data = np.zeros((2, 2), dtype=np.complex64)
    expected_weight = np.zeros((2, 2), dtype=np.float32)
    expected_data[1, 0] = -1.0 - 0.5j
    expected_data[0, 1] = 2.0
    expected_weight[1, 0] = 1.0
    expected_weight[0, 1] = 2.0
    np.testing.assert_array_equal(data, expected_data)
    np.testing.assert_array_equal(weight, expected_weight)
    assert identity["rotation_max_abs"] == 0.0
    assert identity["native_supported_rows"] == 2


def test_norm_terms_preserve_split_formula() -> None:
    projection = np.asarray([[1 + 2j, 3 + 4j]], dtype=np.complex64)
    projection_abs2 = np.abs(projection) ** 2
    summed = np.asarray([[5 + 6j, 0 + 0j]], dtype=np.complex64)
    ctf_prob = np.asarray([[2.0, 0.0]], dtype=np.float32)
    noise = np.asarray([7.0, 11.0], dtype=np.float32)

    terms = analyzer._norm_terms(
        projection,
        projection_abs2,
        summed,
        ctf_prob,
        noise,
    )

    np.testing.assert_array_equal(terms["ctf_probs_raw"], [[14.0, 0.0]])
    np.testing.assert_array_equal(terms["a2"], [[70.0, 0.0]])
    np.testing.assert_array_equal(terms["cross"], [[17.0 + 4.0j, 0.0 + 0.0j]])
    np.testing.assert_array_equal(terms["xa"], [[119.0, 0.0]])
    summary = analyzer._float64_scalar_summary(terms)
    assert summary == {"a2": 70.0, "xa": 119.0, "residual_a2_minus_2xa": -168.0}


def test_native_norm_block_terms_stream_rows_in_native_units() -> None:
    projection = np.asarray(
        [[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]],
        dtype=np.complex64,
    )
    rows = np.zeros(3, dtype=ROW_DTYPE)
    rows["state"] = 1
    rows["flags"] = 3
    rows["orientation_local"] = [4, 4, 5]
    rows["pixel"] = [0, 1, 1]
    rows["source_re"] = [-64.0, -128.0, -192.0]
    rows["source_im"] = [0.0, -64.0, 0.0]
    rows["source_weight"] = [4096.0, 8192.0, 12288.0]

    result = analyzer._native_norm_block_terms(
        projection,
        np.abs(projection) ** 2,
        rows,
        orientation_start=4,
        rectangle_to_reconstruction=np.asarray([0, 1], dtype=np.int32),
        noise_variance=np.asarray([2.0, 3.0], dtype=np.float32),
        physical_image_size=8,
    )

    source = np.asarray([[1.0 + 0.0j, 2.0 + 1.0j], [0.0, 3.0 + 0.0j]], dtype=np.complex64)
    weight = np.asarray([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
    expected = analyzer._norm_terms(
        projection,
        np.abs(projection) ** 2,
        source,
        weight,
        np.asarray([2.0, 3.0], dtype=np.float32),
    )
    assert result["a2"] == float(np.sum(expected["a2"], dtype=np.float64))
    assert result["xa"] == float(np.sum(expected["xa"], dtype=np.float64))
    assert result["row_count"] == 3
    assert result["active_rotation_count"] == 2


def test_counterfactual_totals_split_a2_and_xa() -> None:
    assert analyzer._counterfactual_totals(
        weighted_image=100.0,
        recovar_a2=20.0,
        recovar_xa=5.0,
        native_a2=18.0,
        native_xa=7.0,
    ) == {
        "recovar": 110.0,
        "native_a2_only": 108.0,
        "native_xa_only": 106.0,
        "native_a2_and_xa": 104.0,
    }
