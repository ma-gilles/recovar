import numpy as np

from scripts.run_relion_recovar_prescatter_substitution import scatter_operand


def test_scatter_operand_replays_row_and_neighbor_hermitian_folds():
    output_data = np.zeros(8, dtype=np.complex128)
    output_weight = np.zeros(8, dtype=np.float64)
    source_data = np.asarray([[1 + 2j]], dtype=np.complex64)
    source_weight = np.asarray([[3]], dtype=np.float32)
    row_flags = np.asarray([[64 | 16]], dtype=np.uint32)
    neighbor_indices = np.arange(8, dtype=np.int32).reshape(1, 1, 8)
    coefficients = np.full((1, 1, 8), 0.5, dtype=np.float32)
    neighbor_flags = np.zeros((1, 1, 8), dtype=np.uint32)
    neighbor_flags[..., 1] = 2

    count = scatter_operand(
        output_data,
        output_weight,
        source_data,
        source_weight,
        row_flags,
        neighbor_indices,
        coefficients,
        neighbor_flags,
    )

    assert count == 8
    assert output_data[0] == 0.5 - 1j
    assert output_data[1] == 0.5 + 1j
    np.testing.assert_array_equal(output_weight, np.full(8, 1.5))
