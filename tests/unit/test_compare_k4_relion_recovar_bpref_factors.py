import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import compare_k4_relion_recovar_bpref_factors as comparator
from scripts import validate_relion_bpref_factor_capture as validator


def test_compact_indices_preserve_packed_columns_and_center_rows():
    values = {
        "image_shape": np.asarray([4, 4]),
        "window_indices": np.asarray([0, 2, 3, 8, 10]),
    }

    compact = comparator._compact_indices(values)

    np.testing.assert_array_equal(compact, [6, 8, 9, 2, 4])


def test_dataset_native_processed_reconstruction_inputs_preserve_source_factors():
    raw = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    values = {
        "raw_real_images": raw,
        "relion_preprocess_normalization_factors": np.ones(1, dtype=np.float32),
        "integer_pre_shifts": np.asarray([[1, -1]], dtype=np.int32),
        "relion_cuda_preprocess": np.bool_(False),
        "preprocess_backend": np.asarray("dataset_native"),
        "image_corrections": np.asarray([1.25], dtype=np.float32),
        "scale_corrections": np.asarray([2.0], dtype=np.float32),
    }

    processed, reconstruction_correction = comparator._processed_reconstruction_inputs(values)

    shifted = comparator.apply_relion_integer_pre_shifts(raw, values["integer_pre_shifts"])
    expected = comparator._centered_rfft2_numpy(shifted).reshape(1, -1).astype(np.complex64)
    np.testing.assert_array_equal(processed, expected)
    np.testing.assert_array_equal(reconstruction_correction, values["image_corrections"])


def test_dataset_native_processed_reconstruction_inputs_reject_active_relion_normalization():
    values = {
        "raw_real_images": np.zeros((1, 4, 4), dtype=np.float32),
        "relion_preprocess_normalization_factors": np.asarray([1.25], dtype=np.float32),
        "integer_pre_shifts": np.zeros((1, 2), dtype=np.int32),
        "relion_cuda_preprocess": np.bool_(False),
        "preprocess_backend": np.asarray("dataset_native"),
        "image_corrections": np.ones(1, dtype=np.float32),
        "scale_corrections": np.ones(1, dtype=np.float32),
    }

    with pytest.raises(ValueError, match="active RELION normalization"):
        comparator._processed_reconstruction_inputs(values)


def test_scalar_rotation_records_are_identity_bound(tmp_path):
    report = {
        "classification": "pixel_varying_source_difference_not_explained_by_per_rotation_scalar",
        "particles": [
            {
                "stack_index_one_based": 17,
                "rotation_scalar_fits": [
                    {
                        "recovar_global_rotation_index": 123,
                        "relion_rotation_local_row": 7,
                    },
                    {
                        "recovar_global_rotation_index": 124,
                        "relion_rotation_local_row": 8,
                    },
                ],
            },
            {
                "stack_index_one_based": 23,
                "rotation_scalar_fits": [
                    {
                        "recovar_global_rotation_index": 456,
                        "relion_rotation_local_row": 9,
                    }
                ],
            },
        ],
    }
    path = tmp_path / "scalar.json"
    path.write_text(json.dumps(report))

    assert comparator._scalar_rotation_records(path, [23, 17]) == {
        17: ((123, 7), (124, 8)),
        23: ((456, 9),),
    }


def test_translation_map_aligns_by_vector_not_local_index():
    translations = np.zeros(2, dtype=validator.TRANSLATION_DTYPE)
    fine = np.asarray([[2.0, -1.0], [-3.0, 4.0]], dtype=np.float32)
    increments = -2 * np.pi * fine / comparator.PHYSICAL_IMAGE_SIZE
    translations["x"] = increments[::-1, 0]
    translations["y"] = increments[::-1, 1]
    capture = SimpleNamespace(stack_index=17, translations=translations)

    assert comparator._translation_map(capture, fine) == {0: 1, 1: 0}


def test_translation_map_rejects_rounded_or_duplicated_vectors():
    translations = np.zeros(2, dtype=validator.TRANSLATION_DTYPE)
    translations["x"] = [0.0, 0.0]
    translations["y"] = [0.0, 0.0]
    capture = SimpleNamespace(stack_index=17, translations=translations)

    with pytest.raises(ValueError, match="translation-vector alignment"):
        comparator._translation_map(capture, np.asarray([[0.0, 0.0], [1.0, 0.0]]))


def test_pixel_rows_use_centered_packed_y_coordinates():
    pixels = np.zeros(12, dtype=validator.PIXEL_DTYPE)
    pixels["x"] = np.tile(np.arange(3), 4)
    pixels["y"] = np.repeat([0, 1, 2, -1], 3)
    capture = SimpleNamespace(stack_index=17, pixels=pixels)
    compact = np.asarray(
        [
            (comparator.PHYSICAL_IMAGE_SIZE // 2) * (comparator.PHYSICAL_IMAGE_SIZE // 2 + 1),
            (comparator.PHYSICAL_IMAGE_SIZE // 2 - 1) * (comparator.PHYSICAL_IMAGE_SIZE // 2 + 1) + 2,
        ]
    )

    rows = comparator._pixel_rows(capture, compact)

    np.testing.assert_array_equal(rows, [0, 11])


def test_relion_rotation_matrix_converts_column_major_capture_layout():
    rotations = np.zeros(1, dtype=validator.ROTATION_DTYPE)
    rotations["matrix"][0] = np.arange(9, dtype=np.float32)
    capture = SimpleNamespace(rotations=rotations)

    np.testing.assert_array_equal(
        comparator._relion_rotation_matrix(capture, 0),
        np.arange(9, dtype=np.float32).reshape(3, 3).T,
    )
