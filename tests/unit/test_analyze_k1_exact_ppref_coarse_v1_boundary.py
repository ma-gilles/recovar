import numpy as np
import pytest

from scripts import analyze_k1_exact_ppref_coarse_v1_boundary as analyzer
from scripts.analyze_k1_exact_ppref_coarse_v1_boundary import _candidate_record


@pytest.mark.unit
def test_candidate_record_centers_only_over_active_native_candidates():
    active = np.asarray([[True, True], [False, True]])
    native = np.asarray([[-3.0, -5.0], [np.finfo(np.float32).max, -4.0]])
    recorded = np.asarray([[-3.0, -5.25], [np.finfo(np.float32).max, -4.0]])
    native_texture = np.asarray([[-3.0, -5.125], [np.finfo(np.float32).max, -4.0]])
    model_window_texture = native_texture.copy()
    preprojected = native_texture.copy()
    mask = np.asarray([[True, False], [False, True]])

    record = _candidate_record(
        (0, 1),
        native_scores=native,
        recorded_scores=recorded,
        native_texture_scores=native_texture,
        model_window_texture_scores=model_window_texture,
        preprojected_scores=preprojected,
        active=active,
        native_mask=mask,
        recorded_mask=mask,
        native_texture_mask=mask,
        model_window_texture_mask=mask,
        preprojected_mask=mask,
    )

    assert record["native_centered_raw_score"] == -2.0
    assert record["recorded_minus_native"] == -0.25
    assert record["exact_ppref_native_texture_minus_native"] == -0.125
    assert record["exact_ppref_model_window_texture_minus_native"] == -0.125
    assert record["exact_ppref_preprojected_minus_native"] == -0.125


@pytest.mark.unit
def test_square_score_subset_extracts_native_model_window_from_optics_plus_two():
    physical_size = 8
    half_width = physical_size // 2 + 1
    ky = np.arange(-3, 4, dtype=np.int64)
    kx = np.arange(4, dtype=np.int64)
    indices = (
        (ky[:, None] + physical_size // 2) * half_width + kx[None, :]
    ).reshape(-1)

    subset = analyzer._square_score_subset(
        indices,
        physical_image_size=physical_size,
        score_size=4,
    )

    selected_ky = ky.repeat(kx.size)[subset]
    selected_kx = np.tile(kx, ky.size)[subset]
    assert selected_ky.tolist() == [-1] * 3 + [0] * 3 + [1] * 3 + [2] * 3
    assert selected_kx.tolist() == [0, 1, 2] * 4
