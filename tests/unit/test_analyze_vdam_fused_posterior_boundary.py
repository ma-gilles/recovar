import numpy as np

from scripts.analyze_vdam_fused_posterior_boundary import compare_posteriors


def test_compare_fused_posterior_maps_rotation_rows_and_support():
    identity = np.eye(3, dtype=np.float32)
    quarter_turn = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    posterior = np.zeros((1, 2, 3), dtype=np.float32)
    posterior[0, 1, 2] = 0.75
    posterior[0, 0, 1] = 0.25
    reconstruction_mask = np.zeros_like(posterior, dtype=bool)
    reconstruction_mask[0, 1, 2] = True

    report = compare_posteriors(
        native_rotation_ids=np.array([0, 1], dtype=np.int32),
        native_translation_ids=np.array([2, 1], dtype=np.int32),
        native_rotation_matrices=np.stack([identity, quarter_turn]),
        native_unnormalized_weights=np.array([3.0, 1.0]),
        native_sum_weight=4.0,
        native_reconstruction_mask=np.array([True, False]),
        live={
            "local_rotation_matrices": np.stack([quarter_turn, identity]),
            "posterior": posterior,
            "reconstruction_sample_mask": reconstruction_mask,
        },
    )

    assert report["argmax_equal"]
    assert report["native_best_mapped_key"] == [1, 2]
    assert report["recovar_best_key"] == [1, 2]
    assert report["probability_l1"] == 0.0
    assert report["probability_relative_l2"] == 0.0
    assert report["reconstruction_mask_on_native_equal"]
