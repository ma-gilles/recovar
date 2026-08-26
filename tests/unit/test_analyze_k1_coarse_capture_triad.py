import numpy as np

from scripts.analyze_k1_coarse_capture_triad import (
    _arm_summary,
    _coordinate,
    _score_components,
    _subtract_components,
)
from scripts.analyze_k1_coarse_operand_swap import _restore_square_references, _shapley
from scripts.analyze_k1_native_coarse_image_boundary import (
    _metric,
    _scalar_fit,
)
from scripts.analyze_k1_native_coarse_image_boundary import (
    analyze as analyze_native_coarse_image,
)
from scripts.analyze_k1_native_verbose_coarse_boundary import (
    _native_to_recovar_rotation,
)
from scripts.analyze_k1_native_verbose_coarse_boundary import (
    analyze as analyze_native_verbose_coarse,
)


def _payload():
    raw = np.asarray([[[4.0, 3.0], [2.0, 1.0]]], dtype=np.float32)
    prior = np.asarray([[[0.0, 0.25], [0.5, 0.75]]], dtype=np.float32)
    return {
        "weights_full": np.asarray([0.6, 0.25, 0.1, 0.05], dtype=np.float64),
        "significant_mask": np.asarray([True, True, False, False]),
        "n_significant": np.asarray(2),
        "max_posterior": np.asarray(0.6),
        "hard_assignment": np.asarray(0),
        "scores_pre_prior_per_class": raw,
        "scores_with_prior_per_class": raw + prior,
    }


def test_arm_summary_tracks_stable_ranks_and_masses():
    result = _arm_summary(
        _payload(),
        reference_top_count=2,
        tracked_flat_indices={"winner": 0, "excluded": 2},
    )

    assert result["own_significant_mass"] == 0.85
    assert result["top_reference_count_mass"] == 0.85
    assert result["tracked"]["winner"] == {
        "flat_index": 0,
        "rank": 1,
        "posterior": 0.6,
        "selected": True,
    }
    assert result["tracked"]["excluded"]["rank"] == 3


def test_score_components_separate_raw_and_prior_margins():
    result = _score_components(_payload(), flat_index=2, anchor_flat_index=0)

    assert result == {
        "raw": 2.0,
        "prior": 0.5,
        "total": 2.5,
        "raw_margin_to_exact_winner": -2.0,
        "prior_margin_to_exact_winner": 0.5,
        "total_margin_to_exact_winner": -1.5,
    }
    assert _subtract_components(result, {name: value + 0.25 for name, value in result.items()}) == {
        name: 0.25 for name in result
    }


def test_coordinate_uses_rotation_major_flattening():
    assert _coordinate(738414, 29) == (25462, 16)


def test_shapley_exactly_attributes_additive_three_factor_change():
    coefficients = {"image": 1.0, "weight": 2.0, "initial_diff2": 4.0}
    values = {
        frozenset(subset): 10.0 + sum(coefficients[factor] for factor in subset)
        for size in range(4)
        for subset in __import__("itertools").combinations(coefficients, size)
    }

    assert _shapley(values) == coefficients


def test_restore_square_references_matches_score_pixel_topology():
    image_shape = (8, 8)
    current_size = 6
    from recovar.em.dense_single_volume.helpers.fourier_window import (
        make_fourier_window_indices_np,
    )
    from recovar.em.dense_single_volume.helpers.significance import (
        _compact_projection_window_positions,
    )

    score_indices, _ = make_fourier_window_indices_np(
        image_shape,
        current_size,
        square=True,
        include_dc=False,
    )
    active_indices, _ = make_fourier_window_indices_np(
        image_shape,
        current_size,
        square=False,
        include_dc=False,
    )
    active_positions = _compact_projection_window_positions(score_indices, active_indices)
    compact = np.arange(active_positions.size, dtype=np.float32).astype(np.complex64)[None, :]

    restored = _restore_square_references(
        compact,
        image_shape=image_shape,
        current_size=current_size,
        score_indices=score_indices,
    )

    assert restored.shape == (1, score_indices.size)
    np.testing.assert_array_equal(restored[:, active_positions], compact)
    np.testing.assert_array_equal(
        restored[:, np.setdiff1d(np.arange(score_indices.size), active_positions)],
        0,
    )


def test_native_coarse_metric_counts_float32_components_and_ulps():
    reference = np.asarray([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex64)
    candidate = reference.copy()
    candidate.real[0] = np.nextafter(candidate.real[0], np.float32(np.inf))

    result = _metric(reference, candidate)

    assert result["component_count"] == 4
    assert result["bit_exact_component_count"] == 3
    assert result["mismatch_component_count"] == 1
    assert result["first_mismatch_component"] == 0
    assert result["max_ulp"] == 1


def test_native_coarse_scalar_fit_removes_real_scale():
    reference = np.asarray([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex64)
    candidate = np.asarray(np.float32(1.25) * reference, dtype=np.complex64)

    result = _scalar_fit(reference, candidate)

    assert result["candidate_over_reference_real_alpha"] == 1.25
    assert result["alpha_minus_one"] == 0.25
    assert result["residual_relative_l2_over_reference"] == 0.0


def test_native_coarse_image_analyzer_closes_unit_and_pixel_mapping(tmp_path):
    from recovar.em.dense_single_volume.helpers.fourier_window import (
        make_fourier_window_indices_np,
    )
    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _relion_cuda_fine_full_to_compact_lookup,
    )

    physical_size = 8
    current_size = 6
    native_count = current_size * (current_size // 2 + 1)
    score_indices, _ = make_fourier_window_indices_np(
        (physical_size, physical_size),
        current_size,
        square=True,
        include_dc=True,
    )
    lookup = _relion_cuda_fine_full_to_compact_lookup(
        (physical_size, physical_size),
        current_size,
        score_indices,
    )
    assert score_indices.size == native_count
    assert np.all(lookup >= 0)

    native_raw = (
        np.linspace(-0.2, 0.3, native_count, dtype=np.float32)
        + np.complex64(1j)
        * np.linspace(0.4, -0.1, native_count, dtype=np.float32)
    ).astype(np.complex64)
    native_raw_weight = np.linspace(1.0, 2.0, native_count, dtype=np.float32)
    recovar_image_native_order = np.asarray(
        -np.float32(physical_size**2) * native_raw,
        dtype=np.complex64,
    )
    recovar_weight_native_order = np.asarray(
        native_raw_weight / np.float32(physical_size**4),
        dtype=np.float32,
    )
    recovar_image = np.empty(native_count, dtype=np.complex64)
    recovar_weight = np.empty(native_count, dtype=np.float32)
    recovar_image[lookup] = recovar_image_native_order
    recovar_weight[lookup] = recovar_weight_native_order

    native_dir = tmp_path / "native_image"
    native_dir.mkdir()

    def write_real(name, values):
        values = np.asarray(values, dtype="<f8")
        with (native_dir / name).open("wb") as stream:
            stream.write(np.asarray(values.size, dtype="<i4").tobytes())
            stream.write(values.tobytes())

    write_real("pass1_img0_Fimg_corrected_real.bin", native_raw.real)
    write_real("pass1_img0_Fimg_corrected_imag.bin", native_raw.imag)
    write_real("pass1_img0_corr_img.bin", native_raw_weight)

    def write_capture(path):
        np.savez(
            path,
            current_size=np.asarray(current_size),
            original_index=np.asarray(7),
            coarse_gaussian_score_indices=score_indices,
            coarse_gaussian_unshifted_corrected=recovar_image,
            coarse_gaussian_pixel_weight=recovar_weight,
        )

    exact_path = tmp_path / "exact.npz"
    live_path = tmp_path / "live.npz"
    write_capture(exact_path)
    write_capture(live_path)

    result = analyze_native_coarse_image(
        native_dump_dir=native_dir,
        exact_path=exact_path,
        live_path=live_path,
        physical_image_size=physical_size,
    )

    assert result["corrected_image"]["metrics"]["native_vs_exact"]["mismatch_component_count"] == 0
    assert result["corrected_image"]["metrics"]["native_vs_live"]["relative_l2"] == 0.0
    assert result["corrected_image"]["comparison_qualification"].startswith("unqualified")
    assert result["corrected_image"]["closer_arm_by_relative_l2_unqualified"] == "exact"
    assert result["pixel_weight"]["metrics"]["native_vs_exact"]["mismatch_component_count"] == 0


def test_native_verbose_rotation_mapping_is_direction_to_psi_major():
    native = np.asarray([0, 1, 47, 48, 5697, 5987], dtype=np.int64)

    mapped = _native_to_recovar_rotation(native, n_directions=768, n_psi=48)

    np.testing.assert_array_equal(mapped, [0, 768, 36096, 1, 25462, 27004])


def test_native_verbose_coarse_analyzer_closes_identical_boundary(tmp_path):
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    n_directions = 2
    n_psi = 2
    n_rot = n_directions * n_psi
    n_trans = 3
    native_rotation = np.repeat(np.arange(n_rot, dtype=np.int32), n_trans)
    translation = np.tile(np.arange(n_trans, dtype=np.int32), n_rot)
    raw = np.linspace(1.0, 2.1, n_rot * n_trans, dtype=np.float32)
    total = np.linspace(-2.0, -0.9, n_rot * n_trans, dtype=np.float32)
    probability = np.arange(1, n_rot * n_trans + 1, dtype=np.float32)
    probability /= probability.sum(dtype=np.float32)
    selected = probability >= np.sort(probability)[-3]

    def write_flat(name, values, dtype):
        path = native_dir / f"{name}.bin"
        values = np.asarray(values, dtype=dtype)
        with path.open("wb") as stream:
            stream.write(np.asarray(values.size, dtype="<i4").tobytes())
            stream.write(values.tobytes())

    write_flat("pass0_coarse_candidate_rot_idx", native_rotation, "<i4")
    write_flat("pass0_coarse_candidate_trans_idx", translation, "<i4")
    write_flat("pass0_coarse_candidate_weight_normalized", probability, "<f8")
    write_flat("pass0_coarse_raw_diff2", raw, "<f8")
    write_flat("pass0_coarse_log_weight_preexp", total, "<f8")
    write_flat("pass0_coarse_candidate_in_threshold_set", selected, "<i4")

    recovar_rotation = _native_to_recovar_rotation(
        native_rotation,
        n_directions=n_directions,
        n_psi=n_psi,
    )
    flat = recovar_rotation * n_trans + translation

    def mapped(values, dtype):
        output = np.empty(n_rot * n_trans, dtype=dtype)
        output[flat] = values
        return output.reshape(n_rot, n_trans)

    recovar_path = tmp_path / "recovar.npz"
    np.savez(
        recovar_path,
        current_size=np.asarray(6),
        original_index=np.asarray(7),
        n_rot=np.asarray(n_rot),
        n_trans=np.asarray(n_trans),
        scores_pre_prior_per_class=mapped(-raw, np.float32)[None],
        scores_with_prior_per_class=mapped(total, np.float32)[None],
        weights_per_class=mapped(probability, np.float32)[None].reshape(1, -1),
        significant_mask=mapped(selected, bool).reshape(-1),
        n_significant=np.asarray(np.count_nonzero(selected)),
        hard_assignment=np.asarray(int(np.argmax(mapped(probability, np.float32)))),
    )

    result = analyze_native_verbose_coarse(
        native_dump_dir=native_dir,
        recovar_path=recovar_path,
        n_directions=n_directions,
        n_psi=n_psi,
        target_rotation=1,
        target_translation=2,
    )

    assert result["summary"]["support_mismatch_count"] == 0
    assert result["summary"]["posterior_total_variation"] == 0.0
    assert result["residuals"]["raw_score_centered"]["max_abs"] == 0.0
    assert result["residuals"]["total_log_weight_centered"]["max_abs"] == 0.0
