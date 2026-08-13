from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
    make_frequency_coords_half_np,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _make_relion_wavg_rectangle,
    _prioritize_stopped_pass2_dump_buckets,
    _relion_wavg_atomic_triplet_terms,
    _relion_wavg_direct_modes,
    _relion_wavg_direct_norm_per_image,
    _relion_wavg_rectangle_triplet_terms,
    _replace_low_shell_noise_with_relion_wavg_direct_residual,
)
from scripts.analyze_k1_scale_aa_pixels import analyze


def test_wavg_direct_noise_only_is_independent_from_direct_norm(monkeypatch):
    monkeypatch.delenv("RECOVAR_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL", raising=False)
    monkeypatch.setenv("RECOVAR_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY", "1")

    assert _relion_wavg_direct_modes(
        accumulate_noise=True,
        scale_groups_available=True,
        scale_aa_enabled=True,
    ) == (True, False)
    # Fresh iteration 1 has no scale-group accumulator, so the stopped arm is
    # intentionally dormant rather than changing first-iteration behavior.
    assert _relion_wavg_direct_modes(
        accumulate_noise=True,
        scale_groups_available=False,
        scale_aa_enabled=False,
    ) == (False, False)


def test_wavg_direct_modes_reject_overlapping_factorial_arms(monkeypatch):
    monkeypatch.setenv("RECOVAR_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL", "1")
    monkeypatch.setenv("RECOVAR_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY", "1")

    with pytest.raises(ValueError, match="mutually exclusive"):
        _relion_wavg_direct_modes(
            accumulate_noise=True,
            scale_groups_available=True,
            scale_aa_enabled=True,
        )


def test_wavg_direct_residual_preserves_coupled_noise_and_norm(monkeypatch):
    monkeypatch.setenv("RECOVAR_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL", "1")
    monkeypatch.delenv("RECOVAR_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY", raising=False)

    assert _relion_wavg_direct_modes(
        accumulate_noise=True,
        scale_groups_available=True,
        scale_aa_enabled=True,
    ) == (True, True)


def test_stopped_pass2_dump_prioritizes_only_requested_bucket(monkeypatch):
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET", "1")
    monkeypatch.setattr(
        "recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed._pass2_dump_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed._pass2_dump_requested_for_bucket",
        lambda **kwargs: int(np.asarray(kwargs["image_indices"])[0]) == 7,
    )
    buckets = [
        {"image_indices": np.asarray([3])},
        {"image_indices": np.asarray([7])},
        {"image_indices": np.asarray([11])},
    ]

    prioritized = _prioritize_stopped_pass2_dump_buckets(
        buckets,
        experiment_dataset=object(),
        current_size=80,
    )

    assert [int(bucket["image_indices"][0]) for bucket in prioritized] == [7, 3, 11]


def test_nonstopped_pass2_dump_preserves_bucket_order(monkeypatch):
    monkeypatch.delenv("RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET", raising=False)
    monkeypatch.setattr(
        "recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed._pass2_dump_enabled",
        lambda: True,
    )
    buckets = [
        {"image_indices": np.asarray([3])},
        {"image_indices": np.asarray([7])},
    ]

    preserved = _prioritize_stopped_pass2_dump_buckets(
        buckets,
        experiment_dataset=object(),
        current_size=80,
    )

    assert preserved is buckets


def test_stopped_norm_residual_dump_prioritizes_requested_bucket(monkeypatch):
    monkeypatch.setattr(
        "recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed._pass2_dump_enabled",
        lambda: False,
    )
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS", "1")
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_NORM_RESIDUAL_STOP_AFTER_TARGET", "1")
    monkeypatch.setattr(
        "recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed._pass2_dump_requested_for_bucket",
        lambda **kwargs: int(np.asarray(kwargs["image_indices"])[0]) == 7,
    )
    buckets = [
        {"image_indices": np.asarray([3])},
        {"image_indices": np.asarray([7])},
        {"image_indices": np.asarray([11])},
    ]

    prioritized = _prioritize_stopped_pass2_dump_buckets(
        buckets,
        experiment_dataset=object(),
        current_size=60,
    )

    assert [int(bucket["image_indices"][0]) for bucket in prioritized] == [7, 3, 11]


def test_wavg_atomic_triplet_preserves_scale_units_and_forms_raw_diff2():
    proj = jnp.asarray([[[2.0 + 1.0j, 3.0 - 2.0j]]], dtype=jnp.complex64)
    shifted = jnp.asarray(
        [[[1.0 + 2.0j, 2.0 + 1.0j], [0.0 + 1.0j, 1.0 - 1.0j]]],
        dtype=jnp.complex64,
    )
    posterior = jnp.asarray([[[0.25, 0.75]]], dtype=jnp.float32)
    summed = jnp.einsum("brt,btp->brp", posterior, shifted)
    ctf_posterior = jnp.asarray([[[0.5, 0.25]]], dtype=jnp.float32)
    noise = jnp.asarray([2.0, 4.0], dtype=jnp.float32)
    scale = jnp.asarray([2.0], dtype=jnp.float32)

    result = np.asarray(
        _relion_wavg_atomic_triplet_terms(
            proj,
            jnp.abs(proj) ** 2,
            summed,
            ctf_posterior,
            noise,
            scale,
            shifted,
            posterior,
        )
    )

    image_power = np.sum(
        np.asarray(posterior)[..., None] * np.abs(np.asarray(shifted))[:, None, :, :] ** 2,
        axis=2,
        dtype=np.float32,
    )
    aa_raw = np.asarray(jnp.abs(proj) ** 2) * np.asarray(ctf_posterior) * np.asarray(noise)[None, None]
    xa_raw = np.asarray(noise)[None, None] * np.real(
        np.asarray(proj) * np.conj(np.asarray(summed))
    )
    expected_diff2 = (image_power + aa_raw - 2.0 * xa_raw).astype(np.float32)

    np.testing.assert_allclose(
        result[..., 0],
        xa_raw / 2.0,
    )
    np.testing.assert_allclose(
        result[..., 1],
        aa_raw / 4.0,
    )
    np.testing.assert_allclose(result[..., 2], expected_diff2)


def test_direct_wavg_residual_replaces_only_complete_low_shells():
    residual = np.asarray([100.0, 200.0, 300.0, 400.0])
    image_power = np.asarray([10.0, 20.0, 30.0, 40.0])
    shell_indices = np.asarray([0, 1, 2, 1, 3], dtype=np.int32)
    atomic_diff2 = np.asarray(
        [
            [1.0, 2.0, 1000.0, 3.0, 10000.0],
            [4.0, 5.0, 2000.0, 6.0, 20000.0],
        ],
        dtype=np.float32,
    )

    replaced_residual, replaced_image_power = (
        _replace_low_shell_noise_with_relion_wavg_direct_residual(
            residual,
            image_power,
            atomic_diff2,
            shell_indices,
            exclusive_shell_stop=2,
        )
    )

    # Shells 0 and 1 use fused residuals in image-major, pixel-major order.
    np.testing.assert_array_equal(replaced_residual, [5.0, 16.0, 300.0, 400.0])
    np.testing.assert_array_equal(replaced_image_power, [0.0, 0.0, 30.0, 40.0])
    # The partially represented cutoff shell (2) and all higher shells remain
    # on the original algebraic path despite large direct diagnostic values.
    np.testing.assert_array_equal(residual, [100.0, 200.0, 300.0, 400.0])
    np.testing.assert_array_equal(image_power, [10.0, 20.0, 30.0, 40.0])


def test_relion_wavg_rectangle_matches_native_size60_topology_and_order():
    image_shape = (256, 256)
    current_size = 60
    exact_indices, _ = make_fourier_window_indices_np(
        image_shape,
        current_size,
        include_dc=True,
        exact_radius=True,
    )

    layout = _make_relion_wavg_rectangle(
        image_shape,
        current_size,
        exact_indices,
    )

    assert layout.centered_indices.size == 60 * 31 == 1860
    assert layout.exact_positions.size == 1411
    assert np.count_nonzero(layout.shell_indices >= 0) == 1462
    assert np.count_nonzero(layout.shell_indices < 0) == 398
    assert np.count_nonzero(
        (layout.shell_indices >= 0)
        & ~np.isin(np.arange(layout.shell_indices.size), layout.exact_positions)
    ) == 51
    # Native FFTW row-major order starts at ky=0 and walks kx=0..N/2.
    half_width = image_shape[1] // 2 + 1
    expected_first_row = image_shape[0] // 2 * half_width + np.arange(31)
    np.testing.assert_array_equal(layout.centered_indices[:31], expected_first_row)
    np.testing.assert_array_equal(
        layout.centered_indices[layout.exact_positions],
        exact_indices,
    )


def test_relion_wavg_rectangle_terms_keep_image_only_pixels_in_issue_stream():
    exact_terms = jnp.asarray(
        [
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    shifted = jnp.asarray(
        [
            [
                [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j, 5.0 + 0.0j],
                [0.0 + 1.0j, 0.0 + 2.0j, 0.0 + 3.0j, 0.0 + 4.0j, 0.0 + 5.0j],
            ]
        ],
        dtype=jnp.complex64,
    )
    posterior = jnp.asarray([[[0.25, 0.75], [0.5, 0.5]]], dtype=jnp.float32)
    exact_positions = jnp.asarray([1, 3], dtype=jnp.int32)

    result = np.asarray(
        _relion_wavg_rectangle_triplet_terms(
            exact_terms,
            shifted,
            posterior,
            exact_positions,
        )
    )

    np.testing.assert_array_equal(result[:, :, exact_positions, :], np.asarray(exact_terms))
    image_only = np.asarray([0, 2, 4])
    np.testing.assert_array_equal(result[:, :, image_only, :2], 0.0)
    shifted_power = np.abs(np.asarray(shifted)) ** 2
    expected_power = np.einsum("brt,btp->brp", np.asarray(posterior), shifted_power)
    np.testing.assert_allclose(result[:, :, image_only, 2], expected_power[:, :, image_only])


def test_relion_wavg_direct_norm_uses_valid_pixels_then_high_shell_power():
    atomic_diff2 = np.asarray(
        [[1.0, 1000.0, 2.0, 3.0], [4.0, 2000.0, 5.0, 6.0]],
        dtype=np.float32,
    )
    shells = np.asarray([0, -1, 2, 1], dtype=np.int32)
    high_shell = np.asarray([10.0, 20.0], dtype=np.float64)

    result = _relion_wavg_direct_norm_per_image(
        atomic_diff2,
        shells,
        high_shell,
    )

    np.testing.assert_array_equal(result, [16.0, 35.0])


def test_scale_aa_pixels_joins_fourier_coordinates_and_localizes_operand_delta(tmp_path: Path):
    image_size = 8
    current_size = 4
    divisor = 16.0
    window_indices, _ = make_fourier_window_indices_np(
        (image_size, image_size),
        current_size,
        square=False,
        include_dc=True,
        exact_radius=True,
    )
    coordinates = np.rint(make_frequency_coords_half_np((image_size, image_size))).astype(np.int32)[
        window_indices
    ]
    shells = np.rint(np.linalg.norm(coordinates, axis=1)).astype(np.int32)
    mask = shells <= 1
    aa_native = np.arange(1, window_indices.size + 1, dtype=np.float64) / 100.0
    xa_native = np.arange(2, window_indices.size + 2, dtype=np.float64) / 200.0
    aa_recovar = aa_native * divisor
    aa_recovar[np.flatnonzero(mask)[-1]] *= 1.01
    aa_shell = np.asarray(
        [np.sum(aa_recovar[shells == shell], dtype=np.float64) for shell in range(current_size // 2 + 1)]
    )
    native_direct = np.asarray([0.25, 0.5, 0.75], dtype=np.float64)

    capture = tmp_path / "capture.npz"
    np.savez_compressed(
        capture,
        schema=np.asarray("recovar-k1-scale-xa-aa-chunked-v2"),
        iteration=np.int64(2),
        half=np.int64(1),
        original_index=np.int64(1096),
        group_id=np.int64(109),
        current_size=np.int64(current_size),
        scale_correction_pixel_mask=mask,
        scale_shell_indices=shells,
        scale_aa_per_pixel=aa_recovar.astype(np.float32),
        scale_aa_per_shell=aa_shell,
        scale_aa_atomic_per_pixel=(aa_native * divisor).astype(np.float32),
        scale_xa_per_pixel=(xa_native * divisor * 1.02).astype(np.float32),
        scale_xa_atomic_per_pixel=(xa_native * divisor).astype(np.float32),
        wavg_diff2_atomic_per_shell=(native_direct * divisor).astype(np.float32),
    )

    native = tmp_path / "native.tsv"
    lines = []
    for row in np.flatnonzero(mask):
        x, y = coordinates[row]
        lines.append(
            "acc_scale_pixel\titer=2\tpart_id=109\thalfset=1"
            f"\tj={row}\tx={x}\ty={y}\tshell={shells[row]}"
            f"\taa={aa_native[row]:.17g}\txa={xa_native[row]:.17g}\n"
        )
    native.write_text("".join(reversed(lines)))
    native_components = tmp_path / "native_components.tsv"
    native_components.write_text(
        "".join(
            "acc_components\titer=2\tpart_id=109\thalfset=1"
            f"\tshell={shell}\tdirect_residual={value:.17g}\n"
            for shell, value in enumerate(native_direct)
        )
    )

    report = analyze(
        capture,
        native,
        native_noise_components=native_components,
        expected_iteration=2,
        expected_half=1,
        expected_part_id=109,
        expected_original_index=1096,
        image_size=image_size,
        recovar_term_divisor=divisor,
    )

    assert report["coordinate_join"]["shell_labels_exact"]
    assert report["pixel_aa"]["relative_l2"] > 0.0
    assert report["atomic_aa"]["pixel"]["relative_l2"] < 1e-7
    assert report["atomic_aa"]["fixed_order_shell_reduction"]["relative_l2"] < 1e-7
    assert report["xa"]["pixel"]["relative_l2"] > 0.0
    assert report["xa"]["atomic"]["pixel"]["relative_l2"] < 1e-7
    assert report["xa"]["atomic"]["fixed_order_shell_reduction"]["relative_l2"] < 1e-7
    assert report["wavg_direct_residual"]["relative_l2"] < 1e-7
    assert report["classification"] == "atomic Wavg XA/AA treatment captured"
    assert report["pixel_aa"]["largest_abs_residual_pixels"][0]["x"] == int(
        coordinates[np.flatnonzero(mask)[-1], 0]
    )
