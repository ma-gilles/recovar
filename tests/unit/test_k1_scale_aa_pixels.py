from pathlib import Path

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
    make_frequency_coords_half_np,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_wavg_atomic_triplet_terms,
    _replace_low_shell_noise_with_relion_wavg_direct_residual,
)
from scripts.analyze_k1_scale_aa_pixels import analyze


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
