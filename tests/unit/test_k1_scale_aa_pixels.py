from pathlib import Path

import numpy as np

from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
    make_frequency_coords_half_np,
)
from scripts.analyze_k1_scale_aa_pixels import analyze


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

    report = analyze(
        capture,
        native,
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
    assert report["classification"] == "atomic Wavg XA/AA treatment captured"
    assert report["pixel_aa"]["largest_abs_residual_pixels"][0]["x"] == int(
        coordinates[np.flatnonzero(mask)[-1], 0]
    )
