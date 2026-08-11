import numpy as np

from scripts.analyze_k1_direct_wavg_norm_panel import (
    _direct_wavg_pixel_sum,
    _direct_wavg_posterior_pixel_sum,
    _inferred_average_norm,
)


def test_direct_wavg_pixel_sum_matches_explicit_float32_order():
    projection = np.asarray([1 + 2j, 3 + 4j], dtype=np.complex64)
    translated = np.asarray([0.5 + 1j, -2 + 3j], dtype=np.complex64)
    ctf = np.asarray([2, -0.5], dtype=np.float32)
    expected_pixels = []
    for proj, image, factor in zip(projection, translated, ctf, strict=True):
        ref_real = np.float32(np.float32(proj.real) * factor)
        ref_imag = np.float32(np.float32(proj.imag) * factor)
        diff_real = np.float32(ref_real - np.float32(image.real))
        diff_imag = np.float32(ref_imag - np.float32(image.imag))
        expected_pixels.append(
            np.float32(np.float32(diff_real * diff_real) + np.float32(diff_imag * diff_imag))
        )

    result = _direct_wavg_pixel_sum(projection, translated, ctf)

    assert result == float(np.sum(np.asarray(expected_pixels, dtype=np.float32), dtype=np.float64))


def test_inferred_average_norm_inverts_relion_scale_factor():
    particle_norm_squared = 32.0
    particle_norm = np.sqrt(2.0 * particle_norm_squared)
    average_norm = 20.0
    native_factor = average_norm / particle_norm

    assert _inferred_average_norm(native_factor, particle_norm_squared) == average_norm


def test_direct_wavg_posterior_pixel_sum_accumulates_float32_states():
    projections = np.asarray([[1 + 2j], [3 - 1j]], dtype=np.complex64)
    translated = np.asarray([[0.5 + 1j], [-2 + 3j]], dtype=np.complex64)
    ctf = np.asarray([2], dtype=np.float32)
    probabilities = np.asarray([[0.25, 0.0], [0.5, 0.25]], dtype=np.float32)
    expected = np.float32(0)
    for rotation_row in range(2):
        rotation_total = np.float32(0)
        for translation_row in range(2):
            weight = probabilities[rotation_row, translation_row]
            if weight == 0:
                continue
            proj = projections[rotation_row, 0]
            image = translated[translation_row, 0]
            ref_real = np.float32(np.float32(proj.real) * ctf[0])
            ref_imag = np.float32(np.float32(proj.imag) * ctf[0])
            diff_real = np.float32(ref_real - np.float32(image.real))
            diff_imag = np.float32(ref_imag - np.float32(image.imag))
            diff2 = np.float32(
                np.float32(diff_real * diff_real) + np.float32(diff_imag * diff_imag)
            )
            rotation_total = np.float32(rotation_total + np.float32(weight * diff2))
        expected = np.float32(expected + rotation_total)

    assert _direct_wavg_posterior_pixel_sum(
        projections,
        translated,
        ctf,
        probabilities,
    ) == float(expected)
