import numpy as np
import pytest

from scripts.analyze_vdam_storewavg_reference_decomposition import (
    _current_size_from_rectangle_size,
    _decompose_components,
    _translate_native_rectangle,
)


@pytest.mark.unit
def test_reference_decomposition_separates_posterior_and_projection_effects():
    image = np.asarray([1 + 2j, 2 - 1j, -1 + 0.5j, 0.25 - 0.75j], dtype=np.complex64)
    translated = np.stack((image, image * np.complex64(1j)))
    native_projection = np.asarray(
        [[1 + 0j, 2 + 1j, -1 + 2j, 0.5 - 0.5j]], dtype=np.complex64
    )
    candidate_projection = native_projection + np.complex64(0.125 - 0.0625j)
    ctf = np.asarray([1.0, -0.5, 0.75, 0.25], dtype=np.float32)
    native_posterior = np.asarray([[0.6, 0.4]], dtype=np.float32)
    candidate_posterior = np.asarray([[0.55, 0.45]], dtype=np.float32)
    mask = np.asarray([False, True, True, False])

    provisional = _decompose_components(
        native_projection,
        candidate_projection,
        translated,
        ctf,
        native_posterior,
        candidate_posterior,
        mask,
        {"xa": 0.0, "aa": 0.0},
        {"xa": 0.0, "aa": 0.0},
    )
    captured_reference = {
        name: provisional[name]["native_projection_native_posterior_replay"]
        for name in ("xa", "aa")
    }
    captured_candidate = {
        name: provisional[name]["candidate_projection_candidate_posterior_replay"]
        for name in ("xa", "aa")
    }
    report = _decompose_components(
        native_projection,
        candidate_projection,
        translated,
        ctf,
        native_posterior,
        candidate_posterior,
        mask,
        captured_reference,
        captured_candidate,
    )

    assert report["xa"]["posterior_effect"] != 0.0
    assert report["xa"]["reference_projection_effect"] != 0.0
    assert report["xa"]["unexplained_after_candidate_projection_replay"] == 0.0
    assert report["xa"]["decomposition_closure_error"] == 0.0
    assert report["aa"]["posterior_effect"] == 0.0
    assert report["aa"]["reference_projection_effect"] != 0.0
    assert report["aa"]["decomposition_closure_error"] == 0.0


@pytest.mark.unit
def test_native_rectangle_translation_and_size_inference():
    assert _current_size_from_rectangle_size(12) == 4
    image = np.asarray(
        [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j,
         7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j],
        dtype=np.complex64,
    )
    translated = _translate_native_rectangle(
        image,
        np.asarray([[0.0, 0.0], [np.pi / 2, 0.0]], dtype=np.float32),
        4,
    )

    np.testing.assert_array_equal(translated[0], image)
    assert translated[1, 0] == image[0]
    assert translated[1, 1].real == pytest.approx(0.0, abs=1.0e-6)
    assert translated[1, 1].imag == pytest.approx(2.0, abs=1.0e-6)
    with pytest.raises(ValueError, match="cannot infer current size"):
        _current_size_from_rectangle_size(13)
