import pytest

from recovar.cuda_backproject import _infer_backproject_upsampling, _project_ffi_kwargs
from recovar.em.dense_single_volume.helpers.fourier_window import centered_half_indices_to_fftw_half_indices

pytestmark = pytest.mark.unit


def test_infer_upsampling_accepts_standard_even_grid():
    assert _infer_backproject_upsampling((128, 128), (256, 256, 256)) == 2


def test_infer_upsampling_accepts_final_all_data_relion_pad_without_max_r():
    assert _infer_backproject_upsampling((128, 128), (259, 259, 259), max_r=None) == 2


@pytest.mark.parametrize(
    "image_shape,volume_shape,max_r,expected_ups",
    [
        ((128, 128), (259, 259, 259), 64, 2),
        ((128, 128), (259, 259, 259), 28, 2),
        ((128, 128), (131, 131, 131), 32, 2),
        ((128, 128), (123, 123, 123), 30, 2),
        ((64, 64), (9, 9, 9), 2.5, 1),
    ],
)
def test_infer_upsampling_accepts_relion_odd_pad_size(image_shape, volume_shape, max_r, expected_ups):
    assert _infer_backproject_upsampling(image_shape, volume_shape, max_r=max_r) == expected_ups


def test_infer_upsampling_rejects_unexplained_odd_grid():
    with pytest.raises(ValueError, match="RELION pad_size"):
        _infer_backproject_upsampling((128, 128), (257, 257, 257))


def test_centered_half_indices_to_fftw_half_indices_remaps_rows_only():
    image_shape = (128, 128)
    half_width = 65
    centered = [
        64 * half_width + 0,  # ky=0 -> FFTW row 0
        36 * half_width + 5,  # ky=-28 -> FFTW row 100
        92 * half_width + 7,  # ky=28 -> FFTW row 28
    ]

    mapped = centered_half_indices_to_fftw_half_indices(image_shape, centered)

    assert mapped.tolist() == [
        0 * half_width + 0,
        100 * half_width + 5,
        28 * half_width + 7,
    ]


def test_relion_texture_uses_one_canonical_boolean_ffi_mode(monkeypatch):
    args = ((56, 56), (115, 115, 115), 1, False, True, 28)

    # The retired diagnostic environment variable must not create a hidden
    # third mode: True always means the production RELION texture convention.
    monkeypatch.setenv("RECOVAR_RELION_TEXTURE_POSITIVE_NYQUIST", "1")
    texture, _, _ = _project_ffi_kwargs(*args, relion_texture_interp=True)
    assert int(texture["relion_texture_interp"]) == 1

    non_texture, _, _ = _project_ffi_kwargs(*args, relion_texture_interp=False)
    assert int(non_texture["relion_texture_interp"]) == 0
