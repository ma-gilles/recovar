"""Tests for CryoEMHalfsets wrapper class and halfset splitting."""
import numpy as np
import pytest

from helpers.tiny_synthetic import make_tiny_cryo_dataset
from recovar.data_io.dataset import CryoEMHalfsets, _read_relion_halfsets_from_star

pytestmark = pytest.mark.unit


def test_halfsets_indexing():
    h1 = make_tiny_cryo_dataset(grid_size=4, n_images=4)
    h2 = make_tiny_cryo_dataset(grid_size=4, n_images=6)
    cryos = CryoEMHalfsets(h1, h2)

    assert cryos[0] is h1
    assert cryos[1] is h2
    assert len(cryos) == 2


def test_halfsets_iteration():
    h1 = make_tiny_cryo_dataset(grid_size=4, n_images=4)
    h2 = make_tiny_cryo_dataset(grid_size=4, n_images=6)
    cryos = CryoEMHalfsets(h1, h2)

    halves = list(cryos)
    assert halves[0] is h1
    assert halves[1] is h2

    # Can iterate multiple times
    count = 0
    for cryo in cryos:
        count += 1
    assert count == 2


def test_halfsets_shared_properties():
    h1 = make_tiny_cryo_dataset(grid_size=4, n_images=4)
    h2 = make_tiny_cryo_dataset(grid_size=4, n_images=6)
    cryos = CryoEMHalfsets(h1, h2)

    assert cryos.grid_size == 4
    assert cryos.image_shape == (4, 4)
    assert cryos.image_size == 16
    assert cryos.voxel_size == h1.voxel_size
    assert cryos.padding == h1.padding
    assert cryos.dtype == h1.dtype
    assert cryos.dtype_real == h1.dtype_real
    assert cryos.tilt_series_flag == h1.tilt_series_flag
    assert cryos.premultiplied_ctf == h1.premultiplied_ctf
    assert cryos.volume_mask_threshold == h1.volume_mask_threshold
    assert cryos.hpad == h1.hpad


def test_halfsets_aggregate_properties():
    h1 = make_tiny_cryo_dataset(grid_size=4, n_images=4)
    h2 = make_tiny_cryo_dataset(grid_size=4, n_images=6)
    cryos = CryoEMHalfsets(h1, h2)

    assert cryos.n_total_images == 10
    assert cryos.n_total_units == 10  # SPA: n_units == n_images


def test_halfsets_split_array():
    h1 = make_tiny_cryo_dataset(grid_size=4, n_images=4)
    h2 = make_tiny_cryo_dataset(grid_size=4, n_images=6)
    cryos = CryoEMHalfsets(h1, h2)

    arr = np.arange(10)
    parts = cryos.split_array(arr)
    assert len(parts) == 2
    np.testing.assert_array_equal(parts[0], np.arange(4))
    np.testing.assert_array_equal(parts[1], np.arange(4, 10))


def test_halfsets_delegated_methods():
    h1 = make_tiny_cryo_dataset(grid_size=4, n_images=4)
    h2 = make_tiny_cryo_dataset(grid_size=4, n_images=6)
    cryos = CryoEMHalfsets(h1, h2)

    mask_direct = h1.get_volume_radial_mask()
    mask_via_halfsets = cryos.get_volume_radial_mask()
    np.testing.assert_array_equal(mask_direct, mask_via_halfsets)

    valid_direct = h1.get_valid_frequency_indices()
    valid_via_halfsets = cryos.get_valid_frequency_indices()
    np.testing.assert_array_equal(valid_direct, valid_via_halfsets)


def test_halfsets_repr():
    h1 = make_tiny_cryo_dataset(grid_size=4, n_images=4)
    h2 = make_tiny_cryo_dataset(grid_size=4, n_images=6)
    cryos = CryoEMHalfsets(h1, h2)

    r = repr(cryos)
    assert "CryoEMHalfsets" in r
    assert "grid_size=4" in r
    assert "n_images=[4, 6]" in r


def test_cryo_em_dataset_repr():
    cryo = make_tiny_cryo_dataset(grid_size=4, n_images=8)
    r = repr(cryo)
    assert "CryoEMDataset" in r
    assert "n_images=8" in r
    assert "grid_size=4" in r
    assert "tilt_series=False" in r


def test_cryo_em_dataset_slots():
    """Verify __slots__ prevents accidental attribute creation."""
    cryo = make_tiny_cryo_dataset(grid_size=4, n_images=4)
    with pytest.raises(AttributeError):
        cryo.nonexistent_attribute = 42


# ---------------------------------------------------------------------------
# RELION halfset reading from star files
# ---------------------------------------------------------------------------

def _write_star_with_subsets(path, subsets):
    """Write a minimal star file with _rlnRandomSubset column."""
    with open(path, "w") as f:
        f.write("data_particles\n\nloop_\n")
        f.write("_rlnImageName\n")
        f.write("_rlnRandomSubset\n")
        for i, s in enumerate(subsets):
            f.write(f"{i+1:06d}@particles.mrcs {s}\n")


def test_read_relion_halfsets_from_star(tmp_path):
    star = tmp_path / "particles.star"
    _write_star_with_subsets(star, [1, 2, 1, 2, 1, 2])
    result, n_total = _read_relion_halfsets_from_star(str(star))
    assert result is not None
    assert n_total == 6
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], [0, 2, 4])
    np.testing.assert_array_equal(result[1], [1, 3, 5])


def test_read_relion_halfsets_no_column(tmp_path):
    star = tmp_path / "particles.star"
    with open(star, "w") as f:
        f.write("data_particles\n\nloop_\n_rlnImageName\n")
        f.write("000001@particles.mrcs\n")
    result, n_total = _read_relion_halfsets_from_star(str(star))
    assert result is None
    assert n_total == 1


def test_read_relion_halfsets_non_star():
    result, n_total = _read_relion_halfsets_from_star("particles.mrcs")
    assert result is None
    assert n_total is None


def test_read_relion_halfsets_with_ind_filter(tmp_path):
    import pickle
    star = tmp_path / "particles.star"
    _write_star_with_subsets(star, [1, 2, 1, 2, 1, 2])
    ind_file = tmp_path / "ind.pkl"
    with open(ind_file, "wb") as f:
        pickle.dump(np.array([0, 1, 2, 3], dtype=np.int32), f)
    result, n_total = _read_relion_halfsets_from_star(str(star), ind_file=str(ind_file))
    assert result is not None
    assert n_total == 6
    np.testing.assert_array_equal(result[0], [0, 2])
    np.testing.assert_array_equal(result[1], [1, 3])
