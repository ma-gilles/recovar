"""Tests for CryoEMDataset repr/slots and halfset splitting."""

import numpy as np
import pytest

from helpers.tiny_synthetic import make_tiny_cryo_dataset
from recovar.data_io.halfsets import _read_relion_halfsets_from_star

pytestmark = pytest.mark.unit


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
            f.write(f"{i + 1:06d}@particles.mrcs {s}\n")


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
