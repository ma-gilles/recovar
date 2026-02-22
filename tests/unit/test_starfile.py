import numpy as np
import pandas as pd
import pytest

from recovar.starfile import StarFile, read_star, write_star

pytestmark = pytest.mark.unit


def test_read_star_requires_star_extension(tmp_path):
    bad = tmp_path / "particles.txt"
    bad.write_text("data_\n")
    with pytest.raises(ValueError, match="Expected .star file"):
        read_star(str(bad))


def test_write_and_read_relion30_roundtrip(tmp_path):
    data = pd.DataFrame(
        {
            "_rlnImageName": ["1@a.mrcs", "2@a.mrcs"],
            "_rlnDefocusU": ["10000", "11000"],
        }
    )
    path = tmp_path / "r30.star"
    write_star(str(path), data)

    loaded_data, loaded_optics = read_star(str(path))
    assert loaded_optics is None
    assert loaded_data.equals(data)


def test_write_and_read_relion31_roundtrip(tmp_path):
    data = pd.DataFrame(
        {
            "_rlnImageName": ["1@a.mrcs", "2@a.mrcs", "3@a.mrcs"],
            "_rlnOpticsGroup": ["1", "2", "1"],
        }
    )
    optics = pd.DataFrame(
        {
            "_rlnOpticsGroup": ["1", "2"],
            "_rlnImagePixelSize": ["1.5", "2.0"],
            "_rlnImageSize": ["128", "64"],
        }
    )
    path = tmp_path / "r31.star"
    write_star(str(path), data, optics)

    loaded_data, loaded_optics = read_star(str(path))
    assert loaded_data.equals(data)
    assert loaded_optics.equals(optics)

    sf = StarFile.load(str(path))
    assert sf.has_optics
    np.testing.assert_allclose(sf.apix, np.array([1.5, 2.0, 1.5], dtype=np.float32))
    np.testing.assert_array_equal(sf.resolution, np.array([128, 64, 128], dtype=np.int64))


def test_set_optics_values_moves_per_particle_values_to_main_table():
    data = pd.DataFrame(
        {
            "_rlnImageName": ["1@a.mrcs", "2@a.mrcs", "3@a.mrcs"],
            "_rlnOpticsGroup": ["1", "2", "1"],
        }
    )
    optics = pd.DataFrame(
        {
            "_rlnOpticsGroup": ["1", "2"],
            "_rlnVoltage": ["300", "200"],
        }
    )
    sf = StarFile(data=data, data_optics=optics)
    sf.set_optics_values("_rlnVoltage", [10, 20, 30])

    assert "_rlnVoltage" in sf.df.columns
    assert "_rlnVoltage" not in sf.data_optics.columns
    np.testing.assert_array_equal(sf.df["_rlnVoltage"].values, np.array([10, 20, 30]))


def test_flatten_to_relion30_includes_optics_fields():
    data = pd.DataFrame(
        {
            "_rlnImageName": ["1@a.mrcs", "2@a.mrcs"],
            "_rlnOpticsGroup": ["1", "2"],
        }
    )
    optics = pd.DataFrame(
        {
            "_rlnOpticsGroup": ["1", "2"],
            "_rlnImagePixelSize": ["1.1", "1.2"],
        }
    )
    sf = StarFile(data=data, data_optics=optics)
    flattened = sf.flatten_to_relion30()

    assert "_rlnImagePixelSize" in flattened.columns
    np.testing.assert_allclose(flattened["_rlnImagePixelSize"].astype(float).values, np.array([1.1, 1.2]))


def test_read_star_rejects_duplicate_data_blocks(tmp_path):
    path = tmp_path / "dup.star"
    path.write_text(
        "data_particles\n\nloop_\n_rlnImageName #1\n1@a.mrcs\n\n"
        "data_particles\n\nloop_\n_rlnImageName #1\n2@a.mrcs\n"
    )
    with pytest.raises(ValueError, match="Duplicate data block"):
        read_star(str(path))


def test_read_star_rejects_inconsistent_row_lengths(tmp_path):
    path = tmp_path / "bad_rows.star"
    path.write_text(
        "data_particles\n\nloop_\n"
        "_rlnImageName #1\n"
        "_rlnDefocusU #2\n"
        "1@a.mrcs 10000\n"
        "2@a.mrcs\n"
    )
    with pytest.raises(ValueError, match="Inconsistent row lengths"):
        read_star(str(path))


def test_read_star_rejects_when_no_nonempty_data_table_exists(tmp_path):
    path = tmp_path / "no_data.star"
    path.write_text("data_particles\n\nloop_\n_rlnImageName #1\n")
    with pytest.raises(ValueError, match="No data table found"):
        read_star(str(path))
