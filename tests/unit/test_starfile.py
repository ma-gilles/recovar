import numpy as np
import pandas as pd
import pytest

from recovar.data_io.starfile import StarFile, read_star, write_star

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


def test_read_star_column_count_mismatch(tmp_path):
    path = tmp_path / "col_mismatch.star"
    path.write_text(
        "data_\n\nloop_\n_rlnImageName\n_rlnDefocusU\n"
        "1@a.mrcs 10000 extra_col\n"
    )
    with pytest.raises(ValueError, match="Column count mismatch"):
        read_star(str(path))


def test_read_star_ignores_comments(tmp_path):
    path = tmp_path / "comments.star"
    path.write_text(
        "# This is a comment\n"
        "data_\n\nloop_\n_rlnImageName\n"
        "# Another comment\n"
        "1@a.mrcs\n2@a.mrcs\n"
    )
    main, optics = read_star(str(path))
    assert len(main) == 2


# ---------------------------------------------------------------------------
# StarFile constructor edge cases
# ---------------------------------------------------------------------------

def test_starfile_from_data():
    df = pd.DataFrame({"_rlnImageName": ["a", "b"], "_rlnDefocusU": ["1", "2"]})
    sf = StarFile(data=df)
    assert len(sf) == 2
    assert not sf.has_optics
    assert not sf.relion31


def test_starfile_no_args_raises():
    with pytest.raises(ValueError, match="exactly one"):
        StarFile()


def test_starfile_both_args_raises(tmp_path):
    path = tmp_path / "test.star"
    path.write_text(
        "data_\n\nloop_\n_rlnImageName\n1@a.mrcs\n"
    )
    df = pd.DataFrame({"_rlnImageName": ["a"]})
    with pytest.raises(ValueError, match="exactly one"):
        StarFile(str(path), data=df)


def test_starfile_write_alias(tmp_path):
    df = pd.DataFrame({"_rlnImageName": ["a"], "_rlnDefocusU": ["10000"]})
    sf = StarFile(data=df)
    out = str(tmp_path / "alias.star")
    sf.write(out)
    sf2 = StarFile.load(out)
    assert len(sf2) == 1


def test_starfile_eq_same():
    df = pd.DataFrame({"_rlnImageName": ["a", "b"]})
    sf1 = StarFile(data=df.copy())
    sf2 = StarFile(data=df.copy())
    assert sf1 == sf2


def test_starfile_ne_different_data():
    df1 = pd.DataFrame({"_rlnImageName": ["a"]})
    df2 = pd.DataFrame({"_rlnImageName": ["b"]})
    sf1 = StarFile(data=df1)
    sf2 = StarFile(data=df2)
    assert sf1 != sf2


def test_starfile_ne_optics_mismatch():
    df = pd.DataFrame({"_rlnImageName": ["a"], "_rlnOpticsGroup": ["1"]})
    optics = pd.DataFrame({"_rlnOpticsGroup": ["1"], "_rlnVoltage": ["300"]})
    sf1 = StarFile(data=df.copy(), data_optics=optics.copy())
    sf2 = StarFile(data=df.copy())
    assert sf1 != sf2


# ---------------------------------------------------------------------------
# get_optics_values edge cases
# ---------------------------------------------------------------------------

def test_get_optics_values_missing_field():
    df = pd.DataFrame({"_rlnImageName": ["a"]})
    sf = StarFile(data=df)
    assert sf.get_optics_values("_rlnNonExistent") is None


def test_get_optics_values_dtype_cast():
    df = pd.DataFrame({"_rlnDefocusU": ["15000", "16000"]})
    sf = StarFile(data=df)
    vals = sf.get_optics_values("_rlnDefocusU", dtype=np.float64)
    assert vals.dtype == np.float64
    np.testing.assert_allclose(vals, [15000.0, 16000.0])


# ---------------------------------------------------------------------------
# set_optics_values edge cases
# ---------------------------------------------------------------------------

def test_set_optics_values_scalar():
    df = pd.DataFrame({"_rlnDefocusU": ["100", "200", "300"]})
    sf = StarFile(data=df)
    sf.set_optics_values("_rlnDefocusU", 999)
    assert all(v == 999 for v in sf.df["_rlnDefocusU"])


def test_set_optics_values_invalid_length():
    df = pd.DataFrame({"_rlnDefocusU": ["100", "200", "300"]})
    sf = StarFile(data=df)
    with pytest.raises(ValueError, match="Values length"):
        sf.set_optics_values("_rlnDefocusU", [1, 2])


def test_set_optics_values_unknown_field():
    df = pd.DataFrame({"_rlnImageName": ["a"]})
    sf = StarFile(data=df)
    with pytest.raises(ValueError, match="not found"):
        sf.set_optics_values("_rlnFakeField", [1])


# ---------------------------------------------------------------------------
# apix fallback (old RELION format)
# ---------------------------------------------------------------------------

def test_apix_old_format():
    """Test pixel size from _rlnDetectorPixelSize and _rlnMagnification."""
    df = pd.DataFrame({
        "_rlnImageName": ["a", "b"],
        "_rlnDetectorPixelSize": ["14.0", "14.0"],
        "_rlnMagnification": ["100000", "100000"],
    })
    sf = StarFile(data=df)
    apix = sf.apix
    assert apix is not None
    # 14.0 um * 1e4 / 100000 = 1.4 Å/pixel
    np.testing.assert_allclose(apix, 1.4, rtol=1e-5)


def test_apix_none_when_no_fields():
    df = pd.DataFrame({"_rlnImageName": ["a"]})
    sf = StarFile(data=df)
    assert sf.apix is None


def test_resolution_none_when_no_fields():
    df = pd.DataFrame({"_rlnDefocusU": ["10000"]})
    sf = StarFile(data=df)
    assert sf.resolution is None


# ---------------------------------------------------------------------------
# flatten_to_relion30 passthrough
# ---------------------------------------------------------------------------

def test_flatten_to_relion30_noop_on_relion30():
    df = pd.DataFrame({"_rlnImageName": ["a", "b"]})
    sf = StarFile(data=df)
    flat = sf.flatten_to_relion30()
    pd.testing.assert_frame_equal(flat, sf.df)


def test_to_relion30_alias():
    data = pd.DataFrame({"_rlnImageName": ["a"], "_rlnOpticsGroup": ["1"]})
    optics = pd.DataFrame({"_rlnOpticsGroup": ["1"], "_rlnVoltage": ["300"]})
    sf = StarFile(data=data, data_optics=optics)
    pd.testing.assert_frame_equal(sf.flatten_to_relion30(), sf.to_relion30())
