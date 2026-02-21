import numpy as np
import pandas as pd
import pytest

from recovar import cryodrgn_starfile as cstar


def _example_tables():
    data = pd.DataFrame(
        {
            "_rlnImageName": ["1@a.mrcs", "2@a.mrcs", "3@a.mrcs"],
            "_rlnOpticsGroup": ["1", "2", "1"],
            "_rlnAngleRot": ["0", "90", "180"],
        }
    )
    optics = pd.DataFrame(
        {
            "_rlnOpticsGroup": ["1", "2"],
            "_rlnImagePixelSize": ["1.25", "1.50"],
            "_rlnImageSize": ["64", "64"],
        }
    )
    return data, optics


def test_write_and_parse_star_relion31(tmp_path):
    data, optics = _example_tables()
    star_path = tmp_path / "particles.star"
    cstar.write_star(str(star_path), data=data, data_optics=optics)

    parsed_data, parsed_optics = cstar.parse_star(str(star_path))
    assert parsed_data.shape == data.shape
    assert parsed_optics.shape == optics.shape
    assert list(parsed_optics.columns) == list(optics.columns)


def test_starfile_get_set_optics_values():
    data, optics = _example_tables()
    sf = cstar.Starfile(data=data, data_optics=optics)

    apix = sf.apix
    assert np.allclose(apix, np.array([1.25, 1.50, 1.25], dtype=np.float32))

    sf.set_optics_values("_rlnImagePixelSize", [2.0, 3.0])
    apix2 = sf.apix
    assert np.allclose(apix2, np.array([2.0, 3.0, 2.0], dtype=np.float32))


def test_parse_star_rejects_non_star_extension(tmp_path):
    bad_path = tmp_path / "particles.txt"
    bad_path.write_text("data_\n")
    with pytest.raises(ValueError):
        cstar.parse_star(str(bad_path))

