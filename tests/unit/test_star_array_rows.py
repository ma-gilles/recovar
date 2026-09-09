"""Byte-level formatting parity for the opt-in InitialModel STAR row writer."""

from datetime import datetime
import io

import numpy as np
import pandas as pd
import pytest

from recovar.data_io import starfile

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "table",
    [
        pd.DataFrame({"_image": ["000001@x.mrcs", "000002@x.mrcs"], "_class": [1, 2], "_value": [-0.0, 1e-30]}),
        pd.DataFrame({"_integer": [1, 2], "_float": [1.5, -0.0]}),
        pd.DataFrame({"_string": ["?", "nan"], "_bool": [False, True]}),
        pd.DataFrame({"_f32": np.array([np.nan, np.inf, -np.inf], dtype=np.float32)}),
        pd.DataFrame({"_complex": np.array([1 + 2j, 0j], dtype=np.complex64)}),
        pd.DataFrame({"_bigint": np.array([2**63, 2**63 + 1], dtype=np.uint64)}),
        pd.DataFrame(columns=["_empty"]),
        pd.DataFrame({"_str": pd.Series(["a", "b"], dtype="string")}),
    ],
)
def test_array_rows_match_legacy_without_constructing_series(table, monkeypatch):
    before = table.copy(deep=True)
    expected = io.StringIO()
    starfile._write_block(expected, table, "data_particles")

    def forbidden(*args, **kwargs):
        raise AssertionError("Fast scalar path constructed a Series row")

    monkeypatch.setattr(pd.DataFrame, "iterrows", forbidden)
    actual = io.StringIO()
    starfile._write_block(actual, table, "data_particles", array_rows=True)
    assert actual.getvalue() == expected.getvalue()
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize(
    "table",
    [
        pd.DataFrame({"_date": [datetime(2026, 1, 1), datetime(2026, 1, 2)]}),
        pd.DataFrame({"_object": pd.Series([datetime(2026, 1, 1), "a"], dtype=object)}),
        pd.DataFrame({"_nullable": pd.Series([1, pd.NA], dtype="Int64")}),
    ],
)
def test_other_scalar_types_preserve_legacy_conversion(table):
    expected, actual = io.StringIO(), io.StringIO()
    starfile._write_block(expected, table, "data_")
    starfile._write_block(actual, table, "data_", array_rows=True)
    assert actual.getvalue() == expected.getvalue()


@pytest.mark.parametrize(
    "optics", [None, pd.DataFrame({"_rlnOpticsGroup": [1], "_rlnOpticsGroupName": ["opticsGroup1"]})]
)
def test_complete_file_layout_and_timestamp_match(tmp_path, monkeypatch, optics):
    class FrozenTime:
        @staticmethod
        def now():
            return datetime(2026, 9, 6)

    monkeypatch.setattr(starfile, "datetime", FrozenTime)
    table = pd.DataFrame({"_rlnImageName": ["000001@x.mrcs"], "_rlnClassNumber": [1]})
    old, new = tmp_path / "old.star", tmp_path / "new.star"
    starfile.write_star(old, table, optics)
    starfile.write_star(new, table, optics, array_rows=True)
    assert old.read_bytes() == new.read_bytes()


def test_invalid_initial_model_selector_fails_before_writing(tmp_path, monkeypatch):
    from recovar.em.initial_model import driver

    monkeypatch.setenv("RECOVAR_VDAM_STAR_ARRAY_ROWS", "yes")
    path = tmp_path / "not_written.star"
    with pytest.raises(ValueError, match="must be 0 or 1"):
        driver._write_data_star(str(path), None, None, None, None)
    assert not path.exists()
