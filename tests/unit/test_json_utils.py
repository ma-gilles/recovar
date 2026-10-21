"""JSON report conversion preserves values and leaves rejection to the encoder."""

import json
from pathlib import Path

import numpy as np
import pytest

from recovar.utils.json_utils import to_jsonable

pytestmark = pytest.mark.unit


def test_nested_report_encodes_numpy_values_paths_and_integer_keys():
    report = {
        4: {
            "file": Path("volumes/half1.mrc"),
            "scores": np.array([[0.5, 1.0]], dtype=np.float32),
            "accepted": np.bool_(True),
            "counts": (np.int64(3), np.uint64(2**63 + 1)),
        },
        "missing": None,
    }
    assert json.loads(json.dumps(to_jsonable(report))) == {
        "4": {
            "file": "volumes/half1.mrc",
            "scores": [[0.5, 1.0]],
            "accepted": True,
            "counts": [3, 2**63 + 1],
        },
        "missing": None,
    }


def test_object_and_scalar_arrays_are_normalized_recursively():
    report = {
        "items": np.array([Path("mask.mrc"), np.int64(7)], dtype=object),
        "scalar": np.array(12, dtype=np.int32),
        "empty": np.empty((0, 2), dtype=np.float64),
    }
    assert json.loads(json.dumps(to_jsonable(report))) == {"items": ["mask.mrc", 7], "scalar": 12, "empty": []}


@pytest.mark.parametrize("value", [np.float32(np.nan), np.float64(np.inf), -np.inf])
def test_nonfinite_values_remain_visible_to_strict_json_encoding(value):
    with pytest.raises(ValueError, match="Out of range float values"):
        json.dumps(to_jsonable({"metric": value}), allow_nan=False)


@pytest.mark.parametrize("value", [1 + 2j, {1, 2}, np.complex64(1 + 2j)])
def test_unsupported_values_are_not_silently_stringified(value):
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps(to_jsonable({"value": value}))
