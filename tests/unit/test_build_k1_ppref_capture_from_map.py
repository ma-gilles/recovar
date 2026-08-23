from __future__ import annotations

import numpy as np

from scripts.analyze_k1_exact_ppref_fine_boundary import _load_ppref
from scripts.build_k1_ppref_capture_from_map import write_ppref_capture


def test_write_ppref_capture_roundtrips_schema(tmp_path):
    ppref = (
        np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        + 1j * np.arange(60, 120, dtype=np.float32).reshape(3, 4, 5)
    ).astype(np.complex64)
    path = tmp_path / "ppref.bin"
    write_ppref_capture(
        path,
        ppref,
        iteration=2,
        rank=1,
        model=0,
        current_size=56,
        r_max=28,
        padding_factor=2.0,
    )
    observed, metadata = _load_ppref(path)
    np.testing.assert_array_equal(observed, ppref)
    assert metadata == {
        "version": 1,
        "iteration": 2,
        "rank": 1,
        "model": 0,
        "current_size": 56,
        "shape_zyx": [3, 4, 5],
        "origin_xyz": [0, -2, -1],
        "r_max": 28,
        "padding_factor": 2.0,
        "complex_count": 60,
    }
