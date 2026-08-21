import struct
from pathlib import Path

import numpy as np

from scripts.analyze_k1_projector_aa_replay import _load_native_ppref


def _flat(path: Path, values: np.ndarray, dtype: str) -> None:
    values = np.asarray(values, dtype=dtype).reshape(-1)
    path.write_bytes(struct.pack("<i", values.size) + values.tobytes())


def test_load_native_ppref_preserves_layout(tmp_path: Path):
    prefix = "ppref_"
    values = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    _flat(tmp_path / f"{prefix}dims.bin", [3, 2, 2, 0, -1, -1, 1], "<i4")
    _flat(tmp_path / f"{prefix}real.bin", values, "<f8")
    _flat(tmp_path / f"{prefix}imag.bin", -values, "<f8")

    ppref, metadata = _load_native_ppref(tmp_path, prefix)

    np.testing.assert_array_equal(ppref.real, values)
    np.testing.assert_array_equal(ppref.imag, -values)
    assert metadata["shape_zyx"] == [2, 2, 3]
    assert metadata["r_max"] == 1
