import struct
from pathlib import Path

import numpy as np

from scripts.analyze_k1_projector_boundary import analyze


def _flat(path: Path, values: np.ndarray, dtype: str) -> None:
    values = np.asarray(values, dtype=dtype).reshape(-1)
    path.write_bytes(struct.pack("<i", values.size) + values.tobytes())


def test_projector_boundary_compares_exact_layout_and_values(tmp_path: Path):
    native = tmp_path / "native"
    native.mkdir()
    prefix = "target_"
    reference = np.asarray(
        [[[0.0 + 0.0j, 1.0 + 2.0j], [3.0 + 4.0j, 5.0 + 6.0j]]],
        dtype=np.complex64,
    )
    _flat(native / f"{prefix}dims.bin", [2, 2, 1, 0, -1, 0, 1], "<i4")
    _flat(native / f"{prefix}real.bin", reference.real, "<f8")
    _flat(native / f"{prefix}imag.bin", reference.imag, "<f8")
    (native / f"{prefix}padding_factor.bin").write_bytes(struct.pack("<d", 2.0))

    recovar = tmp_path / "recovar.npz"
    np.savez_compressed(
        recovar,
        projector_half=reference[None, ...],
        projector_r_max=np.int64(1),
        current_size=np.int64(2),
        padding_factor=np.int64(2),
        volume_shape=np.asarray([2, 2, 2], dtype=np.int64),
        n_classes=np.int64(1),
    )

    report = analyze(native, recovar, native_prefix=prefix)

    assert report["complex_values"]["relative_l2"] == 0.0
    assert report["complex_values"]["bit_exact_float32_component_count"] == 8
    assert report["classification"].startswith("Projector::data agrees")
