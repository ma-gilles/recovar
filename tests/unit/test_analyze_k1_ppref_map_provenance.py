from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_k1_ppref_map_provenance import (
    _load_relion_iref,
    _load_setup_ppref,
    _load_verbose_ppref,
    _metric,
    _parse_iref_spec,
    _parse_map_spec,
    _real_metric,
    _shell_metrics,
)


def test_ppref_metric_reports_exact_complex_and_component_counts():
    reference = np.asarray([1 + 2j, 3 + 4j], dtype=np.complex64)
    report = _metric(reference.copy(), reference)
    assert report["bitwise_equal_complex_count"] == 2
    assert report["bitwise_equal_float32_component_count"] == 4
    assert report["relative_l2"] == 0.0


def test_ppref_metric_detects_one_component_change():
    reference = np.asarray([1 + 2j, 3 + 4j], dtype=np.complex64)
    candidate = reference.copy()
    candidate.real[1] = np.nextafter(candidate.real[1], np.float32(np.inf))
    report = _metric(candidate, reference)
    assert report["bitwise_equal_complex_count"] == 1
    assert report["bitwise_equal_float32_component_count"] == 3
    assert report["relative_l2"] > 0.0


def test_ppref_shell_metrics_use_relion_xyz_origin():
    reference = np.ones((3, 3, 2), dtype=np.complex64)
    candidate = reference.copy()
    candidate[1, 1, 1] = np.complex64(2 + 0j)
    shells = _shell_metrics(
        candidate,
        reference,
        origin_xyz=[0, -1, -1],
        r_max=2,
    )
    assert shells["1"]["relative_l2"] > 0.0
    assert shells["0"]["relative_l2"] == 0.0


def test_parse_map_spec_preserves_colons_after_label():
    assert _parse_map_spec("native:/tmp/map.mrc:relion") == (
        "native",
        Path("/tmp/map.mrc"),
        "relion",
    )


def test_parse_iref_spec_preserves_colons_after_label():
    assert _parse_iref_spec("native:/tmp/iref.bin") == (
        "native",
        Path("/tmp/iref.bin"),
    )


@pytest.mark.parametrize("value", ["missing", "label:/tmp/map.mrc:bad", ":/tmp/map.mrc:relion"])
def test_parse_map_spec_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_map_spec(value)


@pytest.mark.parametrize("value", ["missing", ":/tmp/iref.bin", "native:"])
def test_parse_iref_spec_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_iref_spec(value)


def test_load_relion_iref_converts_to_recovar_frame(tmp_path):
    relion = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    path = tmp_path / "iref.bin"
    path.write_bytes(struct.pack("<iii", *relion.shape) + relion.tobytes())

    actual = _load_relion_iref(path)

    np.testing.assert_array_equal(actual, -np.transpose(relion, (2, 1, 0)))


def test_real_metric_detects_float32_roundtrip():
    reference = np.asarray([1.0, 2.0], dtype=np.float64)
    candidate = reference + np.asarray([1.0e-9, -1.0e-9])
    report = _real_metric(candidate, reference)
    assert report["relative_l2"] > 0.0
    assert report["candidate_float32_equals_reference_float32"] is True


def test_load_verbose_ppref_decodes_counted_arrays(tmp_path):
    dims = np.asarray([2, 3, 3, 0, -1, -1, 1], dtype=np.int32)
    values = np.arange(18, dtype=np.float64)
    for name, array in (
        ("pass1_class0_ppref_dims.bin", dims),
        ("pass1_class0_ppref_real.bin", values),
        ("pass1_class0_ppref_imag.bin", -values),
    ):
        (tmp_path / name).write_bytes(
            np.asarray([array.size], dtype=np.int32).tobytes() + array.tobytes()
        )
    (tmp_path / "pass1_class0_ppref_padding_factor.bin").write_bytes(
        np.asarray([2.0], dtype=np.float64).tobytes()
    )
    (tmp_path / "pass1_img0_exp_current_image_size.bin").write_bytes(
        np.asarray([4.0], dtype=np.float64).tobytes()
    )

    ppref, metadata = _load_verbose_ppref(tmp_path)

    assert ppref.shape == (3, 3, 2)
    np.testing.assert_array_equal(ppref.reshape(-1).real, values.astype(np.float32))
    np.testing.assert_array_equal(ppref.reshape(-1).imag, -values.astype(np.float32))
    assert metadata["current_size"] == 2
    assert metadata["image_current_size"] == 4
    assert metadata["origin_xyz"] == [0, -1, -1]
    assert metadata["r_max"] == 1
    assert metadata["padding_factor"] == 2.0


def test_load_setup_ppref_decodes_contiguous_complex_doubles(tmp_path):
    shape = (3, 3, 2)
    values = (np.arange(18) + 1j * -np.arange(18)).astype(np.complex128)
    (tmp_path / "ppref_c0_data_post_setup.bin").write_bytes(
        struct.pack("<iii", *shape) + values.tobytes()
    )
    (tmp_path / "ppref_c0_meta.txt").write_text(
        "\n".join(
            (
                "iter=2",
                "r_max=1",
                "ori_size=8",
                "padding_factor=2.000000",
                "z=3",
                "y=3",
                "x=2",
            )
        )
        + "\n"
    )

    ppref, metadata = _load_setup_ppref(tmp_path)

    assert ppref.shape == shape
    np.testing.assert_array_equal(ppref.reshape(-1), values.astype(np.complex64))
    assert metadata["version"] == "expectation-setup-contiguous"
    assert metadata["iteration"] == 2
    assert metadata["current_size"] == 2
    assert metadata["image_current_size"] == 2
    assert metadata["original_image_size"] == 8
    assert metadata["origin_xyz"] == [0, -1, -1]
    assert metadata["r_max"] == 1
    assert metadata["padding_factor"] == 2.0
