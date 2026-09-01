from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_k1_ppref_map_provenance import _metric, _parse_map_spec, _shell_metrics


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


@pytest.mark.parametrize("value", ["missing", "label:/tmp/map.mrc:bad", ":/tmp/map.mrc:relion"])
def test_parse_map_spec_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_map_spec(value)
