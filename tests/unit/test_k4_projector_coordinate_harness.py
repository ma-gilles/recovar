"""Static and pure-array guards for the K4 projector-coordinate harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts" / "k4_projector_coordinate_harness"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module("k4_projector_prepare", HARNESS / "prepare.py")
validate = load_module("k4_projector_validate", HARNESS / "validate.py")


@pytest.mark.unit
def test_relion_to_recovar_target_mapping_is_exact():
    rows = np.arange(-19, 21, dtype=np.int32)
    columns = np.arange(21, dtype=np.int32)
    full_half_width = 65
    window_indices = ((rows[:, None] + 64) * full_half_width + columns[None, :]).reshape(-1)

    mapping = prepare.relion_to_recovar_columns(window_indices)

    assert mapping.shape == (840,)
    assert sorted(mapping.tolist()) == list(range(840))
    assert int(mapping[242]) == 641
    assert prepare.relion_pixel_coordinates()[242].tolist() == [11, 11]


@pytest.mark.unit
def test_exact_metrics_do_not_hide_one_bit_change():
    expected = np.asarray([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex64)
    actual = expected.copy()
    actual.view(np.uint32)[1] += np.uint32(1)

    metrics = validate.exact_metrics(actual, expected)

    assert metrics["exact_equal"] is False
    assert metrics["different_elements"] == 1
    assert metrics["max_abs"] > 0.0


@pytest.mark.unit
def test_cuda_harness_stages_before_texture_copy_and_keeps_variants_explicit():
    source = (HARNESS / "projector_coordinate_harness.cu").read_text()

    stage_readback = source.index('write_exact(output_dir / "staged_recovar_real.f32"')
    texture_copy = source.index("ProjectorTextures recovar_textures = make_textures")
    assert stage_readback < texture_copy
    assert "static_cast<float>(y) * e[1] + static_cast<float>(x) * e[0]" in source
    assert "e[0] * static_cast<float>(x) + e[1] * static_cast<float>(y)" in source
    assert "__fmaf_rn(y, e[1]" in source
    assert "__fmaf_rn(x, e[0]" in source
    assert "__fadd_rn(__fmul_rn(e[0], x), __fmul_rn(e[1], y))" in source
    assert "floorf(y_scaled) / 256.0f" in source
    assert "ceilf(y_scaled) / 256.0f" in source
