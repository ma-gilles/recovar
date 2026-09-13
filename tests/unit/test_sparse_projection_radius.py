"""Sparse generic projection must preserve the requested image radius."""

import numpy as np
import pytest

from recovar.em.sparse_pass2 import sparse_pass2_projection_blocks as blocks

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("radius_kwargs", [{}, {"max_r": None}, {"max_r": 28.0}, {"max_r": 64.0}])
@pytest.mark.parametrize("cap", [None, 2])
def test_generic_sparse_projection_preserves_radius(monkeypatch, radius_kwargs, cap):
    calls = []

    def project(volume, rotations, image_shape, volume_shape, disc_type, **kwargs):
        calls.append(kwargs)
        return np.zeros((len(rotations), 40), np.complex64), None

    monkeypatch.setattr(blocks, "_compute_projections_block", project)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", raising=False)
    result, abs2 = blocks._compute_sparse_pass2_projections_block(
        np.zeros(512, np.complex64),
        np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)),
        (8, 8),
        (8, 8, 8),
        "linear_interp",
        max_projected_rotations=cap,
        return_abs2=False,
        **radius_kwargs,
    )
    assert len(calls) == (1 if cap is None else 2)
    assert calls == [dict(return_abs2=False, **radius_kwargs)] * len(calls)
    assert result.shape == (3, 40) and abs2 is None


@pytest.mark.parametrize("output_size", [None, 56, 58])
def test_relion_sparse_projection_keeps_separate_crop_and_radius(monkeypatch, output_size):
    calls = []

    def project(volume, rotations, image_shape, **kwargs):
        calls.append(kwargs)
        return np.zeros((len(rotations), 8320), np.complex64), None

    monkeypatch.setattr(blocks, "_compute_relion_projector_projections_block", project)
    blocks._compute_sparse_pass2_projections_block(
        None,
        np.eye(3, dtype=np.float32)[None],
        (128, 128),
        (256, 256, 256),
        "linear_interp",
        max_projected_rotations=1,
        relion_projector_half=np.zeros((117, 117, 59), np.complex64),
        relion_projector_r_max=28,
        projection_padding_factor=2,
        max_r=28.0,
        projector_output_size=output_size,
        return_abs2=False,
    )
    assert calls[0]["r_max"] == 28
    assert calls[0]["projector_output_size"] == (56 if output_size is None else output_size)
    assert "max_r" not in calls[0]


def test_windowed_generic_sparse_projection_preserves_radius(monkeypatch):
    calls = []

    def project(volume, rotations, image_shape, volume_shape, disc_type, **kwargs):
        calls.append(kwargs)
        values = np.broadcast_to(np.arange(40, dtype=np.complex64), (len(rotations), 40)).copy()
        return values, None

    monkeypatch.setattr(blocks, "_compute_projections_block", project)
    score, recon, abs2 = blocks._compute_sparse_pass2_windowed_projections_block(
        np.zeros(512, np.complex64),
        np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)),
        (8, 8),
        (8, 8, 8),
        "linear_interp",
        score_indices=np.array([1, 4, 6]),
        max_projected_rotations=2,
        output_complex_dtype=np.complex64,
        max_r=3.0,
    )
    assert calls == [dict(return_abs2=False, max_r=3.0)] * 2
    np.testing.assert_array_equal(score, np.broadcast_to([1, 4, 6], (3, 3)))
    assert recon is None and abs2 is None
