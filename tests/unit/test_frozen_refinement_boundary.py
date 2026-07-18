from __future__ import annotations

import hashlib

import numpy as np
import pytest

from recovar.em.dense_single_volume.frozen_boundary import (
    FROZEN_BOUNDARY_FILENAME,
    FROZEN_BOUNDARY_MANIFEST,
    FROZEN_BOUNDARY_SCHEMA,
    load_frozen_refinement_boundary,
)


def _write_boundary(root, **overrides):
    root.mkdir(exist_ok=True)
    values = {
        "schema": np.asarray(FROZEN_BOUNDARY_SCHEMA),
        "completed_relion_iteration": np.int32(2),
        "volume_shape": np.asarray([2, 2, 2], dtype=np.int32),
        "current_size": np.int32(92),
        "healpix_order": np.int32(3),
        "relion_incr_size": np.int32(10),
        "has_high_fsc_at_limit": np.bool_(False),
        "half1_mean_ft": np.arange(8, dtype=np.float32).astype(np.complex64),
        "half2_mean_ft": (np.arange(8, dtype=np.float32) + 1j).astype(np.complex64),
        "mean_variance": np.ones(8, dtype=np.float32),
        "half1_noise_radial": np.asarray([1.0, 2.0], dtype=np.float64),
        "half2_noise_radial": np.asarray([1.5, 2.5], dtype=np.float64),
        "fsc": np.asarray([1.0, 0.5], dtype=np.float32),
        "ave_pmax": np.float64(0.25),
        "half1_previous_best_rotation_eulers": np.zeros((2, 3), dtype=np.float32),
        "half2_previous_best_rotation_eulers": np.ones((2, 3), dtype=np.float32),
        "half1_previous_best_translations": np.zeros((2, 2), dtype=np.float32),
        "half2_previous_best_translations": np.ones((2, 2), dtype=np.float32),
        "half1_image_name": np.asarray(["1@a.mrcs", "2@a.mrcs"]),
        "half2_image_name": np.asarray(["3@a.mrcs", "4@a.mrcs"]),
        "state_current_resolution": np.float64(14.97),
        "state_previous_resolution": np.float64(29.94),
        "state_ave_Pmax": np.float64(0.25),
        "state_has_converged": np.bool_(False),
    }
    values.update(overrides)
    path = root / FROZEN_BOUNDARY_FILENAME
    np.savez(path, **values)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / FROZEN_BOUNDARY_MANIFEST).write_text(
        f"{digest}  {FROZEN_BOUNDARY_FILENAME}\n",
        encoding="utf-8",
    )
    return path


def test_frozen_boundary_loader_round_trips_primitive_state(tmp_path):
    _write_boundary(tmp_path)

    boundary = load_frozen_refinement_boundary(tmp_path)

    assert boundary.completed_relion_iteration == 2
    assert boundary.current_size == 92
    assert boundary.means[0].dtype == np.complex64
    assert boundary.image_corrections is None
    assert boundary.refinement_state_fields == {
        "current_resolution": pytest.approx(14.97),
        "previous_resolution": pytest.approx(29.94),
        "ave_Pmax": pytest.approx(0.25),
        "has_converged": False,
    }
    assert len(boundary.boundary_sha256) == 64
    assert len(boundary.source_manifest_sha256) == 64


def test_frozen_boundary_loader_rejects_modified_payload(tmp_path):
    path = _write_boundary(tmp_path)
    with path.open("ab") as stream:
        stream.write(b"modified")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_rejects_duplicate_half_identity(tmp_path):
    _write_boundary(
        tmp_path,
        half2_image_name=np.asarray(["2@a.mrcs", "4@a.mrcs"]),
    )

    with pytest.raises(ValueError, match="globally unique"):
        load_frozen_refinement_boundary(tmp_path)


def test_frozen_boundary_loader_rejects_half_correction_pair_mismatch(tmp_path):
    _write_boundary(
        tmp_path,
        half1_image_corrections=np.ones(2, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="both or neither"):
        load_frozen_refinement_boundary(tmp_path)
