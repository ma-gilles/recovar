from pathlib import Path

import mrcfile
import numpy as np
import pytest

from scripts import audit_em_k1_checkpoint_equivalence as audit

pytestmark = pytest.mark.unit


def _write_checkpoint(root: Path, *, floating_offset: float = 0.0) -> None:
    root.mkdir()
    prefix = "it000"
    for suffix in audit.DISCRETE_NPY_SUFFIXES:
        if suffix == "fsc.npy":
            value = np.linspace(1.0, 0.0, 5, dtype=np.float32)
        elif suffix == "rotations.npy":
            value = np.eye(3, dtype=np.float32)[None]
        elif suffix == "translations.npy":
            value = np.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32)
        else:
            value = np.asarray([0, 2, 1, 3], dtype=np.int32)
        np.save(root / f"{prefix}_{suffix}", value)

    for suffix in audit.DISCRETE_NPZ_SUFFIXES:
        np.savez(
            root / f"{prefix}_{suffix}",
            image_indices=np.arange(4, dtype=np.int32),
            best_class=np.asarray([0, 0, 0, 0], dtype=np.int32),
            pmax=np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        )

    np.save(
        root / f"{prefix}_meta.npy",
        {
            "iteration": 1,
            "local_search": False,
            "symmetry": {"name": "I1", "operator_count": 60},
        },
        allow_pickle=True,
    )

    for index, suffix in enumerate(audit.FLOAT_NPY_SUFFIXES):
        if suffix.startswith("Ft_y"):
            value = np.asarray([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex64)
        else:
            value = np.linspace(1.0, 4.0, 8, dtype=np.float32)
        if floating_offset and index == 0:
            value = value.copy()
            value.reshape(-1)[0] += floating_offset
        np.save(root / f"{prefix}_{suffix}", value)

    for index, suffix in enumerate(audit.MAP_SUFFIXES):
        volume = np.arange(64, dtype=np.float32).reshape(4, 4, 4) + index
        if floating_offset and index == 0:
            volume = volume.copy()
            volume[0, 0, 0] += floating_offset
        with mrcfile.new(root / f"{prefix}_{suffix}", overwrite=True) as output:
            output.set_data(volume)


def test_checkpoint_audit_accepts_exact_state_and_small_float_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_checkpoint(baseline)
    _write_checkpoint(candidate, floating_offset=1e-6)

    report = audit.run_audit(
        baseline,
        candidate,
        0,
        max_relative_l2=1e-5,
    )

    assert report["summary"]["accepted"] is True
    assert report["summary"]["exact_execution_state"] is True
    assert report["maps"]["half1_reg.mrc"]["metrics"]["element_exact"] is False
    assert report["maps"]["half1_reg.mrc"]["metrics"]["centered_correlation"] == pytest.approx(1.0)


def test_checkpoint_audit_rejects_changed_particle_decision(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_checkpoint(baseline)
    _write_checkpoint(candidate)
    np.save(candidate / "it000_ha_half1.npy", np.asarray([0, 2, 1, 4], dtype=np.int32))

    report = audit.run_audit(
        baseline,
        candidate,
        0,
        max_relative_l2=0.0,
    )

    assert report["summary"]["accepted"] is False
    assert report["summary"]["exact_execution_state"] is False
    assert report["discrete_arrays"]["ha_half1.npy"]["metrics"]["element_exact"] is False
