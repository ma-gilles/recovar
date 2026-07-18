import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers.relion_projector_capture import (
    ProjectorLoadError,
    build_relion_projector_replay_state,
)
from recovar.em.dense_single_volume.relion_replay import _parse_relion_projector_replay_state

pytestmark = pytest.mark.unit


def _scalar(path: Path, value):
    path.write_bytes(struct.pack("<d", float(value)))


def _vector(path: Path, values, dtype):
    values = np.asarray(values, dtype=dtype)
    path.write_bytes(struct.pack("<q", values.size) + values.tobytes())


def _fixture(root: Path):
    shape = (5, 5, 3)
    for rank, half in ((7, 2), (9, 1)):
        prefix = root / f"state_iter3_rank{rank}_device0_class0_"
        for field, value in (
            ("state_schema_version", 2), ("iteration", 3), ("mpi_rank", rank),
            ("device_id", 0), ("class_id", 0), ("control_my_halfset", half),
            ("projector_zdim", 5), ("projector_ydim", 5), ("projector_xdim", 3),
            ("projector_zinit", -2), ("projector_yinit", -2), ("projector_xinit", 0),
            ("projector_r_max", 2), ("projector_padding_factor", 2),
            ("projector_element_bytes", 4),
        ):
            _scalar(Path(str(prefix) + field + ".bin"), value)
        _vector(Path(str(prefix) + "control_image_current_size.bin"), [4], "<i8")
        real = np.arange(np.prod(shape), dtype=np.float32) + rank
        imag = -real
        _vector(Path(str(prefix) + "projector_real.bin"), real, "<f4")
        _vector(Path(str(prefix) + "projector_imag.bin"), imag, "<f4")
    manifest = root / "capture.sha256"
    lines = []
    for path in sorted(root.glob("*.bin")):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.resolve()}\n")
    manifest.write_text("".join(lines))
    return manifest


def test_projector_loader_emits_exact_atomic_mapping(tmp_path):
    manifest = _fixture(tmp_path)
    state = build_relion_projector_replay_state(
        tmp_path,
        manifest_path=manifest,
        iteration=3,
        current_size=4,
        volume_shape=(8, 8, 8),
        n_classes=1,
    )

    assert set(state) == {
        "projector_half_by_half", "projector_r_max_by_half", "current_size",
        "padding_factor", "volume_shape", "n_classes", "source_manifest_sha256",
    }
    assert [array.shape for array in state["projector_half_by_half"]] == [(1, 5, 5, 3)] * 2
    assert state["projector_half_by_half"][0][0, 0, 0, 0] == np.complex64(9 - 9j)
    assert state["projector_half_by_half"][1][0, 0, 0, 0] == np.complex64(7 - 7j)
    assert state["projector_r_max_by_half"] == [2, 2]
    assert state["source_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    parsed = _parse_relion_projector_replay_state(state, n_classes=1)
    assert parsed is not None
    assert parsed.projector_half_by_half[0].dtype == np.complex64
    assert parsed.source_manifest_sha256 == state["source_manifest_sha256"]


def test_projector_loader_rejects_nonstandard_start(tmp_path):
    manifest = _fixture(tmp_path)
    target = tmp_path / "state_iter3_rank9_device0_class0_projector_yinit.bin"
    _scalar(target, -1)
    lines = []
    for path in sorted(tmp_path.glob("*.bin")):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.resolve()}\n")
    manifest.write_text("".join(lines))
    with pytest.raises(ProjectorLoadError, match="nonstandard RELION projector starts"):
        build_relion_projector_replay_state(
            tmp_path, manifest_path=manifest, iteration=3, current_size=4,
            volume_shape=(8, 8, 8), n_classes=1,
        )


def test_projector_loader_rejects_manifest_corruption(tmp_path):
    manifest = _fixture(tmp_path)
    target = tmp_path / "state_iter3_rank9_device0_class0_projector_real.bin"
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(ProjectorLoadError, match="manifest verification failed"):
        build_relion_projector_replay_state(
            tmp_path, manifest_path=manifest, iteration=3, current_size=4,
            volume_shape=(8, 8, 8), n_classes=1,
        )


def test_projector_loader_rejects_consumed_field_absent_from_manifest(tmp_path):
    manifest = _fixture(tmp_path)
    lines = [line for line in manifest.read_text().splitlines() if "class0_device_id.bin" not in line]
    manifest.write_text("\n".join(lines) + "\n")

    with pytest.raises(ProjectorLoadError, match="consumed projector field is absent from manifest"):
        build_relion_projector_replay_state(
            tmp_path, manifest_path=manifest, iteration=3, current_size=4,
            volume_shape=(8, 8, 8), n_classes=1,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda root: _scalar(root / "state_iter3_rank9_device0_class0_state_schema_version.bin", 3),
         "unsupported live-state schema"),
        (lambda root: (root / "state_iter3_rank9_device0_class0_state_schema_version.bin").unlink(),
         "unexpected rank/device/class projector topology"),
        (lambda root: _scalar(root / "state_iter3_rank9_device0_class0_device_id.bin", 1),
         "device identity mismatch"),
    ],
)
def test_projector_loader_rejects_schema_or_device_mismatch(tmp_path, mutation, message):
    manifest = _fixture(tmp_path)
    mutation(tmp_path)
    lines = []
    for path in sorted(tmp_path.glob("*.bin")):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.resolve()}\n")
    manifest.write_text("".join(lines))
    with pytest.raises(ProjectorLoadError, match=message):
        build_relion_projector_replay_state(
            tmp_path, manifest_path=manifest, iteration=3, current_size=4,
            volume_shape=(8, 8, 8), n_classes=1,
        )


def test_projector_loader_rejects_unexpected_rank_device_class_topology(tmp_path):
    manifest = _fixture(tmp_path)
    _scalar(tmp_path / "state_iter3_rank9_device0_class1_iteration.bin", 3)
    _vector(
        tmp_path / "state_iter3_rank9_device0_class1_projector_real.bin",
        np.zeros(75, dtype=np.float32),
        "<f4",
    )
    lines = []
    for path in sorted(tmp_path.glob("*.bin")):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.resolve()}\n")
    manifest.write_text("".join(lines))

    with pytest.raises(ProjectorLoadError, match="unexpected rank/device/class projector topology"):
        build_relion_projector_replay_state(
            tmp_path, manifest_path=manifest, iteration=3, current_size=4,
            volume_shape=(8, 8, 8), n_classes=1,
        )
