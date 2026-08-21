from __future__ import annotations

import hashlib
import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "select_k1_bpref_support_cohort.py"
SPEC = importlib.util.spec_from_file_location("select_k1_bpref_support_cohort", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RELION_PATCH = (
    Path(__file__).parents[2]
    / "docs"
    / "patches"
    / "relion_bpref_prescatter_part_id_filter_bc319d0.patch"
)
RELION_PATCH_SHA256 = "82e79e3e07079e553280e2089d2fc5c4887fb43a27c032ee6df3228eb789bd21"
RELION_CHUNKED_PATCH = (
    Path(__file__).parents[2]
    / "docs"
    / "patches"
    / "relion_bpref_prescatter_chunked_capture_bc319d0.patch"
)
RELION_CHUNKED_PATCH_SHA256 = (
    "1a9680d93ae6ab0577a7901999dca464c7929ed10b36c36744fc87672889668f"
)


def _write_header(path: Path, *, part_id: int, stack_index: int, mpi_rank: int) -> None:
    values = [0] * 40
    values[0] = 1
    values[1] = MODULE.HEADER.size
    values[7] = part_id
    values[8] = stack_index
    values[10] = mpi_rank
    path.write_bytes(struct.pack("<16s40Q", MODULE.HEADER_MAGIC, *values))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    trajectory_path = tmp_path / "trajectory.npz"
    support_delta = np.asarray([-3, -2, -2, -1, -1, -1, 0, 0, 0, 0], dtype=np.float64)
    np.savez(
        trajectory_path,
        identity_row_index=np.arange(support_delta.size, dtype=np.int64),
        it002_support_delta=support_delta,
    )
    image_names_path = tmp_path / "image_names.npy"
    np.save(
        image_names_path,
        np.asarray([f"{index + 1}@/frozen/particles.mrcs" for index in range(support_delta.size)]),
        allow_pickle=False,
    )
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    for original_index in range(support_delta.size):
        _write_header(
            capture_dir / f"part{100 + original_index}_stack{original_index + 1}.bpre-v1.bin",
            part_id=100 + original_index,
            stack_index=original_index + 1,
            mpi_rank=1 if original_index != 9 else 2,
        )
    return trajectory_path, image_names_path, capture_dir


def test_relion_part_id_filter_patch_is_exact_and_preallocation() -> None:
    patch_bytes = RELION_PATCH.read_bytes()
    assert hashlib.sha256(patch_bytes).hexdigest() == RELION_PATCH_SHA256
    patch = patch_bytes.decode()
    selection_guard = patch.index("if (!relion_bpre_part_id_selected")
    first_capture_allocation_guard = patch.index("if (data_is_3d)")
    assert selection_guard < first_capture_allocation_guard
    assert "expected_particles != result.part_ids.size()" in patch
    assert "Duplicate particle identity" in patch
    assert "Trailing comma" in patch


def test_relion_chunked_capture_patch_preserves_global_orientation_order() -> None:
    patch_bytes = RELION_CHUNKED_PATCH.read_bytes()
    assert hashlib.sha256(patch_bytes).hexdigest() == RELION_CHUNKED_PATCH_SHA256
    patch = patch_bytes.decode()
    added_lines = "\n".join(
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    selection_guard = patch.index("if (!relion_bpre_part_id_selected")
    device_allocation = patch.index("cudaMalloc(&device_rows, chunk_device_bytes)")
    assert selection_guard < device_allocation
    assert 'relion_bpre_required_u64("RELION_BPRE_CAPTURE_DEVICE_BYTES")' in patch
    assert "config.device_buffer_bytes < bytes_per_orientation" in patch
    assert "chunk_device_bytes > free_device_bytes" in patch
    assert "orientation_offset += orientations_per_chunk" in patch
    assert patch.count("orientation_offset, device_rows);") == 2
    assert "orientation_in_chunk * img_xyz + pixel" in patch
    assert "static_cast<std::uint32_t>(orientation)" in patch
    assert "cudaMalloc(&device_rows, candidate_count" not in added_lines


def test_select_cohort_preserves_deepest_and_fills_controls(tmp_path: Path) -> None:
    trajectory_path, image_names_path, capture_dir = _fixture(tmp_path)
    report = MODULE.select_cohort(
        trajectory_path=trajectory_path,
        image_names_path=image_names_path,
        reference_capture_dir=capture_dir,
        iteration=2,
        mpi_rank=1,
        total_count=6,
        minus_two_count=2,
        minus_one_count=2,
        seed="unit-seed",
    )

    assert report["selected_particle_count"] == 6
    assert report["selected_group_counts"] == {
        "support_delta_le_minus_3": 1,
        "support_delta_minus_2": 2,
        "support_delta_minus_1": 2,
        "exact_support_control": 1,
    }
    assert report["rows"][0]["support_delta"] == -3
    assert {row["mpi_rank"] for row in report["rows"]} == {1}
    assert report["relion_part_ids_csv"] == ",".join(
        str(row["relion_part_id"]) for row in report["rows"]
    )


def test_select_cohort_is_deterministic(tmp_path: Path) -> None:
    trajectory_path, image_names_path, capture_dir = _fixture(tmp_path)
    kwargs = {
        "trajectory_path": trajectory_path,
        "image_names_path": image_names_path,
        "reference_capture_dir": capture_dir,
        "iteration": 2,
        "mpi_rank": 1,
        "total_count": 5,
        "minus_two_count": 1,
        "minus_one_count": 2,
        "seed": "unit-seed",
    }
    first = MODULE.select_cohort(**kwargs)
    second = MODULE.select_cohort(**kwargs)
    assert first["rows"] == second["rows"]
    assert first["cohort_rows_sha256"] == second["cohort_rows_sha256"]


def test_select_cohort_rejects_noncanonical_row_to_stack_mapping(tmp_path: Path) -> None:
    trajectory_path, image_names_path, capture_dir = _fixture(tmp_path)
    image_names = np.load(image_names_path, allow_pickle=False)
    image_names[[0, 1]] = image_names[[1, 0]]
    np.save(image_names_path, image_names, allow_pickle=False)

    with pytest.raises(ValueError, match="canonical"):
        MODULE.select_cohort(
            trajectory_path=trajectory_path,
            image_names_path=image_names_path,
            reference_capture_dir=capture_dir,
            iteration=2,
            mpi_rank=1,
            total_count=5,
            minus_two_count=1,
            minus_one_count=2,
        )


def test_select_cohort_rejects_truncated_reference_header(tmp_path: Path) -> None:
    trajectory_path, image_names_path, capture_dir = _fixture(tmp_path)
    next(capture_dir.glob("*.bpre-v1.bin")).write_bytes(b"short")
    with pytest.raises(ValueError, match="truncated"):
        MODULE.select_cohort(
            trajectory_path=trajectory_path,
            image_names_path=image_names_path,
            reference_capture_dir=capture_dir,
            iteration=2,
            mpi_rank=1,
            total_count=5,
            minus_two_count=1,
            minus_one_count=2,
        )
