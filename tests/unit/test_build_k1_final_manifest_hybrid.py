from pathlib import Path

import numpy as np

from scripts.build_k1_final_manifest_hybrid import build_hybrid


def _write_results(path: Path, half1: list[int], half2: list[int]) -> None:
    np.savez(path, half1_indices=np.asarray(half1), half2_indices=np.asarray(half2))


def _write_manifest(path: Path, half_index: int, scale: list[float], shared: float) -> None:
    np.savez(
        path,
        half_index=np.int32(half_index),
        scale_corrections=np.asarray(scale, dtype=np.float64),
        current_size=np.int32(shared),
    )


def test_build_hybrid_aligns_particle_fields_by_source_identity(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    donor_dir = tmp_path / "donor"
    output_dir = tmp_path / "output"
    base_dir.mkdir()
    donor_dir.mkdir()
    base_results = tmp_path / "base_results.npz"
    donor_results = tmp_path / "donor_results.npz"
    _write_results(base_results, [20, 10], [40, 30])
    _write_results(donor_results, [10, 20], [30, 40])
    _write_manifest(base_dir / "manifest_final_half0.npz", 0, [2.0, 1.0], 64)
    _write_manifest(base_dir / "manifest_final_half1.npz", 1, [4.0, 3.0], 64)
    _write_manifest(donor_dir / "manifest_final_half0.npz", 0, [100.0, 200.0], 96)
    _write_manifest(donor_dir / "manifest_final_half1.npz", 1, [300.0, 400.0], 96)

    report = build_hybrid(
        base_manifest_dir=base_dir,
        base_results=base_results,
        donor_manifest_dir=donor_dir,
        donor_results=donor_results,
        output_dir=output_dir,
        fields=("scale_corrections",),
    )

    with np.load(output_dir / "manifest_final_half0.npz") as half1:
        np.testing.assert_array_equal(half1["scale_corrections"], [200.0, 100.0])
        assert int(half1["current_size"]) == 64
    with np.load(output_dir / "manifest_final_half1.npz") as half2:
        np.testing.assert_array_equal(half2["scale_corrections"], [400.0, 300.0])
        assert int(half2["current_size"]) == 64
    assert report["fields"] == ["scale_corrections"]
    assert all(row["fields"]["scale_corrections"]["changed_count"] == 2 for row in report["halves"])


def test_build_hybrid_rejects_nonbijective_identity_sets(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    donor_dir = tmp_path / "donor"
    base_dir.mkdir()
    donor_dir.mkdir()
    base_results = tmp_path / "base_results.npz"
    donor_results = tmp_path / "donor_results.npz"
    _write_results(base_results, [10, 20], [30, 40])
    _write_results(donor_results, [10, 99], [30, 40])
    for half in range(2):
        _write_manifest(base_dir / f"manifest_final_half{half}.npz", half, [1.0, 2.0], 64)
        _write_manifest(donor_dir / f"manifest_final_half{half}.npz", half, [3.0, 4.0], 64)

    with np.testing.assert_raises_regex(ValueError, "identity sets differ"):
        build_hybrid(
            base_manifest_dir=base_dir,
            base_results=base_results,
            donor_manifest_dir=donor_dir,
            donor_results=donor_results,
            output_dir=tmp_path / "output",
            fields=("scale_corrections",),
        )


def test_build_hybrid_rejects_wrong_manifest_half_before_writing(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    donor_dir = tmp_path / "donor"
    base_dir.mkdir()
    donor_dir.mkdir()
    base_results = tmp_path / "base_results.npz"
    donor_results = tmp_path / "donor_results.npz"
    _write_results(base_results, [10, 20], [30, 40])
    _write_results(donor_results, [10, 20], [30, 40])
    _write_manifest(base_dir / "manifest_final_half0.npz", 0, [1.0, 2.0], 64)
    _write_manifest(base_dir / "manifest_final_half1.npz", 1, [1.0, 2.0], 64)
    _write_manifest(donor_dir / "manifest_final_half0.npz", 0, [3.0, 4.0], 64)
    _write_manifest(donor_dir / "manifest_final_half1.npz", 0, [3.0, 4.0], 64)
    output_dir = tmp_path / "output"

    with np.testing.assert_raises_regex(ValueError, "wrong half_index"):
        build_hybrid(
            base_manifest_dir=base_dir,
            base_results=base_results,
            donor_manifest_dir=donor_dir,
            donor_results=donor_results,
            output_dir=output_dir,
            fields=("scale_corrections",),
        )

    assert not output_dir.exists()
