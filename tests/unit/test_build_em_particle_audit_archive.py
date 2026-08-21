from pathlib import Path

import numpy as np

from scripts.build_em_particle_audit_archive import build_archive


def test_build_archive_scatter_all_iterations_to_source_order(tmp_path: Path):
    output = tmp_path / "output"
    output.mkdir()
    half1 = np.asarray([2, 0], dtype=np.int64)
    half2 = np.asarray([3, 1], dtype=np.int64)
    arrays: dict[str, np.ndarray] = {
        "current_sizes": np.asarray([56, 100], dtype=np.int64),
        "half1_indices": half1,
        "half2_indices": half2,
    }
    for iteration in range(2):
        suffix = f"{iteration:03d}"
        base = 10 * iteration
        arrays[f"pmax_per_image_iter_{suffix}"] = np.asarray(
            [base + 0.1, base + 0.2, base + 0.3, base + 0.4],
            dtype=np.float32,
        )
        arrays[f"sig_counts_by_image_iter_{suffix}"] = np.asarray(
            [base + 1, base + 2, base + 3, base + 4],
            dtype=np.int32,
        )
        arrays[f"best_rotation_eulers_iter_{suffix}_half1"] = np.asarray(
            [[base + 1, 0, 0], [base + 2, 0, 0]],
            dtype=np.float32,
        )
        arrays[f"best_rotation_eulers_iter_{suffix}_half2"] = np.asarray(
            [[base + 3, 0, 0], [base + 4, 0, 0]],
            dtype=np.float32,
        )
        arrays[f"best_translations_iter_{suffix}_half1"] = np.asarray(
            [[base + 1, 0], [base + 2, 0]],
            dtype=np.float32,
        )
        arrays[f"best_translations_iter_{suffix}_half2"] = np.asarray(
            [[base + 3, 0], [base + 4, 0]],
            dtype=np.float32,
        )
    np.savez_compressed(output / "refinement_results.npz", **arrays)

    destination = build_archive(output)

    assert destination == tmp_path / "analysis" / "refinement_results_audit_source_order.npz"
    with np.load(destination, allow_pickle=False) as archive:
        assert np.array_equal(archive["half1_indices"], half1)
        assert np.array_equal(archive["half2_indices"], half2)
        assert np.allclose(
            archive["pmax_per_image_by_image_iter_001"],
            [10.2, 10.4, 10.1, 10.3],
        )
        assert np.array_equal(
            archive["sig_counts_by_image_iter_001"],
            [12, 14, 11, 13],
        )
        assert np.array_equal(
            archive["best_rotation_eulers_by_image_iter_001"][:, 0],
            [12, 14, 11, 13],
        )
        assert np.array_equal(
            archive["best_translations_by_image_iter_001"][:, 0],
            [12, 14, 11, 13],
        )


def test_build_archive_accepts_legacy_zero_based_halves_and_explicit_half_order(
    tmp_path: Path,
):
    output = tmp_path / "output"
    output.mkdir()
    half1 = np.asarray([2, 0], dtype=np.int64)
    half2 = np.asarray([3, 1], dtype=np.int64)
    arrays = {
        "current_sizes": np.asarray([56], dtype=np.int64),
        "half1_indices": half1,
        "half2_indices": half2,
        "pmax_per_image_iter_000": np.full(4, -1, dtype=np.float32),
        "pmax_per_half_order_iter_000": np.asarray([0.1, 0.2, 0.3, 0.4]),
        "sig_counts_by_image_iter_000": np.full(4, -1, dtype=np.int32),
        "sig_counts_half_order_iter_000": np.asarray([1, 2, 3, 4], dtype=np.int32),
        "best_rotation_eulers_iter_000_half0": np.asarray([[1, 0, 0], [2, 0, 0]]),
        "best_rotation_eulers_iter_000_half1": np.asarray([[3, 0, 0], [4, 0, 0]]),
        "best_translations_iter_000_half0": np.asarray([[1, 0], [2, 0]]),
        "best_translations_iter_000_half1": np.asarray([[3, 0], [4, 0]]),
    }
    np.savez_compressed(output / "refinement_results.npz", **arrays)

    destination = build_archive(output)

    with np.load(destination, allow_pickle=False) as archive:
        assert np.allclose(
            archive["pmax_per_image_by_image_iter_000"],
            [0.2, 0.4, 0.1, 0.3],
        )
        assert np.array_equal(
            archive["sig_counts_by_image_iter_000"],
            [2, 4, 1, 3],
        )
        assert np.array_equal(
            archive["best_rotation_eulers_by_image_iter_000"][:, 0],
            [2, 4, 1, 3],
        )
        assert np.array_equal(
            archive["best_translations_by_image_iter_000"][:, 0],
            [2, 4, 1, 3],
        )
