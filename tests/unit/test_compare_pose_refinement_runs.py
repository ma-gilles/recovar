from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from compare_pose_refinement_runs import _load_em_npz


def test_load_em_npz_prefers_final_all_data_pose_keys(tmp_path):
    result_path = tmp_path / "refinement_results.npz"
    n_images = 2

    np.savez(
        result_path,
        best_rotation_eulers_final_by_image=np.zeros((n_images, 3), dtype=np.float32),
        best_rotation_eulers_final_all_data_by_image=np.full((n_images, 3), 10.0, dtype=np.float32),
        best_translations_final_by_image=np.zeros((n_images, 2), dtype=np.float32),
        best_translations_final_all_data_by_image=np.full((n_images, 2), 2.5, dtype=np.float32),
    )

    pose_set = _load_em_npz("em", result_path, n_images)

    np.testing.assert_allclose(pose_set.translations, np.full((n_images, 2), 2.5, dtype=np.float32))
    assert not np.allclose(pose_set.rotations, np.repeat(np.eye(3, dtype=np.float32)[None], n_images, axis=0))
