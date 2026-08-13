import struct
from pathlib import Path

import numpy as np

from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_indices_np
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_translation_angles_f32,
)
from scripts.analyze_k1_scale_aa_candidates import analyze


def _scalar(path: Path, value: float) -> None:
    path.write_bytes(struct.pack("<d", value))


def _real(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values, dtype="<f8").reshape(-1)
    path.write_bytes(struct.pack("<i", values.size) + values.tobytes())


def test_scale_aa_candidates_matches_rotation_and_translation_permutations(tmp_path: Path):
    recovar = tmp_path / "recovar.npz"
    native = tmp_path / "native"
    native.mkdir()
    prefix = "img0_part109_storeWavg_"
    rotations = np.asarray(
        [
            np.eye(3, dtype=np.float32),
            np.diag([-1.0, -1.0, 1.0]).astype(np.float32),
        ]
    )
    translations = np.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32)
    probabilities = np.asarray([[0.1, 0.2], [0.3, 0.399]], dtype=np.float32)
    window_indices, _ = make_fourier_window_indices_np(
        (128, 128), 60, square=False, include_dc=True, exact_radius=True
    )
    np.savez_compressed(
        recovar,
        schema=np.asarray("recovar-k1-scale-xa-aa-chunked-v4"),
        iteration=np.int64(2),
        half=np.int64(1),
        group_id=np.int64(109),
        original_index=np.int64(1096),
        current_size=np.int64(60),
        candidate_posterior_probs=probabilities,
        candidate_rotation_matrices=rotations,
        fine_translations=translations,
        scale_correction_pixel_mask=np.ones(window_indices.size, dtype=bool),
        ctf_probs_raw_sum_per_pixel=np.full(
            window_indices.size,
            np.sum(probabilities, dtype=np.float64),
            dtype=np.float32,
        ),
        scale_aa_per_shell=np.zeros(31, dtype=np.float64),
        candidate_aa_feature_per_shell=np.zeros((2, 2), dtype=np.float32),
        candidate_aa_feature_shell_ids=np.asarray([1, 2], dtype=np.int32),
    )

    native_rotation_order = np.asarray([1, 0])
    native_translation_order = np.asarray([1, 0])
    native_probabilities = probabilities[native_rotation_order][:, native_translation_order]
    phases = np.asarray(_relion_translation_angles_f32(translations, (128, 128)))
    native_phases = phases[native_translation_order]
    native_trans_xyz = np.concatenate(
        (native_phases[:, 0], native_phases[:, 1], np.zeros(2, dtype=np.float32))
    )
    _scalar(native / f"{prefix}orientation_num.bin", 2)
    _scalar(native / f"{prefix}translation_num.bin", 2)
    _scalar(native / f"{prefix}sum_weight.bin", 1.0)
    _scalar(native / f"{prefix}significant_weight.bin", 0.0)
    _real(native / f"{prefix}sorted_weights.bin", native_probabilities)
    _real(
        native / f"{prefix}eulers.bin",
        rotations[native_rotation_order].transpose(0, 2, 1),
    )
    _real(native / f"{prefix}trans_xyz.bin", native_trans_xyz)
    _real(native / f"{prefix}ctfs.bin", np.ones(60 * 31, dtype=np.float32))

    report = analyze(
        recovar,
        native,
        native_prefix=prefix,
        image_size=128,
    )

    assert report["rotation_join"]["exact_match_count"] == 2
    assert report["translation_join"]["exact_phase_match_count"] == 2
    assert report["posterior"]["union_total_variation"] == 0.0
    assert report["posterior"]["common_candidate_metric"]["relative_l2"] == 0.0
    assert report["aa_weight_swap"] is None
