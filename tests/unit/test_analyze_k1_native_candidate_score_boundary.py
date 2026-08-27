import json
from pathlib import Path

import numpy as np

from scripts.analyze_k1_native_candidate_score_boundary import analyze


def _write_flat(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values)
    with path.open("wb") as stream:
        np.asarray([values.size], dtype=np.int32).tofile(stream)
        values.reshape(-1).tofile(stream)


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _fixture(tmp_path: Path, *, omit_native_last: bool = False) -> tuple[Path, Path]:
    rotations = np.stack([np.eye(3), _rotation_z(20.0)]).astype(np.float32)
    keys = [(0, 0), (0, 1), (1, 0)]
    if omit_native_last:
        keys = keys[:-1]
    native_rotation = np.asarray([key[0] for key in keys], dtype=np.int32)
    native_translation = np.asarray([key[1] for key in keys], dtype=np.int32)
    native_raw = np.asarray([10.0, 11.0, 12.0][: len(keys)], dtype=np.float64)
    native_prior = np.asarray([-1.0, -2.0, -1.0][: len(keys)], dtype=np.float64)
    unnormalized = -native_raw + native_prior
    probability = np.exp(unnormalized - np.max(unnormalized))
    probability /= probability.sum()
    significant = probability >= np.min(probability)

    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    _write_flat(dump_dir / "pass1_class0_fine_eulers.bin", rotations.astype(np.float64))
    _write_flat(dump_dir / "pass1_acc_rot_idx.bin", native_rotation)
    _write_flat(dump_dir / "pass1_acc_trans_idx.bin", native_translation)
    _write_flat(dump_dir / "pass1_exp_Mweight_raw_preprior.bin", native_raw)
    _write_flat(dump_dir / "pass1_candidate_combined_log_prior.bin", native_prior)
    _write_flat(dump_dir / "pass1_candidate_weight_normalized.bin", probability.astype(np.float64))
    _write_flat(
        dump_dir / "pass1_candidate_in_reconstruction_set.bin",
        significant.astype(np.int32),
    )

    candidate_mask = np.asarray([[True, True], [True, False]])
    raw = np.zeros((2, 2), dtype=np.float32)
    raw[0, 0], raw[0, 1], raw[1, 0] = 10.0, 11.0, 12.0
    rotation_prior = np.asarray([-1.0, -1.0], dtype=np.float64)
    translation_prior = np.asarray([0.0, -1.0], dtype=np.float64)
    rec_probability = np.zeros((2, 2), dtype=np.float64)
    full_unnormalized = np.asarray([-11.0, -13.0, -13.0])
    full_probability = np.exp(full_unnormalized - np.max(full_unnormalized))
    full_probability /= full_probability.sum()
    rec_probability[0, 0], rec_probability[0, 1], rec_probability[1, 0] = full_probability
    recovar_npz = tmp_path / "recovar.npz"
    np.savez(
        recovar_npz,
        rotations=rotations,
        candidate_mask=candidate_mask,
        raw_operand_raw_diff2=raw,
        rotation_log_prior=rotation_prior,
        translation_log_prior=translation_prior,
        probs=rec_probability,
        reconstruction_probs=rec_probability,
        reconstruction_mask=candidate_mask,
        original_index=np.asarray(84206),
    )
    return dump_dir, recovar_npz


def test_exact_boundary_fixture(tmp_path: Path) -> None:
    dump_dir, recovar_npz = _fixture(tmp_path)
    result = analyze(dump_dir, recovar_npz)

    assert result["first_nonidentical_boundary"] is None
    assert result["candidate_set"]["exact"] is True
    assert result["raw_cost"]["float32_exact"] is True
    assert result["log_prior"]["float32_exact"] is True
    assert result["posterior"]["float32_exact"] is True
    assert result["probability_field"] == "probs"
    assert result["significant_support"]["exact"] is True
    json.dumps(result)


def test_candidate_mismatch_is_reported_without_raising(tmp_path: Path) -> None:
    dump_dir, recovar_npz = _fixture(tmp_path, omit_native_last=True)
    result = analyze(dump_dir, recovar_npz)

    assert result["first_nonidentical_boundary"] == "candidate_set"
    assert result["candidate_set"]["native_count"] == 2
    assert result["candidate_set"]["recovar_count"] == 3
    assert result["candidate_set"]["recovar_only_first20"] == [
        {"rotation_row": 1, "translation_row": 0}
    ]
