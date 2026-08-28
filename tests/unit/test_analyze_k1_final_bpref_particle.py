import numpy as np
import pytest

from scripts.analyze_k1_final_bpref_particle import load_recovar_pair


def _write_pair(tmp_path, *, change_posterior=False):
    local_score = tmp_path / "local_score_it016_image_7.npz"
    contribution = tmp_path / "bpref_contribution_rows_it016_h1.npz"
    posterior = np.arange(6, dtype=np.float64).reshape(1, 2, 3) / 20
    mask = posterior > 0
    shifted = (
        np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        + 1j * np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    ).astype(np.complex64)
    ctf2 = np.arange(4, dtype=np.float32).reshape(1, 4)
    np.savez(
        local_score,
        selected_global_image_indices=np.asarray([7], dtype=np.int64),
        current_size=np.asarray([8], dtype=np.int32),
        debug_iteration=np.asarray([16], dtype=np.int32),
        local_rotation_matrices=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        local_rotation_indices=np.asarray([5, 6], dtype=np.int64),
        translations=np.zeros((3, 2), dtype=np.float32),
        posterior=posterior,
        reconstruction_sample_mask=mask,
        debug_shifted_recon=shifted[0],
        debug_ctf2_over_nv_recon=ctf2[0],
    )
    padded_posterior = np.zeros((1, 4, 3), dtype=np.float64)
    padded_posterior[0, :2] = posterior[0]
    if change_posterior:
        padded_posterior[0, 0, 0] += 1
    padded_mask = np.zeros((1, 4, 3), dtype=bool)
    padded_mask[0, :2] = mask[0]
    np.savez(
        contribution,
        original_indices=np.asarray([7], dtype=np.int64),
        current_size=np.asarray(8, dtype=np.int64),
        iteration=np.asarray(16, dtype=np.int32),
        mstep_shifted_recon=shifted,
        mstep_ctf2_over_nv=ctf2,
        window_indices=np.arange(4, dtype=np.int64),
        active_global_rotation_indices=np.asarray([5, 6], dtype=np.int64),
        posterior_probs=padded_posterior,
        reconstruction_mask=padded_mask,
    )
    return local_score, contribution


@pytest.mark.unit
def test_load_recovar_pair_joins_score_and_mstep_operands(tmp_path):
    local_score, contribution = _write_pair(tmp_path)
    joined = load_recovar_pair(local_score, contribution)
    assert int(joined["original_index"]) == 7
    assert int(joined["current_size"]) == 8
    assert joined["physical_iteration"] == 16
    assert joined["reconstruction_probs"].shape == (2, 3)
    assert joined["shifted_recon"].shape == (3, 4)
    assert joined["ctf2_over_nv_recon"].shape == (4,)


@pytest.mark.unit
def test_load_recovar_pair_rejects_cross_file_posterior_drift(tmp_path):
    local_score, contribution = _write_pair(tmp_path, change_posterior=True)
    with pytest.raises(ValueError, match="posterior changed"):
        load_recovar_pair(local_score, contribution)
