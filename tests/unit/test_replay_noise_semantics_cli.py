"""Mid-trajectory replay noise semantics: RELION continuation versus uninterrupted.

A replay that starts after RELION iteration N > 0 either reproduces a RELION
``--continue`` restart (MPI initialisation scores both halves with the half-1
sigma2_noise) or an uninterrupted trajectory (each half keeps its own spectrum).
Fixed-state comparisons against an uninterrupted RELION reference need the
latter; before this option the CLI could only request the former.
"""

import numpy as np
import pytest

from scripts import run_full_refinement

pytestmark = pytest.mark.unit


def _write_iteration(tmp_path, iteration, half1_noise, half2_noise):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")
    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@particles.mrcs", "2@particles.mrcs"],
            "rlnNormCorrection": [1.0, 1.0],
            "rlnGroupNumber": [1, 1],
        }
    )
    starfile.write({"particles": particles}, tmp_path / f"run_it{iteration:03d}_data.star")
    for half, noise in ((1, half1_noise), (2, half2_noise)):
        starfile.write(
            {
                "model_general": pd.DataFrame({"rlnNormCorrectionAverage": [1.0], "rlnSigmaOffsetsAngst": [2.0]}),
                "model_optics_group_1": pd.DataFrame({"rlnSigma2Noise": np.asarray(noise, dtype=np.float64)}),
                "model_groups": pd.DataFrame({"rlnGroupScaleCorrection": [1.0]}),
            },
            tmp_path / f"run_it{iteration:03d}_half{half}_model.star",
        )


def _slot0_noise(tmp_path, *, semantics, init_relion_iteration):
    broadcast = run_full_refinement._replay_process_start_noise_broadcast(
        semantics,
        init_relion_iteration,
        tmp_path,
    )
    overrides = run_full_refinement.relion_replay._build_replay_iteration_overrides(
        tmp_path,
        half1_idx=np.asarray([0], dtype=np.int64),
        half2_idx=np.asarray([1], dtype=np.int64),
        max_iter=0,
        ds_voxel=2.0,
        ds_grid=8,
        include_normcorr=False,
        init_relion_iteration=init_relion_iteration,
        process_start_noise_broadcast=broadcast,
    )
    return overrides[0]["noise_variance"]


def test_replay_noise_semantics_default_is_continuation():
    args = run_full_refinement._parse_args([])
    assert args.replay_noise_semantics == "continuation"


def test_uninterrupted_replay_keeps_half_specific_slot0_noise(tmp_path):
    _write_iteration(tmp_path, 10, [1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0])

    continuation_h1, continuation_h2 = _slot0_noise(tmp_path, semantics="continuation", init_relion_iteration=10)
    np.testing.assert_array_equal(continuation_h2, continuation_h1)

    uninterrupted_h1, uninterrupted_h2 = _slot0_noise(tmp_path, semantics="uninterrupted", init_relion_iteration=10)
    assert float(np.min(uninterrupted_h1)) == pytest.approx(1.0 * 8**4)
    assert float(np.min(uninterrupted_h2)) == pytest.approx(6.0 * 8**4)
    np.testing.assert_array_equal(uninterrupted_h1, continuation_h1)


@pytest.mark.parametrize(
    ("init_relion_iteration", "replay_dir"),
    [(0, "relion"), (10, None)],
)
def test_uninterrupted_replay_requires_a_mid_trajectory_replay(init_relion_iteration, replay_dir):
    with pytest.raises(ValueError, match="requires --perturb_replay_relion_dir"):
        run_full_refinement._replay_process_start_noise_broadcast(
            "uninterrupted",
            init_relion_iteration,
            replay_dir,
        )


def test_cli_passes_selected_noise_semantics_to_replay_overrides():
    import inspect

    source = inspect.getsource(run_full_refinement.main)
    start = source.index("replay_iteration_overrides = relion_replay._build_replay_iteration_overrides(")
    end = source.index("\n            )", start)
    assert "process_start_noise_broadcast=replay_process_start_noise_broadcast" in source[start:end]
