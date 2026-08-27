from pathlib import Path

import numpy as np

from scripts.analyze_k1_state_swap_target import analyze_state_swap_target


def _write_star(path: Path, rows: list[tuple[str, float, int]]) -> None:
    lines = [
        "data_particles",
        "",
        "loop_",
        "_rlnImageName #1",
        "_rlnMaxValueProbDistribution #2",
        "_rlnNrOfSignificantSamples #3",
    ]
    lines.extend(" ".join(map(str, row)) for row in rows)
    path.write_text("\n".join(lines) + "\n")


def _write_half(path: Path, *, half: int, indices, pmax, support) -> None:
    n = len(indices)
    np.savez(
        path,
        rotation_eulers_deg=np.zeros((n, 3)),
        absolute_translations_pixels=np.zeros((n, 2)),
        max_posterior=np.asarray(pmax),
        significant_counts=np.asarray(support),
        original_image_indices=np.asarray(indices),
        zero_based_iteration=np.asarray([1]),
        one_based_iteration=np.asarray([2]),
        half=np.asarray([half]),
    )


def _fixture(tmp_path: Path, *, swapped_pmax: float):
    identities = ["000001@stack.mrcs", "000002@stack.mrcs", "000003@stack.mrcs"]
    particles = tmp_path / "particles.star"
    relion = tmp_path / "run_it002_data.star"
    _write_star(particles, [(identity, 1.0, 1) for identity in identities])
    _write_star(relion, [(identities[1], 0.23, 13), (identities[0], 1.0, 1), (identities[2], 1.0, 1)])
    _write_half(tmp_path / "it001_particle_state_half1.npz", half=1, indices=[0, 2], pmax=[1, 1], support=[1, 1])
    _write_half(tmp_path / "it001_particle_state_half2.npz", half=2, indices=[1], pmax=[0.18], support=[13])
    capture = tmp_path / "capture.npz"
    probabilities = np.full(10, (1.0 - swapped_pmax) / 9.0)
    probabilities[0] = swapped_pmax
    np.savez(
        capture,
        original_index=np.asarray(1),
        iteration=np.asarray(2),
        probs=probabilities.reshape(2, 5),
        candidate_mask=np.ones((2, 5), dtype=bool),
        reconstruction_n_significant=np.asarray(47),
    )
    significance = tmp_path / "significance.npz"
    np.savez(
        significance,
        original_index=np.asarray(1),
        one_based_iteration=np.asarray(2),
        n_significant=np.asarray(13),
    )
    return capture, significance, particles, relion


def test_state_swap_target_classifies_inherited_state(tmp_path):
    capture, significance, particles, relion = _fixture(tmp_path, swapped_pmax=0.229)
    report = analyze_state_swap_target(
        capture_path=capture,
        significance_capture_path=significance,
        autonomous_intermediates_dir=tmp_path,
        recovar_particles_star=particles,
        relion_star=relion,
        recovar_iteration=1,
    )
    assert report["classification"] == "inherited_state_or_reference"
    assert report["source_row_zero_based"] == 1
    assert report["identity"] == "000002@stack.mrcs"
    assert report["comparison"]["state_swap_to_autonomous_abs_pmax_error_ratio"] < 0.1
    assert report["state_swap"]["coarse_significant_support"] == 13
    assert report["state_swap"]["fine_reconstruction_support"] == 47


def test_state_swap_target_classifies_persistent_identical_input_residual(tmp_path):
    capture, significance, particles, relion = _fixture(tmp_path, swapped_pmax=0.18)
    report = analyze_state_swap_target(
        capture_path=capture,
        significance_capture_path=significance,
        autonomous_intermediates_dir=tmp_path,
        recovar_particles_star=particles,
        relion_star=relion,
        recovar_iteration=1,
    )
    assert report["classification"] == "identical_input_scoring_or_support_residual"
    assert report["comparison"]["state_swap_to_autonomous_abs_pmax_error_ratio"] == 1.0
