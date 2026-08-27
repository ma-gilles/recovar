from pathlib import Path

import numpy as np
import pytest

from scripts.audit_em_particle_state_distribution import AuditError
from scripts.audit_k1_live_particle_state import audit_live_particle_state


def _write_star(path: Path, rows: list[tuple[str, float, int, float, float, float, float, float]]) -> None:
    lines = [
        "data_particles",
        "",
        "loop_",
        "_rlnImageName #1",
        "_rlnMaxValueProbDistribution #2",
        "_rlnNrOfSignificantSamples #3",
        "_rlnAngleRot #4",
        "_rlnAngleTilt #5",
        "_rlnAnglePsi #6",
        "_rlnOriginXAngst #7",
        "_rlnOriginYAngst #8",
    ]
    lines.extend(" ".join(map(str, row)) for row in rows)
    path.write_text("\n".join(lines) + "\n")


def _write_half(
    path: Path,
    *,
    half: int,
    indices: np.ndarray,
    pmax: np.ndarray,
    support: np.ndarray,
    eulers: np.ndarray,
    translations_pixels: np.ndarray,
) -> None:
    np.savez(
        path,
        rotation_eulers_deg=eulers,
        absolute_translations_pixels=translations_pixels,
        max_posterior=pmax,
        significant_counts=support,
        original_image_indices=indices,
        zero_based_iteration=np.asarray([1], dtype=np.int32),
        one_based_iteration=np.asarray([2], dtype=np.int32),
        half=np.asarray([half], dtype=np.int32),
    )


def test_audit_live_particle_state_aligns_halves_and_converts_translations(tmp_path):
    input_star = tmp_path / "particles.star"
    relion_star = tmp_path / "run_it002_data.star"
    identities = ["000001@stack.mrcs", "000002@stack.mrcs", "000003@stack.mrcs"]
    _write_star(
        input_star,
        [(identity, 1.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0) for identity in identities],
    )
    _write_star(
        relion_star,
        [
            (identities[2], 0.3, 3, 30.0, 31.0, 32.0, 3.0, -3.0),
            (identities[0], 0.1, 1, 10.0, 11.0, 12.0, 1.0, -1.0),
            (identities[1], 0.2, 2, 20.0, 21.0, 22.0, 2.0, -2.0),
        ],
    )
    _write_half(
        tmp_path / "it001_particle_state_half1.npz",
        half=1,
        indices=np.asarray([2, 0]),
        pmax=np.asarray([0.3, 0.1]),
        support=np.asarray([3, 1]),
        eulers=np.asarray([[30.0, 31.0, 32.0], [10.0, 11.0, 12.0]]),
        translations_pixels=np.asarray([[1.5, -1.5], [0.5, -0.5]]),
    )
    _write_half(
        tmp_path / "it001_particle_state_half2.npz",
        half=2,
        indices=np.asarray([1]),
        pmax=np.asarray([0.2]),
        support=np.asarray([2]),
        eulers=np.asarray([[20.0, 21.0, 22.0]]),
        translations_pixels=np.asarray([[1.0, -1.0]]),
    )

    report = audit_live_particle_state(
        intermediates_dir=tmp_path,
        recovar_particles_star=input_star,
        relion_star=relion_star,
        recovar_iteration=1,
        pixel_size_angstrom=2.0,
    )

    metrics = report["recovar_vs_relion"]
    assert report["status"] == "complete"
    assert report["half_counts"] == {"1": 2, "2": 1}
    assert metrics["pmax"]["absolute"]["max"] == 0.0
    assert metrics["significant_support"]["different_count"] == 0
    assert metrics["angular_error_deg"]["max"] < 1e-5
    assert metrics["translation_error"]["max"] == 0.0
    assert report["schema"] == "recovar.em.k1_live_particle_state_audit.v2"
    top_pmax = report["largest_discrepancies"]["pmax_signed_delta"]
    assert [row["source_row_zero_based"] for row in top_pmax[:3]] == [0, 1, 2]
    assert top_pmax[0] == {
        "source_row_zero_based": 0,
        "identity": identities[0],
        "half": 1,
        "metric": 0.0,
        "metric_is_signed": True,
        "recovar": 0.1,
        "relion": 0.1,
    }
    top_translation = report["largest_discrepancies"]["translation_error"]
    assert top_translation[0]["recovar"] == [1.0, -1.0]
    assert top_translation[0]["relion"] == [1.0, -1.0]


def test_audit_live_particle_state_rejects_incomplete_partition(tmp_path):
    input_star = tmp_path / "particles.star"
    relion_star = tmp_path / "run_it002_data.star"
    rows = [
        ("000001@stack.mrcs", 1.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0),
        ("000002@stack.mrcs", 1.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    _write_star(input_star, rows)
    _write_star(relion_star, rows)
    for half in (1, 2):
        _write_half(
            tmp_path / f"it001_particle_state_half{half}.npz",
            half=half,
            indices=np.asarray([0]),
            pmax=np.asarray([1.0]),
            support=np.asarray([1]),
            eulers=np.zeros((1, 3)),
            translations_pixels=np.zeros((1, 2)),
        )

    with pytest.raises(AuditError, match="disjoint complete"):
        audit_live_particle_state(
            intermediates_dir=tmp_path,
            recovar_particles_star=input_star,
            relion_star=relion_star,
            recovar_iteration=1,
            pixel_size_angstrom=2.0,
        )
