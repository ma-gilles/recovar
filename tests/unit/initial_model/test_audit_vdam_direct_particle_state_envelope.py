from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from recovar.data_io.starfile import write_star
from scripts.audit_vdam_candidate_state_envelope import CandidateStateEnvelopeError
from scripts.audit_vdam_direct_particle_state_envelope import (
    audit_direct_particle_state_envelope,
)


def _particles(a_rotation: float, b_rotation: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_rlnImageName": ["1@particles.mrcs", "2@particles.mrcs"],
            "_rlnAngleRot": [a_rotation, b_rotation],
            "_rlnAngleTilt": [0.0, 0.0],
            "_rlnAnglePsi": [0.0, 0.0],
            "_rlnOriginXAngst": [0.0, 0.0],
            "_rlnOriginYAngst": [0.0, 0.0],
            "_rlnMaxValueProbDistribution": [0.5, 0.5],
        }
    )


def _write_checkpoint(root: Path, iteration: int, table: pd.DataFrame) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_star(str(root / f"run_it{iteration:03d}_data.star"), table)


def test_direct_particle_envelope_localizes_first_active_failure(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    native1 = tmp_path / "native1"
    native2 = tmp_path / "native2"
    fixture = tmp_path / "particles.star"
    write_star(str(fixture), _particles(0.0, 10.0))

    for iteration in (1, 2):
        _write_checkpoint(candidate, iteration, _particles(0.0 if iteration == 1 else 5.0, 10.0))
        _write_checkpoint(native1, iteration, _particles(0.0, 40.0))
        _write_checkpoint(native2, iteration, _particles(30.0 if iteration == 1 else 10.0, 10.0))
        (candidate / f"run_it{iteration:03d}_recovar_meta.json").write_text(
            json.dumps({"selected_particle_ids": [0, 1] if iteration == 1 else [0]})
        )

    report = audit_direct_particle_state_envelope(
        candidate_dir=candidate,
        native_dirs=[native1, native2],
        fixture_star=fixture,
        iterations=(1, 2),
    )

    assert report["result"] == "fail"
    assert report["first_failure_iteration"] == 2
    assert report["failure_count"] == 1
    assert report["checkpoints"][0]["pass"] is True
    assert report["checkpoints"][1]["first_particles_matching_no_native_repeat"] == [
        "1@particles.mrcs"
    ]


def test_direct_particle_envelope_rejects_noncanonical_iterations(tmp_path: Path) -> None:
    fixture = tmp_path / "particles.star"
    write_star(str(fixture), _particles(0.0, 10.0))

    with pytest.raises(CandidateStateEnvelopeError, match="sorted, unique, and positive"):
        audit_direct_particle_state_envelope(
            candidate_dir=tmp_path / "candidate",
            native_dirs=[tmp_path / "native1", tmp_path / "native2"],
            fixture_star=fixture,
            iterations=(2, 1),
        )
