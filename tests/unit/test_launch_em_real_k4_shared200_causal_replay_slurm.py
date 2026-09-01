from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import launch_em_real_k4_shared200_causal_replay_slurm as launcher


def _particles(*, assigned: bool = False, bad_half: bool = False) -> pd.DataFrame:
    halves = [1] * 93 + [2] * 107
    if bad_half:
        halves[-1] = 1
    values: dict[str, object] = {
        "_rlnImageName": [f"{index}@particles.256.mrcs" for index in range(1, 201)],
        "_rlnRandomSubset": halves,
    }
    if assigned:
        values["_rlnClassNumber"] = [1] * 200
    return pd.DataFrame(values)


def _shared(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": launcher.SHARED_SET_SCHEMA,
                "iteration": launcher.CASE.iteration,
                "same_visited_particle_ids": True,
                "relion_assigned_count": 200,
                "recovar_assigned_count": 200,
                "visited_particle_ids": [f"{index}@particles.256.mrcs" for index in range(1, 201)],
            }
        )
    )


def test_shared_target_rows_enforces_exact_set_and_half_split(monkeypatch, tmp_path):
    fixture = tmp_path / "fixture.star"
    relion = tmp_path / "relion.star"
    recovar = tmp_path / "recovar.star"
    shared = tmp_path / "shared.json"
    for path in (fixture, relion, recovar):
        path.touch()
    _shared(shared)
    tables = {
        fixture: (_particles(), pd.DataFrame({"_rlnOpticsGroup": [1]})),
        relion: (_particles(assigned=True), None),
        recovar: (_particles(assigned=True), None),
    }
    monkeypatch.setattr(launcher, "read_star", lambda value: tables[Path(value)])

    selected, optics, targets = launcher._shared_target_rows(
        fixture_star=fixture,
        shared_set_path=shared,
        relion_it1_data=relion,
        recovar_it1_data=recovar,
    )

    assert len(selected) == 200
    assert optics is not None
    assert targets["half_counts"] == {"1": 93, "2": 107}
    assert targets["original_indices_zero_based"] == list(range(200))


@pytest.mark.parametrize("failure", ["half", "assigned"])
def test_shared_target_rows_rejects_topology_drift(monkeypatch, tmp_path, failure):
    fixture = tmp_path / "fixture.star"
    relion = tmp_path / "relion.star"
    recovar = tmp_path / "recovar.star"
    shared = tmp_path / "shared.json"
    for path in (fixture, relion, recovar):
        path.touch()
    _shared(shared)
    fixture_table = _particles(bad_half=failure == "half")
    recovar_table = _particles(assigned=True)
    if failure == "assigned":
        recovar_table.loc[0, "_rlnClassNumber"] = 0
    tables = {
        fixture: (fixture_table, None),
        relion: (_particles(assigned=True), None),
        recovar: (recovar_table, None),
    }
    monkeypatch.setattr(launcher, "read_star", lambda value: tables[Path(value)])

    with pytest.raises(launcher.PreflightError):
        launcher._shared_target_rows(
            fixture_star=fixture,
            shared_set_path=shared,
            relion_it1_data=relion,
            recovar_it1_data=recovar,
        )


def test_shared_target_rows_rejects_old_balanced_half_assumption(monkeypatch, tmp_path):
    fixture = tmp_path / "fixture.star"
    relion = tmp_path / "relion.star"
    recovar = tmp_path / "recovar.star"
    shared = tmp_path / "shared.json"
    for path in (fixture, relion, recovar):
        path.touch()
    _shared(shared)
    balanced = _particles()
    balanced["_rlnRandomSubset"] = [1] * 100 + [2] * 100
    tables = {
        fixture: (balanced, None),
        relion: (_particles(assigned=True), None),
        recovar: (_particles(assigned=True), None),
    }
    monkeypatch.setattr(launcher, "read_star", lambda value: tables[Path(value)])

    with pytest.raises(launcher.PreflightError, match="half split drift"):
        launcher._shared_target_rows(
            fixture_star=fixture,
            shared_set_path=shared,
            relion_it1_data=relion,
            recovar_it1_data=recovar,
        )


def test_subset_star_is_deterministic_and_uses_absolute_stack(tmp_path):
    selected = _particles().iloc[:2].copy()
    optics = pd.DataFrame({"_rlnOpticsGroup": [1], "_rlnVoltage": [300]})
    stack = tmp_path / "particles.256.mrcs"
    stack.touch()
    first = tmp_path / "first.star"
    second = tmp_path / "second.star"

    launcher.write_deterministic_subset_star(output=first, selected=selected, optics=optics, particle_stack=stack)
    launcher.write_deterministic_subset_star(output=second, selected=selected, optics=optics, particle_stack=stack)

    assert first.read_bytes() == second.read_bytes()
    particles, reread_optics = launcher.read_star(str(first))
    assert reread_optics is not None
    assert particles["_rlnImageName"].tolist() == [f"1@{stack}", f"2@{stack}"]


def test_rendered_sbatch_is_single_gpu_nonexclusive_and_runs_all_arms(tmp_path):
    args = argparse.Namespace(
        output_root=tmp_path / "run",
        runtime_root=tmp_path / "runtime",
        control_pair_root=tmp_path / "pair",
        fixture_dir=tmp_path / "fixture",
        pixi_python=tmp_path / "python",
        relion_bind_source=tmp_path / "relion-src",
        relion_capture_binary=tmp_path / "relion_refine_mpi",
        partition="cryoem",
        account="gilles",
        constraint="h100",
        mem="192G",
        time_limit="02:00:00",
        relion_module="relion/5.0.0/gcc-11.5.0",
        cuda_module="cudatoolkit/12.8",
    )
    script = launcher.render_sbatch(
        args, expected_head="a" * 40, manifest_path=args.output_root / "launch_manifest.json"
    )

    assert "#SBATCH --gres=gpu:1" in script
    assert "--exclusive" not in script
    assert "run_native_arm control_a 0" in script
    assert "run_native_arm control_b 0" in script
    assert 'for class_id in 1 2 3 4; do run_native_arm "class${class_id}"' in script
    assert "--data-star" in script and "particles_shared200.star" in script
    assert "-eq 800" in script
    assert "audit_em_real_k4_shared200_causal_replay" in script
    assert "\n+  " not in script
    assert 'test "${REQ_TRES}" = "${ALLOC_TRES}"' in script
    assert '[[ "${ALLOC_TRES}" == *"gres/gpu=1"* ]]' in script


def test_manifest_record_rejects_checksum_drift(tmp_path):
    path = tmp_path / "input"
    path.write_text("sealed")
    record = launcher._file_record(path, role="unit input")
    launcher._validate_record(record)
    path.write_text("drifted")
    with pytest.raises(launcher.PreflightError, match="drift"):
        launcher._validate_record(record)


def test_cli_is_dry_run_by_default(tmp_path):
    args = launcher.parse_args(["--output-root", str(tmp_path / "run")])
    assert args.submit is False
