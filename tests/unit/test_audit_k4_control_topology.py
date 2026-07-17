from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starfile

from recovar.em.dense_single_volume.relion_worker_scale import (
    make_relion_dispatch_schedule_from_chunks,
    relion_class3d_follower_owners_from_schedule,
    relion_oracle_id,
    relion_oracle_manifest_sha256,
    relion_ordered_particle_sha256,
)
from scripts import audit_k4_control_topology as auditor


def _write_particles(path: Path, names: np.ndarray) -> None:
    starfile.write(
        {
            "particles": pd.DataFrame(
                {
                    "rlnImageName": names,
                    "rlnOpticsGroup": np.ones(names.size, dtype=np.int64),
                    "rlnRandomSubset": np.ones(names.size, dtype=np.int64),
                    "rlnGroupNumber": np.arange(1, names.size + 1, dtype=np.int64),
                }
            )
        },
        path,
    )


def _write_model(path: Path, *, current_size: int) -> None:
    path.write_text(f"data_model_general\n\n_rlnCurrentImageSize {current_size}\n_rlnNrClasses 4\n")


def _write_optimiser(path: Path, *, iteration: int, converged: bool) -> None:
    path.write_text(f"data_optimiser_general\n\n_rlnCurrentIteration {iteration}\n_rlnHasConverged {int(converged)}\n")


def _save_schedule(path: Path, schedule) -> None:
    np.savez_compressed(
        path,
        schema_version=np.int64(3),
        relion_iterations=schedule.relion_iterations,
        owner_by_sorted_position=schedule.owner_by_sorted_position,
        original_particle_id_by_sorted_position=schedule.original_particle_id_by_sorted_position,
        n_followers=np.int64(schedule.n_followers),
        pool_size=np.int64(schedule.pool_size),
        random_seed=np.int64(schedule.random_seed),
        oracle_id=np.asarray(schedule.oracle_id),
        oracle_manifest_sha256=np.asarray(schedule.oracle_manifest_sha256),
        oracle_artifact_paths=np.asarray(schedule.oracle_artifact_paths),
        particle_order_sha256=np.asarray(schedule.particle_order_sha256),
        particle_star_relative_path=np.asarray(schedule.particle_star_relative_path),
        dispatch_log_relative_path=np.asarray(schedule.dispatch_log_relative_path),
        source=np.asarray(schedule.source),
    )


def _fixture(tmp_path: Path, *, final_all_data: bool = False) -> dict[str, Path]:
    n_images = 8
    numbered_iterations = np.asarray([1, 2], dtype=np.int64)
    schedule_iterations = np.asarray([1, 2, 3] if final_all_data else [1, 2], dtype=np.int64)
    owners = np.asarray(
        [
            [0, 0, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
        ],
        dtype=np.int64,
    )[: schedule_iterations.size]
    originals = np.asarray(
        [
            [3, 0, 7, 1, 5, 2, 6, 4],
            [4, 6, 2, 5, 1, 7, 0, 3],
            [2, 7, 0, 6, 3, 5, 4, 1],
        ],
        dtype=np.int64,
    )[: schedule_iterations.size]
    names = np.asarray([f"{index + 1:06d}@particles.mrcs" for index in range(n_images)])
    oracle = tmp_path / "relion"
    oracle.mkdir()
    _write_particles(oracle / "run_it000_data.star", names)
    for row, iteration in enumerate(numbered_iterations):
        (oracle / f"run_it{iteration:03d}_data.star").write_text("data_particles\n")
        _write_model(oracle / f"run_it{iteration:03d}_model.star", current_size=(40, 52)[row])
        (oracle / f"run_it{iteration:03d}_sampling.star").write_text("data_sampling_general\n")
        _write_optimiser(
            oracle / f"run_it{iteration:03d}_optimiser.star",
            iteration=int(iteration),
            converged=False,
        )
    if final_all_data:
        (oracle / "run_data.star").write_text("data_particles\n")
        (oracle / "run_model.star").write_text("data_model_general\n")
        (oracle / "run_sampling.star").write_text("data_sampling_general\n")
        _write_optimiser(oracle / "run_optimiser.star", iteration=-1, converged=True)

    dispatch_rows = []
    for row, iteration in enumerate(schedule_iterations):
        dispatch_rows.extend(
            (2, int(iteration), int(owners[row, position]) + 1, position, int(originals[row, position]))
            for position in range(n_images)
        )
    dispatch_path = oracle / "dispatch.tsv"
    dispatch_path.write_text(
        "# RELION_DISPATCH_LOG_SCHEMA_V2\n"
        + "\n".join("\t".join(str(value) for value in row) for row in dispatch_rows)
        + "\n"
    )
    metadata = {
        "schema_version": 2,
        "dispatch_log_schema_version": 2,
        "schedule_schema_version": 3,
        "dispatch_log_relative_path": "dispatch.tsv",
        "n_particles": n_images,
        "n_followers": 2,
        "pool_size": 1,
        "random_seed": 17,
    }
    (oracle / "dispatch.tsv.recovar_schedule.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    artifacts = sorted(path.name for path in oracle.iterdir() if path.is_file())
    manifest = relion_oracle_manifest_sha256(oracle, artifacts)
    oracle_particles = starfile.read(oracle / "run_it000_data.star")
    if isinstance(oracle_particles, dict):
        oracle_particles = oracle_particles["particles"]
    particle_order = relion_ordered_particle_sha256(oracle_particles)
    oracle_id = relion_oracle_id(
        manifest_sha256=manifest,
        particle_order_sha256=particle_order,
    )
    flat_iterations = np.repeat(schedule_iterations, n_images)
    flat_positions = np.tile(np.arange(n_images, dtype=np.int64), schedule_iterations.size)
    schedule = make_relion_dispatch_schedule_from_chunks(
        relion_iterations=schedule_iterations,
        chunk_iterations=flat_iterations,
        chunk_first=flat_positions,
        chunk_last=flat_positions,
        chunk_ranks=(owners.reshape(-1) + 1),
        n_particles=n_images,
        original_particle_id_by_sorted_position=originals,
        n_followers=2,
        pool_size=1,
        random_seed=17,
        oracle_id=oracle_id,
        oracle_manifest_sha256=manifest,
        oracle_artifact_paths=artifacts,
        particle_order_sha256=particle_order,
        particle_star_relative_path="run_it000_data.star",
        dispatch_log_relative_path="dispatch.tsv",
        source=str(dispatch_path),
    )
    schedule_path = tmp_path / "dispatch_schedule.npz"
    _save_schedule(schedule_path, schedule)

    recovar_order = np.asarray([5, 1, 7, 0, 3, 6, 2, 4], dtype=np.int64)
    recovar_particles = tmp_path / "particles.star"
    _write_particles(recovar_particles, names[recovar_order])
    oracle_row = {name: index for index, name in enumerate(names.tolist())}
    particle_ids = np.asarray([oracle_row[name] for name in names[recovar_order].tolist()], dtype=np.int64)
    expected_owners = np.stack(
        [
            relion_class3d_follower_owners_from_schedule(
                schedule,
                particle_ids_by_image=particle_ids,
                optics_group_ids_by_image=np.zeros(n_images, dtype=np.int64),
                random_seed=17,
                relion_iteration=int(iteration),
            )
            for iteration in schedule_iterations
        ]
    )
    results = tmp_path / "refinement_results.npz"
    np.savez_compressed(
        results,
        n_images=np.int64(n_images),
        half1_indices=np.arange(n_images, dtype=np.int64),
        half2_indices=np.empty(0, dtype=np.int64),
        current_sizes=np.asarray([40, 52], dtype=np.int64),
        class_weights=np.full(4, 0.25, dtype=np.float64),
        class_weight_trajectory=np.full((2, 4), 0.25, dtype=np.float64),
        relion_dispatch_oracle_id=np.asarray(oracle_id),
        relion_dispatch_oracle_manifest_sha256=np.asarray(manifest),
        relion_dispatch_particle_order_sha256=np.asarray(particle_order),
        relion_scale_follower_owners_half1_trajectory=expected_owners[: numbered_iterations.size],
        relion_scale_follower_owners_half1=expected_owners[-1],
        relion_scale_follower_scales_numbered_pre_score_trajectory=np.ones((2, 2, 3)),
        relion_scale_follower_scales_numbered_post_mstep_trajectory=np.ones((2, 2, 3)),
        convergence_iteration=np.int64(2),
        convergence_has_converged=np.bool_(final_all_data),
        final_all_data_ran=np.bool_(final_all_data),
    )
    return {
        "results": results,
        "particles": recovar_particles,
        "relion": oracle,
        "schedule": schedule_path,
    }


def _audit(paths: dict[str, Path]) -> dict:
    return auditor.audit(
        recovar_results=paths["results"],
        recovar_particles_star=paths["particles"],
        relion_dir=paths["relion"],
        dispatch_schedule=paths["schedule"],
    )


def _rewrite_npz(path: Path, **updates) -> None:
    with np.load(path, allow_pickle=False) as payload:
        values = {key: payload[key] for key in payload.files}
    values.update(updates)
    np.savez_compressed(path, **values)


@pytest.mark.unit
def test_strict_k4_control_topology_passes_complete_nonconverged_trajectory(tmp_path):
    report = _audit(_fixture(tmp_path))

    assert report["status"] == "pass"
    assert report["combined_control_pass"] is True
    assert report["dispatch"]["hashes_exact"] is True
    assert report["dispatch"]["all_iterations_consumed_exactly"] is True
    assert [item["recovar"] for item in report["current_size_schedule"]] == [40, 52]
    assert report["convergence"]["exact"] is True
    assert report["finalization"]["exact"] is True
    assert "no correlation" in report["metric_policy"]
    assert "FSC-AUC" in report["metric_policy"]


@pytest.mark.unit
def test_strict_k4_control_topology_accepts_complete_converged_final_branch(tmp_path):
    report = _audit(_fixture(tmp_path, final_all_data=True))

    assert report["status"] == "pass"
    assert report["convergence"]["recovar"] == {
        "iteration": 2,
        "has_converged": True,
    }
    assert report["finalization"]["recovar_final_all_data_ran"] is True
    assert report["finalization"]["relion_final_all_data_ran"] is True


@pytest.mark.unit
def test_exact_mismatches_fail_combined_control_gate(tmp_path):
    paths = _fixture(tmp_path)
    with np.load(paths["results"], allow_pickle=False) as payload:
        owners = payload["relion_scale_follower_owners_half1_trajectory"].copy()
    owners[1, 3] = 1 - owners[1, 3]
    _rewrite_npz(
        paths["results"],
        relion_scale_follower_owners_half1_trajectory=owners,
        current_sizes=np.asarray([40, 54], dtype=np.int64),
        convergence_iteration=np.int64(1),
        final_all_data_ran=np.bool_(True),
    )

    report = _audit(paths)

    assert report["status"] == "fail"
    assert report["combined_control_pass"] is False
    assert any("dispatch owners differ" in failure for failure in report["failures"])
    assert any("current_size differs" in failure for failure in report["failures"])
    assert any("convergence iteration differs" in failure for failure in report["failures"])
    assert any("finalization path differs" in failure for failure in report["failures"])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"class_weight_trajectory": np.asarray([[0.25] * 4, [0.25, 0.25, np.nan, 0.25]])}, "non-finite"),
        ({"convergence_has_converged": np.asarray(0, dtype=np.int64)}, "boolean scalar"),
        ({"half2_indices": np.asarray([0], dtype=np.int64)}, "strict K=4 Class3D topology"),
    ],
)
def test_missing_or_malformed_control_artifacts_fail_closed(tmp_path, updates, match):
    paths = _fixture(tmp_path)
    _rewrite_npz(paths["results"], **updates)

    with pytest.raises(auditor.AuditError, match=match):
        _audit(paths)


@pytest.mark.unit
def test_unmanifested_numbered_sampling_artifact_fails_closed(tmp_path):
    paths = _fixture(tmp_path)
    (paths["relion"] / "run_it003_sampling.star").write_text("data_sampling_general\n")

    with pytest.raises(auditor.AuditError, match="not manifest-bound"):
        _audit(paths)


@pytest.mark.unit
def test_cli_writes_error_json_and_returns_two_for_missing_artifact(tmp_path):
    paths = _fixture(tmp_path)
    output = tmp_path / "audit.json"
    paths["schedule"].unlink()

    status = auditor.main(
        [
            "--recovar-results",
            str(paths["results"]),
            "--recovar-particles-star",
            str(paths["particles"]),
            "--relion-dir",
            str(paths["relion"]),
            "--dispatch-schedule",
            str(paths["schedule"]),
            "--output-json",
            str(output),
        ]
    )

    assert status == 2
    report = json.loads(output.read_text())
    assert report["schema"] == auditor.SCHEMA
    assert report["status"] == "error"
    assert report["combined_control_pass"] is False


@pytest.mark.unit
def test_cli_writes_error_json_for_corrupt_schedule(tmp_path):
    paths = _fixture(tmp_path)
    output = tmp_path / "audit.json"
    paths["schedule"].write_bytes(b"not an npz")

    status = auditor.main(
        [
            "--recovar-results",
            str(paths["results"]),
            "--recovar-particles-star",
            str(paths["particles"]),
            "--relion-dir",
            str(paths["relion"]),
            "--dispatch-schedule",
            str(paths["schedule"]),
            "--output-json",
            str(output),
        ]
    )

    assert status == 2
    report = json.loads(output.read_text())
    assert report["status"] == "error"
    assert report["combined_control_pass"] is False
