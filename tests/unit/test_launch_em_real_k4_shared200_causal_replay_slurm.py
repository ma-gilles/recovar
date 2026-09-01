from __future__ import annotations

import argparse
import json
import subprocess
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


def test_image_identity_mapping_is_fixed_width_absolute_and_deterministic(tmp_path):
    stack = tmp_path / "particles.256.mrcs"
    stack.touch()
    particles = _particles().iloc[:2].copy()
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"

    launcher.write_fixed_image_identity_mapping(
        output=first, particles=particles, particle_stack=stack, stack_image_count=2
    )
    launcher.write_fixed_image_identity_mapping(
        output=second, particles=particles, particle_stack=stack, stack_image_count=2
    )

    assert first.read_bytes() == second.read_bytes()
    identities = launcher.np.load(first, allow_pickle=False)
    assert identities.dtype.kind == "S"
    assert identities.astype(str).tolist() == [f"1@{stack}", f"2@{stack}"]

    shuffled = particles.iloc[::-1].reset_index(drop=True)
    launcher.write_fixed_image_identity_mapping(
        output=second, particles=shuffled, particle_stack=stack, stack_image_count=2
    )
    assert launcher.np.array_equal(launcher.np.load(second, allow_pickle=False), identities)


def test_image_identity_mapping_indexes_sparse_physical_stack_ids(tmp_path):
    stack = tmp_path / "particles.256.mrcs"
    stack.touch()
    particles = _particles().iloc[:2].copy()
    particles["_rlnImageName"] = [f"2@{stack}", f"5@{stack}"]
    output = tmp_path / "mapping.npy"

    launcher.write_fixed_image_identity_mapping(
        output=output, particles=particles, particle_stack=stack, stack_image_count=5
    )

    identities = launcher.np.load(output, allow_pickle=False).astype(str)
    assert identities.tolist() == ["", f"2@{stack}", "", "", f"5@{stack}"]


def test_iteration0_continuation_bundle_restores_preinitialisation_offsets(tmp_path):
    pair = tmp_path / "pair"
    output = tmp_path / "output"
    pair.mkdir()
    model = pair / "run_it000_model.star"
    data = pair / "run_it000_data.star"
    model.write_text("model\n")
    data.write_text("data\n")
    missing_sampling = pair / "run_it000_sampling.star"
    optimiser = pair / "run_it000_optimiser.star"
    optimiser.write_text(
        "data_optimiser_general\n\n"
        f"_rlnModelStarFile {model}\n"
        f"_rlnExperimentalDataStarFile {data}\n"
        f"_rlnOrientSamplingStarFile {missing_sampling}\n"
        "_rlnDoGradientRefine 1\n"
    )
    sampling = pair / "run_it001_sampling.star"
    sampling.write_text(
        "data_sampling_general\n\n"
        "_rlnHealpixOrder 1\n"
        "_rlnSymmetryGroup C1\n"
        "_rlnPsiStep 30.000000\n"
        "_rlnOffsetRange 9.825000\n"
        "_rlnOffsetStep 3.275000\n"
        "_rlnSamplingPerturbInstance -0.07991\n"
        "_rlnSamplingPerturbFactor 0.500000\n"
        "_rlnOffsetRangeOriginal 9.825000\n"
        "_rlnOffsetStepOriginal 3.275000\n"
    )

    replay_optimiser, replay_sampling, provenance = launcher.materialize_iteration0_continuation_bundle(
        source_optimiser=optimiser,
        source_sampling=sampling,
        output_dir=output,
    )

    metadata = launcher.read_relion_sampling_metadata(replay_sampling)
    assert metadata["offset_range"] == 6.0
    assert metadata["offset_step"] == 2.0
    assert metadata["random_perturbation"] == 0.0
    replay_text = replay_optimiser.read_text()
    assert launcher._star_scalar(replay_text, "_rlnOrientSamplingStarFile") == str(replay_sampling.resolve())
    assert launcher._star_scalar(replay_text, "_rlnModelStarFile") == str(model)
    assert launcher._star_scalar(replay_text, "_rlnDoGradientRefine") == "1"
    assert provenance["method"].startswith("iteration-1 topology")


def test_iteration0_continuation_bundle_rejects_angstrom_geometry_drift(tmp_path):
    model = tmp_path / "run_it000_model.star"
    data = tmp_path / "run_it000_data.star"
    model.touch()
    data.touch()
    optimiser = tmp_path / "run_it000_optimiser.star"
    optimiser.write_text(
        f"_rlnModelStarFile {model}\n"
        f"_rlnExperimentalDataStarFile {data}\n"
        f"_rlnOrientSamplingStarFile {tmp_path / 'run_it000_sampling.star'}\n"
        "_rlnDoGradientRefine 1\n"
    )
    sampling = tmp_path / "run_it001_sampling.star"
    sampling.write_text(
        "_rlnHealpixOrder 1\n_rlnSymmetryGroup C1\n_rlnPsiStep 30\n"
        "_rlnOffsetRange 9.9\n_rlnOffsetStep 3.275\n"
        "_rlnSamplingPerturbInstance 0\n_rlnSamplingPerturbFactor 0.5\n"
        "_rlnOffsetRangeOriginal 9.9\n_rlnOffsetStepOriginal 3.275\n"
    )
    with pytest.raises(launcher.PreflightError, match="offset range drift"):
        launcher.materialize_iteration0_continuation_bundle(
            source_optimiser=optimiser,
            source_sampling=sampling,
            output_dir=tmp_path / "output",
        )


def test_rendered_sbatch_is_single_gpu_nonexclusive_and_runs_all_arms(tmp_path):
    args = argparse.Namespace(
        output_root=tmp_path / "run",
        runtime_root=tmp_path / "runtime",
        control_pair_root=tmp_path / "pair",
        fixture_dir=tmp_path / "fixture",
        pixi_python=tmp_path / "python",
        relion_bind_source=tmp_path / "relion-src",
        relion_capture_binary=tmp_path / "relion_refine",
        partition="cryoem",
        account="gilles",
        constraint="h100",
        mem="192G",
        time_limit="02:00:00",
        cuda_module="cudatoolkit/12.8",
        native_smoke_only=False,
    )
    script = launcher.render_sbatch(
        args, expected_head="a" * 40, manifest_path=args.output_root / "launch_manifest.json"
    )
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    assert "#SBATCH --gres=gpu:1" in script
    assert "--exclusive" not in script
    assert "run_native_arm control_a 0" in script
    assert "run_native_arm control_b 0" in script
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "srun --ntasks=1 --cpus-per-task=8" in script
    assert "--mpi=pmix" not in script
    assert "--gpu 0 --j 8" in script
    assert "--auto_iter_max 1" in script
    assert "--iter 1" not in script
    assert "RELION_BPRE_CAPTURE_MAX_PARTICLES_PER_RANK=200" in script
    assert "RELION_BPRE_CAPTURE_EXPECTED_FOLLOWERS=1" in script
    assert "RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR" in script
    assert "RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE=1" in script
    assert "RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY" in script
    assert f"RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256={launcher.EXPECTED_PARTICLE_STACK_SHA256}" in script
    assert "unset RECOVAR_BPREF_CONTRIBUTION_DUMP_CLASS RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF" in script
    assert "unset RECOVAR_BPREF_CONTRIBUTION_TARGET_ONLY RECOVAR_BPREF_CONTRIBUTION_STOP_AFTER_TARGET" in script
    assert "inputs/continuation/run_it000_optimiser_replay.star" in script
    assert "pair/relion/run_it000_optimiser.star" not in script
    assert launcher.EXPECTED_CONTINUED_ITER0_MARKER in script
    assert "grep -Fxc" in script
    assert 'for class_id in 1 2 3 4; do run_native_arm "class${class_id}"' in script
    assert "--data-star" in script and "particles_shared200.star" in script
    assert "-eq 800" in script
    assert "audit_em_real_k4_shared200_causal_replay" in script
    assert "\n+  " not in script
    assert 'test "${REQ_TRES}" = "${ALLOC_TRES}"' in script
    assert '[[ "${ALLOC_TRES}" == *"gres/gpu=1"* ]]' in script
    assert "module load relion" not in script
    assert "module load cudatoolkit/12.8" in script
    assert "PIXI_NVIDIA_ROOT=" in script
    assert "PIXI_NVIDIA_LIB_DIRS=" in script
    assert "libcusparse.so*" in script
    assert 'export LD_LIBRARY_PATH="${PIXI_NVIDIA_LIB_DIRS}:${CUDA_TARGET_LIB_DIR}:${PIXI_ENV_ROOT}/lib:' in script
    assert '${PIXI_ENV_ROOT}/include/fftw/fftw3.h' in script
    assert 'export CMAKE_INCLUDE_PATH="${PIXI_ENV_ROOT}/include/fftw:${PIXI_ENV_ROOT}/include:' in script
    assert 'export CMAKE_LIBRARY_PATH="${PIXI_ENV_ROOT}/lib:' in script


def test_rendered_native_smoke_exits_after_one_continuation_arm(tmp_path):
    args = argparse.Namespace(
        output_root=tmp_path / "run",
        runtime_root=tmp_path / "runtime",
        control_pair_root=tmp_path / "pair",
        fixture_dir=tmp_path / "fixture",
        pixi_python=tmp_path / "python",
        relion_bind_source=tmp_path / "relion-src",
        relion_capture_binary=tmp_path / "relion_refine",
        partition="cryoem",
        account="gilles",
        constraint="h100",
        mem="192G",
        time_limit="02:00:00",
        cuda_module="cudatoolkit/12.8",
        native_smoke_only=True,
    )
    script = launcher.render_sbatch(
        args, expected_head="a" * 40, manifest_path=args.output_root / "launch_manifest.json"
    )
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    first_arm = script.index("run_native_arm control_a 0")
    smoke_exit = script.index("exit 0", first_arm)
    second_arm = script.index("run_native_arm control_b 0")
    assert first_arm < smoke_exit < second_arm
    assert "run_it001_sampling.star" in script
    assert "RELION_SAMPLING_PERTURBATION_OVERRIDE" in script
    assert "scripts.audit_em_real_kclass_initialmodel" in script
    assert "--minimum-fsc-auc 0.999" in script
    assert "--minimum-assignment-accuracy 0.995" in script
    assert "native_smoke_trajectory.json" in script


def test_manifest_record_rejects_checksum_drift(tmp_path):
    path = tmp_path / "input"
    path.write_text("sealed")
    record = launcher._file_record(path, role="unit input")
    launcher._validate_record(record)
    path.write_text("drifted")
    with pytest.raises(launcher.PreflightError, match="drift"):
        launcher._validate_record(record)


def test_gradient_replay_rejects_mpi_capture_binary(tmp_path):
    launcher._validate_capture_binary_mode(tmp_path / "relion_refine")
    with pytest.raises(launcher.PreflightError, match="non-MPI"):
        launcher._validate_capture_binary_mode(tmp_path / "relion_refine_mpi")


def test_cli_is_dry_run_by_default(tmp_path):
    args = launcher.parse_args(["--output-root", str(tmp_path / "run")])
    assert args.submit is False
    assert args.native_smoke_only is False
    assert args.relion_capture_source == launcher.DEFAULT_RELION_CAPTURE_SOURCE.resolve()
    assert args.relion_capture_binary == launcher.DEFAULT_RELION_CAPTURE_BINARY.resolve()
    assert launcher.EXPECTED_CAPTURE_RELION_HEAD == "8680e84c906a5eeedad7eeda2703d617b1f9e9e5"
    assert launcher.EXPECTED_CAPTURE_RELION_TREE == "0ed8161a7dd10ddcb01ff4bb10d41ddd25e9fd69"
    assert launcher.EXPECTED_CAPTURE_RELION_BINARY_SHA256 == (
        "882a37de3449ede0132f3bed29638603b880ee3abf46d753b5f6a28ef9f90afd"
    )


def test_input_closure_includes_gradient_moment_maps(tmp_path):
    args = argparse.Namespace(
        fixture_dir=tmp_path / "fixture",
        shared_set=tmp_path / "shared.json",
        control_pair_root=tmp_path / "pair",
        relion_capture_binary=tmp_path / "relion_refine",
        relion_bind_source=tmp_path / "relion-src",
        pixi_python=tmp_path / "python",
    )
    roles = {role for role, _ in launcher._input_paths(args)}
    for class_id in range(1, 5):
        assert f"RELION initial first moment {class_id}" in roles
        assert f"RELION initial second moment {class_id}" in roles
