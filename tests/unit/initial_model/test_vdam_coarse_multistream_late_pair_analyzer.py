import hashlib
import json
import subprocess
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile

from scripts import analyze_vdam_coarse_multistream_late_pair as analyzer

GPU_UUID = "GPU-00000000-1111-2222-3333-444444444444"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_manifest(path: Path, entries: list[tuple[Path, str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{_sha256(artifact)}  {name}\n" for artifact, name in entries))
    return _sha256(path)


def _audit(*, workers: int, atomic: bool) -> dict:
    multistream = workers == 8
    return {
        "score_mode": "gaussian",
        "translation_count": 29,
        "requested_fused": True,
        "effective_fused": True,
        "requested_workers": workers,
        "effective_workers": workers,
        "requested_atomic": atomic,
        "effective_atomic": atomic,
        "wrapper": (
            "relion_coarse_diff2_projector_multistream_f32" if multistream else "relion_coarse_diff2_projector_f32"
        ),
        "target": (
            "cuda_relion_coarse_diff2_projector_multistream_f32"
            if multistream
            else "cuda_relion_coarse_diff2_projector_f32"
        ),
        "counts": {
            "fused_calls": 3,
            "actual_rows": 3000,
            "multistream_calls": 3 if multistream else 0,
            "native_atomic_selected_calls": 3 if atomic else 0,
        },
    }


def _metadata(*, workers: int, atomic: bool) -> dict:
    return {
        "n_translations": 116,
        "oversampling": 1,
        "joint_halfset_particle_stream": True,
        "halfset_ids": [0, 1],
        "selected_particle_ids": [0, 1, 2],
        "best_pose_rotation_ids": [3, 4, 5],
        "best_pose_rotations": [[0.0, 0.0, 0.0]] * 3,
        "best_pose_translations": [[0.0, 0.0]] * 3,
        "class_assignments": [0, 0, 0],
        "max_posterior_per_image": [0.75, 0.8, 0.9],
        "pose_assignments": [1, 2, 3],
        "halfset_0_class_assignments": [0, 0, 0],
        "sparse_pass2_profile_summary": {
            "pass1_time_s": 4.0,
            "pass2_time_s": 1.5,
        },
        "halfset_0_profile_summary": {
            "em_time_s": 6.0,
            "coarse_selector_audit": _audit(workers=workers, atomic=atomic),
        },
    }


def _write_star(path: Path) -> None:
    optics = pd.DataFrame(
        {
            "rlnOpticsGroup": [1],
            "rlnImagePixelSize": [1.5],
            "rlnImageSize": [4],
        }
    )
    particles = pd.DataFrame(
        {
            "rlnImageName": ["000001@particles.mrcs", "000002@particles.mrcs", "000003@particles.mrcs"],
            "rlnAngleRot": [1.0, 2.0, 3.0],
            "rlnAngleTilt": [4.0, 5.0, 6.0],
            "rlnAnglePsi": [7.0, 8.0, 9.0],
            "rlnOriginXAngst": [0.0, 0.1, 0.2],
            "rlnOriginYAngst": [0.0, -0.1, -0.2],
            "rlnClassNumber": [1, 1, 1],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    starfile.write({"optics": optics, "particles": particles}, path, overwrite=True)


def _write_map(path: Path, value: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.arange(64, dtype=np.float32).reshape(4, 4, 4) if value is None else value
    with mrcfile.new(path, overwrite=True) as stream:
        stream.set_data(np.asarray(data, dtype=np.float32))


def _write_junit(path: Path) -> None:
    path.write_text(
        '<testsuite name="focused" tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="cuda" name="lane_envelope"/></testsuite>\n'
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _build_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "late_pair"
    repo = tmp_path / "repo"
    provenance = root / "provenance"
    repo.mkdir()
    provenance.mkdir(parents=True)
    (root / "COMPLETED").touch()

    source = repo / "source.py"
    source.write_text("VALUE = 1\n")
    gate_source_artifact = repo / "gate_source.py"
    gate_source_artifact.write_text("GATE_VALUE = 1\n")
    interpreter = repo / "python"
    interpreter.write_bytes(b"pinned pixi interpreter")
    _git(repo, "init", "--quiet")
    _git(repo, "add", "source.py", "gate_source.py", "python")
    _git(
        repo,
        "-c",
        "user.name=RECOVAR test",
        "-c",
        "user.email=recovar-test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    git_head = _git(repo, "rev-parse", "HEAD")
    git_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    source_manifest = provenance / "source_manifest.sha256"
    source_digest = _write_manifest(source_manifest, [(source, "source.py")])
    (provenance / "source_manifest.final.sha256").write_bytes(source_manifest.read_bytes())

    input_file = tmp_path / "input.dat"
    input_file.write_bytes(b"fixture")
    input_digest = _write_manifest(
        provenance / "input_manifest.sha256",
        [(input_file, str(input_file.resolve()))],
    )

    cuda = root / "runtime" / "cuda" / "libcuda_backproject.so"
    binding = root / "runtime" / "relion_bind" / "_relion_bind_core.so"
    cuda.parent.mkdir(parents=True)
    binding.parent.mkdir(parents=True)
    cuda.write_bytes(b"qualified cuda")
    binding.write_bytes(b"qualified relion binding")
    demangled = provenance / "cuda_resource_usage.demangled.txt"
    demangled.write_text("relion coarse CUDA resource report\n")
    resource_report = provenance / "coarse_atomic_resources.json"
    _write_json(
        resource_report,
        {
            "binary_sha256": _sha256(cuda),
            "demangled_resource_report_sha256": _sha256(demangled),
            "resources": {
                "canonical": {"registers_per_thread": 56, "static_shared_bytes": 15232},
                "atomic": {"registers_per_thread": 48, "static_shared_bytes": 7040},
            },
        },
    )
    (provenance / "qualified_cuda.sha256").write_text(f"{_sha256(cuda)}  {cuda.resolve()}\n")
    (provenance / "relion_bind.sha256").write_text(f"{_sha256(binding)}  {binding.resolve()}\n")
    (provenance / "interpreter.sha256").write_text(f"{_sha256(interpreter)}  {interpreter.resolve()}\n")

    qualified = tmp_path / "qualified_gpu_gate"
    (qualified / "provenance").mkdir(parents=True)
    (qualified / "COMPLETED").touch()

    run = {
        "schema": analyzer.RUN_SCHEMA,
        "classification": "diagnostic_performance_only",
        "job_id": "12345",
        "git_head": git_head,
        "git_tree": git_tree,
        "gpu_uuid": GPU_UUID,
        "node": "della-h21g4",
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "cuda_sha256": _sha256(cuda),
        "relion_bind_sha256": _sha256(binding),
        "interpreter_sha256": _sha256(interpreter),
        "resource_report_sha256": _sha256(resource_report),
        "resource_demangled_sha256": _sha256(demangled),
        "source_manifest_sha256": source_digest,
        "source_manifest_scope": "selected_high_risk_files",
        "input_manifest_sha256": input_digest,
        "qualified_gpu_gate_root": str(qualified.resolve()),
        "execution_order": list(analyzer.ARM_LABELS),
        "coarse_multistream_workers": [spec[1] for spec in analyzer.ARM_SPECS],
        "single_lane_canonical": [False] * len(analyzer.ARM_SPECS),
        "native_atomic_reduction": [spec[2] for spec in analyzer.ARM_SPECS],
        "focused_test_node": analyzer.FOCUSED_TEST_NODE,
        "raw_image_cache": "off",
        "exact_local_bucket_radix": 4,
        "exact_local_physical_order_chunk_size": 0,
        "acceptance_rule": analyzer.ACCEPTANCE_RULE,
        "science_promotion_allowed": False,
    }
    _write_json(provenance / "run.json", run)
    scalar_files = {
        "repo_head.txt": run["git_head"],
        "repo_tree.txt": run["git_tree"],
        "slurm_job_id.txt": run["job_id"],
        "selected_gpu_uuid.txt": run["gpu_uuid"],
        "node.txt": run["node"],
        "gpu_name.txt": run["gpu_name"],
        "allocated_gpu_uuids.csv": run["gpu_uuid"],
        "visible_gpu_uuids.csv": run["gpu_uuid"],
    }
    for name, value in scalar_files.items():
        (provenance / name).write_text(f"{value}\n")
    (provenance / "repo_status.txt").write_text("")
    (provenance / "repo_diff.sha256").write_text(f"{hashlib.sha256(b'').hexdigest()}  -\n")

    gate_provenance = qualified / "provenance"
    gate_source = gate_provenance / "source_manifest.sha256"
    gate_source_digest = _write_manifest(gate_source, [(gate_source_artifact, "gate_source.py")])
    (gate_provenance / "source_manifest.final.sha256").write_bytes(gate_source.read_bytes())
    gate_cuda = qualified / "runtime" / "cuda" / "libcuda_backproject.so"
    gate_cuda.parent.mkdir(parents=True)
    gate_cuda.write_bytes(cuda.read_bytes())
    (gate_provenance / "qualified_cuda.sha256").write_text(f"{_sha256(gate_cuda)}  {gate_cuda.resolve()}\n")
    (gate_provenance / "interpreter.sha256").write_text(f"{_sha256(interpreter)}  {interpreter.resolve()}\n")
    (gate_provenance / "coarse_kernel_resources.json").write_bytes(resource_report.read_bytes())
    (gate_provenance / "cuda_resource_usage.demangled.txt").write_bytes(demangled.read_bytes())
    gate_run = {
        "schema": analyzer.FOCUSED_GATE_SCHEMA,
        "classification": "performance_only_qualification",
        "job_id": "12344",
        "git_head": run["git_head"],
        "git_tree": run["git_tree"],
        "interpreter_sha256": run["interpreter_sha256"],
        "gpu_uuid": run["gpu_uuid"],
        "node": run["node"],
        "gpu_name": run["gpu_name"],
        "cuda_sha256": run["cuda_sha256"],
        "source_manifest_sha256": gate_source_digest,
        "source_manifest_scope": "selected_high_risk_files",
        "cuda_stage_wall_s": 2.5,
        "test_wall_s": 3.5,
        "pytest_node": "focused_a focused_b focused_c",
        "tests": 3,
        "passed": 3,
        "science_promotion_allowed": False,
    }
    _write_json(gate_provenance / "run.json", gate_run)
    gate_scalar_files = {
        "repo_head.txt": gate_run["git_head"],
        "repo_tree.txt": gate_run["git_tree"],
        "slurm_job_id.txt": gate_run["job_id"],
        "selected_gpu_uuid.txt": gate_run["gpu_uuid"],
        "node.txt": gate_run["node"],
        "gpu_name.txt": gate_run["gpu_name"],
        "allocated_gpu_uuids.csv": gate_run["gpu_uuid"],
        "visible_gpu_uuids.csv": gate_run["gpu_uuid"],
    }
    for name, value in gate_scalar_files.items():
        (gate_provenance / name).write_text(f"{value}\n")
    (gate_provenance / "repo_status.txt").write_text("")
    (gate_provenance / "repo_diff.sha256").write_text(f"{hashlib.sha256(b'').hexdigest()}  -\n")
    _write_junit(root / "focused_pytest.junit.xml")

    execution_rows = ["order\tlabel\tworkers\tsingle_lane_canonical\tnative_atomic_reduction\tnsys_base\n"]
    wall_by_config = {
        "canonical_serial": (10.0, 10.2),
        "atomic_serial": (9.8, 10.0),
        "canonical_multistream": (9.5, 9.7),
        "atomic_multistream": (8.8, 9.0),
    }
    for order, (label, workers, atomic, repeat) in enumerate(analyzer.ARM_SPECS, start=1):
        nsys_base = root / "nsight" / f"{label}_it181_warm"
        execution_rows.append(f"{order}\t{label}\t{workers}\t0\t{int(atomic)}\t{nsys_base.resolve()}\n")
        (provenance / f"{label}_command.sh").write_text(
            "nsys profile env "
            f"RECOVAR_K1_COARSE_MULTISTREAM_WORKERS={workers} "
            "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL=0 "
            f"RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION={int(atomic)} "
            "python -m scripts.run_vdam_late_iteration_profile\n"
        )
        profile_root = root / "runs" / label / "profile"
        warm_root = profile_root / "warm"
        warm_root.mkdir(parents=True)
        metadata_path = warm_root / "run_it181_recovar_meta.json"
        _write_json(metadata_path, _metadata(workers=workers, atomic=atomic))
        config = label.rsplit("_", 1)[0]
        wall = wall_by_config[config][repeat - 1]
        summary = {
            "schema": analyzer.PROFILE_SCHEMA,
            "classification": "diagnostic_performance_only",
            "checkpoint_iteration": 180,
            "profiled_iteration": 181,
            "nr_iter_schedule": 200,
            "exact_local_bucket_radix": 4,
            "exact_local_physical_order_chunk_size": 0,
            "cuda_profiler_range": True,
            "cold": {},
            "warm": {
                "wall_s": wall,
                "meta_path": str(metadata_path.resolve()),
                "meta_sha256": _sha256(metadata_path),
                "iteration_profile": {
                    "expectation_time_s": wall - 1.5,
                    "mstep_time_s": 0.25,
                },
            },
        }
        _write_json(profile_root / "profile_summary.json", summary)
        _write_star(warm_root / "run_it181_data.star")
        _write_map(warm_root / "run_it181_class001.mrc")
        _write_json(
            root / "nsight" / f"{label}_summary.json",
            {
                "schema": analyzer.NSIGHT_SCHEMA,
                "devices": {"0": {"gpu_busy_ns": int(wall * 0.4 * 1e9)}},
            },
        )
    (provenance / "execution_order.tsv").write_text("".join(execution_rows))
    return root, repo


def _mutate_metadata(root: Path, label: str, mutate) -> None:
    profile_root = root / "runs" / label / "profile"
    path = profile_root / "warm" / "run_it181_recovar_meta.json"
    metadata = json.loads(path.read_text())
    mutate(metadata)
    _write_json(path, metadata)
    summary_path = profile_root / "profile_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["warm"]["meta_sha256"] = _sha256(path)
    _write_json(summary_path, summary)


@pytest.mark.unit
def test_complete_crossed_fixture_passes_and_renders_outputs(tmp_path):
    root, repo = _build_root(tmp_path)

    report = analyzer.analyze(root, repo=repo)

    assert report["acceptance"] == {
        "topology_and_provenance": True,
        "effective_selector_audits": True,
        "selector_audit_count": 8,
        "exact_discrete_and_star_parity": True,
        "map_deltas_within_repeat_envelope": True,
        "material_warm_wall_win": True,
        "pass": True,
    }
    assert report["performance"]["percent_change_vs_canonical_serial"]["atomic_multistream"]["warm_wall_s"] < -10.0
    assert (
        report["provenance"]["source_manifest"]["sha256"]
        != report["provenance"]["qualified_gpu_gate"]["source_manifest"]["sha256"]
    )
    assert "| atomic_multistream |" in report["markdown"]
    assert "canonical_serial_1__atomic_multistream_1" in report["markdown"]

    output_json = tmp_path / "analysis" / "report.json"
    output_markdown = tmp_path / "analysis" / "report.md"
    assert (
        analyzer.main(
            [
                "--root",
                str(root),
                "--repo",
                str(repo),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_markdown),
            ]
        )
        == 0
    )
    assert json.loads(output_json.read_text())["acceptance"]["pass"] is True
    assert output_markdown.read_text().startswith("# VDAM coarse multistream late-pair gate")


@pytest.mark.unit
def test_extra_arm_fails_closed_on_topology(tmp_path):
    root, repo = _build_root(tmp_path)
    (root / "runs" / "unexpected_arm").mkdir()

    with pytest.raises(analyzer.LatePairSetupError, match="run-directory topology"):
        analyzer.analyze(root, repo=repo)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "mutate", "message"),
    [
        (
            "canonical_serial_1",
            lambda audit: audit["counts"].update(fused_calls=0),
            "zero calls",
        ),
        (
            "canonical_serial_1",
            lambda audit: audit["counts"].update(native_atomic_selected_calls=3),
            "native-atomic call count",
        ),
        (
            "atomic_multistream_1",
            lambda audit: audit.update(wrapper="relion_coarse_diff2_projector_f32"),
            "wrong wrapper/target",
        ),
        (
            "canonical_serial_1",
            lambda audit: audit.update(translation_count=30),
            "effective selector differs",
        ),
    ],
)
def test_selector_audit_mutations_fail_closed(tmp_path, label, mutate, message):
    root, repo = _build_root(tmp_path)

    def mutate_metadata(metadata):
        mutate(metadata["halfset_0_profile_summary"]["coarse_selector_audit"])

    _mutate_metadata(root, label, mutate_metadata)

    with pytest.raises(analyzer.LatePairSetupError, match=message):
        analyzer.analyze(root, repo=repo)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda metadata: metadata.update(joint_halfset_particle_stream=False),
            "joint-halfset particle stream",
        ),
        (
            lambda metadata: metadata.update(halfset_ids=[1, 0]),
            "joint-halfset IDs",
        ),
        (
            lambda metadata: metadata.update(halfset_1_profile_summary=metadata["halfset_0_profile_summary"]),
            "joint-halfset profile topology",
        ),
    ],
)
def test_joint_halfset_topology_mutations_fail_closed(tmp_path, mutate, message):
    root, repo = _build_root(tmp_path)
    _mutate_metadata(root, "canonical_serial_1", mutate)

    with pytest.raises(analyzer.LatePairSetupError, match=message):
        analyzer.analyze(root, repo=repo)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda metadata: metadata.update(n_translations=115),
            "not divisible by its oversampling factor",
        ),
        (
            lambda metadata: metadata.update(oversampling=-1),
            "oversampling must be non-negative",
        ),
        (
            lambda metadata: metadata.pop("oversampling"),
            "oversampling must be an integer",
        ),
    ],
)
def test_coarse_translation_topology_mutations_fail_closed(tmp_path, mutate, message):
    root, repo = _build_root(tmp_path)
    _mutate_metadata(root, "canonical_serial_1", mutate)

    with pytest.raises(analyzer.LatePairSetupError, match=message):
        analyzer.analyze(root, repo=repo)


@pytest.mark.unit
def test_discrete_mutation_is_reported_as_a_science_failure(tmp_path):
    root, repo = _build_root(tmp_path)
    _mutate_metadata(
        root,
        "atomic_multistream_1",
        lambda metadata: metadata["pose_assignments"].__setitem__(0, 99),
    )

    report = analyzer.analyze(root, repo=repo)

    pair = report["science"]["candidate_pairs"]["canonical_serial_1__atomic_multistream_1"]
    assert pair["metadata_exact"]["pose_assignments"] is False
    assert report["acceptance"]["exact_discrete_and_star_parity"] is False
    assert report["acceptance"]["pass"] is False


@pytest.mark.unit
def test_candidate_map_shift_outside_zero_repeat_envelope_fails(tmp_path):
    root, repo = _build_root(tmp_path)
    shifted = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    shifted[0, 0, 0] += 1.0
    for repeat in (1, 2):
        _write_map(
            root / "runs" / f"atomic_multistream_{repeat}" / "profile" / "warm" / "run_it181_class001.mrc",
            shifted,
        )

    report = analyzer.analyze(root, repo=repo)

    assert report["science"]["repeat_relative_l2_envelope"] == 0.0
    assert report["acceptance"]["map_deltas_within_repeat_envelope"] is False
    assert report["acceptance"]["pass"] is False


@pytest.mark.unit
def test_late_source_artifact_mutation_fails_closed(tmp_path):
    root, repo = _build_root(tmp_path)
    (repo / "source.py").write_text("VALUE = 2\n")

    with pytest.raises(analyzer.LatePairSetupError, match="artifact digest differs"):
        analyzer.analyze(root, repo=repo)


@pytest.mark.unit
def test_late_source_manifest_corruption_fails_closed(tmp_path):
    root, repo = _build_root(tmp_path)
    manifest = root / "provenance" / "source_manifest.sha256"
    manifest.write_text(manifest.read_text() + "\n")

    with pytest.raises(analyzer.LatePairSetupError, match="source manifest digest differs"):
        analyzer.analyze(root, repo=repo)


@pytest.mark.unit
def test_qualified_gate_source_manifest_corruption_fails_closed(tmp_path):
    root, repo = _build_root(tmp_path)
    run = json.loads((root / "provenance" / "run.json").read_text())
    manifest = Path(run["qualified_gpu_gate_root"]) / "provenance" / "source_manifest.sha256"
    manifest.write_text(manifest.read_text() + "\n")

    with pytest.raises(analyzer.LatePairSetupError, match="qualified gate source manifest digest differs"):
        analyzer.analyze(root, repo=repo)


@pytest.mark.unit
def test_duplicate_input_manifest_artifact_fails_closed(tmp_path):
    root, repo = _build_root(tmp_path)
    provenance = root / "provenance"
    manifest = provenance / "input_manifest.sha256"
    manifest.write_text(manifest.read_text() * 2)
    run_path = provenance / "run.json"
    run = json.loads(run_path.read_text())
    run["input_manifest_sha256"] = _sha256(manifest)
    _write_json(run_path, run)

    with pytest.raises(analyzer.LatePairSetupError, match="input manifest repeats an artifact"):
        analyzer.analyze(root, repo=repo)


@pytest.mark.unit
def test_recorded_git_tree_mismatch_fails_closed(tmp_path):
    root, repo = _build_root(tmp_path)
    run_path = root / "provenance" / "run.json"
    run = json.loads(run_path.read_text())
    run["git_tree"] = "f" * 40
    _write_json(run_path, run)
    (root / "provenance" / "repo_tree.txt").write_text(f"{run['git_tree']}\n")

    with pytest.raises(analyzer.LatePairSetupError, match="git tree differs"):
        analyzer.analyze(root, repo=repo)


@pytest.mark.unit
def test_qualified_gate_v2_gpu_mutation_fails_closed(tmp_path):
    root, repo = _build_root(tmp_path)
    gate_run_path = (
        Path(json.loads((root / "provenance" / "run.json").read_text())["qualified_gpu_gate_root"])
        / "provenance"
        / "run.json"
    )
    gate_run = json.loads(gate_run_path.read_text())
    gate_run["gpu_uuid"] = "GPU-wrong"
    _write_json(gate_run_path, gate_run)

    with pytest.raises(analyzer.LatePairSetupError, match="qualified focused GPU gate differs"):
        analyzer.analyze(root, repo=repo)


@pytest.mark.unit
def test_resource_report_mutation_fails_closed(tmp_path):
    root, repo = _build_root(tmp_path)
    (root / "provenance" / "coarse_atomic_resources.json").write_text("{}\n")

    with pytest.raises(analyzer.LatePairSetupError, match="resource report digest differs"):
        analyzer.analyze(root, repo=repo)


@pytest.mark.unit
def test_partial_nsight_panel_fails_closed(tmp_path):
    root, repo = _build_root(tmp_path)
    (root / "nsight" / "atomic_serial_1_summary.json").unlink()

    with pytest.raises(analyzer.LatePairSetupError, match="partially present"):
        analyzer.analyze(root, repo=repo)
