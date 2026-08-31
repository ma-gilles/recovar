from __future__ import annotations

import dataclasses
import importlib.util
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/audit_empiar10202_set6_i1_native_harness.py"
SPEC = importlib.util.spec_from_file_location("audit_empiar10202_set6_i1_native_harness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

FINALIZER_SCRIPT = Path(__file__).parents[2] / "scripts/finalize_empiar10202_set6_i1_evidence.py"
FINALIZER_SPEC = importlib.util.spec_from_file_location("native_evidence_finalizer_for_test", FINALIZER_SCRIPT)
assert FINALIZER_SPEC is not None and FINALIZER_SPEC.loader is not None
FINALIZER = importlib.util.module_from_spec(FINALIZER_SPEC)
sys.modules[FINALIZER_SPEC.name] = FINALIZER
FINALIZER_SPEC.loader.exec_module(FINALIZER)

SUMMARIZER_SCRIPT = Path(__file__).parents[2] / "scripts/summarize_em_k1_realdata_science_equivalence.py"
SUMMARIZER_SPEC = importlib.util.spec_from_file_location("native_evidence_summarizer_for_test", SUMMARIZER_SCRIPT)
assert SUMMARIZER_SPEC is not None and SUMMARIZER_SPEC.loader is not None
SUMMARIZER = importlib.util.module_from_spec(SUMMARIZER_SPEC)
sys.modules[SUMMARIZER_SPEC.name] = SUMMARIZER
SUMMARIZER_SPEC.loader.exec_module(SUMMARIZER)


def _write(path: Path, text: str = "sealed\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _scontrol(
    *,
    job_id: str,
    job_name: str,
    script: Path,
    root: Path,
    key: str,
    resources: dict,
    allocated: bool,
) -> str:
    cpu = resources["ntasks"] * resources["cpus_per_task"]
    tres = f"cpu={cpu},mem={resources['memory']},node=1,billing=40,gres/gpu={resources['h100_gpus']}"
    alloc = tres if allocated else "(null)"
    return (
        f"JobId={job_id} JobName={job_name} NumCPUs={cpu} NumTasks={resources['ntasks']} "
        f"CPUs/Task={resources['cpus_per_task']} Features=h100 Command={script} "
        f"StdOut={root}/logs/{key}-{job_id}.out StdErr={root}/logs/{key}-{job_id}.err "
        f"ReqTRES={tres} AllocTRES={alloc} TresPerNode=gres/gpu:h100:{resources['h100_gpus']}\n"
    )


def _gpu_inventory(count: int) -> str:
    rows = [f"Attached GPUs : {count}"]
    for index in range(count):
        rows.extend(
            [
                "    Product Name : NVIDIA H100 80GB HBM3",
                f"    GPU UUID : GPU-00000000-0000-0000-0000-{index:012d}",
            ]
        )
    return "\n".join(rows) + "\n"


def _gpu_identity(count: int) -> str:
    return "".join(f"GPU-00000000-0000-0000-0000-{index:012d}, NVIDIA H100 80GB HBM3\n" for index in range(count))


def _subject_values(repo: Path, cuda_lib: Path, cuda_sha: str) -> dict[str, str]:
    return {
        "SUBJECT_REPO": str(repo),
        "SUBJECT_COMMIT": "a" * 40,
        "SUBJECT_TREE": "b" * 40,
        "CUDA_LIB": str(cuda_lib),
        "CUDA_LIB_SHA256": cuda_sha,
    }


def _write_subject_file(path: Path, values: dict[str, str], *, lowercase: bool) -> Path:
    return _write(
        path,
        "".join(f"{name.lower() if lowercase else name}={shlex.quote(value)}\n" for name, value in values.items()),
    )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    completed_full: bool = False,
) -> tuple[Path, dict[str, str], dict[str, str], object]:
    root = tmp_path / "native"
    runtime = tmp_path / "runtime" / root.name
    for path in (
        root / "config",
        root / "scripts",
        root / "logs",
        root / "provenance",
        root / "inputs/smoke",
        root / "inputs/full",
        root / "outputs",
        runtime,
    ):
        path.mkdir(parents=True)
    _write(root / "SAFE_TO_DELETE", "")
    _write(runtime / "SAFE_TO_DELETE", "")
    monkeypatch.setattr(MODULE, "ALLOWED_ROOT", tmp_path)

    subject_repo = tmp_path / "subject"
    python = _write(subject_repo / ".pixi/envs/default/bin/python")
    python.chmod(0o755)
    _write(subject_repo / "scripts/run_full_refinement.py", "# driver\n")
    cuda_source = _write(subject_repo / "recovar/cuda/cuda_backproject.cu", "// cuda source\n")
    cuda_lib = _write(tmp_path / "native-lib/libcuda.so", "cuda library\n")
    cuda_sha = MODULE.sha256_file(cuda_lib)
    subject_values = _subject_values(subject_repo, cuda_lib, cuda_sha)
    _write_subject_file(root / "config/subject.env", subject_values, lowercase=False)
    monkeypatch.setattr(MODULE, "CUDA_SOURCE_SHA256", MODULE.sha256_file(cuda_source))

    preparation = _write(tmp_path / "prepared/preparation_manifest.json", "{}\n")
    full_star = _write(root / "inputs/full/particles.star", "full star\n")
    smoke_star = _write(root / "inputs/smoke/particles.star", "smoke star\n")
    rec_reference = _write(tmp_path / "prepared/reference_rec.mrc", "rec ref\n")
    rel_reference = _write(tmp_path / "prepared/reference_rel.mrc", "rel ref\n")
    particle_stack = _write(tmp_path / "particles.mrcs", "particles\n")
    for phase in ("smoke", "full"):
        _write(root / f"inputs/{phase}/reference_init.mrc", rec_reference.read_text())
        _write(root / f"inputs/{phase}/reference_init_relion.mrc", rel_reference.read_text())
    monkeypatch.setattr(MODULE, "PREPARATION_SHA256", MODULE.sha256_file(preparation))
    monkeypatch.setattr(MODULE, "FULL_STAR_SHA256", MODULE.sha256_file(full_star))
    monkeypatch.setattr(MODULE, "SMOKE_STAR_SHA256", MODULE.sha256_file(smoke_star))
    monkeypatch.setattr(MODULE, "RECOVAR_REFERENCE_SHA256", MODULE.sha256_file(rec_reference))
    monkeypatch.setattr(MODULE, "RELION_REFERENCE_SHA256", MODULE.sha256_file(rel_reference))
    monkeypatch.setattr(MODULE, "PARTICLE_STACK_SHA256", MODULE.sha256_file(particle_stack))
    monkeypatch.setattr(MODULE, "PARTICLE_STACK_SIZE_BYTES", particle_stack.stat().st_size)

    relion_mpi = _write(tmp_path / "native-lib/relion_refine_mpi", "relion\n")
    relion_bind = _write(tmp_path / "native-lib/relion_bind.so", "bind\n")
    relion_source = tmp_path / "relion-source"
    relion_source.mkdir()
    monkeypatch.setattr(MODULE, "RELION_MPI_SHA256", MODULE.sha256_file(relion_mpi))
    monkeypatch.setattr(MODULE, "RELION_BIND_SHA256", MODULE.sha256_file(relion_bind))
    monkeypatch.setattr(MODULE, "RELION_SOURCE_COMMIT", "c" * 40)
    monkeypatch.setattr(MODULE, "RELION_SOURCE_TREE", "d" * 40)

    def git_output(repo: Path, *arguments: str) -> str:
        if repo == subject_repo:
            if arguments == ("rev-parse", "HEAD"):
                return subject_values["SUBJECT_COMMIT"]
            if arguments == ("rev-parse", "HEAD^{tree}"):
                return subject_values["SUBJECT_TREE"]
        elif repo == relion_source:
            if arguments == ("rev-parse", "HEAD"):
                return MODULE.RELION_SOURCE_COMMIT
            if arguments == ("rev-parse", "HEAD^{tree}"):
                return MODULE.RELION_SOURCE_TREE
        assert arguments == ("status", "--porcelain=v1", "--untracked-files=all")
        return ""

    monkeypatch.setattr(MODULE, "_git_output", git_output)
    source_manifest = _write(tmp_path / "source-harness/launch_manifest.json", "source\n")
    common = _write(root / "scripts/common.sh")
    validate = _write(root / "scripts/validate_harness.sh")
    submit = _write(root / "submit_when_configured.sh")

    launcher = MODULE._load_launcher()
    specs = launcher.build_run_specs(subject_repo, root)
    job_ids = {key: str(101 + index) for index, key in enumerate(MODULE.RUN_KEYS)}
    states = {
        "recovar_smoke": "FAILED",
        "relion_smoke": "COMPLETED",
        "recovar_full": "COMPLETED" if completed_full else "PENDING",
        "relion_full": "COMPLETED" if completed_full else "PENDING",
    }
    science = {
        "recovar_smoke": {
            "particles": "smoke",
            "iterations": 1,
            "current_size": 800,
            "symmetry": "I1",
            "seed": 10202,
            "classes": 1,
        },
        "relion_smoke": {
            "particles": "smoke",
            "iterations": 1,
            "incr_size": 800,
            "symmetry": "I1",
            "seed": 10202,
            "classes": 1,
        },
        "recovar_full": {"particles": "full", "max_iterations": 50, "symmetry": "I1", "seed": 10202, "classes": 1},
        "relion_full": {"particles": "full", "max_iterations": 50, "symmetry": "I1", "seed": 10202, "classes": 1},
    }
    runs = {}
    for spec in specs:
        output_dir = root / "outputs" / spec.key
        output_dir.mkdir()
        script = _write(
            root / "scripts" / f"{spec.key}.sbatch",
            f"#!/bin/bash\n#SBATCH --job-name=10202-s6-{spec.key.replace('_', '-')}-fixture\n",
        )
        resources = dataclasses.asdict(spec.resources)
        row = {
            "script": str(script),
            "script_sha256": MODULE.sha256_file(script),
            "resources": resources,
            "science": science[spec.key],
        }
        if spec.depends_on:
            row["depends_on"] = list(spec.depends_on)
        runs[spec.key] = row
        job_id = job_ids[spec.key]
        submit_text = _scontrol(
            job_id=job_id,
            job_name=f"10202-s6-{spec.key.replace('_', '-')}-fixture",
            script=script,
            root=root,
            key=spec.key,
            resources=resources,
            allocated=False,
        )
        _write(root / f"provenance/scontrol_submit_{job_id}.txt", submit_text)
        if states[spec.key] != "PENDING":
            runtime_text = _scontrol(
                job_id=job_id,
                job_name=f"10202-s6-{spec.key.replace('_', '-')}-fixture",
                script=script,
                root=root,
                key=spec.key,
                resources=resources,
                allocated=True,
            )
            _write(root / f"provenance/scontrol_{job_id}.txt", runtime_text)
            command = (
                launcher._recovar_command(subject_repo, spec.data_dir, output_dir, smoke=spec.phase == "smoke")
                if spec.engine == "recovar"
                else launcher._relion_command(spec.data_dir, output_dir, smoke=spec.phase == "smoke")
            )
            _write(root / f"provenance/command_{job_id}.sh", shlex.join(command) + "\n")
            _write_subject_file(root / f"provenance/subject_{job_id}.txt", subject_values, lowercase=True)
            environment = "PYTHONNOUSERSITE=1\nXLA_PYTHON_CLIENT_PREALLOCATE=false\n"
            if spec.engine == "recovar":
                environment += f"RECOVAR_CUDA_LIB={cuda_lib}\nRECOVAR_EXPECTED_REPO_ROOT={subject_repo}\n"
            _write(root / f"provenance/environment_{job_id}.txt", environment)
            _write(
                root / f"provenance/final_env_{job_id}.txt",
                "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=unset\nRECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=unset\n",
            )
            _write(root / f"provenance/nvidia_smi_{job_id}.txt", _gpu_inventory(resources["h100_gpus"]))
            ldd = "ldd_relion_bind" if spec.engine == "recovar" else "ldd_relion"
            _write(root / f"provenance/{ldd}_{job_id}.txt", "all resolved\n")
            _write(root / f"logs/{spec.key}-{job_id}.out", "started\n")
            error = "CUDA: out of memory\n" if states[spec.key] == "FAILED" else ""
            _write(root / f"logs/{spec.key}-{job_id}.err", error)
        if states[spec.key] == "COMPLETED":
            for expected in spec.expected_outputs:
                _write(expected)
            _write(output_dir / "COMPLETED", "")
            _write(
                output_dir / "slurm_walltime.json",
                json.dumps({"job_id": job_id, "run_key": spec.key, "wall_s": 1}),
            )
            _write(output_dir / "gpu_identity.txt", _gpu_identity(resources["h100_gpus"]))

    payload = {
        "schema": MODULE.NATIVE_LAUNCH_SCHEMA,
        "created_utc": "2026-08-30T23:19:05Z",
        "status": "prepared_unsubmitted_subject_unconfigured",
        "run_root": str(root),
        "runtime_root": str(runtime),
        "safe_to_delete": {
            "run_marker": str(root / "SAFE_TO_DELETE"),
            "runtime_marker": str(runtime / "SAFE_TO_DELETE"),
        },
        "source_harness": {
            "run_root": str(source_manifest.parent),
            "launch_manifest": str(source_manifest),
            "launch_manifest_sha256": MODULE.sha256_file(source_manifest),
            "recovar_smoke_job_id": "1",
            "relion_smoke_job_id": "2",
            "recovar_full_job_id": "3",
            "relion_full_job_id": "4",
        },
        "subject": {
            "configuration": str(root / "config/subject.env"),
            "required_fields": list(MODULE.SUBJECT_FIELDS),
            "configured": False,
            "submission_gate": "sealed",
        },
        "harness_delta_from_r3": {
            "scientific_command_delta": "none after normalizing paths",
            "resource_delta": "none",
            "python_cache_fix": [],
            "command_equivalence_gate": {key: "passed" for key in MODULE.RUN_KEYS},
        },
        "inputs": {
            "preparation_manifest": {"path": str(preparation), "sha256": MODULE.PREPARATION_SHA256},
            "particle_stack": {
                "path": str(particle_stack),
                "size_bytes": particle_stack.stat().st_size,
                "first_mib_sha256": "unused",
                "full_sha256": MODULE.PARTICLE_STACK_SHA256,
            },
            "full_particles_star": {"path": str(full_star), "sha256": MODULE.FULL_STAR_SHA256},
            "smoke_particles_star": {"path": str(smoke_star), "sha256": MODULE.SMOKE_STAR_SHA256},
            "recovar_reference": {"canonical_path": str(rec_reference), "sha256": MODULE.RECOVAR_REFERENCE_SHA256},
            "relion_reference": {"canonical_path": str(rel_reference), "sha256": MODULE.RELION_REFERENCE_SHA256},
        },
        "immutable_native_artifacts": {
            "relion_refine_mpi": {"path": str(relion_mpi), "sha256": MODULE.RELION_MPI_SHA256},
            "relion_binding": {"path": str(relion_bind), "sha256": MODULE.RELION_BIND_SHA256},
            "relion_binding_source": {
                "path": str(relion_source),
                "commit": MODULE.RELION_SOURCE_COMMIT,
                "tree": MODULE.RELION_SOURCE_TREE,
            },
        },
        "runs": runs,
        "harness_scripts": {
            "common": {"path": str(common), "sha256": MODULE.sha256_file(common)},
            "validate": {"path": str(validate), "sha256": MODULE.sha256_file(validate)},
            "submit": {"path": str(submit), "sha256": MODULE.sha256_file(submit)},
        },
        "submission": None,
    }
    manifest = _write(root / "launch_manifest.json", json.dumps(payload))
    _write(
        root / "submission.json",
        json.dumps(
            {
                "schema": MODULE.NATIVE_SUBMISSION_SCHEMA,
                "dependency": f"afterok:{job_ids['recovar_smoke']}:{job_ids['relion_smoke']}",
                "job_ids": job_ids,
            }
        ),
    )

    def runner(command, **_kwargs):
        job_id = command[command.index("--jobs") + 1]
        key = next(name for name, value in job_ids.items() if value == job_id)
        state = states[key]
        if command[0] == "squeue":
            reason = "DependencyNeverSatisfied" if state == "PENDING" else "None"
            return SimpleNamespace(stdout=f"{job_id}|{state}|{reason}\n", returncode=0)
        resources = dataclasses.asdict(next(spec.resources for spec in specs if spec.key == key))
        cpu = resources["ntasks"] * resources["cpus_per_task"]
        tres = f"cpu={cpu},mem={resources['memory']},node=1,billing=40,gres/gpu={resources['h100_gpus']}"
        exit_code = "0:0" if state in {"COMPLETED", "PENDING"} else "1:0"
        allocated = "" if state == "PENDING" else tres
        return SimpleNamespace(
            stdout=f"{job_id}|{state}|{exit_code}|{tres}|{allocated}|1024K\n",
            returncode=0,
        )

    return manifest, job_ids, states, runner


def test_native_audit_classifies_oom_and_dependencies_without_promoting_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, _, _, runner = _fixture(tmp_path, monkeypatch)

    _, audit = MODULE.audit_native_launch(manifest, runner=runner)

    assert audit["jobs"]["recovar_smoke"]["classification"] == "oom"
    assert audit["jobs"]["relion_smoke"]["classification"] == "completed"
    assert audit["jobs"]["recovar_full"]["classification"] == "dependency_never_satisfied"
    assert audit["jobs"]["relion_full"]["classification"] == "dependency_never_satisfied"
    assert audit["jobs"]["relion_smoke"]["science_role"] == "capability_only"
    assert audit["jobs"]["relion_smoke"]["smoke_can_promote"] is False
    assert audit["science_scoring"]["eligible"] is False
    assert audit["science_scoring"]["smoke_only_promotion_forbidden"] is True
    with pytest.raises(ValueError, match="not science-ready"):
        MODULE.audit_native_launch(manifest, runner=runner, require_science_ready=True)


def test_native_audit_accepts_only_completed_full_pair_for_science(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, _, _, runner = _fixture(tmp_path, monkeypatch, completed_full=True)

    normalized, audit = MODULE.audit_native_launch(
        manifest,
        runner=runner,
        require_science_ready=True,
    )

    assert audit["science_scoring"]["eligible"] is True
    assert normalized["subject"] == audit["jobs"]["recovar_full"]["subject"]
    assert audit["jobs"]["recovar_full"]["expected_outputs"]
    assert audit["jobs"]["relion_full"]["expected_outputs"]


@pytest.mark.parametrize("mutation", ("command", "tres", "identity"))
def test_native_audit_rejects_mutated_execution_provenance(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, job_ids, _, runner = _fixture(tmp_path, monkeypatch)
    root = manifest.parent
    job_id = job_ids["relion_smoke"]
    if mutation == "command":
        path = root / f"provenance/command_{job_id}.sh"
        path.write_text(path.read_text().rstrip() + " --local\n")
        message = "executed command mismatch"
    elif mutation == "tres":
        path = root / f"provenance/scontrol_{job_id}.txt"
        path.write_text(path.read_text().replace("AllocTRES=cpu=12", "AllocTRES=cpu=8"))
        message = "ReqTRES != AllocTRES"
    else:
        path = root / f"provenance/nvidia_smi_{job_id}.txt"
        path.write_text(path.read_text().replace("NVIDIA H100", "NVIDIA A100"))
        message = "non-H100 GPU"

    with pytest.raises(ValueError, match=message):
        MODULE.audit_native_launch(manifest, runner=runner)


def test_native_binding_rejects_subject_config_hash_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, _, _, runner = _fixture(tmp_path, monkeypatch)
    subject = manifest.parent / "config/subject.env"
    subject.write_text(subject.read_text().replace("SUBJECT_TREE=", "EXTRA=bad\nSUBJECT_TREE="))

    with pytest.raises(ValueError, match="unexpected key set"):
        MODULE.audit_native_launch(manifest, runner=runner)


def test_explicit_smoke_replacement_is_bound_as_oom_capability_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, job_ids, _, base_runner = _fixture(tmp_path, monkeypatch)
    parent_root = manifest.parent
    replacement_root = tmp_path / "replacement"
    for path in (
        replacement_root / "scripts",
        replacement_root / "logs",
        replacement_root / "provenance",
        replacement_root / "outputs/recovar_smoke",
    ):
        path.mkdir(parents=True)
    _write(replacement_root / "SAFE_TO_DELETE", "")
    subject_values = MODULE._parse_key_values(parent_root / "config/subject.env", MODULE.SUBJECT_FIELDS)
    subject_repo = Path(subject_values["SUBJECT_REPO"])
    replacement_id = "999"
    script = _write(
        replacement_root / "scripts/recovar_smoke.sbatch",
        "#!/bin/bash\n#SBATCH --job-name=10202-s6-recovar-replacement\n",
    )
    launcher = MODULE._load_launcher()
    resources = dataclasses.asdict(launcher.build_run_specs(subject_repo, parent_root)[0].resources)
    _write(
        replacement_root / f"provenance/scontrol_{replacement_id}.txt",
        _scontrol(
            job_id=replacement_id,
            job_name="10202-s6-recovar-replacement",
            script=script,
            root=replacement_root,
            key="recovar_smoke",
            resources=resources,
            allocated=True,
        ),
    )
    command = launcher._recovar_command(
        subject_repo,
        parent_root / "inputs/smoke",
        replacement_root / "outputs/recovar_smoke",
        smoke=True,
    )
    _write(replacement_root / f"provenance/command_{replacement_id}.sh", shlex.join(command) + "\n")
    _write_subject_file(
        replacement_root / f"provenance/subject_{replacement_id}.txt",
        subject_values,
        lowercase=True,
    )
    _write(
        replacement_root / f"provenance/environment_{replacement_id}.txt",
        "PYTHONNOUSERSITE=1\n"
        "XLA_PYTHON_CLIENT_PREALLOCATE=false\n"
        f"RECOVAR_CUDA_LIB={subject_values['CUDA_LIB']}\n"
        f"RECOVAR_EXPECTED_REPO_ROOT={subject_values['SUBJECT_REPO']}\n",
    )
    _write(
        replacement_root / f"provenance/final_env_{replacement_id}.txt",
        "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=unset\nRECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=unset\n",
    )
    _write(replacement_root / f"provenance/nvidia_smi_{replacement_id}.txt", _gpu_inventory(1))
    _write(replacement_root / f"provenance/ldd_relion_bind_{replacement_id}.txt", "all resolved\n")
    _write(replacement_root / f"logs/recovar_smoke-{replacement_id}.out", "started\n")
    _write(replacement_root / f"logs/recovar_smoke-{replacement_id}.err", "CUDA: out of memory\n")
    replacement_payload = {
        "schema": MODULE.REPLACEMENT_SCHEMA,
        "parent_launch_manifest": {
            "path": str(manifest),
            "sha256": MODULE.sha256_file(manifest),
        },
        "run_key": "recovar_smoke",
        "replaces_job_id": job_ids["recovar_smoke"],
        "job_id": replacement_id,
        "run_root": str(replacement_root),
        "script_sha256": MODULE.sha256_file(script),
        "subject": subject_values,
        "reason": "capability-only retry",
    }
    replacement = _write(tmp_path / "replacement.json", json.dumps(replacement_payload))

    def runner(command, **kwargs):
        job_id = command[command.index("--jobs") + 1]
        if job_id != replacement_id:
            return base_runner(command, **kwargs)
        tres = "cpu=4,mem=500G,node=1,billing=40,gres/gpu=1"
        return SimpleNamespace(
            stdout=f"{replacement_id}|FAILED|1:0|{tres}|{tres}|1024K\n",
            returncode=0,
        )

    _, audit = MODULE.audit_native_launch(
        manifest,
        replacement_records=(replacement,),
        runner=runner,
    )

    selected = audit["jobs"]["recovar_smoke"]
    assert selected["job_id"] == replacement_id
    assert selected["source"] == "replacement"
    assert selected["classification"] == "oom"
    assert selected["science_role"] == "capability_only"
    assert selected["provenance_complete"] is False
    assert audit["science_scoring"]["eligible"] is False


def test_finalizer_dispatches_native_adapter_and_binds_replacement_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "native"
    root.mkdir()
    manifest = _write(
        root / "launch_manifest.json",
        json.dumps({"schema": FINALIZER.NATIVE_LAUNCH_SCHEMA}),
    )
    replacement = _write(tmp_path / "replacement.json", "{}\n")
    subject = {"commit": "a" * 40}
    audit = {
        "launch_manifest": {
            "path": str(manifest),
            "sha256": FINALIZER.sha256_file(manifest),
        },
        "jobs": {},
    }

    class Adapter:
        @staticmethod
        def audit_native_launch(path, *, replacement_records, runner, require_science_ready):
            assert path == manifest
            assert replacement_records == (replacement,)
            assert require_science_ready is True
            return {"run_root": str(root), "subject": subject}, audit

    monkeypatch.setattr(FINALIZER, "_load_native_adapter", lambda: Adapter)
    monkeypatch.setattr(
        FINALIZER,
        "_run_analysis",
        lambda *_args, **_kwargs: (
            {"schema": "collector"},
            {"commands": {}, "analysis_artifacts": {}},
        ),
    )
    monkeypatch.setattr(FINALIZER, "_input_contract", lambda _payload: {"k": 1})

    output = FINALIZER.finalize(manifest, replacement_records=(replacement,))
    evidence = json.loads(output.read_text())

    binding = evidence["execution_binding"]
    assert binding["launch_manifest_schema"] == FINALIZER.NATIVE_LAUNCH_SCHEMA
    assert binding["replacement_records"][0]["sha256"] == FINALIZER.sha256_file(replacement)
    assert binding["finalizer_argv"][-2:] == ["--replacement-record", str(replacement)]


def _native_binding_fixture(tmp_path: Path) -> dict:
    run_root = tmp_path / "run"
    run_root.mkdir()
    manifest = _write(
        run_root / "launch_manifest.json",
        json.dumps(
            {
                "schema": SUMMARIZER.NATIVE_LAUNCH_MANIFEST_SCHEMA,
                "run_root": str(run_root),
            }
        ),
    )
    subject = {"commit": "a" * 40}
    jobs = {
        key: {
            "phase": "smoke" if key.endswith("smoke") else "full",
            "science_role": "capability_only" if key.endswith("smoke") else "science_candidate",
            "smoke_can_promote": False,
            "classification": "completed",
            "provenance_complete": True,
            "expected_outputs": [{"sha256": "b" * 64}],
        }
        for key in MODULE.RUN_KEYS
    }
    launch = {
        "schema": SUMMARIZER.NATIVE_EXECUTION_AUDIT_SCHEMA,
        "launch_manifest": {
            "path": str(manifest),
            "sha256": SUMMARIZER.sha256_file(manifest),
        },
        "replacement_records": [],
        "science_subject": subject,
        "science_scoring": {
            "eligible": True,
            "smoke_only_promotion_forbidden": True,
        },
        "jobs": jobs,
    }
    collector = {"schema": "collector"}
    commands: dict = {}
    artifacts: dict = {}
    envelope = _write(
        run_root / "execution_envelope.json",
        json.dumps(
            {
                "schema": SUMMARIZER.EXECUTION_BINDING_SCHEMA,
                "launch": launch,
                "collector": collector,
                "analysis_commands": commands,
                "analysis_artifacts": artifacts,
            }
        ),
    )
    finalizer = (Path(__file__).parents[2] / SUMMARIZER.FINALIZER_RELATIVE_PATH).resolve()
    argv = [sys.executable, str(finalizer), "--launch-manifest", str(manifest)]
    return {
        "subject": subject,
        "launch": launch,
        "collector": collector,
        "analysis_commands": commands,
        "analysis_artifacts": artifacts,
        "execution_binding": {
            "schema": SUMMARIZER.EXECUTION_BINDING_SCHEMA,
            "launch_manifest_path": str(manifest),
            "launch_manifest_sha256": SUMMARIZER.sha256_file(manifest),
            "launch_manifest_schema": SUMMARIZER.NATIVE_LAUNCH_MANIFEST_SCHEMA,
            "replacement_records": [],
            "finalizer_command_path": str(finalizer),
            "finalizer_command_sha256": SUMMARIZER.sha256_file(finalizer),
            "finalizer_argv": argv,
            "finalizer_argv_sha256": SUMMARIZER.sha256_json(argv),
            "evidence_envelope_path": str(envelope),
            "evidence_envelope_sha256": SUMMARIZER.sha256_file(envelope),
        },
    }


def test_scorecard_binding_accepts_native_full_pair_and_rejects_smoke_promotion(
    tmp_path: Path,
) -> None:
    evidence = _native_binding_fixture(tmp_path)

    assert SUMMARIZER._validate_execution_binding(evidence, required=True) == []

    evidence["launch"]["jobs"]["recovar_smoke"]["science_role"] = "science_candidate"
    failures = SUMMARIZER._validate_execution_binding(evidence, required=True)
    assert "execution_binding:native_recovar_smoke_capability_only" in failures


def test_native_subject_change_requires_deliberate_scorecard_contract_update(
    tmp_path: Path,
) -> None:
    scorecard = SUMMARIZER.load_and_validate_scorecard()
    target = next(case for case in scorecard["cases"] if case["id"] == SUMMARIZER.TARGET_CASE_ID)
    evidence = _native_binding_fixture(tmp_path)
    evidence.update(
        {
            "schema": SUMMARIZER.EVIDENCE_SCHEMA,
            "case_id": SUMMARIZER.TARGET_CASE_ID,
            "subject": {
                "commit": "a" * 40,
                "tree_clean": True,
                "diff_sha256": SUMMARIZER.EMPTY_SHA256,
            },
        }
    )

    failures = SUMMARIZER.validate_case_evidence_provenance(scorecard, target, evidence)

    assert "subject_commit" in failures
    assert "native_subject_requires_deliberate_scorecard_contract_update" in failures
