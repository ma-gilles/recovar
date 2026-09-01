import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile

from scripts import analyze_vdam_raw_cache_abba as analyzer

GPU_UUID = "GPU-00000000-1111-2222-3333-444444444444"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_manifest(path: Path, entries: list[tuple[Path, str]]) -> str:
    path.write_text("".join(f"{_sha256(artifact)}  {name}\n" for artifact, name in entries))
    return _sha256(path)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _audit() -> dict:
    return {
        "score_mode": "gaussian",
        "translation_count": 29,
        "requested_fused": True,
        "effective_fused": True,
        "requested_workers": 8,
        "effective_workers": 8,
        "requested_atomic": True,
        "effective_atomic": True,
        "wrapper": "relion_coarse_diff2_projector_multistream_f32",
        "target": "cuda_relion_coarse_diff2_projector_multistream_f32",
        "counts": {
            "fused_calls": 3,
            "actual_rows": 3000,
            "multistream_calls": 3,
            "native_atomic_selected_calls": 3,
        },
    }


def _schedule() -> dict:
    return dict(analyzer.EXPECTED_SCHEDULE)


def _metadata(pass1_s: float, pass2_s: float) -> dict:
    return {
        **_schedule(),
        "oversampling": 1,
        "joint_halfset_particle_stream": True,
        "halfset_ids": [0, 1],
        "selected_particle_ids": list(range(3000)),
        "best_pose_rotation_ids": [3] * 3000,
        "best_pose_rotations": [[0.0, 0.0, 0.0]] * 3000,
        "best_pose_translations": [[0.0, 0.0]] * 3000,
        "class_assignments": [0] * 3000,
        "max_posterior_per_image": [0.75] * 3000,
        "pose_assignments": [1] * 3000,
        "halfset_0_class_assignments": [0] * 3000,
        "sparse_pass2_profile_summary": {
            "pass1_time_s": pass1_s,
            "pass2_time_s": pass2_s,
        },
        "halfset_0_profile_summary": {
            "em_time_s": pass1_s + pass2_s,
            "coarse_selector_audit": _audit(),
        },
    }


def _write_star(path: Path, *, changed: bool = False) -> None:
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnImagePixelSize": [1.5], "rlnImageSize": [128]})
    particles = pd.DataFrame(
        {
            "rlnImageName": [f"{index + 1:06d}@particles.mrcs" for index in range(3000)],
            "rlnAngleRot": np.arange(3000, dtype=np.float64) + float(changed),
            "rlnAngleTilt": np.zeros(3000),
            "rlnAnglePsi": np.zeros(3000),
            "rlnClassNumber": np.ones(3000, dtype=np.int64),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    starfile.write({"optics": optics, "particles": particles}, path, overwrite=True)


def _write_map(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as stream:
        stream.set_data(np.asarray(value, dtype=np.float32))


def _resource_snapshot(*, hwm_kb: int) -> dict:
    before = {
        "user_cpu_s": 1.0,
        "system_cpu_s": 1.0,
        "max_rss_kb": hwm_kb - 1024,
        "minor_faults": 1,
        "major_faults": 0,
        "input_blocks": 0,
        "output_blocks": 0,
        "voluntary_context_switches": 1,
        "involuntary_context_switches": 0,
        "current_rss_kb": hwm_kb - 2048,
        "high_water_rss_kb": hwm_kb - 1024,
        "proc_io": {"read_bytes": 100, "write_bytes": 100},
    }
    after = {
        **before,
        "user_cpu_s": 4.0,
        "system_cpu_s": 2.0,
        "max_rss_kb": hwm_kb,
        "current_rss_kb": hwm_kb - 512,
        "high_water_rss_kb": hwm_kb,
        "proc_io": {"read_bytes": 10100, "write_bytes": 2100},
    }
    return {
        "before": before,
        "after": after,
        "delta": {
            "user_cpu_s": 3.0,
            "system_cpu_s": 1.0,
            "minor_faults": 0.0,
            "major_faults": 0.0,
            "input_blocks": 0.0,
            "output_blocks": 0.0,
            "voluntary_context_switches": 0.0,
            "involuntary_context_switches": 0.0,
            "proc_io": {"read_bytes": 10000, "write_bytes": 2000},
        },
    }


def _cache_event() -> dict:
    return {
        "loader_type": "recovar.data_io.image_loader.MRCLoader",
        "num_images": 3000,
        "image_size": 128,
        "dtype": "<f4",
        "estimated_bytes": analyzer.EXPECTED_CACHE_BYTES,
        "cached_before": False,
        "cached_after": True,
        "cached_nbytes": analyzer.EXPECTED_CACHE_BYTES,
        "elapsed_s": 0.20,
    }


def _write_nsight(
    sqlite_path: Path,
    summary_path: Path,
    report_path: Path,
    *,
    mode: str,
    pass1_s: float,
    pass2_s: float,
    launches: int = 1000,
) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    pass1 = (2_000_000_000, 2_000_000_000 + int(pass1_s * 1e9))
    pass2 = (8_000_000_000, 8_000_000_000 + int(pass2_s * 1e9))
    strings = {
        1: analyzer.COARSE_KERNEL_NAME,
        2: analyzer.GETITEM_RANGE,
        3: analyzer.LOADER_RANGE,
        4: analyzer.PASS1_RANGE,
        5: analyzer.PASS2_RANGE,
    }
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        connection.executemany("INSERT INTO StringIds(id, value) VALUES (?, ?)", strings.items())
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(start INTEGER, end INTEGER, deviceId INTEGER, shortName INTEGER, "
            "gridX INTEGER, gridY INTEGER, gridZ INTEGER, blockX INTEGER, blockY INTEGER, blockZ INTEGER)"
        )
        kernel_rows = []
        for index in range(launches):
            bounds = pass1 if index < launches // 2 else pass2
            local = index if index < launches // 2 else index - launches // 2
            start = bounds[0] + local * 2_000_000
            kernel_rows.append((start, start + 1_000_000, 0, 1, 1, 1, 1, 32, 1, 1))
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", kernel_rows
        )
        connection.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, textId INTEGER, text TEXT)")
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL)",
            ((*pass1, 4), (*pass2, 5)),
        )
        nvtx_rows = []
        for bounds, text_id in ((pass1, 2), (pass2, 2)):
            for index in range(1000):
                start = bounds[0] + index * 1_000_000
                nvtx_rows.append((start, start + 500_000, text_id, None))
        if mode == "off":
            for bounds in (pass1, pass2):
                for index in range(1000):
                    start = bounds[0] + index * 1_000_000 + 100_000
                    nvtx_rows.append((start, start + 200_000, 3, None))
        else:
            nvtx_rows.append((1_000_000_000, 1_200_000_000, 3, None))
        connection.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)", nvtx_rows)
    kernel_sum = launches * 1_000_000
    _write_json(
        summary_path,
        {
            "schema": analyzer.NSIGHT_SCHEMA,
            "sqlite": str(sqlite_path.resolve()),
            "devices": {"0": {"kernel_count": launches, "gpu_busy_ns": kernel_sum}},
            "kernels": [
                {
                    "name": analyzer.COARSE_KERNEL_NAME,
                    "count": launches,
                    "total_ns": kernel_sum,
                    "max_ns": 1_000_000,
                    "mean_ns": 1_000_000.0,
                }
            ],
        },
    )
    report_path.write_bytes(b"synthetic nsys report")


def _build_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "raw_cache_abba"
    repo = tmp_path / "repo"
    provenance = root / "provenance"
    repo.mkdir()
    provenance.mkdir(parents=True)
    (root / "ARMS_COMPLETED").touch()
    source = repo / "source.py"
    external_interpreter = tmp_path / "external_python"
    external_interpreter.write_bytes(b"pixi interpreter")
    interpreter = repo / "python"
    interpreter.symlink_to(external_interpreter)
    source.write_text("VALUE = 1\n")
    _git(repo, "init", "--quiet")
    _git(repo, "add", "source.py", "python")
    _git(repo, "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "fixture")
    git_head = _git(repo, "rev-parse", "HEAD")
    git_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    source_digest = _write_manifest(provenance / "source_manifest.sha256", [(source, "source.py")])

    particle_stack = tmp_path / "particles.128.mrcs"
    particle_stack.write_bytes(b"synthetic particle stack")
    input_file = tmp_path / "checkpoint.star"
    input_file.write_bytes(b"synthetic checkpoint")
    input_digest = _write_manifest(
        provenance / "input_manifest.sha256",
        [(input_file, str(input_file.resolve())), (particle_stack, str(particle_stack.resolve()))],
    )
    cuda = root / "runtime" / "cuda" / "libcuda_backproject.so"
    binding = root / "runtime" / "relion_bind" / "_relion_bind_core.so"
    cuda.parent.mkdir(parents=True)
    binding.parent.mkdir(parents=True)
    cuda.write_bytes(b"cuda")
    binding.write_bytes(b"binding")
    (provenance / "qualified_cuda.sha256").write_text(f"{_sha256(cuda)}  {cuda.resolve()}\n")
    (provenance / "relion_bind.sha256").write_text(f"{_sha256(binding)}  {binding.resolve()}\n")
    (provenance / "interpreter.sha256").write_text(f"{_sha256(interpreter)}  {interpreter.resolve()}\n")
    gate_ledger = provenance / "qualified_gate.SHA256SUMS"
    gate_ledger.write_text("sealed focused gate\n")
    (provenance / "qualified_gate.SHA256SUMS.sha256").write_text(f"{_sha256(gate_ledger)}  SHA256SUMS\n")
    nsys_binary = tmp_path / "nsys"
    cusparse_library = tmp_path / "libcusparse.so.12"
    nsys_binary.write_bytes(b"nsys")
    cusparse_library.write_bytes(b"cusparse")

    run = {
        "schema": analyzer.RUN_SCHEMA,
        "classification": "diagnostic_performance_only",
        "job_id": "12345",
        "git_head": git_head,
        "git_tree": git_tree,
        "science_base_head": analyzer.SCIENCE_BASE_HEAD,
        "science_base_tree": analyzer.SCIENCE_BASE_TREE,
        "source_manifest_sha256": source_digest,
        "source_manifest_scope": "selected_high_risk_files",
        "input_manifest_sha256": input_digest,
        "particle_stack_sha256": _sha256(particle_stack),
        "gpu_uuid": GPU_UUID,
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "node": "della-h21g4",
        "qualified_gpu_gate_root": "/sealed/gate",
        "qualified_gate_sha256sums_sha256": _sha256(gate_ledger),
        "cuda_sha256": _sha256(cuda),
        "relion_bind_sha256": _sha256(binding),
        "interpreter_sha256": _sha256(interpreter),
        "nsys_sha256": _sha256(nsys_binary),
        "cusparse_sha256": _sha256(cusparse_library),
        "execution_order": list(analyzer.ARM_LABELS),
        "raw_image_cache_modes": [spec[1] for spec in analyzer.ARM_SPECS],
        "raw_image_cache_max_gb": 16.0,
        "raw_image_cache_expected_bytes": analyzer.EXPECTED_CACHE_BYTES,
        "raw_image_cache_force_used": False,
        "checkpoint_iteration": 180,
        "profiled_iteration": 181,
        "nr_iter_schedule": 200,
        "random_seed": 29,
        "image_batch_size": 500,
        "coarse_multistream_workers": 8,
        "single_lane_canonical": False,
        "native_atomic_reduction": True,
        "exact_local_bucket_radix": 4,
        "exact_local_physical_order_chunk_size": 0,
        "science_promotion_allowed": False,
    }
    _write_json(provenance / "run.json", run)
    for name, value in {
        "repo_head.txt": git_head,
        "repo_tree.txt": git_tree,
        "slurm_job_id.txt": run["job_id"],
        "selected_gpu_uuid.txt": GPU_UUID,
        "node.txt": run["node"],
        "gpu_name.txt": run["gpu_name"],
        "allocated_gpu_uuids.csv": GPU_UUID,
        "visible_gpu_uuids.csv": GPU_UUID,
    }.items():
        (provenance / name).write_text(f"{value}\n")
    (provenance / "repo_status.txt").write_text("")
    (provenance / "repo_diff.sha256").write_text(f"{hashlib.sha256(b'').hexdigest()}  -\n")
    (provenance / "science_base_head.txt").write_text(f"{analyzer.SCIENCE_BASE_HEAD}\n")
    (provenance / "science_base_tree.txt").write_text(f"{analyzer.SCIENCE_BASE_TREE}\n")
    (provenance / "nsys.sha256").write_text(f"{_sha256(nsys_binary)}  {nsys_binary.resolve()}\n")
    (provenance / "cusparse.sha256").write_text(f"{_sha256(cusparse_library)}  {cusparse_library.resolve()}\n")

    execution = [
        "order\tlabel\traw_image_cache_mode\traw_image_cache_max_gb\tworkers\tsingle_lane_canonical\tnative_atomic_reduction\tnsys_base\n"
    ]
    base_map = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    map_values = {}
    for label, delta in (("cache_off_1", 0.0), ("cache_off_2", 0.010), ("cache_auto_1", 0.004), ("cache_auto_2", 0.006)):
        value = base_map.copy()
        value[0, 0, 0] += delta
        value[0, 0, 1] -= delta
        map_values[label] = value
    timings = {
        "cache_off_1": (10.0, 8.0, 4.0, 3.0),
        "cache_auto_1": (9.0, 7.0, 3.6, 2.8),
        "cache_auto_2": (8.9, 6.9, 3.5, 2.8),
        "cache_off_2": (10.2, 8.2, 4.1, 3.0),
    }
    for order, (label, mode, _repeat) in enumerate(analyzer.ARM_SPECS, start=1):
        run_root = root / "runs" / label
        profile_root = run_root / "profile"
        cache = run_root / "jax_cache"
        cache.mkdir(parents=True)
        (cache / "SAFE_TO_DELETE").touch()
        (cache / "compiled-cache").write_bytes(b"jax")
        files = sorted(path.name for path in cache.iterdir())
        (run_root / "jax_cache_files.txt").write_text("".join(f"{name}\n" for name in files))
        (run_root / "jax_cache_file_count.txt").write_text(f"{len(files)}\n")
        (run_root / "process.time").write_text("Maximum resident set size: synthetic\n")
        (run_root / "runner.stdout").write_text("profile complete\n")
        (run_root / "runner.stderr").write_text("")
        nsys_base = root / "nsight" / f"{label}_it181_warm"
        execution.append(f"{order}\t{label}\t{mode}\t16\t8\t0\t1\t{nsys_base.resolve()}\n")
        (provenance / f"{label}_command.sh").write_text(
            "nsys profile env "
            f"RECOVAR_EM_RAW_IMAGE_CACHE={mode} RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB=16 "
            "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS=8 "
            "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL=0 "
            "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION=1 "
            f"JAX_COMPILATION_CACHE_DIR={cache.resolve()} python -m scripts.run_vdam_late_iteration_profile "
            "--checkpoint-iteration 180 --nr-iter 200 --image-batch-size 500 "
            "--exact-local-bucket-radix 4 --exact-local-physical-order-chunk-size 0 "
            "--audit-raw-image-cache --cuda-profiler-range\n"
        )
        wall, expectation, pass1_s, pass2_s = timings[label]
        phases = {}
        phase_audits = {}
        for phase_name in ("cold", "warm"):
            phase_root = profile_root / phase_name
            metadata_path = phase_root / "run_it181_recovar_meta.json"
            _write_json(metadata_path, _metadata(pass1_s, pass2_s))
            _write_star(phase_root / "run_it181_data.star")
            _write_map(phase_root / "run_it181_class001.mrc", map_values[label])
            events = [] if mode == "off" else [_cache_event()]
            audit = {"mode": mode, "max_gb": 16.0, "load_all_events": events}
            phase_audits[phase_name] = audit
            phases[phase_name] = {
                "wall_s": wall + (1.0 if phase_name == "cold" else 0.0),
                "meta_path": str(metadata_path.resolve()),
                "meta_sha256": _sha256(metadata_path),
                "iteration_profile": {"expectation_time_s": expectation},
                "schedule": _schedule(),
                "process_resources": _resource_snapshot(hwm_kb=1_000_000 + (100_000 if mode == "auto" else 0)),
                "raw_image_cache_audit": audit,
            }
        _write_json(
            profile_root / "profile_summary.json",
            {
                "schema": analyzer.PROFILE_SCHEMA,
                "classification": "diagnostic_performance_only",
                "checkpoint_iteration": 180,
                "profiled_iteration": 181,
                "nr_iter_schedule": 200,
                "cuda_profiler_range": True,
                "raw_image_cache_audit_enabled": True,
                "exact_local_bucket_radix": 4,
                "exact_local_physical_order_chunk_size": 0,
                **phases,
            },
        )
        _write_json(
            run_root / "cache_admission.json",
            {
                "schema": "recovar.vdam_raw_cache_admission.v1",
                "label": label,
                "mode": mode,
                "max_gb": 16.0,
                "expected_bytes": analyzer.EXPECTED_CACHE_BYTES,
                "phases": {
                    phase: {
                        "load_all_count": len(audit["load_all_events"]),
                        "load_all_events": audit["load_all_events"],
                        "admitted": mode == "auto",
                    }
                    for phase, audit in phase_audits.items()
                },
            },
        )
        _write_nsight(
            root / "nsight" / f"{label}.sqlite",
            root / "nsight" / f"{label}_summary.json",
            root / "nsight" / f"{label}_it181_warm.nsys-rep",
            mode=mode,
            pass1_s=pass1_s,
            pass2_s=pass2_s,
        )
    (provenance / "execution_order.tsv").write_text("".join(execution))
    return root, repo


def _rewrite_summary(root: Path, label: str, mutate) -> None:
    path = root / "runs" / label / "profile" / "profile_summary.json"
    value = json.loads(path.read_text())
    mutate(value)
    _write_json(path, value)


@pytest.mark.unit
def test_complete_fixture_passes_and_cli_writes_json_markdown(tmp_path):
    root, repo = _build_root(tmp_path)

    report = analyzer.analyze(root, repo=repo)

    assert report["decision"]["status"] == "GO"
    assert report["decision"]["pass"] is True
    assert report["performance"]["median_percent_improvement"]["warm_wall_s"] > 5.0
    assert report["performance"]["median_off_minus_auto"]["stage_no_kernel_s"] >= 0.3
    assert report["performance"]["preload"]["break_even_iterations_ceiling"] == 1
    assert report["performance"]["hwm"]["auto_minus_off_median_hwm_bytes"] == 100_000 * 1024
    assert report["science"]["all_cross_mode_maps_within_repeat_envelope"] is True
    assert report["provenance"]["final_source_manifest_state"] == "pending_runner_seal"
    assert Path(report["provenance"]["interpreter"]["path"]).resolve() == (
        tmp_path / "external_python"
    ).resolve()
    assert report["markdown"].startswith("# GO")

    output_json = tmp_path / "analysis" / "report.json"
    output_markdown = tmp_path / "analysis" / "report.md"
    assert analyzer.main(
        ["--root", str(root), "--repo", str(repo), "--output-json", str(output_json), "--output-markdown", str(output_markdown)]
    ) == 0
    assert json.loads(output_json.read_text())["decision"]["status"] == "GO"
    assert output_markdown.read_text().startswith("# GO")


@pytest.mark.unit
def test_valid_but_slow_auto_reports_no_go_and_cli_exit_one(tmp_path):
    root, repo = _build_root(tmp_path)
    for repeat in (1, 2):
        _rewrite_summary(
            root,
            f"cache_auto_{repeat}",
            lambda value: value["warm"].update(wall_s=11.0, iteration_profile={"expectation_time_s": 9.0}),
        )

    report = analyzer.analyze(root, repo=repo)

    assert report["decision"]["status"] == "NO_GO"
    assert report["decision"]["gates"]["both_adjacent_warm_wall_faster"] is False
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    assert analyzer.main(
        ["--root", str(root), "--repo", str(repo), "--output-json", str(output_json), "--output-markdown", str(output_md)]
    ) == 1
    assert json.loads(output_json.read_text())["decision"]["status"] == "NO_GO"


@pytest.mark.unit
def test_discrete_mismatch_is_no_go_not_setup_error(tmp_path):
    root, repo = _build_root(tmp_path)
    metadata = root / "runs" / "cache_auto_1" / "profile" / "warm" / "run_it181_recovar_meta.json"
    value = json.loads(metadata.read_text())
    value["pose_assignments"][0] = 99
    _write_json(metadata, value)
    _rewrite_summary(root, "cache_auto_1", lambda summary: summary["warm"].update(meta_sha256=_sha256(metadata)))

    report = analyzer.analyze(root, repo=repo)

    assert report["decision"]["status"] == "NO_GO"
    assert report["science"]["all_particle_star_and_discrete_metadata_exact"] is False


@pytest.mark.unit
def test_directional_cross_mode_map_drift_is_no_go(tmp_path):
    root, repo = _build_root(tmp_path)
    shifted = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    shifted += 0.001
    for repeat in (1, 2):
        _write_map(root / "runs" / f"cache_auto_{repeat}" / "profile" / "warm" / "run_it181_class001.mrc", shifted)

    report = analyzer.analyze(root, repo=repo)

    assert report["decision"]["status"] == "NO_GO"
    assert report["science"]["all_cross_mode_signed_drift_nondirectional"] is False


@pytest.mark.unit
def test_hwm_overhead_gate_is_no_go(tmp_path):
    root, repo = _build_root(tmp_path)
    excess_kb = (analyzer.EXPECTED_CACHE_BYTES + analyzer.HWM_SLACK_BYTES) // 1024 + 1
    for repeat in (1, 2):
        _rewrite_summary(
            root,
            f"cache_auto_{repeat}",
            lambda value: value["warm"].update(process_resources=_resource_snapshot(hwm_kb=1_000_000 + excess_kb)),
        )

    report = analyzer.analyze(root, repo=repo)

    assert report["decision"]["status"] == "NO_GO"
    assert report["performance"]["gates"]["hwm_auto_minus_off_overhead_within_cache_plus_128mib"] is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda root: (root / "ARMS_COMPLETED").unlink(), "arms are incomplete"),
        (
            lambda root: _rewrite_summary(
                root,
                "cache_off_1",
                lambda value: value["warm"]["raw_image_cache_audit"].update(load_all_events=[_cache_event()]),
            ),
            "load_all event count differs",
        ),
        (
            lambda root: _rewrite_summary(
                root,
                "cache_auto_1",
                lambda value: value["warm"]["raw_image_cache_audit"]["load_all_events"][0].update(cached_nbytes=1),
            ),
            "cache admission differs",
        ),
        (
            lambda root: (root / "runs" / "unexpected").mkdir(),
            "run-directory topology differs",
        ),
        (
            lambda root: _rewrite_summary(
                root,
                "cache_off_1",
                lambda value: value["warm"]["schedule"].update(healpix_order=2),
            ),
            "GF46 iteration-181 schedule differs",
        ),
    ],
)
def test_structural_evidence_mutations_fail_closed(tmp_path, mutate, message):
    root, repo = _build_root(tmp_path)
    mutate(root)

    with pytest.raises(analyzer.RawCacheSetupError, match=message):
        analyzer.analyze(root, repo=repo)


@pytest.mark.unit
def test_source_manifest_corruption_fails_closed(tmp_path):
    root, repo = _build_root(tmp_path)
    (repo / "source.py").write_text("VALUE = 2\n")

    with pytest.raises(analyzer.RawCacheSetupError, match="artifact digest differs"):
        analyzer.analyze(root, repo=repo)


@pytest.mark.unit
def test_nvtx_loader_count_corruption_fails_closed(tmp_path):
    root, repo = _build_root(tmp_path)
    sqlite_path = root / "nsight" / "cache_auto_1.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute("DELETE FROM NVTX_EVENTS WHERE textId = 3")

    with pytest.raises(analyzer.RawCacheSetupError, match="loader NVTX count differs"):
        analyzer.analyze(root, repo=repo)


@pytest.mark.unit
def test_final_source_manifest_is_verified_when_present(tmp_path):
    root, repo = _build_root(tmp_path)
    initial = root / "provenance" / "source_manifest.sha256"
    (root / "provenance" / "source_manifest.final.sha256").write_bytes(initial.read_bytes())
    (root / "COMPLETED").touch()

    report = analyzer.analyze(root, repo=repo)

    assert report["provenance"]["final_source_manifest_state"] == "verified"


@pytest.mark.unit
def test_completed_result_without_final_source_manifest_fails_closed(tmp_path):
    root, repo = _build_root(tmp_path)
    (root / "COMPLETED").touch()

    with pytest.raises(analyzer.RawCacheSetupError, match="lacks its final source manifest"):
        analyzer.analyze(root, repo=repo)
