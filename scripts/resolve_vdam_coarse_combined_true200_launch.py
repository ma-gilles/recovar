#!/usr/bin/env python3
"""Create or validate the sealed launch manifest for the VDAM combined-coarse true-200 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "recovar.vdam_coarse_combined_true200_launch.v2"


class LaunchResolutionError(RuntimeError):
    """Raised when a launch input is unresolved, stale, or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchResolutionError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchResolutionError(f"cannot read {label} at {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must contain one JSON object")
    return value


def _absolute_existing_file(path: Path, label: str) -> Path:
    _require(path.is_absolute(), f"{label} path must be absolute: {path}")
    absolute = Path(os.path.abspath(path))
    _require(absolute.is_file() and absolute.stat().st_size > 0, f"{label} is missing or empty: {absolute}")
    _require("\n" not in str(absolute), f"{label} path contains a newline")
    return absolute


def _absolute_existing_directory(path: Path, label: str) -> Path:
    _require(path.is_absolute(), f"{label} path must be absolute: {path}")
    resolved = path.resolve()
    _require(resolved.is_dir(), f"{label} directory is missing: {resolved}")
    _require("\n" not in str(resolved), f"{label} path contains a newline")
    return resolved


def _git_output(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LaunchResolutionError(f"cannot inspect source checkout {repo}: {exc}") from exc


def _validate_repo(repo: Path, expected_head: str, production_head: str) -> None:
    _require(_git_output(repo, "rev-parse", "HEAD") == expected_head, "source checkout HEAD differs")
    _require(not _git_output(repo, "status", "--porcelain=v1"), "source checkout is dirty")
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", production_head, expected_head],
        check=False,
    ).returncode
    _require(ancestor == 0, "production candidate is not an ancestor of the launch checkout")


def _source_manifest(repo: Path, relative_paths: Sequence[str]) -> tuple[str, list[dict[str, str]]]:
    raw_paths = []
    for value in relative_paths:
        relative = Path(str(value))
        _require(not relative.is_absolute() and ".." not in relative.parts, f"source path escapes repo: {value}")
        raw_paths.append(relative.as_posix())
    _require(len(set(raw_paths)) == len(raw_paths), "source manifest repeats a canonical path")
    rows = []
    entries = []
    for raw in sorted(raw_paths):
        relative = Path(raw)
        path = _absolute_existing_file(repo / relative, f"source file {raw}")
        digest = _sha256(path)
        rows.append(f"{digest}  {relative.as_posix()}\n")
        entries.append({"path": relative.as_posix(), "sha256": digest})
    manifest_digest = hashlib.sha256("".join(rows).encode()).hexdigest()
    return manifest_digest, entries


def _file_entry(path: Path, label: str, expected_sha256: str | None = None) -> dict[str, str]:
    absolute = _absolute_existing_file(path, label)
    digest = _sha256(absolute)
    if expected_sha256 is not None:
        _require(digest == expected_sha256, f"{label} hash differs: {digest} != {expected_sha256}")
    result = {"path": str(absolute), "resolved_path": str(absolute.resolve()), "sha256": digest}
    if expected_sha256 is not None:
        result["contract_sha256"] = expected_sha256
    return result


def _validate_override_environment(
    environment: dict[str, str],
    runtime_contract: dict[str, Any],
    *,
    include_science: bool = False,
) -> dict[str, str]:
    """Reject inherited RECOVAR/VDAM/JAX controls outside the sealed allowlist."""

    prefixes = tuple(str(value) for value in runtime_contract["override_prefixes_rejected_unless_allowlisted"])
    exact_names = tuple(
        str(value) for value in runtime_contract["override_exact_names_rejected_unless_allowlisted"]
    )
    _require(
        prefixes
        and len(set(prefixes)) == len(prefixes)
        and all(prefix and "\x00" not in prefix and "\n" not in prefix for prefix in prefixes),
        "override-prefix contract is invalid",
    )
    _require(
        len(set(exact_names)) == len(exact_names)
        and all(name and "\x00" not in name and "\n" not in name for name in exact_names)
        and "PATH" not in exact_names,
        "exact-name override contract is invalid",
    )
    allowlist = {str(value) for value in runtime_contract["launch_environment_allowlist"]}
    if include_science:
        allowlist.update(str(value) for value in runtime_contract["science_environment_allowlist"])
    observed = {
        str(name): str(value)
        for name, value in environment.items()
        if str(name).startswith(prefixes) or str(name) in exact_names
    }
    _require(
        all("\x00" not in name and "\n" not in name and "\x00" not in value and "\n" not in value for name, value in observed.items()),
        "override environment contains a NUL or newline",
    )
    undeclared = sorted(set(observed).difference(allowlist))
    _require(not undeclared, f"undeclared override environment variables: {', '.join(undeclared)}")
    return {name: observed[name] for name in sorted(observed)}


def _science_environment_snapshot(
    environment: dict[str, str],
    runtime_contract: dict[str, Any],
) -> dict[str, str]:
    """Return the canonical, curated effective environment used by one arm."""

    override_environment = _validate_override_environment(
        environment,
        runtime_contract,
        include_science=True,
    )
    capture_names = tuple(str(value) for value in runtime_contract["science_environment_capture_names"])
    _require(
        capture_names
        and len(set(capture_names)) == len(capture_names)
        and all(name and "\x00" not in name and "\n" not in name for name in capture_names),
        "science-environment capture contract is invalid",
    )
    missing = sorted(set(capture_names).difference(environment))
    _require(not missing, f"effective science environment is missing: {', '.join(missing)}")
    captured = {name: str(environment[name]) for name in capture_names}
    captured.update(override_environment)
    _require(
        all("\x00" not in value and "\n" not in value for value in captured.values()),
        "effective science environment contains a NUL or newline",
    )
    return {name: captured[name] for name in sorted(captured)}


def _native_repeat_file_entries(
    native_root: Path,
    acceptance: dict[str, Any],
) -> dict[str, dict[str, str]]:
    contract = acceptance["native_reference"]
    repeat_count = int(contract["repeat_count"])
    definitions = (
        ("paired_gpu_uuid", "paired_gpu_uuid.json", "paired_gpu_uuid_sha256"),
        ("run_provenance", "run_provenance.json", "run_provenance_sha256"),
        ("relion_timing", "relion/relion.timing.json", "relion_timing_sha256"),
        ("relion_command", "relion/relion_command.json", "relion_command_sha256"),
    )
    for _, _, contract_name in definitions:
        values = contract.get(contract_name)
        _require(
            isinstance(values, list)
            and len(values) == repeat_count
            and all(isinstance(value, str) and len(value) == 64 for value in values),
            f"native repeat hash contract is invalid: {contract_name}",
        )
    entries = {}
    for index in range(1, repeat_count + 1):
        repeat = native_root / f"repeat-{index:02d}" / "vdam-gf46"
        for label, relative, contract_name in definitions:
            key = f"native_repeat_{index:02d}_{label}"
            entries[key] = _file_entry(
                repeat / relative,
                key.replace("_", " "),
                contract[contract_name][index - 1],
            )
    return entries


def build_launch_manifest(
    *,
    repo_root: Path,
    fixture_dir: Path,
    native_reference_root: Path,
    qualified_cuda: Path,
    relion_bind: Path,
    interpreter: Path,
    gpu_selection_helper: Path,
    cusparse_library: Path,
    expected_overlay_head: str,
    expected_node_name: str,
    target_gpu_uuid: str,
) -> dict[str, Any]:
    """Resolve all launch inputs and return a self-validating manifest payload."""

    repo = _absolute_existing_directory(repo_root, "repo root")
    fixture = _absolute_existing_directory(fixture_dir, "GF46 fixture")
    native = _absolute_existing_directory(native_reference_root, "native reference")
    acceptance_path = repo / "scripts/vdam_coarse_combined_true200_acceptance.json"
    acceptance = _load_json(acceptance_path, "acceptance contract")
    _require(
        acceptance.get("schema") == "recovar.vdam_coarse_combined_true200_acceptance.v2",
        "unsupported acceptance contract",
    )
    production_head = str(acceptance["qualified_candidate"]["production_head"])
    _require(len(expected_overlay_head) == 40, "expected overlay HEAD is not a full commit")
    _require(bool(expected_node_name), "expected node name is empty")
    _require(target_gpu_uuid.startswith("GPU-"), "target GPU UUID is invalid")
    _require(
        target_gpu_uuid == acceptance["native_reference"]["physical_gpu_uuid"],
        "target GPU UUID differs from the native reference",
    )
    _validate_repo(repo, expected_overlay_head, production_head)

    fixture_contract = acceptance["case"]
    qualified = acceptance["qualified_candidate"]
    sealed_override_environment = _validate_override_environment(
        dict(os.environ), acceptance["runtime_contract"]
    )
    files = {
        "acceptance": _file_entry(acceptance_path, "acceptance contract"),
        "analyzer": _file_entry(
            repo / "scripts/analyze_vdam_coarse_combined_true200.py",
            "true-200 analyzer",
        ),
        "harness": _file_entry(
            repo / "scripts/run_vdam_coarse_combined_true200.sbatch",
            "true-200 harness",
        ),
        "resolver": _file_entry(Path(__file__).resolve(), "launch resolver"),
        "scorecard": _file_entry(
            repo / "docs/math/vdam_k1_full_trajectory_expansion_v3.json",
            "frozen v3 scorecard",
            fixture_contract["scorecard_sha256"],
        ),
        "fixture_manifest": _file_entry(
            fixture / "fixture_materialization.json",
            "GF46 fixture manifest",
            fixture_contract["fixture_manifest_sha256"],
        ),
        "fixture_particles_star": _file_entry(
            fixture / "particles.star",
            "GF46 particle STAR",
            fixture_contract["particle_star_sha256"],
        ),
        "fixture_particle_stack": _file_entry(
            fixture / "particles.128.mrcs",
            "GF46 particle stack",
            fixture_contract["particle_stack_sha256"],
        ),
        "fixture_gt_map": _file_entry(
            fixture / "reference_gt_relion.mrc",
            "GF46 ground-truth map",
            fixture_contract["gt_relion_mrc_sha256"],
        ),
        "native_science_manifest": _file_entry(
            native / "science_manifest.json",
            "native science manifest",
            acceptance["native_reference"]["science_manifest_sha256"],
        ),
        "qualified_cuda": _file_entry(
            qualified_cuda,
            "qualified CUDA library",
            qualified["cuda_sha256"],
        ),
        "relion_bind": _file_entry(
            relion_bind,
            "qualified RELION binding",
            qualified["relion_bind_sha256"],
        ),
        "interpreter": _file_entry(
            interpreter,
            "qualified interpreter",
            qualified["interpreter_sha256"],
        ),
        "gpu_selection_helper": _file_entry(gpu_selection_helper, "GPU selection helper"),
        "cusparse_library": _file_entry(cusparse_library, "LD_PRELOAD cuSPARSE library"),
    }
    files.update(_native_repeat_file_entries(native, acceptance))
    _require(
        files["resolver"]["path"]
        == str((repo / "scripts/resolve_vdam_coarse_combined_true200_launch.py").resolve()),
        "launch resolver is not loaded from the recorded repo",
    )
    source_digest, source_entries = _source_manifest(repo, acceptance["source_contract"]["files"])
    return {
        "schema": SCHEMA,
        "classification": "sealed_true200_launch_inputs",
        "repo_root": str(repo),
        "fixture_dir": str(fixture),
        "native_reference_root": str(native),
        "expected_overlay_head": expected_overlay_head,
        "production_candidate_head": production_head,
        "expected_node_name": expected_node_name,
        "target_gpu_uuid": target_gpu_uuid,
        "files": files,
        "source_manifest": {
            "sha256": source_digest,
            "entries": source_entries,
        },
        "runtime_contract": acceptance["runtime_contract"],
        "sealed_override_environment": sealed_override_environment,
    }


def validate_launch_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Validate the external seal and every path/hash in a launch manifest."""

    manifest_path = _absolute_existing_file(path, "launch manifest")
    observed_manifest_sha = _sha256(manifest_path)
    _require(len(expected_sha256) == 64, "expected launch-manifest SHA-256 is invalid")
    _require(observed_manifest_sha == expected_sha256, "launch manifest SHA-256 differs from the external seal")
    payload = _load_json(manifest_path, "launch manifest")
    _require(payload.get("schema") == SCHEMA, "unsupported launch manifest schema")
    repo = _absolute_existing_directory(Path(payload["repo_root"]), "recorded repo root")
    fixture = _absolute_existing_directory(Path(payload["fixture_dir"]), "recorded fixture")
    native = _absolute_existing_directory(Path(payload["native_reference_root"]), "recorded native reference")
    _validate_repo(repo, payload["expected_overlay_head"], payload["production_candidate_head"])
    files = payload.get("files")
    _require(isinstance(files, dict) and files, "launch manifest has no file entries")
    for label, entry in files.items():
        _require(isinstance(entry, dict), f"launch file entry is invalid: {label}")
        resolved = _absolute_existing_file(Path(entry.get("path", "")), f"launch file {label}")
        _require(_sha256(resolved) == entry.get("sha256"), f"launch file hash differs: {label}")
        recorded_resolved_path = entry.get("resolved_path")
        if recorded_resolved_path is not None:
            _require(
                str(resolved.resolve()) == recorded_resolved_path,
                f"launch file symlink target differs: {label}",
            )
        contract_sha = entry.get("contract_sha256")
        if contract_sha is not None:
            _require(entry["sha256"] == contract_sha, f"launch file no longer matches contract: {label}")
    _require(
        Path(files["fixture_manifest"]["path"]).parent == fixture,
        "fixture manifest is outside the recorded fixture directory",
    )
    _require(
        Path(files["native_science_manifest"]["path"]).parent == native,
        "native science manifest is outside the recorded native root",
    )
    acceptance = _load_json(Path(files["acceptance"]["path"]), "recorded acceptance contract")
    _require(
        acceptance.get("schema") == "recovar.vdam_coarse_combined_true200_acceptance.v2",
        "recorded acceptance contract schema differs",
    )
    _require(payload.get("runtime_contract") == acceptance["runtime_contract"], "runtime contract differs")
    _validate_override_environment(dict(os.environ), acceptance["runtime_contract"])
    expected_native_entries = _native_repeat_file_entries(native, acceptance)
    for label, expected_entry in expected_native_entries.items():
        _require(files.get(label) == expected_entry, f"native repeat launch entry differs: {label}")
    source = payload.get("source_manifest")
    _require(isinstance(source, dict), "launch manifest has no source manifest")
    digest, entries = _source_manifest(repo, acceptance["source_contract"]["files"])
    _require(digest == source.get("sha256"), "launch source-manifest digest differs")
    _require(entries == source.get("entries"), "launch source-manifest entries differ")
    _require(
        payload["target_gpu_uuid"] == acceptance["native_reference"]["physical_gpu_uuid"],
        "recorded target GPU differs from acceptance",
    )
    return payload


def _path_or_env(value: Path | None, env_name: str) -> Path | None:
    if value is not None:
        return value
    raw = os.environ.get(env_name)
    return None if raw is None else Path(raw)


def _create(args: argparse.Namespace) -> int:
    named_paths = {
        "repo_root": _path_or_env(args.repo_root, "REPO_ROOT"),
        "fixture_dir": _path_or_env(args.fixture_dir, "VDAM_GF46_FIXTURE_DIR"),
        "native_reference_root": _path_or_env(args.native_reference_root, "VDAM_NATIVE_REFERENCE_ROOT"),
        "qualified_cuda": _path_or_env(args.qualified_cuda, "QUALIFIED_CUDA_OVERRIDE"),
        "relion_bind": _path_or_env(args.relion_bind, "RELION_BIND_BINARY_OVERRIDE"),
        "interpreter": _path_or_env(args.interpreter, "PIXI_PY_OVERRIDE"),
        "gpu_selection_helper": _path_or_env(args.gpu_selection_helper, "GPU_SELECTION_HELPER"),
        "cusparse_library": _path_or_env(args.cusparse_library, "CUSPARSE_LIBRARY"),
    }
    unresolved = sorted(name for name, value in named_paths.items() if value is None)
    scalar_values = {
        "expected_overlay_head": args.expected_overlay_head or os.environ.get("EXPECTED_OVERLAY_HEAD"),
        "expected_node_name": args.expected_node_name or os.environ.get("EXPECTED_NODE_NAME"),
        "target_gpu_uuid": args.target_gpu_uuid or os.environ.get("TARGET_GPU_UUID"),
    }
    unresolved.extend(sorted(name for name, value in scalar_values.items() if not value))
    _require(not unresolved, f"unresolved launch inputs: {', '.join(unresolved)}")
    payload = build_launch_manifest(
        **named_paths,
        **scalar_values,
    )
    output = args.output.resolve()
    repo = Path(payload["repo_root"])
    _require(
        not output.is_relative_to(repo),
        "launch manifest must be written outside the sealed source checkout",
    )
    _require(not output.exists(), f"refusing to overwrite launch manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"launch_manifest": str(output), "sha256": _sha256(output)}, sort_keys=True))
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="resolve and seal launch inputs")
    create.add_argument("--repo-root", type=Path)
    create.add_argument("--fixture-dir", type=Path)
    create.add_argument("--native-reference-root", type=Path)
    create.add_argument("--qualified-cuda", type=Path)
    create.add_argument("--relion-bind", type=Path)
    create.add_argument("--interpreter", type=Path)
    create.add_argument("--gpu-selection-helper", type=Path)
    create.add_argument("--cusparse-library", type=Path)
    create.add_argument("--expected-overlay-head")
    create.add_argument("--expected-node-name")
    create.add_argument("--target-gpu-uuid")
    create.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="validate a sealed launch manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--expected-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "create":
            return _create(args)
        payload = validate_launch_manifest(args.manifest, args.expected_sha256)
    except (LaunchResolutionError, KeyError, TypeError, ValueError) as exc:
        print(f"RESOLUTION_FAILURE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "valid", "schema": payload["schema"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
