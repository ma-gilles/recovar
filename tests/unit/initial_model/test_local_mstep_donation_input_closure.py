"""Focused adversarial tests for the donation A/B input and launch closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mrcfile
import numpy as np
import pytest

from scripts import run_local_mstep_donation_ab as runner

pytestmark = pytest.mark.unit


def _write_stack(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=False) as handle:
        handle.set_data(np.asarray(values, dtype=np.float32))


def _make_gf46_fixture(root: Path) -> dict[str, object]:
    checkpoint_dir = root / "checkpoint"
    data_dir = root / "particle_data"
    checkpoint_dir.mkdir(parents=True)
    data_dir.mkdir()
    prefix = checkpoint_dir / "run_it180"
    checkpoint = Path(f"{prefix}_optimiser.star")
    model = Path(f"{prefix}_model.star")
    data = Path(f"{prefix}_data.star")
    sampling = Path(f"{prefix}_sampling.star")
    reference = Path(f"{prefix}_class001.mrc")
    moment1 = Path(f"{prefix}_1moment001.mrc")
    moment1_second = Path(f"{prefix}_1moment002.mrc")
    moment2 = Path(f"{prefix}_2moment001.mrc")
    stack_a = data_dir / "stack-a.mrcs"
    stack_b = data_dir / "nested/stack-b.mrcs"

    checkpoint.write_text(
        "data_optimiser_general\n\n"
        "_rlnModelStarFile run_it180_model.star\n"
        "_rlnExperimentalDataStarFile run_it180_data.star\n"
        "_rlnOrientSamplingStarFile run_it180_sampling.star\n"
    )
    model.write_text(
        "data_model_general\n\n"
        "_rlnNrClasses 1\n\n"
        "data_model_classes\n\n"
        "loop_\n"
        "_rlnReferenceImage #1\n"
        "_rlnGradMoment1 #2\n"
        "_rlnGradMoment2 #3\n"
        "run_it180_class001.mrc run_it180_1moment001.mrc run_it180_2moment001.mrc\n"
    )
    data.write_text(
        "data_particles\n\n"
        "loop_\n"
        "_rlnImageName #1\n"
        "1@stack-a.mrcs\n"
        "2@stack-a.mrcs\n"
        "1@nested/stack-b.mrcs\n"
    )
    sampling.write_text("data_sampling_general\n\n_rlnHealpixOrder 1\n")
    for index, path in enumerate((reference, moment1, moment1_second, moment2), start=1):
        path.write_bytes(f"checkpoint-volume-{index}\n".encode())
    _write_stack(stack_a, np.arange(32, dtype=np.float32).reshape(2, 4, 4))
    _write_stack(stack_b, np.arange(16, dtype=np.float32).reshape(1, 4, 4))
    stacks = sorted((stack_a.resolve(), stack_b.resolve()), key=lambda path: path.as_posix())
    return {
        "checkpoint": checkpoint,
        "model": model,
        "data": data,
        "sampling": sampling,
        "reference": reference,
        "moment1": moment1,
        "moment1_second": moment1_second,
        "moment2": moment2,
        "data_dir": data_dir,
        "stack_a": stack_a,
        "stack_b": stack_b,
        "stacks": stacks,
    }


def _manifest_args(fixture: dict[str, object]) -> tuple[Path, Path, Path, list[Path]]:
    return (
        Path(fixture["checkpoint"]),
        Path(fixture["data"]),
        Path(fixture["data_dir"]),
        list(fixture["stacks"]),
    )


def _clone_stack_bytes(source: dict[str, object], destination: dict[str, object]) -> None:
    for name in ("stack_a", "stack_b"):
        Path(destination[name]).write_bytes(Path(source[name]).read_bytes())


def test_transitive_manifest_is_root_independent_and_pins_two_actual_stacks(tmp_path):
    left = _make_gf46_fixture(tmp_path / "left")
    right = _make_gf46_fixture(tmp_path / "right")
    _clone_stack_bytes(left, right)

    left_payload = runner.gf46_input_manifest_payload(*_manifest_args(left))
    right_payload = runner.gf46_input_manifest_payload(*_manifest_args(right))

    assert left_payload == right_payload
    roles = [entry["relative_name"] for entry in left_payload["entries"]]
    assert roles[:8] == [
        "checkpoint/optimiser.star",
        "checkpoint/model.star",
        "checkpoint/data.star",
        "checkpoint/sampling.star",
        "checkpoint/class001.mrc",
        "checkpoint/1moment001.mrc",
        "checkpoint/1moment002.mrc",
        "checkpoint/2moment001.mrc",
    ]
    assert roles[-2:] == [
        "particles/000/stack-b.mrcs",
        "particles/001/stack-a.mrcs",
    ]
    assert str(tmp_path) not in json.dumps(left_payload)


@pytest.mark.parametrize(
    "member",
    (
        "checkpoint",
        "model",
        "data",
        "sampling",
        "reference",
        "moment1",
        "moment1_second",
        "moment2",
        "stack_a",
        "stack_b",
    ),
)
def test_every_transitively_consumed_file_is_content_sensitive(tmp_path, member: str):
    fixture = _make_gf46_fixture(tmp_path / member)
    baseline = runner.gf46_input_manifest_payload(*_manifest_args(fixture))
    path = Path(fixture[member])
    if member.startswith("stack_"):
        with mrcfile.open(path, mode="r+") as handle:
            handle.data.reshape(-1)[0] += np.float32(1.0)
    else:
        path.write_bytes(path.read_bytes() + b"\n# manifest-drift\n")

    assert runner.gf46_input_manifest_payload(*_manifest_args(fixture)) != baseline


def test_same_content_particle_substitution_is_rejected(tmp_path):
    fixture = _make_gf46_fixture(tmp_path / "fixture")
    alternate = Path(fixture["data_dir"]) / "alternate-a.mrcs"
    alternate.write_bytes(Path(fixture["stack_a"]).read_bytes())
    data = Path(fixture["data"])
    data.write_text(data.read_text().replace("stack-a.mrcs", alternate.name))

    with pytest.raises(RuntimeError, match="particle stacks differ"):
        runner.gf46_input_manifest_payload(*_manifest_args(fixture))


@pytest.mark.parametrize("optimiser_label,target_key,suffix", [
    ("rlnModelStarFile", "model", "alternate_model.star"),
    ("rlnExperimentalDataStarFile", "data", "alternate_data.star"),
])
def test_same_content_star_reference_substitution_is_rejected(
    tmp_path,
    optimiser_label: str,
    target_key: str,
    suffix: str,
):
    fixture = _make_gf46_fixture(tmp_path / target_key)
    target = Path(fixture[target_key])
    alternate = target.with_name(suffix)
    alternate.write_bytes(target.read_bytes())
    optimiser = Path(fixture["checkpoint"])
    lines = optimiser.read_text().splitlines()
    optimiser.write_text(
        "\n".join(
            f"_{optimiser_label} {alternate.name}" if line.startswith(f"_{optimiser_label} ") else line
            for line in lines
        )
        + "\n"
    )

    expected = "optimiser data STAR differs" if target_key == "data" else "non-canonical checkpoint"
    with pytest.raises(RuntimeError, match=expected):
        runner.gf46_input_manifest_payload(*_manifest_args(fixture))


def test_duplicate_or_missing_pinned_particle_targets_are_rejected(tmp_path):
    fixture = _make_gf46_fixture(tmp_path / "fixture")
    checkpoint, data, data_dir, stacks = _manifest_args(fixture)
    with pytest.raises(RuntimeError, match="duplicate canonical paths"):
        runner.gf46_input_manifest_payload(checkpoint, data, data_dir, [*stacks, stacks[0]])
    with pytest.raises(RuntimeError, match="particle stacks differ"):
        runner.gf46_input_manifest_payload(checkpoint, data, data_dir, stacks[:1])


def test_starloader_extension_fallback_pins_the_file_actually_opened(tmp_path):
    fixture = _make_gf46_fixture(tmp_path / "fixture")
    stack_a = Path(fixture["stack_a"])
    Path(fixture["data"]).write_text(
        Path(fixture["data"]).read_text().replace("stack-a.mrcs", "stack-a.mrc")
    )

    payload = runner.gf46_input_manifest_payload(*_manifest_args(fixture))

    stack_entry = next(
        entry for entry in payload["entries"]
        if entry["relative_name"].endswith("/stack-a.mrcs")
    )
    assert stack_entry["sha256"] == hashlib.sha256(stack_a.read_bytes()).hexdigest()


def test_missing_derived_pseudo_half_moment_is_rejected(tmp_path):
    fixture = _make_gf46_fixture(tmp_path / "fixture")
    Path(fixture["moment1_second"]).unlink()

    with pytest.raises(FileNotFoundError):
        runner.gf46_input_manifest_payload(*_manifest_args(fixture))


def test_manifest_resolution_never_creates_a_staging_cache(tmp_path, monkeypatch):
    fixture = _make_gf46_fixture(tmp_path / "fixture")
    temp_dir = tmp_path / "tmp"
    cache_dir = tmp_path / "cache"
    temp_dir.mkdir()
    cache_dir.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp_dir))
    monkeypatch.setenv("RECOVAR_CACHE_DIR", str(cache_dir))

    runner.gf46_input_manifest_payload(*_manifest_args(fixture))

    assert list(temp_dir.iterdir()) == []
    assert list(cache_dir.iterdir()) == []


def _launch_payload(tmp_path: Path) -> dict[str, object]:
    fixture = _make_gf46_fixture(tmp_path / "fixture")
    relion_bind = tmp_path / "relion_bind.so"
    relion_bind.write_bytes(b"relion-bind")
    return {
        "schema": runner.LAUNCH_MANIFEST_SCHEMA,
        "output_root": str((tmp_path / "output").resolve()),
        "repo_root": str(tmp_path.resolve()),
        "checkpoint_optimiser": str(Path(fixture["checkpoint"]).resolve()),
        "input_star": str(Path(fixture["data"]).resolve()),
        "data_dir": str(Path(fixture["data_dir"]).resolve()),
        "particle_stacks": [str(path) for path in fixture["stacks"]],
        "expected_repo_head": "a" * 40,
        "expected_source_manifest_sha256": "b" * 64,
        "expected_input_manifest_sha256": "c" * 64,
        "expected_node_name": "test-h100-node",
        "target_gpu_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "relion_bind_binary": str(relion_bind.resolve()),
        "expected_relion_bind_sha256": hashlib.sha256(relion_bind.read_bytes()).hexdigest(),
        "expected_focused_test_count": 32,
    }


def test_launch_manifest_is_exact_canonical_and_digest_pinned(tmp_path):
    payload = _launch_payload(tmp_path)
    assert runner._validate_launch_manifest_payload(payload) == payload
    manifest = tmp_path / "launch.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    verified = runner._verify_launch_manifest(manifest, digest)
    assert verified["payload"] == payload

    with pytest.raises(RuntimeError, match="SHA mismatch"):
        runner._verify_launch_manifest(manifest, "0" * 64)


@pytest.mark.parametrize("exploit", ("extra_key", "duplicate_stack", "noncanonical_stack"))
def test_launch_manifest_rejects_structural_exploits(tmp_path, exploit: str):
    payload = _launch_payload(tmp_path)
    if exploit == "extra_key":
        payload["MAKEFILES"] = "/tmp/attacker.mk"
    elif exploit == "duplicate_stack":
        payload["particle_stacks"] = [
            payload["particle_stacks"][0],
            payload["particle_stacks"][0],
        ]
    else:
        particle = Path(payload["particle_stacks"][0])
        alias = tmp_path / "particle-alias.mrcs"
        alias.symlink_to(particle)
        payload["particle_stacks"] = [str(alias)]

    with pytest.raises(RuntimeError, match="keys drifted|sorted, unique canonical paths"):
        runner._validate_launch_manifest_payload(payload)
