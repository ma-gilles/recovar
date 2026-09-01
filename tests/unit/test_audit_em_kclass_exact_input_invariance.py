from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import audit_em_kclass_exact_input_invariance as audit


def test_script_path_invocation_resolves_sibling_imports(tmp_path) -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "audit_em_kclass_exact_input_invariance.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--suite-root" in result.stdout


def _axis() -> audit.AxisExpectation:
    return audit.AxisExpectation(
        baseline_case=25,
        variant_case=26,
        key="image_batch_size",
        baseline_value=50,
        variant_value=17,
    )


def _config(*, batch_size: int, name: str, case_root: str) -> dict:
    return {
        "index": 25 if batch_size == 50 else 26,
        "name": name,
        "base_name": name.rsplit("_seed", 1)[0],
        "case_root": case_root,
        "slurm_job_id": "1" if batch_size == 50 else "2",
        "shared_input_role": "producer" if batch_size == 50 else "consumer",
        "n_classes": 4,
        "seed": 41001,
        "grid_size": 128,
        "image_batch_size": batch_size,
        "rotation_block_size": 8192,
        "shared_input_manifest": "/sealed/inputs.sha256",
        "shared_relion_manifest": "/sealed/relion.sha256",
    }


def _slurm_record(**updates: str) -> dict[str, str]:
    record = {
        "job_id": "123",
        "state": "COMPLETED",
        "elapsed_s": "900",
        "exit_code": "0:0",
        "alloc_cpus": "24",
        "req_mem": "192G",
        "req_tres": "billing=24,cpu=24,gres/gpu=1,mem=192G,node=1",
        "alloc_tres": "billing=24,cpu=24,gres/gpu=1,mem=192G,node=1",
        "node": "della-h19g1",
    }
    record.update(updates)
    return record


def _validate_slurm(record: dict[str, str], script: str = "#!/bin/bash\n") -> dict:
    return audit.validate_slurm_allocation(
        record,
        job_script_text=script,
        expected_cpus=24,
        expected_memory="192G",
        expected_gpus=1,
    )


def test_declared_effective_axes_pin_8192_instead_of_submission_default() -> None:
    image_batch = audit.AxisExpectation.parse("25:26:image_batch_size:50:17")
    rotation_block = audit.AxisExpectation.parse("25:27:rotation_block_size:8192:257")

    assert (image_batch.baseline_value, image_batch.variant_value) == (50, 17)
    assert (rotation_block.baseline_value, rotation_block.variant_value) == (8192, 257)
    assert rotation_block.baseline_value != 2000


def test_config_comparison_accepts_only_declared_axis_and_metadata() -> None:
    baseline = _config(batch_size=50, name="baseline_seed41001", case_root="/base")
    variant = _config(batch_size=17, name="variant_seed41001", case_root="/variant")

    report = audit.compare_configurations(baseline, variant, _axis())

    assert report["axis"] == "image_batch_size"
    assert report["undeclared_differences"] == {}


def test_config_comparison_fails_closed_on_axis_or_scientific_drift() -> None:
    baseline = _config(batch_size=50, name="baseline_seed41001", case_root="/base")
    variant = _config(batch_size=17, name="variant_seed41001", case_root="/variant")
    variant["rotation_block_size"] = 2000

    with pytest.raises(audit.AuditError, match="undeclared configuration differences"):
        audit.compare_configurations(baseline, variant, _axis())

    variant["rotation_block_size"] = 8192
    variant["image_batch_size"] = 16
    with pytest.raises(audit.AuditError, match="expected"):
        audit.compare_configurations(baseline, variant, _axis())


def test_manifest_verification_is_absolute_nonempty_and_checksum_exact(tmp_path) -> None:
    artifact = tmp_path / "particles.mrcs"
    artifact.write_bytes(b"sealed particles")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "sealed.sha256"
    manifest.write_text(f"{digest}  {artifact.resolve()}\n")

    report = audit.validate_sha256_manifest(manifest)

    assert report["entry_count"] == 1
    assert report["entries"][0]["sha256"] == digest


@pytest.mark.parametrize("failure", ["corrupt", "relative", "empty"])
def test_manifest_verification_fails_closed(tmp_path, failure: str) -> None:
    artifact = tmp_path / "particles.mrcs"
    artifact.write_bytes(b"sealed particles")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "sealed.sha256"
    if failure == "corrupt":
        manifest.write_text(f"{'0' * 64}  {artifact.resolve()}\n")
        match = "checksum mismatch"
    elif failure == "relative":
        manifest.write_text(f"{digest}  {artifact.name}\n")
        match = "invalid manifest line"
    else:
        manifest.write_text("")
        match = "manifest is empty"

    with pytest.raises(audit.AuditError, match=match):
        audit.validate_sha256_manifest(manifest)


def test_slurm_allocation_accepts_exact_nonexclusive_one_gpu() -> None:
    report = _validate_slurm(_slurm_record())

    assert report["valid"] is True
    assert report["nonexclusive"] is True


@pytest.mark.parametrize(
    ("record", "script", "match"),
    [
        (_slurm_record(state="FAILED"), "#!/bin/bash\n", "state=FAILED"),
        (
            _slurm_record(alloc_tres="billing=48,cpu=24,gres/gpu=2,mem=192G,node=1"),
            "#!/bin/bash\n",
            "ReqTRES != AllocTRES",
        ),
        (_slurm_record(), "#SBATCH --exclusive\n", "requests --exclusive"),
    ],
)
def test_slurm_allocation_fails_closed(
    record: dict[str, str], script: str, match: str
) -> None:
    with pytest.raises(audit.AuditError, match=match):
        _validate_slurm(record, script)


def test_frozen_science_gate_accepts_science_equivalence_not_bitwise_identity() -> None:
    result = audit.evaluate_science_gate(
        complete=True,
        controller_equal=True,
        min_numbered_fsc_auc=0.99999997,
        min_final_fsc_auc=0.99999998,
        final_class_assignment_agreement=1.0,
        min_gt_fsc_auc_delta=-9e-7,
        thresholds=audit.Thresholds(),
    )

    assert result == {"accepted": True, "failures": []}


@pytest.mark.parametrize(
    "updates",
    [
        {"complete": False},
        {"controller_equal": False},
        {"min_numbered_fsc_auc": 0.994999},
        {"min_final_fsc_auc": 0.994999},
        {"final_class_assignment_agreement": 0.989999},
        {"min_gt_fsc_auc_delta": -0.002001},
    ],
)
def test_frozen_science_gate_fails_closed_on_every_required_axis(updates) -> None:
    values = {
        "complete": True,
        "controller_equal": True,
        "min_numbered_fsc_auc": 0.99999997,
        "min_final_fsc_auc": 0.99999998,
        "final_class_assignment_agreement": 1.0,
        "min_gt_fsc_auc_delta": -9e-7,
        "thresholds": audit.Thresholds(),
    }
    values.update(updates)

    result = audit.evaluate_science_gate(**values)

    assert result["accepted"] is False
    assert result["failures"]
