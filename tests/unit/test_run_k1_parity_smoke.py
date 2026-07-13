from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts import run_k1_parity_smoke as smoke


def _fixture(tmp_path: Path, *, prefix: str = "run") -> smoke.SmokeInputs:
    data_star = tmp_path / "particles.star"
    stack = tmp_path / "particles.mrcs"
    stack.write_bytes(b"stack")
    data_star.write_text("data_particles\n\nloop_\n_rlnImageName #1\n000001@particles.mrcs\n")
    gt = tmp_path / "reference_gt.mrc"
    gt.write_bytes(b"gt")
    relion = tmp_path / "relion"
    relion.mkdir()
    inputs = smoke.SmokeInputs(data_star, gt, relion, tmp_path / "out", tmp_path, 3, prefix)
    for path in smoke.required_relion_paths(inputs):
        path.write_bytes(b"fixture")
    return inputs


def test_validate_portable_fixture_contract(tmp_path):
    inputs = _fixture(tmp_path)
    assert smoke.validate_inputs(inputs) == [(tmp_path / "particles.mrcs").resolve()]
    assert len(smoke.required_relion_paths(inputs)) == 14


def test_nondefault_prefix_is_validated_portably(tmp_path):
    inputs = _fixture(tmp_path, prefix="other")
    assert smoke.validate_inputs(inputs) == [(tmp_path / "particles.mrcs").resolve()]


def test_runner_command_is_one_iteration_split_half_smoke(tmp_path):
    inputs = _fixture(tmp_path)
    args = Namespace(image_batch_size=32, rotation_block_size=512, max_particles=None)
    command = smoke.build_runner_command(args, inputs, Path("/env/python"))
    assert command[0] == "/env/python"
    assert command[command.index("--max_iter") + 1] == "1"
    assert "--skip_final_iteration" in command
    assert "--gt_volume" in command


def test_cli_default_uses_immutable_k1_direct_map_gate(tmp_path):
    parsed = smoke.parser().parse_args(["--output-dir", str(tmp_path / "out")])
    assert parsed.min_relion_fsc_auc == pytest.approx(0.995)


def test_fsc_auc_is_gate_and_correlation_is_auxiliary(tmp_path):
    result = tmp_path / "refinement_results.npz"
    arrays = {
        "final_merged_fsc_vs_relion": np.array([1.0, 1.0, 0.999]),
        "final_merged_fsc_auc_vs_relion": np.float64(0.9995),
        "final_merged_corr_vs_relion": np.float64(-1.0),
    }
    for label in smoke.QUALITY_LABELS:
        arrays[f"{label}_fsc_vs_gt"] = np.array([1.0, 0.8, 0.4])
        arrays[f"{label}_fsc_auc_vs_gt"] = np.float64(0.6)
        arrays[f"{label}_shell_05"] = np.int32(2)
        arrays[f"{label}_shell_0143"] = np.int32(-1)
        arrays[f"{label}_corr_vs_gt"] = np.float64(-1.0)
    np.savez(result, **arrays)
    summary = smoke.quality_summary(result, auc_tolerance=1e-4, min_relion_fsc_auc=0.99)
    assert summary["passed"]
    assert summary["auxiliary_correlations_not_gates"]["final_merged_corr_vs_relion"] == -1.0


def test_fsc_auc_deficit_fails_quality_gate(tmp_path):
    result = tmp_path / "refinement_results.npz"
    arrays = {
        "final_merged_fsc_vs_relion": np.array([1.0, 1.0, 0.999]),
        "final_merged_fsc_auc_vs_relion": np.float64(0.9995),
    }
    for label in smoke.QUALITY_LABELS:
        arrays[f"{label}_fsc_vs_gt"] = np.array([1.0, 0.8, 0.4])
        arrays[f"{label}_fsc_auc_vs_gt"] = np.float64(0.6)
        arrays[f"{label}_shell_05"] = np.int32(2)
        arrays[f"{label}_shell_0143"] = np.int32(-1)
    arrays["recovar_merged_fsc_auc_vs_gt"] = np.float64(0.59)
    np.savez(result, **arrays)
    summary = smoke.quality_summary(result, auc_tolerance=1e-4, min_relion_fsc_auc=0.99)
    assert not summary["passed"]
    assert any("trails RELION" in failure for failure in summary["failures"])


def test_nonfinite_high_frequency_shell_is_json_safe(tmp_path):
    result = tmp_path / "refinement_results.npz"
    arrays = {
        "final_merged_fsc_vs_relion": np.array([1.0, 1.0, 0.999, np.nan]),
        "final_merged_fsc_auc_vs_relion": np.float64(0.9995),
    }
    for label in smoke.QUALITY_LABELS:
        arrays[f"{label}_fsc_vs_gt"] = np.array([1.0, 0.8, 0.4, np.nan])
        arrays[f"{label}_fsc_auc_vs_gt"] = np.float64(0.6)
        arrays[f"{label}_shell_05"] = np.int32(2)
        arrays[f"{label}_shell_0143"] = np.int32(-1)
    np.savez(result, **arrays)
    summary = smoke.quality_summary(result, auc_tolerance=1e-4, min_relion_fsc_auc=0.99)
    assert summary["passed"]
    assert summary["primary_fsc_quality"]["recovar_merged"]["fsc_curve"][-1] is None
