from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from recovar.em.initial_model.gt_metrics import (
    DEFAULT_GT_ALIGN_HEALPIX_ORDER,
    DEFAULT_GT_ALIGN_MAX_SHELL,
    relion_alignment_rotations,
)
from scripts.run_vdam_abinitio_merge_guard import build_guard_commands, run_guard


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_long_guard_module():
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    module_path = REPO_ROOT / "tests/long_test/test_em_parity_long.py"
    spec = importlib.util.spec_from_file_location("em_parity_long_guard_for_unit_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vdam_alignment_defaults_are_merge_guarded():
    assert DEFAULT_GT_ALIGN_HEALPIX_ORDER == 2
    assert DEFAULT_GT_ALIGN_MAX_SHELL == 8
    assert relion_alignment_rotations(DEFAULT_GT_ALIGN_HEALPIX_ORDER).shape == (4608, 3, 3)


def test_gt_alignment_cli_wiring_survives_merges():
    expected_tokens = [
        "--gt_align",
        "--gt_align_healpix_order",
        "DEFAULT_GT_ALIGN_HEALPIX_ORDER",
        "--gt_align_max_shell",
        "DEFAULT_GT_ALIGN_MAX_SHELL",
        "aligned_fsc_vs_gt",
        "gt_align_rotation_index",
    ]
    for relative_path in [
        "scripts/run_multi_iter_parity.py",
        "scripts/postprocess_multi_iter_gt.py",
    ]:
        text = (REPO_ROOT / relative_path).read_text()
        missing = [token for token in expected_tokens if token not in text]
        assert not missing, f"{relative_path} lost GT-alignment wiring: {missing}"


def test_standalone_abinitio_evaluator_cli_wiring_survives_merges():
    text = (REPO_ROOT / "scripts/evaluate_ab_initio_gt.py").read_text()
    expected_tokens = [
        "--volume_frame",
        "--gt_frame",
        "choices=(\"relion\", \"recovar\")",
        "--gt_align",
        "aligned_prefix = f\"{label}_aligned\"",
        "_fsc_vs_gt",
        "Native InitialModel writes RELION-frame",
        "load_relion_volume",
        "output_json",
        "output_npz",
    ]
    missing = [token for token in expected_tokens if token not in text]
    assert not missing, f"standalone evaluator lost merge-guarded contract: {missing}"


def test_merge_guard_plan_contains_cpu_and_gpu_gates():
    cpu_names = [command.name for command in build_guard_commands("cpu")]
    assert cpu_names == [
        "py_compile",
        "vdam_abinitio_contracts",
        "initial_model_unit_suite",
        "em_fast_guard",
    ]

    quick_names = [command.name for command in build_guard_commands("cpu", quick=True)]
    assert "initial_model_unit_suite" not in quick_names
    assert "em_fast_guard" in quick_names

    all_names = [command.name for command in build_guard_commands("all", quick=True)]
    assert "em_parity_fast_gpu" in all_names
    assert "extract_em_parity_fast_tables" in all_names


def test_long_native_quality_guard_uses_relion_initialmodel_reference():
    long_test = (REPO_ROOT / "tests/long_test/test_em_parity_long.py").read_text()
    slurm_launcher = (REPO_ROOT / "scripts/run_em_parity_long_slurm.sh").read_text()

    expected_long_tokens = [
        "K1_NATIVE_RELION_DIR",
        "relion_initialmodel_k1_it008",
        "_assert_relion_initialmodel_reference",
        "--grad",
        "--denovo_3dref",
        "_rlnDoGradientRefine",
        "_rlnDoAutoRefine",
        "run_it008_class001.mrc",
        "relion_initialmodel_it008",
    ]
    missing = [token for token in expected_long_tokens if token not in long_test]
    assert not missing, f"native InitialModel long guard lost RELION --grad reference checks: {missing}"

    expected_launcher_tokens = [
        "em_parity_long_k1_native_ref",
        "relion_initialmodel_k1_it008",
        "--grad",
        "--denovo_3dref",
        "--dependency=afterok:${K1_NATIVE_REF_JOB}",
        "_rlnDoGradientRefine[[:space:]]*1",
        "_rlnDoAutoRefine[[:space:]]*0",
    ]
    missing = [token for token in expected_launcher_tokens if token not in slurm_launcher]
    assert not missing, f"EM-long launcher no longer prepares the RELION InitialModel reference: {missing}"


def test_relion_initialmodel_reference_checker_rejects_autorefine(tmp_path):
    mod = _load_long_guard_module()

    initialmodel_dir = tmp_path / "initialmodel"
    initialmodel_dir.mkdir()
    (initialmodel_dir / "run_it008_optimiser.star").write_text(
        "\n".join(
            [
                "# RELION optimiser; version test",
                (
                    "# --o out/run --iter 8 --grad --denovo_3dref --i particles.star "
                    "--ctf --K 1 --sym C1 --flatten_solvent --zero_mask "
                    "--dont_combine_weights_via_disc --pool 3 --pad 1 "
                    "--particle_diameter 200 --oversampling 1 --healpix_order 1 "
                    "--offset_range 6 --offset_step 2 --auto_sampling --tau2_fudge 4 "
                    "--j 4 --random_seed 0"
                ),
                "",
                "data_optimiser_general",
                "_rlnCurrentIteration 8",
                "_rlnNumberOfIterations 8",
                "_rlnDoSplitRandomHalves 0",
                "_rlnDoGradientRefine 1",
                "_rlnDoAutoRefine 0",
            ]
        )
    )
    mod._assert_relion_initialmodel_reference(initialmodel_dir, expected_iter=8)

    autorefine_dir = tmp_path / "autorefine"
    autorefine_dir.mkdir()
    (autorefine_dir / "run_it008_optimiser.star").write_text(
        "\n".join(
            [
                "# RELION optimiser; version test",
                "# --i particles.star --ref reference_init_relion.mrc --o out/run --auto_refine --split_random_halves --pad 2 --iter 8",
                "",
                "data_optimiser_general",
                "_rlnCurrentIteration 8",
                "_rlnNumberOfIterations 999",
                "_rlnDoSplitRandomHalves 1",
                "_rlnDoGradientRefine 0",
                "_rlnDoAutoRefine 1",
            ]
        )
    )
    try:
        mod._assert_relion_initialmodel_reference(autorefine_dir, expected_iter=8)
    except AssertionError as exc:
        assert "--grad" in str(exc)
        assert "--denovo_3dref" in str(exc)
    else:
        raise AssertionError("auto-refine fixture was accepted as native InitialModel reference")


def test_merge_guard_dry_run_writes_reproducibility_ledger(tmp_path):
    ledger = run_guard(tier="cpu", quick=True, output_dir=tmp_path, dry_run=True)

    assert ledger["schema"] == "vdam_abinitio_merge_guard.v1"
    assert ledger["ok"] is True
    assert ledger["dry_run"] is True
    assert ledger["provenance"]["ok"] is True
    assert [command["name"] for command in ledger["commands"]] == [
        "py_compile",
        "vdam_abinitio_contracts",
        "em_fast_guard",
    ]

    summary_path = Path(ledger["summary_path"])
    assert summary_path.exists()
    stored = json.loads(summary_path.read_text())
    assert stored["git"]["commit"]
    assert stored["commands"][0]["skipped"] is True
