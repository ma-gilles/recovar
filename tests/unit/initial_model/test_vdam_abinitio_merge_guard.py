from __future__ import annotations

import json
from pathlib import Path

from recovar.em.initial_model.gt_metrics import (
    DEFAULT_GT_ALIGN_HEALPIX_ORDER,
    DEFAULT_GT_ALIGN_MAX_SHELL,
    relion_alignment_rotations,
)
from scripts.run_vdam_abinitio_merge_guard import build_guard_commands, run_guard


REPO_ROOT = Path(__file__).resolve().parents[3]


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
