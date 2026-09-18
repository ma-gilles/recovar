"""CPU-only checks for parity result isolation; no scientific jobs are executed."""

import importlib.util
import json
from pathlib import Path

import pytest

from scripts import extract_em_parity_tables as tables

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit


def load_test_module(relative_path):
    spec = importlib.util.spec_from_file_location("ledger_writer", REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tier", ["fast", "long"])
def test_quality_ledgers_keep_runs_and_baselines_separate(tmp_path, monkeypatch, tier):
    folder = "integration" if tier == "fast" else "long_test"
    module = load_test_module(f"tests/{folder}/test_em_parity_{tier}.py")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    name = f"em_parity_quality_{tier}_ledger_k1_replay.json"
    historical = baseline / name
    historical.write_text('{"historical": true}')
    monkeypatch.setattr(module, "BASELINES_DIR", baseline, raising=False)
    payload = {"timestamp": "fixed", "pmax": 0.75}
    first = module._write_quality_ledger("k1_replay", payload, output_dir=tmp_path / "run1")
    second = module._write_quality_ledger("k1_replay", {"pmax": 0.5}, output_dir=tmp_path / "run2")
    assert first == tmp_path / "run1" / name
    assert json.loads(first.read_text()) == payload
    assert json.loads(second.read_text())["pmax"] == 0.5
    assert payload == {"timestamp": "fixed", "pmax": 0.75}
    assert historical.read_text() == '{"historical": true}'
    assert list(baseline.iterdir()) == [historical]


def test_initial_model_ledgers_are_case_local(tmp_path):
    module = load_test_module("tests/integration/test_initial_model_iter10_parity.py")
    for case in ["k2", "k4"]:
        output = tmp_path / case
        payload = {"case": case, "score": 0.75}
        module._write_ledger(payload, output_dir=output)
        assert json.loads((output / "initial_model_iter10_ledger.json").read_text()) == {case: payload}


@pytest.mark.parametrize(
    "tier,case,metric",
    [
        ("fast", "k1_replay", "k1_replay_half1_corr_vs_relion"),
        ("long", "k1_long", "k1_long_recovar_fsc05_resolution_A"),
    ],
)
def test_extract_tables_reads_nested_run_and_original_baseline(tmp_path, monkeypatch, capsys, tier, case, metric):
    baseline = tmp_path / "baselines"
    baseline.mkdir()
    expected = baseline / f"em_parity_quality_{tier}_baseline.json"
    expected.write_text(json.dumps({metric: 0.8}))
    monkeypatch.setattr(tables, "BASELINES_DIR", baseline)
    root = tmp_path / "run"
    output = root / "job" / "case"
    output.mkdir(parents=True)
    payloads = {
        "fast": {
            "k1_replay_half1_corr_vs_relion": 0.9,
            "k1_replay_half2_corr_vs_relion": 0.9,
            "k1_replay_pmax_abs_diff": 0.0001,
            "k1_replay_walltime_s": 3.0,
        },
        "long": {
            "k1_long_recovar_fsc05_resolution_A": 0.9,
            "k1_long_relion_fsc05_resolution_A": 0.9,
            "k1_long_fsc05_resolution_diff_A": 0.0,
            "k1_long_pmax_diff_max_iter3plus": 0.0001,
            "k1_long_walltime_s": 3.0,
        },
    }
    (output / f"em_parity_quality_{tier}_ledger_{case}.json").write_text(json.dumps(payloads[tier]))
    monkeypatch.setattr("sys.argv", ["extract", "--tier", tier, "--ledger-root", str(root)])
    assert tables.main() == 0
    text = capsys.readouterr().out
    assert metric in text
    assert "0.8" in text and "0.9" in text
    assert json.loads(expected.read_text()) == {metric: 0.8}


def test_explicit_empty_run_does_not_reuse_historical_results(tmp_path, monkeypatch, capsys):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "em_parity_quality_fast_ledger_k1_replay.json").write_text('{"k1_replay_half1_corr_vs_relion": 1.0}')
    monkeypatch.setattr(tables, "BASELINES_DIR", baseline)
    run = tmp_path / "empty"
    run.mkdir()
    monkeypatch.setattr("sys.argv", ["extract", "--tier", "fast", "--ledger-root", str(run)])
    assert tables.main() == 1
    assert "no ledger files found" in capsys.readouterr().out


@pytest.mark.parametrize("contents", ["{", "[]"])
def test_corrupt_current_ledger_is_an_error(tmp_path, contents):
    (tmp_path / "ledger.json").write_text(contents)
    with pytest.raises(ValueError):
        tables._load_ledger(tmp_path, "ledger.json")


def test_duplicate_run_ledgers_are_rejected(tmp_path):
    for run in ["one", "two"]:
        folder = tmp_path / run
        folder.mkdir()
        (folder / "ledger.json").write_text("{}")
    with pytest.raises(ValueError, match="Ambiguous ledger"):
        tables._load_ledger(tmp_path, "ledger.json")


def test_missing_explicit_root_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["extract", "--ledger-root", str(tmp_path / "missing")])
    with pytest.raises(SystemExit) as exc:
        tables.main()
    assert exc.value.code == 2


def test_guard_producer_and_reporter_share_explicit_run_root(tmp_path):
    from scripts.run_vdam_abinitio_merge_guard import build_guard_commands

    root = tmp_path / "selected_run"
    plan = {command.name: command.argv for command in build_guard_commands("gpu", ledger_root=root)}
    producer = plan["em_parity_fast_gpu"]
    reporter = plan["extract_em_parity_fast_tables"]
    assert producer[producer.index("--basetemp") + 1] == str(root)
    assert reporter[reporter.index("--ledger-root") + 1] == str(root)
