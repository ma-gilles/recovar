"""Report inventory validation is independent of scientific acceptance."""

import ast
import json
from pathlib import Path

import pytest

from scripts import extract_em_parity_tables as tables
from scripts.run_vdam_abinitio_merge_guard import build_guard_commands

pytestmark = pytest.mark.unit


def payload(case):
    result = {key: 0.9 for key, _, _ in tables.CASE_METRICS[case]}
    result[f"{case}_walltime_s"] = 1.0
    if case in tables.PER_CLASS_METRICS:
        key, count = tables.PER_CLASS_METRICS[case]
        result[key] = [0.9] * count
    return result


def write_case(root, case):
    tier = next(t for t, cases in tables.TIER_CASES.items() if case in cases)
    path = root / f"em_parity_quality_{tier}_ledger_{case}.json"
    path.write_text(json.dumps(payload(case)))
    return path


@pytest.mark.parametrize("case", tuple(tables.CASE_METRICS))
def test_complete_case_and_each_missing_summary_field(case):
    valid = payload(case)
    tables._validate_case(case, valid)
    for key in valid:
        incomplete = dict(valid)
        del incomplete[key]
        with pytest.raises(ValueError, match=case):
            tables._validate_case(case, incomplete)


@pytest.mark.parametrize("value", [None, True, "0.9", float("nan"), float("inf"), -float("inf")])
def test_invalid_scalar_is_not_a_measurement(value):
    data = payload("k1_replay")
    data["k1_replay_half1_corr_vs_relion"] = value
    with pytest.raises(ValueError, match="k1_replay_half1_corr_vs_relion"):
        tables._validate_case("k1_replay", data)


@pytest.mark.parametrize("values", [[0.9] * 3, [0.9] * 5, [0.9, 0.9, 0.9, float("nan")], [True] * 4])
def test_exact_k4_class_inventory(values):
    data = payload("kclass_strict")
    data["kclass_strict_per_class_corrs_after_hungarian"] = values
    with pytest.raises(ValueError, match="exactly 4 finite class values"):
        tables._validate_case("kclass_strict", data)


def test_missing_time_is_not_reported_as_zero(capsys):
    # Historical partial files stay readable, but do not invent a measurement.
    tables._emit_tier("fast", {"k1_replay": {"k1_replay_half1_corr_vs_relion": 0.9}})
    row = next(line for line in capsys.readouterr().out.splitlines() if "k1_replay_walltime_s" in line)
    assert row == "| k1_replay_walltime_s | — | — | N/A |"


def test_new_k1_cases_and_each_k4_class_are_visible(tmp_path, monkeypatch, capsys):
    cases = ["k1_coldstart", "k1_perturbreplay", "kclass_strict"]
    for case in cases:
        write_case(tmp_path, case)
    monkeypatch.setattr(
        "sys.argv", ["extract", "--tier", "fast", "--ledger-root", str(tmp_path), "--require-case", *cases]
    )
    assert tables.main() == 0
    output = capsys.readouterr().out
    for case in cases:
        for key, _, _ in tables.CASE_METRICS[case]:
            assert key in output
    for index in range(4):
        assert f"kclass_strict_per_class_corrs_after_hungarian[{index}]" in output
    assert "Cases not measured in this report:" in output
    assert "does not establish scientific acceptance" in output


def test_missing_required_case_fails_before_tables(tmp_path, monkeypatch, capsys):
    write_case(tmp_path, "k1_replay")
    monkeypatch.setattr(
        "sys.argv",
        ["extract", "--tier", "fast", "--ledger-root", str(tmp_path), "--require-case", "k1_replay", "k1_coldstart"],
    )
    with pytest.raises(SystemExit) as exc:
        tables.main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "Missing required fast ledger: k1_coldstart" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("args", [["--require-case", "k1_replay"], ["--tier", "long", "--require-case", "k1_replay"]])
def test_required_cases_cannot_use_history_or_wrong_tier(tmp_path, monkeypatch, args):
    if "--tier" in args:
        args = [*args, "--ledger-root", str(tmp_path)]
    monkeypatch.setattr("sys.argv", ["extract", *args])
    with pytest.raises(SystemExit) as exc:
        tables.main()
    assert exc.value.code == 2


def test_report_inventory_matches_actual_producers():
    repo = Path(__file__).resolve().parents[2]
    for tier, folder in [("fast", "integration"), ("long", "long_test")]:
        tree = ast.parse((repo / f"tests/{folder}/test_em_parity_{tier}.py").read_text())
        producers = {}
        for fn in tree.body:
            if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("test_"):
                continue
            calls = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_write_quality_ledger"
            ]
            assignments = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "payload" for x in n.targets)
            ]
            assert len(calls) == len(assignments) == 1
            producers[ast.literal_eval(calls[0].args[0])] = {ast.literal_eval(key) for key in assignments[0].value.keys}
        assert set(producers) == set(tables.TIER_CASES[tier])
        for case, keys in producers.items():
            assert set(payload(case)) <= keys
    plan = {command.name: command.argv for command in build_guard_commands("gpu")}
    argv = plan["extract_em_parity_fast_tables"]
    assert set(argv[argv.index("--require-case") + 1 :]) == set(tables.TIER_CASES["fast"])


def test_guard_requires_current_report_success():
    report = next(command for command in build_guard_commands("gpu") if command.name == "extract_em_parity_fast_tables")
    assert report.required is True


def test_negative_walltime_is_rejected():
    data = payload("k1_replay")
    data["k1_replay_walltime_s"] = -1.0
    with pytest.raises(ValueError, match="negative wall time"):
        tables._validate_case("k1_replay", data)


def test_report_failure_fails_guard_without_running_workloads(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import run_vdam_abinitio_merge_guard as guard

    monkeypatch.setattr(guard, "_git_snapshot", lambda: {})
    monkeypatch.setattr(guard, "_provenance", lambda env: {"ok": True})
    monkeypatch.setattr(guard, "_gpu_snapshot", lambda: "not queried")
    monkeypatch.setattr(guard, "_env_for", lambda command: {})

    def fake_run(argv, **kwargs):
        report = "scripts/extract_em_parity_tables.py" in argv
        return SimpleNamespace(returncode=2 if report else 0, stdout="fixture result")

    monkeypatch.setattr(guard.subprocess, "run", fake_run)
    result = guard.run_guard(tier="gpu", quick=True, output_dir=tmp_path)
    assert result["ok"] is False
    assert len(result["commands"]) == 3
    assert result["commands"][-1]["returncode"] == 2
    assert result["commands"][-1]["required"] is True
