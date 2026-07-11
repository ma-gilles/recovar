from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_monitor_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "monitor_em_robustness_roots.py"
    spec = importlib.util.spec_from_file_location("monitor_em_robustness_roots", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compact_jobs_deduplicates_commas_and_whitespace():
    monitor = _load_monitor_module()

    assert monitor.compact_jobs(["1,2", " 2 3 ", "1"]) == "1,2,3"


def test_write_report_records_queue_accounting_and_summary(tmp_path, monkeypatch):
    monitor = _load_monitor_module()
    calls = []
    root_a = tmp_path / "root_a"
    root_a.mkdir()
    (root_a / "summary.md").write_text("# One-Off Summary\n\nmetric: 1\n")

    def fake_run_text(cmd, *, cwd=monitor.REPO_ROOT):
        calls.append(cmd)
        if cmd[0] == "squeue":
            return 0, "QUEUE\n"
        if cmd[0] == "sacct":
            return 0, "ACCT\n"
        raise AssertionError(cmd)

    def fake_summarize(roots, output_dir):
        markdown = output_dir / "latest_robustness_summary.md"
        json_path = output_dir / "latest_robustness_summary.json"
        markdown.write_text("# Summary\n")
        json_path.write_text('{"cases": [{"job_id": "2"}, {"job_id": "3"}, {"job_id": null}]}\n')
        return 0, "summary output\n", markdown, json_path

    monkeypatch.setattr(monitor, "run_text", fake_run_text)
    monkeypatch.setattr(monitor, "summarize_roots", fake_summarize)

    args = SimpleNamespace(output_dir=tmp_path, root=[root_a, Path("/root/b")], job_id=["1,2", "2"])
    started = dt.datetime(2026, 6, 28, 16, 0, tzinfo=dt.timezone.utc)
    monitor.write_report(args, started, started + dt.timedelta(hours=24))

    report = (tmp_path / "latest_report.md").read_text()
    assert "`1,2,3`" in report
    assert f"`{root_a}`" in report
    assert "QUEUE" in report
    assert "ACCT" in report
    assert "# Summary" in report
    assert "# One-Off Summary" in report
    assert "metric: 1" in report
    assert calls[0][0] == "squeue"
    assert calls[0][2] == "1,2,3"
    assert calls[1][0] == "sacct"
