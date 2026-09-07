"""Command behavior shared by fixed historical scorecard renderers."""

import json
import sys

import pytest

from scripts.scorecard_cli import run_scorecard_cli


@pytest.fixture
def scorecard_paths(tmp_path):
    source = tmp_path / "scorecard.json"
    source.write_text('{"title": "Historical result"}')
    markdown = tmp_path / "scorecard.md"
    return source, markdown


def load_scorecard(path):
    return json.loads(path.read_text())


def render_scorecard(scorecard):
    return f"# {scorecard['title']}\n"


@pytest.mark.unit
def test_cli_prints_default_scorecard_without_writing(monkeypatch, capsys, scorecard_paths):
    source, markdown = scorecard_paths
    monkeypatch.setattr(sys, "argv", ["scorecard"])
    run_scorecard_cli(source, markdown, load_scorecard, render_scorecard)
    assert capsys.readouterr().out == "# Historical result\n"
    assert not markdown.exists()


@pytest.mark.unit
def test_cli_reads_explicit_source_and_overwrites_explicit_output(monkeypatch, capsys, scorecard_paths, tmp_path):
    source, markdown = scorecard_paths
    alternative = tmp_path / "alternative.json"
    alternative.write_text('{"title": "Selected result"}')
    markdown.write_text("old output")
    monkeypatch.setattr(sys, "argv", ["scorecard", "--scorecard", str(alternative), "--output", str(markdown)])
    run_scorecard_cli(source, markdown, load_scorecard, render_scorecard)
    assert markdown.read_text() == "# Selected result\n"
    assert capsys.readouterr().out == ""


@pytest.mark.unit
@pytest.mark.parametrize("explicit_output", [False, True])
def test_cli_checks_current_output_without_printing(monkeypatch, capsys, scorecard_paths, explicit_output):
    source, markdown = scorecard_paths
    markdown.write_text("# Historical result\n")
    arguments = ["scorecard", "--check"]
    if explicit_output:
        arguments += ["--output", str(markdown)]
    monkeypatch.setattr(sys, "argv", arguments)
    run_scorecard_cli(source, markdown, load_scorecard, render_scorecard)
    assert markdown.read_text() == "# Historical result\n"
    assert capsys.readouterr().out == ""


@pytest.mark.unit
def test_cli_rejects_stale_output_without_overwriting(monkeypatch, scorecard_paths):
    source, markdown = scorecard_paths
    markdown.write_text("stale output")
    monkeypatch.setattr(sys, "argv", ["scorecard", "--check"])
    with pytest.raises(SystemExit) as error:
        run_scorecard_cli(source, markdown, load_scorecard, render_scorecard)
    assert str(error.value) == f"{markdown} is stale; regenerate it"
    assert markdown.read_text() == "stale output"


@pytest.mark.unit
def test_cli_propagates_missing_check_target(monkeypatch, scorecard_paths):
    source, markdown = scorecard_paths
    monkeypatch.setattr(sys, "argv", ["scorecard", "--check"])
    with pytest.raises(FileNotFoundError):
        run_scorecard_cli(source, markdown, load_scorecard, render_scorecard)
    assert not markdown.exists()


@pytest.mark.unit
def test_cli_propagates_loader_failure_without_overwriting(monkeypatch, scorecard_paths):
    source, markdown = scorecard_paths
    source.write_text("invalid JSON")
    markdown.write_text("existing output")
    monkeypatch.setattr(sys, "argv", ["scorecard", "--output", str(markdown)])
    with pytest.raises(json.JSONDecodeError):
        run_scorecard_cli(source, markdown, load_scorecard, render_scorecard)
    assert markdown.read_text() == "existing output"
