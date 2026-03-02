from types import SimpleNamespace

import pytest

import recovar.command_line as command_line

pytestmark = pytest.mark.unit


def test_main_commands_shows_usage_when_no_subcommand(monkeypatch, capsys):
    monkeypatch.setattr(command_line.os, "listdir", lambda _: ["foo.py", "__init__.py"])
    monkeypatch.setattr(command_line.sys, "argv", ["recovar"])

    with pytest.raises(SystemExit) as exc:
        command_line.main_commands()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Usage: recovar <command>" in err
    assert "foo" in err


def test_main_commands_rejects_unknown_subcommand(monkeypatch, capsys):
    monkeypatch.setattr(command_line.os, "listdir", lambda _: ["foo.py"])
    monkeypatch.setattr(command_line.sys, "argv", ["recovar", "bar"])

    with pytest.raises(SystemExit) as exc:
        command_line.main_commands()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Command 'bar' not found." in err
    assert "foo" in err


def test_main_commands_dispatches_and_strips_subcommand(monkeypatch):
    captured = {}

    def fake_main():
        captured["argv"] = list(command_line.sys.argv)

    monkeypatch.setattr(command_line.os, "listdir", lambda _: ["foo.py"])
    monkeypatch.setattr(command_line.sys, "argv", ["recovar", "foo", "--x", "1"])
    monkeypatch.setattr(command_line.importlib, "import_module", lambda _: SimpleNamespace(main=fake_main))

    command_line.main_commands()
    assert captured["argv"] == ["recovar", "--x", "1"]


def test_main_commands_import_error_exits(monkeypatch, capsys):
    monkeypatch.setattr(command_line.os, "listdir", lambda _: ["foo.py"])
    monkeypatch.setattr(command_line.sys, "argv", ["recovar", "foo"])

    def raise_import_error(_):
        raise ImportError("boom")

    monkeypatch.setattr(command_line.importlib, "import_module", raise_import_error)

    with pytest.raises(SystemExit) as exc:
        command_line.main_commands()
    assert exc.value.code == 1
    assert "Error importing recovar.commands.foo: boom" in capsys.readouterr().err


def test_main_commands_requires_module_main(monkeypatch, capsys):
    monkeypatch.setattr(command_line.os, "listdir", lambda _: ["foo.py"])
    monkeypatch.setattr(command_line.sys, "argv", ["recovar", "foo"])
    monkeypatch.setattr(command_line.importlib, "import_module", lambda _: SimpleNamespace())

    with pytest.raises(SystemExit) as exc:
        command_line.main_commands()
    assert exc.value.code == 1
    assert "does not define a main() function" in capsys.readouterr().err


def test_main_commands_ignores_non_python_files(monkeypatch, capsys):
    monkeypatch.setattr(command_line.os, "listdir", lambda _: ["foo.py", "README.md", "bar.txt", "__init__.py"])
    monkeypatch.setattr(command_line.sys, "argv", ["recovar"])

    with pytest.raises(SystemExit):
        command_line.main_commands()
    err = capsys.readouterr().err
    assert "foo" in err
    assert "README" not in err
    assert "bar.txt" not in err


def test_main_commands_lists_available_commands_in_sorted_order(monkeypatch, capsys):
    monkeypatch.setattr(command_line.os, "listdir", lambda _: ["zeta.py", "alpha.py", "__init__.py"])
    monkeypatch.setattr(command_line.sys, "argv", ["recovar"])

    with pytest.raises(SystemExit):
        command_line.main_commands()
    err = capsys.readouterr().err

    assert err.index("alpha") < err.index("zeta")
