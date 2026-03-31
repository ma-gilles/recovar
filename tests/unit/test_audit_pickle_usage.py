import importlib.util
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.unit


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "audit_pickle_usage.py"
)
_SPEC = importlib.util.spec_from_file_location("audit_pickle_usage", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_audit_path_finds_pickle_module_and_function_aliases(tmp_path):
    target = tmp_path / "example.py"
    target.write_text(
        "\n".join(
            [
                "import pickle as p",
                "from pickle import load as pl, dumps",
                "p.dump({'x': 1}, sink)",
                "value = pl(source)",
                "payload = dumps(value)",
            ]
        ),
        encoding="utf-8",
    )

    uses = _MODULE.audit_path(target, root=tmp_path)

    assert [(use.operation, use.line) for use in uses] == [("dump", 3), ("load", 4), ("dumps", 5)]


def test_build_summary_groups_counts_by_file(tmp_path):
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text("import pickle\npickle.load(src)\npickle.dump(obj, dst)\n", encoding="utf-8")
    second.write_text("from pickle import loads\nloads(blob)\n", encoding="utf-8")

    uses = _MODULE.audit_paths([first, second], root=tmp_path)
    summary = _MODULE.build_summary(uses)
    markdown = _MODULE.render_markdown(summary, uses, details=True)

    assert summary["total_calls"] == 3
    assert summary["total_files"] == 2
    assert summary["by_file"]["a.py"]["load"] == 1
    assert summary["by_file"]["a.py"]["dump"] == 1
    assert summary["by_file"]["b.py"]["loads"] == 1
    assert "`a.py:2` `load`" in markdown
