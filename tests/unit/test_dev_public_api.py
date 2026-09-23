"""The non-EM public API of origin/dev stays available on dev2 (relax split, PLAN.md section 1.3).

``tests/fixtures/dev_public_api.json`` freezes every public top-level function, class and name of
the non-EM recovar modules on origin/dev 76f786b1f (regenerate with
``scripts/freeze_dev_public_api.py``). Each must still exist; functions may gain trailing optional
parameters but must keep their positional order, keep every parameter and add no required one.
The two recorded exceptions are user decisions of 2026-09-23.
"""

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "dev_public_api.json"

# Deliberate removal (user decision 2026-09-23): undocumented, no callers, 801 lines of benchmark CUDA.
ALLOWED_MISSING = {"recovar/cuda_backproject.py::CudaBenchmarker"}
ALLOWED_SIGNATURE_BREAKS: set[str] = set()


def _load_freezer():
    spec = importlib.util.spec_from_file_location(
        "freeze_dev_public_api", ROOT / "scripts" / "freeze_dev_public_api.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _problems(dev, head):
    problems = []
    if head["pos"][: len(dev["pos"])] != dev["pos"] and not (head["var"] or head["varkw"]):
        problems.append("positional order changed")
    removed = [p for p in dev["pos"] + dev["kw"] if p not in head["pos"] + head["kw"] and not head["varkw"]]
    if removed:
        problems.append("removed parameters " + ",".join(removed))
    new_required = [p for p in head["req"] + head["kwreq"] if p not in dev["req"] + dev["kwreq"]]
    if new_required:
        problems.append("new required parameters " + ",".join(new_required))
    return problems


def test_dev_public_api_is_preserved():
    frozen = json.loads(FIXTURE.read_text())
    assert frozen["commit"].startswith("76f786b1f")
    signatures = _load_freezer().signatures
    missing, breaks = [], []
    for path, symbols in frozen["modules"].items():
        source = ROOT / path
        head = signatures(source.read_text()) if source.is_file() else {}
        for name, dev in symbols.items():
            key = f"{path}::{name}"
            if name not in head:
                if key not in ALLOWED_MISSING:
                    missing.append(key)
                continue
            if dev["kind"] == "def" and head[name]["kind"] == "def":
                problems = _problems(dev, head[name])
                if problems and key not in ALLOWED_SIGNATURE_BREAKS:
                    breaks.append(f"{key}: {'; '.join(problems)}")
    assert missing == []
    assert breaks == []


def test_recorded_exceptions_are_still_exceptions():
    frozen = json.loads(FIXTURE.read_text())
    signatures = _load_freezer().signatures
    for key in ALLOWED_MISSING:
        path, name = key.split("::")
        assert name in frozen["modules"][path]
        assert name not in signatures((ROOT / path).read_text())
    for key in ALLOWED_SIGNATURE_BREAKS:
        path, name = key.split("::")
        assert _problems(frozen["modules"][path][name], signatures((ROOT / path).read_text())[name])
