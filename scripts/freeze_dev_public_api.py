"""Freeze the non-EM public API of origin/dev for tests/unit/test_dev_public_api.py.

Usage: python scripts/freeze_dev_public_api.py [REF] > tests/fixtures/dev_public_api.json

For every ``recovar/**.py`` module on REF (default ``origin/dev``) outside ``recovar/em`` and the GUI
frontend, records each public top-level function (positional, keyword-only, required parameters and
*args/**kwargs), class and assigned name. The relax split must keep all of them (PLAN.md section 1.3).
"""

import ast
import json
import subprocess
import sys


def signatures(source):
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            a = node.args
            pos = [x.arg for x in a.posonlyargs + a.args]
            n_default = len(a.defaults)
            out[node.name] = {
                "kind": "def",
                "pos": pos,
                "kw": [x.arg for x in a.kwonlyargs],
                "req": pos[: len(pos) - n_default] if n_default else pos,
                "kwreq": [x.arg for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is None],
                "var": bool(a.vararg),
                "varkw": bool(a.kwarg),
            }
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            out[node.name] = {"kind": "class"}
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    out[target.id] = {"kind": "var"}
    return out


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else "origin/dev"
    git = lambda *args: subprocess.check_output(["git", *args], text=True)  # noqa: E731
    files = [
        f
        for f in git("ls-tree", "-r", "--name-only", ref, "recovar").split()
        if f.endswith(".py") and not f.startswith(("recovar/em/", "recovar/gui_v2/frontend"))
    ]
    modules = {f: signatures(git("show", f"{ref}:{f}")) for f in files}
    record = {"ref": ref, "commit": git("rev-parse", ref).strip(), "modules": {f: s for f, s in modules.items() if s}}
    json.dump(record, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
