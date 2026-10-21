"""Every command module must actually run when invoked with ``python -m``.

`recovar/commands/build_custom_cuda.py` and `check_paths.py` defined `main()` but
had no `if __name__ == "__main__"` guard, so `python -m recovar.commands.<name>`
imported the module, ran nothing, and exited 0. A silent success is worse than an
error here: a build or check step wired into a script reports that it worked while
doing nothing at all, which is how a long-parity tier ended up with no library
built and no indication of it.
"""

import ast
import pathlib

import pytest

pytestmark = pytest.mark.unit

COMMANDS_DIR = pathlib.Path(__file__).resolve().parents[2] / "recovar" / "commands"


def _defines_main(tree: ast.Module) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
        for node in tree.body
    )


def _has_main_guard(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and any(
                isinstance(comparator, ast.Constant) and comparator.value == "__main__"
                for comparator in test.comparators
            )
        ):
            return True
    return False


def test_every_command_module_with_a_main_can_be_run_as_a_module():
    modules = sorted(p for p in COMMANDS_DIR.glob("*.py") if p.name != "__init__.py")
    assert modules, f"no command modules found under {COMMANDS_DIR}"

    missing = []
    for path in modules:
        tree = ast.parse(path.read_text())
        if _defines_main(tree) and not _has_main_guard(tree):
            missing.append(path.name)

    assert not missing, (
        "these command modules define main() but never call it under `python -m`, "
        f"so running them exits 0 without doing anything: {missing}"
    )
