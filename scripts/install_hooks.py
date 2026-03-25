#!/usr/bin/env python3
"""Install a git pre-commit hook that runs ruff format check.

Works with both regular clones and git worktrees.
Usage: pixi run install-hooks
"""
import os
import stat

HOOK_CONTENT = """\
#!/bin/sh
# Auto-installed by: pixi run install-hooks
# Checks ruff formatting before each commit.
pixi run ruff format --check recovar/ tests/ scripts/ 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Files are not ruff-formatted."
    echo "Run:   pixi run ruff format recovar/ tests/ scripts/"
    echo ""
    exit 1
fi
"""


def main():
    # Find the git hooks directory (handle worktrees)
    if os.path.isfile(".git"):
        with open(".git") as f:
            gitdir = f.read().strip().split("gitdir: ", 1)[1]
    else:
        gitdir = ".git"

    hooks_dir = os.path.join(gitdir, "hooks")
    os.makedirs(hooks_dir, exist_ok=True)

    hook_path = os.path.join(hooks_dir, "pre-commit")
    with open(hook_path, "w") as f:
        f.write(HOOK_CONTENT)
    os.chmod(hook_path, os.stat(hook_path).st_mode | stat.S_IEXEC)
    print(f"Pre-commit hook installed: {hook_path}")


if __name__ == "__main__":
    main()
