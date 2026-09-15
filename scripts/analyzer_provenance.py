"""Source identity checks shared by standalone EM diagnostic analyzers."""

import subprocess
from pathlib import Path


def clean_repo_head(repo: Path) -> str:
    """Return HEAD only when tracked and untracked source changes are absent."""
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain=v1"], text=True)
    if status:
        raise ValueError("analyzer repository is dirty")
    return head
