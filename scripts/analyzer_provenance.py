"""Source and device identity checks shared by standalone EM diagnostic analyzers."""

import subprocess
from pathlib import Path


def clean_repo_head(repo: Path) -> str:
    """Return HEAD only when tracked and untracked source changes are absent."""
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain=v1"], text=True)
    if status:
        raise ValueError("analyzer repository is dirty")
    return head


def allocated_gpu_uuid(expected_gpu_uuid: str) -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            expected_gpu_uuid,
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip()
    if actual != expected_gpu_uuid:
        raise ValueError("allocated GPU UUID mismatch")
    return actual
