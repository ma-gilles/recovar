from __future__ import annotations

import subprocess

from recovar.utils.parity_provenance import git_worktree_provenance


def test_git_worktree_fingerprint_changes_with_dirty_patch_content(tmp_path, monkeypatch):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "RECOVAR Tests"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)

    monkeypatch.chdir(tmp_path)
    tracked.write_text("dirty one\n", encoding="utf-8")
    first = git_worktree_provenance()
    tracked.write_text("dirty two\n", encoding="utf-8")
    second = git_worktree_provenance()

    assert first["head"] == second["head"]
    assert first["status_porcelain"] == second["status_porcelain"]
    assert first["diff_sha256"] != second["diff_sha256"]
    assert first["worktree_fingerprint_sha256"] != second["worktree_fingerprint_sha256"]


def test_git_worktree_fingerprint_includes_untracked_file_content(tmp_path, monkeypatch):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "RECOVAR Tests"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)

    monkeypatch.chdir(tmp_path)
    untracked = tmp_path / "new_script.py"
    untracked.write_text("print('one')\n", encoding="utf-8")
    first = git_worktree_provenance()
    untracked.write_text("print('two')\n", encoding="utf-8")
    second = git_worktree_provenance()

    assert first["head"] == second["head"]
    assert first["status_porcelain"] == second["status_porcelain"]
    assert first["diff_sha256"] == second["diff_sha256"]
    assert first["worktree_fingerprint_sha256"] != second["worktree_fingerprint_sha256"]
