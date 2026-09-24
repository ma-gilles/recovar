"""Tests for recovar.data_io.staging — transparent MRC local staging."""

import os
import time

import mrcfile
import numpy as np
import pytest

from recovar.data_io import staging
from recovar.data_io.staging import _cache_key, get_cache_dir, stage_mrc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_mrc(path, n: int = 8, D: int = 16, seed: int = 0) -> np.ndarray:
    """Write a minimal MRC stack; return the raw float32 data."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, D, D)).astype(np.float32)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data)
    return data


# ---------------------------------------------------------------------------
# get_cache_dir
# ---------------------------------------------------------------------------


class TestGetCacheDir:
    def test_returns_recovar_cache_dir(self, monkeypatch):
        monkeypatch.setenv("RECOVAR_CACHE_DIR", "/some/fast/disk")
        monkeypatch.delenv("TMPDIR", raising=False)
        assert get_cache_dir() == "/some/fast/disk"

    def test_falls_back_to_tmpdir(self, monkeypatch):
        monkeypatch.delenv("RECOVAR_CACHE_DIR", raising=False)
        monkeypatch.setenv("TMPDIR", "/tmp/slurm_job123")
        assert get_cache_dir() == "/tmp/slurm_job123"

    def test_network_tmpdir_is_not_staged_to(self, monkeypatch):
        monkeypatch.delenv("RECOVAR_CACHE_DIR", raising=False)
        monkeypatch.setenv("TMPDIR", "/network/job123/tmp")
        monkeypatch.setattr(staging, "filesystem_type", lambda path: "gpfs")
        assert get_cache_dir() is None

    def test_explicit_cache_dir_on_a_network_filesystem_is_honored(self, monkeypatch):
        monkeypatch.setenv("RECOVAR_CACHE_DIR", "/network/job123/tmp")
        monkeypatch.setattr(staging, "filesystem_type", lambda path: "gpfs")
        assert get_cache_dir() == "/network/job123/tmp"

    def test_local_tmpdir_types_still_stage(self, monkeypatch):
        monkeypatch.delenv("RECOVAR_CACHE_DIR", raising=False)
        monkeypatch.setenv("TMPDIR", "/scratch/local/job123")
        for local_type in ("ext4", "xfs", "tmpfs", "btrfs", "overlay"):
            monkeypatch.setattr(staging, "filesystem_type", lambda path, t=local_type: t)
            assert get_cache_dir() == "/scratch/local/job123"

    def test_filesystem_type_resolves_the_nearest_existing_ancestor(self, tmp_path):
        missing = tmp_path / "not" / "created" / "yet"
        assert staging.filesystem_type(str(missing)) == staging.filesystem_type(str(tmp_path))
        assert staging.filesystem_type(str(tmp_path)) is not None

    def test_recovar_cache_dir_takes_precedence_over_tmpdir(self, monkeypatch):
        monkeypatch.setenv("RECOVAR_CACHE_DIR", "/fast")
        monkeypatch.setenv("TMPDIR", "/tmp")
        assert get_cache_dir() == "/fast"

    def test_empty_string_disables_even_when_tmpdir_set(self, monkeypatch):
        monkeypatch.setenv("RECOVAR_CACHE_DIR", "")
        monkeypatch.setenv("TMPDIR", "/tmp/slurm_job123")
        assert get_cache_dir() is None

    def test_both_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv("RECOVAR_CACHE_DIR", raising=False)
        monkeypatch.delenv("TMPDIR", raising=False)
        assert get_cache_dir() is None


# ---------------------------------------------------------------------------
# _cache_key
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_different_paths_give_different_keys(self, tmp_path):
        a = tmp_path / "a.mrcs"
        b = tmp_path / "b.mrcs"
        _write_mrc(a)
        _write_mrc(b)
        assert _cache_key(str(a), os.stat(a)) != _cache_key(str(b), os.stat(b))

    def test_same_path_same_stat_gives_same_key(self, tmp_path):
        f = tmp_path / "p.mrcs"
        _write_mrc(f)
        stat = os.stat(f)
        assert _cache_key(str(f), stat) == _cache_key(str(f), stat)

    def test_changed_mtime_gives_different_key(self, tmp_path):
        f = tmp_path / "p.mrcs"
        _write_mrc(f)
        stat1 = os.stat(f)
        os.utime(f, (stat1.st_atime + 1, stat1.st_mtime + 1))
        stat2 = os.stat(f)
        assert _cache_key(str(f), stat1) != _cache_key(str(f), stat2)

    def test_key_is_short_hex(self, tmp_path):
        f = tmp_path / "p.mrcs"
        _write_mrc(f)
        key = _cache_key(str(f), os.stat(f))
        assert len(key) == 20
        int(key, 16)  # must be valid hex


# ---------------------------------------------------------------------------
# stage_mrc
# ---------------------------------------------------------------------------


class TestStageMrc:
    def test_copies_file_to_cache_subdir(self, tmp_path):
        src = tmp_path / "particles.mrcs"
        _write_mrc(src)
        cache_dir = tmp_path / "cache"

        staged = stage_mrc(str(src), str(cache_dir))

        assert staged != str(src)
        assert os.path.exists(staged)
        assert "recovar_cache" in staged
        assert staged.endswith(".mrcs")

    def test_staged_content_matches_source(self, tmp_path):
        src = tmp_path / "particles.mrcs"
        data = _write_mrc(src)
        cache_dir = tmp_path / "cache"

        staged = stage_mrc(str(src), str(cache_dir))

        with mrcfile.open(staged, mode="r") as mrc:
            np.testing.assert_array_equal(mrc.data, data)

    def test_idempotent_returns_same_path(self, tmp_path):
        src = tmp_path / "particles.mrcs"
        _write_mrc(src)
        cache_dir = tmp_path / "cache"

        staged1 = stage_mrc(str(src), str(cache_dir))
        staged2 = stage_mrc(str(src), str(cache_dir))

        assert staged1 == staged2

    def test_second_call_does_not_recopy(self, tmp_path):
        src = tmp_path / "particles.mrcs"
        _write_mrc(src)
        cache_dir = tmp_path / "cache"

        staged = stage_mrc(str(src), str(cache_dir))
        mtime_after_first = os.path.getmtime(staged)
        time.sleep(0.05)

        stage_mrc(str(src), str(cache_dir))
        mtime_after_second = os.path.getmtime(staged)

        assert mtime_after_first == mtime_after_second  # not re-copied

    def test_source_update_triggers_new_cache_entry(self, tmp_path):
        src = tmp_path / "particles.mrcs"
        _write_mrc(src, seed=1)
        cache_dir = tmp_path / "cache"

        staged1 = stage_mrc(str(src), str(cache_dir))

        # Rewrite source with different data → different mtime
        time.sleep(0.01)
        _write_mrc(src, seed=2)
        os.utime(src, None)  # ensure mtime changes even on coarse filesystems

        staged2 = stage_mrc(str(src), str(cache_dir))

        assert staged1 != staged2

    def test_source_already_under_cache_dir_is_skipped(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        src = cache_dir / "particles.mrcs"
        _write_mrc(src)

        staged = stage_mrc(str(src), str(cache_dir))

        assert staged == str(src)

    def test_nonexistent_source_returns_src_path(self, tmp_path):
        src = str(tmp_path / "missing.mrcs")
        cache_dir = tmp_path / "cache"

        result = stage_mrc(src, str(cache_dir))

        assert result == src  # graceful fallback

    def test_preserves_mrc_extension(self, tmp_path):
        src = tmp_path / "map.mrc"
        _write_mrc(src)
        cache_dir = tmp_path / "cache"

        staged = stage_mrc(str(src), str(cache_dir))

        assert staged.endswith(".mrc")

    def test_sentinel_file_created(self, tmp_path):
        src = tmp_path / "particles.mrcs"
        _write_mrc(src)
        cache_dir = tmp_path / "cache"

        staged = stage_mrc(str(src), str(cache_dir))

        sentinel = staged.replace(".mrcs", ".ok")
        assert os.path.exists(sentinel)


# ---------------------------------------------------------------------------
# Integration: MRCLoader reads through staged file
# ---------------------------------------------------------------------------


class TestMRCLoaderWithStaging:
    def test_loader_reads_from_staged_path(self, tmp_path, monkeypatch):
        src = tmp_path / "particles.mrcs"
        data = _write_mrc(src, n=8, D=16)
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("RECOVAR_CACHE_DIR", str(cache_dir))

        from recovar.data_io.image_loader import MRCLoader

        loader = MRCLoader(str(src))

        assert "recovar_cache" in loader._filepath
        np.testing.assert_array_almost_equal(loader.get(), data)

    def test_loader_uses_source_when_staging_disabled(self, tmp_path, monkeypatch):
        src = tmp_path / "particles.mrcs"
        _write_mrc(src)
        monkeypatch.setenv("RECOVAR_CACHE_DIR", "")  # explicitly disabled
        monkeypatch.delenv("TMPDIR", raising=False)

        from recovar.data_io.image_loader import MRCLoader

        loader = MRCLoader(str(src))

        assert loader._filepath == str(src)

    def test_loader_uses_source_when_no_cache_env(self, tmp_path, monkeypatch):
        src = tmp_path / "particles.mrcs"
        _write_mrc(src)
        monkeypatch.delenv("RECOVAR_CACHE_DIR", raising=False)
        monkeypatch.delenv("TMPDIR", raising=False)

        from recovar.data_io.image_loader import MRCLoader

        loader = MRCLoader(str(src))

        assert loader._filepath == str(src)

    def test_subset_indices_read_correctly_via_staged(self, tmp_path, monkeypatch):
        src = tmp_path / "particles.mrcs"
        data = _write_mrc(src, n=20, D=16)
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("RECOVAR_CACHE_DIR", str(cache_dir))

        from recovar.data_io.image_loader import MRCLoader

        indices = np.array([0, 5, 12, 19])
        loader = MRCLoader(str(src), indices=indices)

        result = loader.get()
        np.testing.assert_array_almost_equal(result, data[indices])

    def test_two_loaders_same_src_reuse_cache(self, tmp_path, monkeypatch):
        src = tmp_path / "particles.mrcs"
        _write_mrc(src)
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("RECOVAR_CACHE_DIR", str(cache_dir))

        from recovar.data_io.image_loader import MRCLoader

        loader1 = MRCLoader(str(src))
        loader2 = MRCLoader(str(src))

        assert loader1._filepath == loader2._filepath
        # Only one staged file in the cache dir
        staged_files = list((cache_dir / "recovar_cache").glob("*.mrcs"))
        assert len(staged_files) == 1


# ---------------------------------------------------------------------------
# stage_stacks (strict) and ImageLoader.stack_files
# ---------------------------------------------------------------------------


def _write_star(path, stack_names, n_per_stack):
    lines = ["data_particles", "", "loop_", "_rlnImageName #1"]
    for name in stack_names:
        lines += [f"{i + 1:06d}@{name}" for i in range(n_per_stack)]
    path.write_text("\n".join(lines) + "\n")


class TestStageStacks:
    def test_copies_every_file_with_the_stage_mrc_layout(self, tmp_path):
        srcs = [tmp_path / "a.mrcs", tmp_path / "b.mrcs"]
        for seed, src in enumerate(srcs):
            _write_mrc(src, seed=seed)
        cache_dir = tmp_path / "cache"

        staged = staging.stage_stacks([str(s) for s in srcs + srcs[:1]], str(cache_dir))

        assert list(staged) == [str(s) for s in srcs]
        for src in srcs:
            assert open(staged[str(src)], "rb").read() == src.read_bytes()
            # A later lenient stage_mrc (what MRCLoader calls) is a cache hit on the same file.
            assert stage_mrc(str(src), str(cache_dir)) == staged[str(src)]

    def test_refuses_before_copying_when_space_is_short(self, tmp_path, monkeypatch):
        src = tmp_path / "a.mrcs"
        _write_mrc(src)
        cache_dir = tmp_path / "cache"
        size = src.stat().st_size

        class _Usage:
            free = size + 99

        monkeypatch.setattr(staging.shutil, "disk_usage", lambda path: _Usage)
        with pytest.raises(staging.StagingSpaceError, match="kept free"):
            staging.stage_stacks([str(src)], str(cache_dir), keep_free_bytes=100)
        assert list((cache_dir / "recovar_cache").iterdir()) == []

        staged = staging.stage_stacks([str(src)], str(cache_dir), keep_free_bytes=99)
        assert os.path.exists(staged[str(src)])

    def test_already_staged_files_need_no_space(self, tmp_path, monkeypatch):
        src = tmp_path / "a.mrcs"
        _write_mrc(src)
        cache_dir = tmp_path / "cache"
        first = staging.stage_stacks([str(src)], str(cache_dir))

        class _Usage:
            free = 0

        monkeypatch.setattr(staging.shutil, "disk_usage", lambda path: _Usage)
        assert staging.stage_stacks([str(src)], str(cache_dir)) == first

    def test_copy_failure_raises_and_leaves_no_partial_file(self, tmp_path, monkeypatch):
        src = tmp_path / "a.mrcs"
        _write_mrc(src)
        cache_dir = tmp_path / "cache"

        def _fail(*args, **kwargs):
            raise OSError("device full")

        monkeypatch.setattr(staging.shutil, "copy2", _fail)
        with pytest.raises(OSError, match="device full"):
            staging.stage_stacks([str(src)], str(cache_dir))
        assert list((cache_dir / "recovar_cache").iterdir()) == []

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            staging.stage_stacks([str(tmp_path / "missing.mrcs")], str(tmp_path / "cache"))


class TestStackFiles:
    def test_star_loader_reports_resolved_source_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RECOVAR_CACHE_DIR", "")
        (tmp_path / "stacks").mkdir()
        for seed, name in enumerate(("a.mrcs", "b.mrcs")):
            _write_mrc(tmp_path / "stacks" / name, n=4, seed=seed)
        star = tmp_path / "particles.star"
        _write_star(star, ["stacks/a.mrcs", "stacks/b.mrcs"], 4)

        from recovar.data_io.image_loader import load_images

        loader = load_images(str(star), lazy=True)
        assert sorted(loader.stack_files()) == sorted(str(tmp_path / "stacks" / name) for name in ("a.mrcs", "b.mrcs"))
        subset = load_images(str(star), indices=np.arange(4), lazy=True)
        assert subset.stack_files() == [str(tmp_path / "stacks" / "a.mrcs")]

    def test_strictly_staged_stacks_serve_every_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RECOVAR_CACHE_DIR", "")
        (tmp_path / "stacks").mkdir()
        data = [
            _write_mrc(tmp_path / "stacks" / name, n=4, seed=seed) for seed, name in enumerate(("a.mrcs", "b.mrcs"))
        ]
        star = tmp_path / "particles.star"
        _write_star(star, ["stacks/a.mrcs", "stacks/b.mrcs"], 4)

        from recovar.data_io.image_loader import load_images

        sources = load_images(str(star), lazy=True, skip_staging=True).stack_files()
        cache_dir = tmp_path / "cache"
        staged = staging.stage_stacks(sources, str(cache_dir))
        monkeypatch.setenv("RECOVAR_CACHE_DIR", str(cache_dir))
        loader = load_images(str(star), lazy=True)
        assert sorted(loader.stack_files()) == sorted(staged.values())

        # The sources are no longer needed: every read comes from the staged copies.
        for src in sources:
            os.rename(src, src + ".moved")
        np.testing.assert_array_equal(loader.get(np.array([6, 1, 4])), np.stack([data[1][2], data[0][1], data[1][0]]))
