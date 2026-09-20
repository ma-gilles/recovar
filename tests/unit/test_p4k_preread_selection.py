"""P4-K: ``RECOVAR_PREREAD_IMAGES`` must read only the rows the metadata selects.

``MultiMRCLoader`` (and therefore ``StarLoader`` / ``CryoSparcLoader``) builds one
child ``MRCLoader`` per stack file, without indices, so every child used to preread
its whole file: a 10,045-particle STAR over a 130,000-image stack read 34 GB to serve
2.6 GB.  These tests hold the served images **bitwise** identical to the lazy path
while asserting that only the selected rows are read, and that a selection covering
the whole stack still issues the single sequential read it always did.
"""

import numpy as np
import pandas as pd
import pytest

from recovar.data_io import image_loader
from recovar.data_io.image_loader import MRCLoader, MultiMRCLoader, StarLoader

pytestmark = pytest.mark.unit


def _write_stack(tmp_path, n=40, d=8, seed=0, name="stack.mrcs"):
    import mrcfile

    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, d, d)).astype(np.float32)
    path = tmp_path / name
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.set_image_stack()
    return str(path), data


def _file_map(path, rows):
    return pd.DataFrame({"mrc_file": [path] * len(rows), "mrc_index": np.asarray(rows)})


def _same_bytes(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


class _CountingLoader(MRCLoader):
    """MRCLoader that records every contiguous read it issues."""

    def __init__(self, *args, **kwargs):
        self.reads = []
        super().__init__(*args, **kwargs)

    def _read_contiguous(self, first_index, count):
        self.reads.append((int(first_index), int(count)))
        return super()._read_contiguous(first_index, count)

    @property
    def images_read(self):
        return sum(count for _, count in self.reads)


@pytest.fixture(autouse=True)
def _no_staging(monkeypatch):
    monkeypatch.setenv("RECOVAR_CACHE_DIR", "")


@pytest.fixture
def counting(monkeypatch):
    """Make MultiMRCLoader build counting children."""
    monkeypatch.setattr(image_loader, "MRCLoader", _CountingLoader)
    yield


# --------------------------------------------------------- only the selection ---


@pytest.mark.parametrize(
    "rows",
    [
        [3, 17, 18, 19, 31, 0, 39],  # one run plus scattered rows, unsorted
        [0],  # a single row
        [39, 38, 37],  # a descending run
        [5, 5, 6, 5],  # duplicates
        list(range(0, 40, 7)),  # a stride
        list(range(12, 24)),  # one contiguous block
    ],
)
def test_wrapper_prereads_only_the_selected_rows(tmp_path, monkeypatch, counting, rows):
    path, data = _write_stack(tmp_path)
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "1")
    loader = MultiMRCLoader(_file_map(path, rows), lazy=True, skip_staging=True)
    child = next(iter(loader._loaders.values()))

    expected = len(np.unique(rows))
    assert child.images_read == expected, child.reads
    assert child._cached is not None and child._cached.shape[0] == expected
    assert child._cache_slot_of_position is not None

    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "0")
    lazy = MultiMRCLoader(_file_map(path, rows), lazy=True, skip_staging=True)
    request = np.arange(len(rows))[::-1]
    assert _same_bytes(loader.get(request), lazy.get(request))
    assert _same_bytes(loader.get(request), data[np.asarray(rows)][request])
    assert _same_bytes(loader.get(None), data[np.asarray(rows)])


def test_a_contiguous_selection_is_one_sequential_read(tmp_path, monkeypatch, counting):
    path, _ = _write_stack(tmp_path)
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "1")
    loader = MultiMRCLoader(_file_map(path, list(range(10, 30))), lazy=True, skip_staging=True)
    child = next(iter(loader._loaders.values()))
    assert child.reads == [(10, 20)]


def test_adjacent_rows_coalesce_into_runs(tmp_path, monkeypatch, counting):
    path, _ = _write_stack(tmp_path)
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "1")
    rows = [2, 3, 4, 20, 21, 35]
    loader = MultiMRCLoader(_file_map(path, rows), lazy=True, skip_staging=True)
    child = next(iter(loader._loaders.values()))
    assert sorted(child.reads) == [(2, 3), (20, 2), (35, 1)]


# ------------------------------------------- the full-stack contract is intact ---


def test_a_full_stack_selection_still_issues_the_single_read(tmp_path, monkeypatch, counting):
    path, data = _write_stack(tmp_path)
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "1")
    loader = MultiMRCLoader(_file_map(path, list(range(40))), lazy=True, skip_staging=True)
    child = next(iter(loader._loaders.values()))
    assert child.reads == [(0, 40)]
    assert _same_bytes(child._cached, data)
    assert _same_bytes(loader.get(None), data)


def test_direct_mrcloader_preread_is_unchanged(tmp_path, monkeypatch):
    """A loader constructed with its own selection never took the wrapper path."""
    path, data = _write_stack(tmp_path)
    selection = np.asarray([9, 2, 5, 0, 11], dtype=np.int64)
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "1")
    loader = MRCLoader(path, indices=selection, skip_staging=True)
    assert loader._cached is not None
    assert loader._cache_slot_of_position is None  # the plain, in-order cache
    request = np.asarray([3, 0, 3, 4, 1], dtype=np.int64)
    assert _same_bytes(loader.get(request), data[selection][request])


def test_preread_false_suppresses_the_constructor_read(tmp_path, monkeypatch):
    path, _ = _write_stack(tmp_path)
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "1")
    assert MRCLoader(path, skip_staging=True, preread=False)._cached is None
    assert MRCLoader(path, skip_staging=True, preread=True)._cached is not None
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "0")
    assert MRCLoader(path, skip_staging=True, preread=True)._cached is not None
    assert MRCLoader(path, skip_staging=True)._cached is None


# ------------------------------------------------------------ cache behaviour ---


def test_a_row_outside_the_subset_is_read_not_mis_served(tmp_path, monkeypatch, counting):
    """The wrapper only asks for selected rows, but a wrong answer here would be silent."""
    path, data = _write_stack(tmp_path)
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "1")
    loader = MultiMRCLoader(_file_map(path, [3, 17, 31]), lazy=True, skip_staging=True)
    child = next(iter(loader._loaders.values()))
    served = child._load(np.asarray([5, 17, 5], dtype=np.int64))
    assert _same_bytes(served, data[[5, 17, 5]])


def test_preread_positions_is_idempotent_and_range_checked(tmp_path, monkeypatch):
    path, _ = _write_stack(tmp_path)
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "0")
    child = MRCLoader(path, skip_staging=True, preread=False)
    child.preread_positions([1, 2, 3])
    first = child._cached
    child.preread_positions([10, 11])  # already populated: a no-op, not a re-read
    assert child._cached is first
    fresh = MRCLoader(path, skip_staging=True, preread=False)
    with pytest.raises(IndexError):
        fresh.preread_positions([40])
    fresh.preread_positions([])  # empty is a no-op
    assert fresh._cached is None


def test_the_host_memory_cap_is_now_scoped_to_the_subset(tmp_path, monkeypatch):
    """The cap sizes the rows actually read, not the file.

    A 40-image stack is 10240 bytes here and the three selected rows are 768.
    A cap between the two used to reject the preread outright, because the old
    code sized it from the whole file; it now admits the subset, which is the
    behaviour change this ticket is about.
    """
    path, data = _write_stack(tmp_path)
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "1")

    monkeypatch.setenv(image_loader.PREREAD_MAX_GB_ENV, "0.000005")  # 5000 B
    admitted = MultiMRCLoader(_file_map(path, [1, 2, 3]), lazy=True, skip_staging=True)
    child = next(iter(admitted._loaders.values()))
    assert child._cached is not None and child._cached.shape[0] == 3
    assert _same_bytes(admitted.get(None), data[[1, 2, 3]])

    monkeypatch.setenv(image_loader.PREREAD_MAX_GB_ENV, "0.0000001")  # 100 B
    refused = MultiMRCLoader(_file_map(path, [1, 2, 3]), lazy=True, skip_staging=True)
    assert next(iter(refused._loaders.values()))._cached is None
    assert refused.get(np.asarray([0, 2])).shape == (2, 8, 8)
    assert _same_bytes(refused.get(None), data[[1, 2, 3]])


# ----------------------------------------------------------- several stacks ---


def test_each_stack_prereads_only_its_own_selection(tmp_path, monkeypatch, counting):
    path_a, data_a = _write_stack(tmp_path, n=30, seed=1, name="a.mrcs")
    path_b, data_b = _write_stack(tmp_path, n=30, seed=2, name="b.mrcs")
    frame = pd.DataFrame(
        {
            "mrc_file": [path_a, path_b, path_a, path_b, path_a],
            "mrc_index": [4, 9, 5, 21, 28],
        }
    )
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "1")
    loader = MultiMRCLoader(frame, lazy=True, skip_staging=True)
    assert loader._loaders[path_a].images_read == 3
    assert loader._loaders[path_b].images_read == 2
    expected = np.stack([data_a[4], data_b[9], data_a[5], data_b[21], data_a[28]])
    assert _same_bytes(loader.get(None), expected)


# ------------------------------------------------------------------ StarLoader ---


def test_star_loader_prereads_only_its_particles(tmp_path, monkeypatch, counting):
    path, data = _write_stack(tmp_path, n=50, name="particles.mrcs")
    rows = [41, 2, 3, 4, 17]
    star = tmp_path / "particles.star"
    star.write_text("data_particles\n\nloop_\n_rlnImageName #1\n" + "".join(f"{r + 1:06d}@{path}\n" for r in rows))
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "1")
    loader = StarLoader(str(star), skip_staging=True)
    child = next(iter(loader._loaders.values()))
    assert child.images_read == len(rows)
    assert _same_bytes(loader.get(None), data[np.asarray(rows)])


def test_a_wrapper_subset_prereads_only_that_subset(tmp_path, monkeypatch, counting):
    """The production path: `ImageLoader.from_file(..., indices=ind)`.

    `image_backends` passes the halfset indices down, so the file map is already
    subset before the preread decides anything; the preread must follow that
    subset, not the STAR's full row list.
    """
    path, data = _write_stack(tmp_path, n=60)
    star_rows = list(range(0, 60, 2))  # the STAR lists 30 particles
    subset = np.asarray([0, 1, 2, 20, 29], dtype=np.int64)  # the halfset picks 5
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "1")
    loader = MultiMRCLoader(_file_map(path, star_rows), indices=subset, lazy=True, skip_staging=True)
    child = next(iter(loader._loaders.values()))
    assert child.images_read == subset.size
    expected_rows = np.asarray(star_rows)[subset]
    assert _same_bytes(loader.get(None), data[expected_rows])
    # NOTE: `loader.selection_indices` reports arange(n) rather than `subset` here.
    # That is a pre-existing defect of MultiMRCLoader, unrelated to the preread and
    # present identically at the base: __init__ stores the requested indices, then
    # `super().__init__` overwrites the attribute with arange, and the line after it
    # only re-casts the overwritten value. The served images are correct either way,
    # so it is reported rather than fixed inside this performance change.


def test_star_loader_through_load_images_with_indices(tmp_path, monkeypatch, counting):
    path, data = _write_stack(tmp_path, n=50, name="particles.mrcs")
    rows = [41, 2, 3, 4, 17, 30]
    star = tmp_path / "particles.star"
    star.write_text("data_particles\n\nloop_\n_rlnImageName #1\n" + "".join(f"{r + 1:06d}@{path}\n" for r in rows))
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "1")
    subset = np.asarray([1, 2, 3], dtype=np.int64)
    loader = image_loader.load_images(str(star), indices=subset, lazy=True)
    child = next(iter(loader._loaders.values()))
    assert child.images_read == 3
    assert _same_bytes(loader.get(None), data[np.asarray(rows)[subset]])
