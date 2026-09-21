"""The local raw-image cache must not change when it stops going through JAX.

`_fetch_local_raw_rows_once` used to route every image through
`CryoEMDataset.iter_batches`, which collates each item into a JAX array on the
device; the caller then pulled it straight back to host NumPy. Under
``RECOVAR_PREREAD_IMAGES`` the loader already holds those images in host memory,
so the round trip bought nothing and cost about 270 s of a 6673 s K=1 100k/256
run at 200 mini-batches -- 4.1%.

The fast path asks the image source for host images directly. These tests pin
the property that makes it safe to prefer: identical images, identical CTF rows,
identical index bookkeeping, for arbitrary index order and repeats -- checked
against the original route rather than against my reading of it. They also pin
that a source without a host path still works.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.data_io.cryoem_dataset as dataset  # noqa: E402
from recovar.em.helpers.batch_fetch import fetch_indexed_batch  # noqa: E402
from recovar.em.local.local_caches import _fetch_local_raw_rows_once  # noqa: E402

pytestmark = pytest.mark.unit

N_IMAGES = 6
D = 4


class _HostImageStack:
    """Minimal non-tilt backend serving distinguishable host images."""

    def __init__(self, n_images=N_IMAGES, d=D):
        self.n_images = n_images
        self.D = d
        self.unpadded_D = d
        self.padding = 0
        self.image_shape = (d, d)
        self.Np = n_images
        self.mask = np.ones((d, d), dtype=np.float32)
        self.particles = [np.asarray([i], dtype=np.int32) for i in range(n_images)]
        self.dataset_tilt_indices = np.arange(n_images, dtype=np.int32)
        # Every image is a distinct constant, so a mis-ordered gather cannot pass.
        self._images = np.arange(n_images, dtype=np.float32)[:, None, None] * np.ones(
            (1, d, d), dtype=np.float32
        )

    def __getitem__(self, index):
        idx = np.atleast_1d(np.asarray(index, dtype=np.int32))
        return self._images[idx], idx, idx

    def get_image_generator(self, batch_size, num_workers=0):
        _ = (batch_size, num_workers)
        idx = np.arange(self.n_images, dtype=np.int32)
        yield self._images, idx, idx

    def get_image_subset_generator(self, batch_size, subset_indices, num_workers=0):
        _ = (batch_size, num_workers)
        idx = np.asarray(subset_indices, dtype=np.int32)
        yield self._images[idx], idx, idx

    def get_dataset_generator(self, batch_size, num_workers=0, **kwargs):
        yield from self.get_image_generator(batch_size, num_workers)

    def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
        yield from self.get_image_subset_generator(batch_size, subset_indices, num_workers)

    def process_images(self, images, apply_image_mask=False):
        return images


def _dataset():
    source = dataset.BackendImageSource(
        _HostImageStack(), info=dataset.ImageSourceInfo(tilt_series=False),
    )
    rots = np.tile(np.eye(3, dtype=np.float32), (N_IMAGES, 1, 1))
    trans = np.zeros((N_IMAGES, 2), dtype=np.float32)
    ctf = np.arange(N_IMAGES * 9, dtype=np.float32).reshape(N_IMAGES, 9)
    return dataset.CryoEMDataset(
        image_source=source,
        voxel_size=1.0,
        metadata=dataset.ImageMetadata(rots, trans, ctf),
        dataset_indices=np.arange(N_IMAGES, dtype=np.int32),
        tilt_series_flag=False,
    )


@pytest.mark.parametrize(
    "indices",
    [
        np.arange(N_IMAGES, dtype=np.int32),
        np.asarray([3, 0, 5, 1], dtype=np.int32),      # out of order
        np.asarray([2, 2, 4, 2], dtype=np.int32),      # repeated
        np.asarray([4], dtype=np.int32),               # single
    ],
)
def test_host_path_matches_the_batch_pipeline(indices):
    cryo = _dataset()

    images, ctf, fetched = _fetch_local_raw_rows_once(cryo, indices)

    reference_images, reference_ctf, reference_indices = fetch_indexed_batch(cryo, indices)
    # The reference route may return its own ordering, so compare row by row through it
    # rather than assuming the two agree positionally.
    lookup = {int(i): pos for pos, i in enumerate(np.asarray(reference_indices))}
    for row, index in enumerate(np.asarray(fetched)):
        ref_row = lookup[int(index)]
        np.testing.assert_array_equal(images[row], np.asarray(reference_images)[ref_row])
        np.testing.assert_array_equal(ctf[row], np.asarray(reference_ctf)[ref_row])
    np.testing.assert_array_equal(np.asarray(fetched), indices)


def test_source_without_a_host_path_still_works(monkeypatch):
    """A source that does not implement `host_images` keeps the original route."""

    cryo = _dataset()
    monkeypatch.delattr(type(cryo.image_source), "host_images", raising=True)

    indices = np.asarray([1, 3], dtype=np.int32)
    images, ctf, fetched = _fetch_local_raw_rows_once(cryo, indices)
    assert images.shape[0] == indices.size
    assert ctf.shape[0] == indices.size
    np.testing.assert_array_equal(np.sort(np.asarray(fetched)), np.sort(indices))
