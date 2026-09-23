"""``ImageSource.mult`` stays settable on the concrete image sources, as on ``dev``.

``mult`` is the public alias of ``data_multiplier``: assigning it on a
``BackendImageSource`` or a ``SubsetImageSource`` view changes the sign every
view of that stack applies when processing images.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

from helpers import tiny_synthetic

from recovar.data_io import cryoem_dataset as dataset
from recovar.data_io import image_sources

pytestmark = pytest.mark.unit


def _loaded(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    return dataset.load_dataset(
        particles_file=files["particles_mrcs"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(tmp_path),
        lazy=True,
    )


def _processed(cryo):
    images = next(iter(cryo.iter_batches(batch_size=cryo.n_images, by_image=True, prefetch=False)))[0]
    return np.asarray(cryo.process_images(images))


@pytest.mark.parametrize("view", ["backend", "subset"])
def test_setting_mult_flips_the_processing_sign_of_every_view(tmp_path, view):
    cryo = _loaded(tmp_path)
    subset = cryo.subset(np.array([5, 0, 3], dtype=np.int32))
    assert isinstance(cryo.image_source, image_sources.BackendImageSource)
    assert isinstance(subset.image_source, image_sources.SubsetImageSource)
    before_full, before_subset = _processed(cryo), _processed(subset)
    assert cryo.image_source.mult == 1

    source = cryo.image_source if view == "backend" else subset.image_source
    source.mult = -1

    assert cryo.image_source.mult == -1
    assert subset.image_source.mult == -1
    np.testing.assert_array_equal(_processed(cryo), -before_full)
    np.testing.assert_array_equal(_processed(subset), -before_subset)


@pytest.mark.parametrize("view", ["backend", "subset"])
def test_mult_and_data_multiplier_stay_one_state(tmp_path, view):
    cryo = _loaded(tmp_path)
    subset = cryo.subset(np.array([5, 0, 3], dtype=np.int32))
    source = cryo.image_source if view == "backend" else subset.image_source

    source.mult = -1

    for src in (cryo.image_source, subset.image_source):
        assert src.data_multiplier == -1
    assert cryo.data_multiplier == -1
    assert cryo.image_source.info.invert_data
