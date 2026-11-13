"""The public CryoEMDataset constructor keeps its complex64 default.

``CryoEMDataset(image_source, voxel_size, metadata)`` builds a single-precision
dataset whatever its image source stores, as on ``dev``. Double precision is an
explicit request: ``load_dataset(dtype=np.complex128)`` and the dataset's own
subset and reload views pass it on.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")

from helpers import tiny_synthetic

from recovar import core
from recovar.data_io import cryoem_dataset as dataset
from recovar.data_io import halfsets

pytestmark = pytest.mark.unit


def _stack_with_storage_dtype(images, storage_dtype):
    stack = tiny_synthetic.TinyFTImageStack(images)
    if storage_dtype is not None:
        stack.dtype = storage_dtype
    return stack


def _dataset(image_source, **kwargs):
    images, ctf_params, rots, trans, _, voxel_size, _ = tiny_synthetic.make_tiny_simulation(grid_size=4, n_images=6)
    return dataset.CryoEMDataset(
        image_source=image_source(images),
        voxel_size=voxel_size,
        metadata=dataset.ImageMetadata(rots, trans, ctf_params),
        ctf_evaluator=core.CTFEvaluator(),
        grid_size=4,
        **kwargs,
    )


@pytest.mark.parametrize("storage_dtype", [None, np.complex64, np.complex128, np.float32], ids=str)
def test_constructor_default_is_complex64_whatever_the_image_source_stores(storage_dtype):
    cryo = _dataset(lambda images: _stack_with_storage_dtype(images, storage_dtype))

    assert cryo.dtype is np.complex64
    assert cryo.dtype_real == np.dtype(np.float32)
    subset = cryo.subset(np.array([4, 1], dtype=np.int32))
    assert subset.dtype is np.complex64
    assert subset.dtype_real == np.dtype(np.float32)


def test_explicit_double_precision_is_kept_by_subsets():
    cryo = _dataset(lambda images: _stack_with_storage_dtype(images, np.complex128), dtype=np.complex128)

    assert cryo.dtype is np.complex128
    assert cryo.dtype_real == np.dtype(np.float64)
    assert cryo.subset(np.array([0, 3], dtype=np.int32)).dtype is np.complex128


def test_pipeline_halfset_load_is_single_precision(tmp_path):
    """The heterogeneity pipeline's loader (pipeline.py: HalfsetDatasetSpec.from_args) stays float32."""

    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    args = SimpleNamespace(
        particles=files["particles_mrcs"],
        ctf=files["ctf_pkl"],
        poses=files["poses_pkl"],
        datadir=str(tmp_path),
        uninvert_data="automatic",
    )
    spec = halfsets.HalfsetDatasetSpec.from_args(args)
    cryo = halfsets.load_halfset_dataset(
        spec,
        ind_split=[np.array([0, 2, 4], dtype=np.int32), np.array([1, 3, 5], dtype=np.int32)],
        lazy=True,
    )

    assert cryo.dtype is np.complex64
    assert cryo.dtype_real == np.dtype(np.float32)
    assert cryo.CTF_params.dtype == np.float32
    assert cryo.rotation_matrices.dtype == np.float32
    assert cryo.translations.dtype == np.float32
    images = next(iter(cryo.iter_batches(batch_size=6, by_image=True, prefetch=False)))[0]
    assert np.asarray(cryo.process_images(images)).dtype == np.complex64
