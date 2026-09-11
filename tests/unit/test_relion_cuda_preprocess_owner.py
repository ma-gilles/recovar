"""RELION CUDA image-preprocessing detection has one owner, and the fresh K=1 defaults fail closed early without it."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import recovar.em.dense_single_volume.iteration_loop as iteration_loop
import recovar.em.dense_single_volume.local_em_engine as local_em_engine
import recovar.em.initial_model.dense_adapter as dense_adapter
from recovar.em.dense_single_volume.helpers import preprocessing

pytestmark = pytest.mark.unit


def _dataset(backend_name, *, depth=0):
    backend = SimpleNamespace(relion_fourier_backend=backend_name)
    source = SimpleNamespace(backend=backend)
    for _ in range(depth):
        source = SimpleNamespace(parent=source)
    return SimpleNamespace(image_source=source)


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_detection_follows_subset_parents(depth):
    assert preprocessing.uses_relion_cuda_image_preprocessing(_dataset("relion_cuda", depth=depth)) is True
    assert preprocessing.uses_relion_cuda_image_preprocessing(_dataset("host_numpy", depth=depth)) is False
    assert preprocessing.relion_preprocess_backend(_dataset("relion_cuda", depth=depth)).relion_fourier_backend == "relion_cuda"


def test_datasets_without_a_backend_are_not_cuda():
    assert preprocessing.uses_relion_cuda_image_preprocessing(SimpleNamespace()) is False
    assert preprocessing.uses_relion_cuda_image_preprocessing(SimpleNamespace(image_source=object())) is False


def test_initial_model_patch_point_is_the_owner():
    assert dense_adapter._uses_relion_cuda_image_preprocessing is preprocessing.uses_relion_cuda_image_preprocessing


def test_local_engine_and_controller_use_the_owner():
    engine_source = inspect.getsource(local_em_engine.run_local_em_exact)
    assert "uses_relion_cuda_image_preprocessing(experiment_dataset)" in engine_source
    assert 'while hasattr(image_source, "parent")' not in engine_source
    controller_source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    assert "require RELION CUDA image preprocessing; pass" in controller_source
    assert controller_source.count('getattr(backend, "relion_fourier_backend", None) not in (None, "relion_cuda")') == 1
