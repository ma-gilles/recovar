"""Frozen RELION InitialModel offset-prior values and coarse-parent inheritance."""
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.vdam import dense_adapter, native_options, native_sampling

pytestmark = pytest.mark.unit


def test_initialmodel_fine_children_inherit_frozen_relion_parent_prior():
    plan = native_sampling.NativeSamplingPlan(
        rotations=np.zeros((1, 3, 3), np.float32),
        translations=np.asarray([[0, -6.5], [0, -5.5]], np.float32),
        random_perturbation=0,
        coarse_translations=np.asarray([[99, 0]], np.float32),
        coarse_prior_translations=np.asarray([[0, -6]], np.float32),
    )
    # The protocol also accepts externally materialized plans. Supplying the
    # parent explicitly isolates adapter behavior from sampling construction.
    fields = vars(plan).copy()
    fields['translation_parent'] = np.asarray([0, 0], np.int64)
    config = dense_adapter._dense_estep_config(
        SimpleNamespace(voxel_size=1.6375, n_images=1, image_shape=(8, 8)),
        native_options.NativeInitialModelOptions(fn_img='particles.star', oversampling=1),
        np.ones(5, np.float32), SimpleNamespace(**fields),
        np.asarray([[4.2, -14.1]], np.float32),
        sigma_offset_angstrom=10,
        class_log_priors=np.zeros(4, np.float64), pass1_healpix_order=1,
    )
    # RELION row114 capture retained by Q donor d1f2f9f934f1.
    expected = np.asarray([[-7.8247542]], np.float32)
    np.testing.assert_array_equal(config.engine_kwargs['coarse_translation_log_prior'], expected)
    np.testing.assert_array_equal(config.engine_kwargs['translation_log_prior'], expected[:, [0, 0]])
    np.testing.assert_array_equal(config.engine_kwargs['image_pre_shifts'], [[4, -14]])
    np.testing.assert_array_equal(config.engine_kwargs['translation_prior_centers'], [[-4, 14]])
