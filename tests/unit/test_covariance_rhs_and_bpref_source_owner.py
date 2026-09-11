"""Owners for the fixed-rotation covariance images and the BPref centered sources."""

import inspect

import numpy as np
import pytest

import recovar.em.heterogeneity as hetero
from recovar.em.initial_model import layout, relion_layout


def test_covariance_accumulators_share_the_image_owner():
    classic = inspect.getsource(hetero.sum_up_images_fixed_rots_covariance_with_precompute)
    eqx = inspect.getsource(hetero.sum_up_images_fixed_rots_covariance_with_precompute_eqx)
    for source in (classic, eqx):
        assert source.count("images = _fixed_rotation_covariance_images(") == 1
        assert "e2_p1 = " not in source and "noise_piece" not in source
        assert "images.before_adj_B2" in source and "images.H_before_adj" in source
    owner = inspect.getsource(hetero._fixed_rotation_covariance_images)
    assert owner.count("evaluate_kernel_on_grid(") == 1
    assert hetero._FixedRotationCovarianceImages._fields == ("before_adj_B2", "H_before_adj")


def test_covariance_images_have_per_rotation_layout():
    n_images, n_rot, n_trans, image_size = 2, 3, 2, 16
    rng = np.random.default_rng(0)
    shifted = (rng.standard_normal((n_images, n_trans, image_size)) + 0j).astype(np.complex64)
    mean_projections = (rng.standard_normal((n_rot, image_size)) + 0j).astype(np.complex64)
    ctf = rng.standard_normal((n_images, image_size)).astype(np.float32)
    probs = rng.random((n_images, n_rot, n_trans)).astype(np.float32)
    noise = rng.random(image_size).astype(np.float32) + 0.1
    gridpoints = rng.standard_normal((n_rot, image_size, 3)).astype(np.float32)
    images = hetero._fixed_rotation_covariance_images(
        shifted, mean_projections, ctf, gridpoints, probs, n_rot, noise, np.zeros(3, dtype=np.float32),
        right_kernel="triangular", right_kernel_width=2,
    )
    assert images.before_adj_B2.shape == (n_rot, image_size)
    assert images.H_before_adj.shape == (n_rot, image_size)
    assert np.iscomplexobj(np.asarray(images.before_adj_B2))
    assert np.all(np.isfinite(np.asarray(images.H_before_adj)))


def test_centered_bpref_sources_validate_and_agree():
    cube = (np.arange(8**3, dtype=np.float32) + 0j).astype(np.complex64)
    data, weight, center, radius = layout._centered_bpref_sources(cube, cube, ori_size=8, r_max=3, padding_factor=1)
    assert data.shape == weight.shape == (8, 8, 8) and (center, radius) == (4, 3)
    with pytest.raises(NotImplementedError):
        layout._centered_bpref_sources(cube, cube, ori_size=8, r_max=3, padding_factor=3)
    with pytest.raises(ValueError, match="non-negative"):
        layout._centered_bpref_sources(cube, cube, ori_size=8, r_max=-1, padding_factor=1)
    full_10 = np.zeros(10**3, dtype=np.complex64)  # center 5
    compact = np.zeros(9**3, dtype=np.complex64)  # current-size cube for r_max=3, center 4
    with pytest.raises(ValueError, match="different centered layouts"):
        layout._centered_bpref_sources(full_10, compact, ori_size=10, r_max=3, padding_factor=1)


def test_bpref_slab_outputs_cast_and_clamp():
    data = np.asarray([1 + 2j], dtype=np.complex64)
    weight = np.asarray([1e-16 + 5j, 2.0, -1e-16], dtype=np.complex64)
    out_data, out_weight = layout._bpref_slab_outputs(data, weight)
    assert out_data.dtype == np.complex128 and out_weight.dtype == np.float64
    assert np.array_equal(out_weight, [0.0, 2.0, 0.0])
    assert out_data[0] == 1 + 2j and out_data is not data


def test_converters_use_the_owners():
    for fn in (layout.run_em_output_to_bpref, relion_layout.relion_x_public_output_to_bpref):
        source = inspect.getsource(fn)
        assert source.count("_centered_bpref_sources(") == 1
        assert source.count("_bpref_slab_outputs(") == 1
        assert "_as_centered_bpref_source(" not in source and "1e-15" not in source
    assert "transpose(2, 1, 0)" in inspect.getsource(relion_layout.relion_x_public_output_to_bpref)
    assert "transpose" not in inspect.getsource(layout.run_em_output_to_bpref)
