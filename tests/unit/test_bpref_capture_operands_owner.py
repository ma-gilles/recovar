"""The exact-local BPref contribution capture has one owner for its fixed operands and priors."""

import inspect

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume import local_em_engine


class _Dataset:
    voxel_size = 1.25


class _Bucket:
    def __init__(self, rotation_mask, sample_mask, translation_log_prior):
        self.local_rotation_mask = rotation_mask
        self.local_sample_mask = sample_mask
        self.translation_log_prior = translation_log_prior


def test_static_capture_operands_record_absent_fields_and_run_geometry():
    window = np.arange(6, dtype=np.int32)
    kwargs = local_em_engine._exact_local_bpref_capture_static_kwargs(
        experiment_dataset=_Dataset(),
        score_with_masked_images=True,
        disc_type="linear_interp",
        projection_padding_factor=2,
        reconstruction_padding_factor=2,
        mstep_relion_x_half=True,
        mstep_adjoint_max_r=7.0,
        mstep_recon_window_indices=window,
        image_shape=(8, 8),
        recon_volume_shape=(16, 16, 9),
    )
    absent = {
        "raw_batch_data", "ctf_params", "noise_variance_half", "integer_pre_shifts", "batch_image_corrections",
        "batch_scale_corrections", "relion_preprocess_normalization_factors", "image_mask", "shadow_reduction_agreement",
    }
    assert all(kwargs[name] is None for name in absent)
    assert kwargs["image_mask_mode"] == "not-captured" and kwargs["ctf_mode"] == "not-captured"
    assert kwargs["high_precision_operand_bundle"] is False and kwargs["relion_cuda_preprocess"] is False
    assert kwargs["winner_take_all"] is False and kwargs["shadow_only_mode"] is False
    assert kwargs["shadow_score_bitwise_equal"] is True
    assert kwargs["ctf_dose_per_tilt"] == 0.0 and kwargs["ctf_angle_per_tilt"] == 0.0
    assert kwargs["voxel_size"] == 1.25 and kwargs["score_with_masked_images"] is True
    assert kwargs["disc_type"] == "linear_interp"
    assert kwargs["use_relion_x_half_mstep"] is True and kwargs["max_r"] == 7.0
    assert kwargs["window_indices"] is window
    assert kwargs["image_shape"] == (8, 8) and kwargs["volume_shape"] == (16, 16, 9)
    assert kwargs["projection_padding_factor"] == 2 and kwargs["reconstruction_padding_factor"] == 2
    assert len(kwargs) == 28


def test_capture_priors_mask_and_remove_pose_priors():
    scores = jnp.asarray([[[0.0, jnp.inf], [1.0, 2.0]]], dtype=jnp.float32)
    rotation_log_prior = jnp.asarray([[0.5, -0.5]], dtype=jnp.float32)
    bucket = _Bucket(np.asarray([[True, False]]), None, np.asarray([[0.25, 0.75]], dtype=np.float32))
    priors = local_em_engine._bpref_capture_priors(scores, scores.shape, bucket=bucket, rotation_log_prior=rotation_log_prior)
    assert np.array_equal(np.asarray(priors.candidate_mask), [[[True, True], [False, False]]])
    assert priors.rotation_log_prior is rotation_log_prior
    assert np.array_equal(np.asarray(priors.translation_log_prior), [[0.25, 0.75]])
    assert np.array_equal(np.asarray(priors.preprior_scores), [[[-0.75, -np.inf], [-np.inf, -np.inf]]])
    masked = _Bucket(np.asarray([[True, True]]), np.asarray([[[True, False], [False, True]]]), np.zeros((1, 2), dtype=np.float32))
    priors = local_em_engine._bpref_capture_priors(scores, scores.shape, bucket=masked, rotation_log_prior=jnp.zeros((1, 2), dtype=jnp.float32))
    assert np.array_equal(np.asarray(priors.candidate_mask), [[[True, False], [False, True]]])
    assert np.array_equal(np.asarray(priors.preprior_scores), [[[0.0, -np.inf], [-np.inf, 2.0]]])


def test_exact_local_capture_sites_use_the_owners():
    source = inspect.getsource(local_em_engine.run_local_em_exact)
    assert source.count("_maybe_dump_exact_local_bpref_contribution_rows(") == 2
    assert source.count("**capture_static_kwargs,") == 2
    assert source.count("capture_priors = _bpref_capture_priors(") == 2
    assert source.count("capture_static_kwargs = (") == 1
    assert "if bpref_contribution_capture_active\n        else None" in source
    assert source.index("_local_mstep_adjoint_window(") < source.index("capture_static_kwargs = (") < source.index("_maybe_dump_exact_local_bpref_contribution_rows(")
    assert 'image_mask_mode="not-captured"' not in source
    assert "preprior_scores = jnp.where(" not in source
