from pathlib import Path

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as sparse


def test_norm_residual_input_capture_preserves_exact_target_arrays(tmp_path, monkeypatch):
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES", "66")
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_CURRENT_SIZE", "56")
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ITERATION", "1")
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS", "1")
    monkeypatch.setitem(sparse._bpref_contribution_context, "iteration", 1)
    monkeypatch.setitem(sparse._bpref_contribution_context, "half", 2)
    proj = jnp.asarray([[[1 + 2j, 3 + 4j]]], dtype=jnp.complex64)
    proj_abs2 = jnp.abs(proj) ** 2
    summed = jnp.asarray([[[5 + 6j, 7 + 8j]]], dtype=jnp.complex64)
    ctf_probs = jnp.asarray([[[2.0, 3.0]]], dtype=jnp.float32)
    noise = jnp.asarray([11.0, 13.0], dtype=jnp.float32)
    residual = jnp.asarray([17.0], dtype=jnp.float32)
    processed = jnp.asarray([[2 + 3j, 4 + 5j]], dtype=jnp.complex64)
    shells = jnp.asarray([0, 1], dtype=jnp.int32)
    support_mass = jnp.asarray([1.0], dtype=jnp.float32)
    high_shell = jnp.asarray([19.0], dtype=jnp.float32)
    weighted_image_power = jnp.asarray([23.0], dtype=jnp.float32)

    count = sparse._maybe_dump_norm_residual_inputs(
        experiment_dataset=object(),
        image_indices=np.asarray([66], dtype=np.int64),
        current_size=56,
        proj_for_noise=proj,
        proj_abs2_for_noise=proj_abs2,
        summed_masked_noise=summed,
        ctf_probs=ctf_probs,
        noise_variance_for_noise=noise,
        block_norm_residual=residual,
        processed_score_half_for_noise=processed,
        shell_indices_half=shells,
        support_mass=support_mass,
        relion_norm_high_shell=high_shell,
        weighted_img_per_image=weighted_image_power,
        relion_score_translation_angles=None,
        recon_window_indices=jnp.asarray([0, 1], dtype=jnp.int32),
        score_window_indices=jnp.asarray([0, 1], dtype=jnp.int32),
        image_shape=(2, 2),
    )

    assert count == 1
    path = Path(tmp_path) / "norm_residual_orig000066_half2_cs056.npz"
    with np.load(path, allow_pickle=False) as capture:
        assert capture["schema"].item() == "recovar-k1-norm-residual-inputs-v1"
        np.testing.assert_array_equal(capture["proj_for_noise"], np.asarray(proj[0]))
        np.testing.assert_array_equal(capture["proj_abs2_for_noise"], np.asarray(proj_abs2[0]))
        np.testing.assert_array_equal(capture["summed_masked_noise"], np.asarray(summed[0]))
        np.testing.assert_array_equal(capture["ctf_probs"], np.asarray(ctf_probs[0]))
        np.testing.assert_array_equal(capture["noise_variance_for_noise"], np.asarray(noise))
        assert float(capture["block_norm_residual"]) == 17.0
        np.testing.assert_array_equal(capture["processed_score_half_for_noise"], np.asarray(processed[0]))
        np.testing.assert_array_equal(capture["shell_indices_half"], np.asarray(shells))
        assert float(capture["support_mass"]) == 1.0
        assert float(capture["relion_norm_high_shell"]) == 19.0
        assert float(capture["weighted_img_per_image"]) == 23.0
        assert capture["raw_translated_recon"].shape == (0, 0)
        assert capture["raw_translated_wavg"].shape == (0, 0)


def test_deterministic_norm_reduction_uses_float64_sum(monkeypatch):
    monkeypatch.setenv("RECOVAR_K1_RELION_DETERMINISTIC_NORM_REDUCTION", "1")
    processed = np.asarray(
        [[10000 + 0j, 1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j]],
        dtype=np.complex64,
    )
    shells = np.zeros(processed.shape[1], dtype=np.int32)

    _, per_image = sparse._weighted_image_power_shells_and_per_image(
        jnp.asarray(processed),
        jnp.asarray(shells),
        jnp.ones(1, dtype=jnp.float32),
        shell_count=1,
    )

    expected = np.float32(np.sum(np.abs(processed[0]) ** 2, dtype=np.float64))
    assert np.asarray(per_image).dtype == np.float32
    assert np.float32(per_image[0]) == expected


def test_norm_capture_slices_and_reshapes_raw_translation_rows(tmp_path, monkeypatch):
    from recovar import cuda_backproject

    monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES", "66")
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_CURRENT_SIZE", "56")
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ITERATION", "1")
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS", "1")
    monkeypatch.setitem(sparse._bpref_contribution_context, "iteration", 1)
    monkeypatch.setitem(sparse._bpref_contribution_context, "half", 2)

    def fake_translate(images, angles, pixel_indices, image_shape):
        assert angles.shape == (2, 2)
        assert image_shape == (2, 2)
        values = jnp.arange(2 * images.shape[1], dtype=jnp.float32).astype(jnp.complex64)
        return values.reshape(2, images.shape[1])

    monkeypatch.setattr(cuda_backproject, "relion_translate_score_f32", fake_translate)
    sparse._maybe_dump_norm_residual_inputs(
        experiment_dataset=object(),
        image_indices=np.asarray([66], dtype=np.int64),
        current_size=56,
        proj_for_noise=jnp.ones((1, 1, 2), dtype=jnp.complex64),
        proj_abs2_for_noise=jnp.ones((1, 1, 2), dtype=jnp.float32),
        summed_masked_noise=jnp.ones((1, 1, 2), dtype=jnp.complex64),
        ctf_probs=jnp.ones((1, 1, 2), dtype=jnp.float32),
        noise_variance_for_noise=jnp.ones(2, dtype=jnp.float32),
        block_norm_residual=jnp.ones(1, dtype=jnp.float32),
        processed_score_half_for_noise=jnp.arange(4, dtype=jnp.float32).astype(jnp.complex64)[None],
        shell_indices_half=jnp.zeros(4, dtype=jnp.int32),
        support_mass=jnp.ones(1, dtype=jnp.float32),
        relion_norm_high_shell=jnp.ones(1, dtype=jnp.float32),
        weighted_img_per_image=jnp.ones(1, dtype=jnp.float32),
        relion_score_translation_angles=jnp.zeros((2, 2), dtype=jnp.float32),
        recon_window_indices=jnp.asarray([0, 2], dtype=jnp.int32),
        score_window_indices=jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        image_shape=(2, 2),
    )

    with np.load(
        tmp_path / "norm_residual_orig000066_half2_cs056.npz",
        allow_pickle=False,
    ) as capture:
        np.testing.assert_array_equal(
            capture["raw_translated_recon"],
            np.arange(4, dtype=np.float32).astype(np.complex64).reshape(2, 2),
        )
        np.testing.assert_array_equal(
            capture["raw_translated_wavg"],
            np.arange(8, dtype=np.float32).astype(np.complex64).reshape(2, 4),
        )
        np.testing.assert_array_equal(
            capture["wavg_window_indices"],
            np.asarray([0, 1, 2, 3], dtype=np.int32),
        )
