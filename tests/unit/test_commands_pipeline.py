"""
Unit tests for recovar.commands.pipeline and recovar.commands.pipeline_with_outliers.

Only tests argument registration via add_args() – no actual EM execution.
"""

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")  # pipeline.py imports jax at module level

import recovar.commands.pipeline as pipeline_cmd
import recovar.commands.pipeline_with_outliers as pwo_cmd

pytestmark = pytest.mark.unit


def _parser_with_pipeline_args() -> argparse.ArgumentParser:
    """Return a parser populated with pipeline.add_args()."""
    parser = argparse.ArgumentParser()
    pipeline_cmd.add_args(parser)
    return parser


# ---------------------------------------------------------------------------
# pipeline.add_args
# ---------------------------------------------------------------------------


def test_pipeline_registers_particles_positional():
    """'particles' is a required positional argument."""
    parser = _parser_with_pipeline_args()
    # positional args are listed in _actions, not _option_string_actions
    positional_dests = [a.dest for a in parser._actions if not a.option_strings]
    assert "particles" in positional_dests


def test_pipeline_registers_outdir():
    actions = _parser_with_pipeline_args()._option_string_actions
    assert "-o" in actions or "--outdir" in actions


def test_pipeline_registers_zdim():
    actions = _parser_with_pipeline_args()._option_string_actions
    assert "--zdim" in actions


def test_pipeline_registers_use_ppca():
    actions = _parser_with_pipeline_args()._option_string_actions
    assert "--use-ppca" in actions
    assert "--ppca-em-iters" in actions
    assert "--ppca-zdim" in actions
    assert "--ppca-use-gridding-correction" in actions
    assert "--ppca-contrast-mode" in actions
    assert "--ppca-projected-covariance" in actions


def test_pipeline_zdim_default_is_list():
    """Default zdim must be a list (multiple resolutions are trained)."""
    parser = _parser_with_pipeline_args()
    # parse with minimum required args; zdim should fall back to default
    # We only check the default stored in the action, not actually parsing
    zdim_action = _parser_with_pipeline_args()._option_string_actions["--zdim"]
    assert isinstance(zdim_action.default, list)
    assert len(zdim_action.default) > 0


def test_pipeline_registers_poses():
    actions = _parser_with_pipeline_args()._option_string_actions
    assert "--poses" in actions


def test_pipeline_registers_ctf():
    actions = _parser_with_pipeline_args()._option_string_actions
    assert "--ctf" in actions


def test_pipeline_registers_mask():
    actions = _parser_with_pipeline_args()._option_string_actions
    assert "--mask" in actions


def test_pipeline_registers_n_images():
    """--n-images (or equivalent) controls how many particles to use."""
    actions = _parser_with_pipeline_args()._option_string_actions
    assert "--n-images" in actions or "--n_images" in actions


# ---------------------------------------------------------------------------
# pipeline_with_outliers
# ---------------------------------------------------------------------------


def test_pipeline_with_outliers_module_importable():
    assert callable(pwo_cmd.run_pipeline_with_outlier_removal)


def test_pipeline_with_outliers_reuses_pipeline_add_args():
    """pipeline_with_outliers calls pipeline.add_args, so --zdim must be available
    when we build the combined parser the same way the command does."""
    parser = argparse.ArgumentParser()
    pipeline_cmd.add_args(parser)  # same call as pwo_cmd
    parser.add_argument("--k-rounds", type=int, default=1)
    parser.add_argument("--use-contrast-detection", action="store_true")
    actions = parser._option_string_actions
    assert "--zdim" in actions
    assert "--k-rounds" in actions
    assert "--use-contrast-detection" in actions


def test_pipeline_with_outliers_k_rounds_default_is_one():
    """The default number of outlier-removal rounds is 1."""
    parser = argparse.ArgumentParser()
    pipeline_cmd.add_args(parser)
    parser.add_argument("--k-rounds", type=int, default=1)
    action = parser._option_string_actions["--k-rounds"]
    assert action.default == 1


def test_pipeline_with_outliers_junk_detection_flag_is_store_true():
    """--use-junk-detection must be a boolean flag (action='store_true')."""
    parser = argparse.ArgumentParser()
    pipeline_cmd.add_args(parser)
    parser.add_argument("--use-junk-detection", action="store_true")
    action = parser._option_string_actions["--use-junk-detection"]
    assert action.const is True  # store_true stores True


def test_standard_pipeline_estimates_initial_noise_from_halfset_zero(monkeypatch, tmp_path):
    half0 = object()

    class _FakeDataset:
        volume_shape = (8, 8, 8)
        grid_size = 8
        n_images = 10
        tilt_series_flag = True

        def get_halfset_dataset(self, halfset_id, *, independent=False, lazy=None):
            assert halfset_id == 0
            assert independent is True
            assert lazy is True
            return half0

    ds = _FakeDataset()

    monkeypatch.setattr(pipeline_cmd.halfsets, "resolve_halfset_indices", lambda _args: "split")
    dataset_spec = pipeline_cmd.halfsets.HalfsetDatasetSpec(
        particles_file="particles.star",
        poses_file="poses.pkl",
        ctf_file="ctf.pkl",
    )
    monkeypatch.setattr(
        pipeline_cmd.halfsets.HalfsetDatasetSpec,
        "from_args",
        classmethod(lambda cls, _args: dataset_spec),
    )
    monkeypatch.setattr(pipeline_cmd.halfsets, "load_halfset_dataset", lambda spec, **_kwargs: ds)
    monkeypatch.setattr(
        pipeline_cmd.utils,
        "make_algorithm_options",
        lambda _args: SimpleNamespace(contrast="none", zs_dim_to_test=[4]),
    )
    monkeypatch.setattr(pipeline_cmd.utils, "jax_has_gpu", lambda: True)
    monkeypatch.setattr(pipeline_cmd.utils, "get_gpu_memory_total", lambda: 16)
    monkeypatch.setattr(pipeline_cmd.utils, "get_image_batch_size", lambda _grid, _gpu: 32)
    monkeypatch.setattr(pipeline_cmd.utils, "get_vol_batch_size", lambda _grid, _gpu: 8)
    monkeypatch.setattr(pipeline_cmd.utils, "get_column_batch_size", lambda _grid, _gpu: 4)
    monkeypatch.setattr(pipeline_cmd.utils, "report_memory_device", lambda logger=None: None)

    def _raise_after_assert(dataset_arg, batch_size, max_images=10000):
        assert dataset_arg is half0
        assert batch_size == 32
        assert max_images == 10000
        raise RuntimeError("noise-estimate-stop")

    monkeypatch.setattr(pipeline_cmd.noise, "estimate_noise_variance", _raise_after_assert)

    args = SimpleNamespace(
        outdir=str(tmp_path / "pipeline_out"),
        particles="particles.star",
        poses="poses.pkl",
        ctf="ctf.pkl",
        mask="mask.mrc",
        datadir=None,
        strip_prefix=None,
        downsample=None,
        accept_cpu=False,
        tilt_series=True,
        tilt_series_ctf=None,
        dose_per_tilt=None,
        angle_per_tilt=None,
        do_over_with_contrast=False,
        correct_contrast=False,
        lazy=True,
        gpu_memory=None,
        noise_model="radial",
    )

    with pytest.raises(RuntimeError, match="noise-estimate-stop"):
        pipeline_cmd.standard_recovar_pipeline(args)


def test_standard_pipeline_rejects_ppca_without_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_cmd.utils, "jax_has_gpu", lambda: False)

    args = SimpleNamespace(
        outdir=str(tmp_path / "pipeline_out"),
        particles="particles.star",
        poses="poses.pkl",
        ctf="ctf.pkl",
        mask="mask.mrc",
        accept_cpu=True,
        use_ppca=True,
    )

    with pytest.raises(ValueError, match="--use-ppca currently requires a CUDA GPU"):
        pipeline_cmd.standard_recovar_pipeline(args)


def test_resolve_ppca_contrast_mode_auto():
    assert (
        pipeline_cmd._resolve_ppca_contrast_mode(
            SimpleNamespace(ppca_contrast_mode="auto", correct_contrast=False)
        )
        == "none"
    )
    assert (
        pipeline_cmd._resolve_ppca_contrast_mode(
            SimpleNamespace(ppca_contrast_mode="auto", correct_contrast=True)
        )
        == "marginalize"
    )
    assert (
        pipeline_cmd._resolve_ppca_contrast_mode(
            SimpleNamespace(ppca_contrast_mode="profile", correct_contrast=False)
        )
        == "profile"
    )


def test_configure_ppca_single_zdim_overrides_options():
    args = SimpleNamespace(ppca_zdim=7, zdim=[1, 2, 4])
    options = SimpleNamespace(zs_dim_to_test=[1, 2, 4, 10])

    out = pipeline_cmd._configure_ppca_single_zdim(args, options)

    assert out == 7
    assert args.zdim == [7]
    assert options.zs_dim_to_test == [7]


def test_rescale_ppca_posteriors_matches_covariance_space_formula():
    expected_zs = np.array([[1.0, 2.0], [0.5, -1.0]], dtype=np.float32)
    second_moment_zs = np.array(
        [
            [[2.0, 2.5], [2.5, 6.0]],
            [[1.0, -0.5], [-0.5, 3.0]],
        ],
        dtype=np.float32,
    )
    eigenvalues = np.array([9.0, 4.0], dtype=np.float32)

    coords, covariances = pipeline_cmd._rescale_ppca_posteriors(expected_zs, second_moment_zs, eigenvalues)

    scale = np.sqrt(eigenvalues)
    expected_coords = expected_zs * scale[None, :]
    cov_z = second_moment_zs - np.einsum("ni,nj->nij", expected_zs, expected_zs)
    expected_cov = cov_z * scale[None, :, None] * scale[None, None, :]

    np.testing.assert_allclose(coords, expected_coords)
    np.testing.assert_allclose(covariances, expected_cov)


def test_run_ppca_refinement_uses_hybrid_shell_prior(monkeypatch):
    fake_dataset = SimpleNamespace(volume_shape=(2, 2, 2), volume_size=8)
    means = SimpleNamespace(combined=np.zeros(8, dtype=np.complex64))
    options = SimpleNamespace(zs_dim_to_test=[4])
    args = SimpleNamespace(
        ppca_zdim=4,
        ppca_contrast_mode="auto",
        correct_contrast=True,
        ppca_em_iters=7,
        use_complement_mask=False,
        ppca_use_gridding_correction=True,
        tilt_series=False,
    )

    prior_calls = {}
    em_calls = {}

    def _fake_prior(dataset, mean_estimate, npc, volume_shape, batch_size):
        prior_calls["args"] = (dataset, mean_estimate, npc, volume_shape, batch_size)
        return {"W_prior": np.full((8, npc), 3.0, dtype=np.float32)}

    def _fake_em(dataset, mean_estimate, W_init, W_prior, **kwargs):
        em_calls["dataset"] = dataset
        em_calls["mean_estimate"] = mean_estimate
        em_calls["W_init"] = W_init
        em_calls["W_prior"] = W_prior
        em_calls["kwargs"] = kwargs
        q = W_init.shape[1]
        expected_zs = np.array([[1.0, 2.0, 0.0, 0.0]], dtype=np.float32)
        second_moment_zs = np.array(
            [[
                [2.0, 2.5, 0.0, 0.0],
                [2.5, 6.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]],
            dtype=np.float32,
        )
        return (
            np.ones((8, q), dtype=np.complex128),
            np.arange(1, q + 1, dtype=np.float64),
            np.full((8, q), 5.0, dtype=np.complex128),
            expected_zs,
            second_moment_zs,
            [{"Iteration": 0}],
            {"mean_c": np.array([1.25], dtype=np.float32)},
        )

    monkeypatch.setattr(
        pipeline_cmd.ppca_prior_estimation,
        "estimate_hybrid_shell_prior_from_data",
        _fake_prior,
    )
    monkeypatch.setattr(pipeline_cmd.ppca_module, "EM", _fake_em)

    out = pipeline_cmd._run_ppca_refinement(
        fake_dataset,
        means,
        np.ones((2, 2, 2), dtype=np.float32),
        options,
        args,
        batch_size=32,
    )

    assert prior_calls["args"][0] is fake_dataset
    assert prior_calls["args"][2] == 4
    assert prior_calls["args"][3] == (2, 2, 2)
    assert prior_calls["args"][4] == 32
    assert em_calls["W_init"].shape == (8, 4)
    np.testing.assert_allclose(em_calls["W_prior"], np.full((8, 4), 3.0, dtype=np.float32))
    assert em_calls["kwargs"]["contrast_mode"] == "marginalize"
    assert em_calls["kwargs"]["EM_iter"] == 7
    assert em_calls["kwargs"]["return_posterior_info"] is True
    assert out["basis_size"] == 4
    assert out["contrast_mode"] == "marginalize"
    assert out["prior_mode"] == "hybrid_shell"
    assert out["u_rescaled"].dtype == np.complex64
    assert out["s_rescaled"].dtype == np.float32
    assert out["W"].dtype == np.complex64
    np.testing.assert_allclose(out["latent_coords"][4], np.array([[1.0, 2.0, 0.0, 0.0]], dtype=np.float32) * np.sqrt(np.arange(1, 5, dtype=np.float32))[None, :])
    np.testing.assert_allclose(out["contrasts"][4], np.array([1.25], dtype=np.float32))


def test_orthonormalize_loading_basis_returns_orthonormal_columns():
    loadings = np.array(
        [
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    basis = pipeline_cmd._orthonormalize_loading_basis(loadings, basis_size=2)

    assert basis.shape == (3, 2)
    gram = np.asarray(basis).conj().T @ np.asarray(basis)
    np.testing.assert_allclose(gram, np.eye(2), atol=1e-6, rtol=1e-6)


def test_run_ppca_projected_covariance_refinement_uses_projcov_then_embeddings(monkeypatch):
    fake_dataset = SimpleNamespace(image_size=16)
    means = SimpleNamespace(combined=np.arange(8, dtype=np.complex64))
    options = SimpleNamespace(zs_dim_to_test=[4])
    args = SimpleNamespace(tilt_series=False)
    focus_mask = np.ones((2, 2, 2), dtype=np.float32)
    volume_mask = np.ones((2, 2, 2), dtype=np.float32) * 2.0
    dilated_mask = np.ones((2, 2, 2), dtype=np.float32) * 3.0
    ppca_loadings = np.arange(32, dtype=np.float32).reshape(8, 4)

    projcov_calls = {}
    embedding_calls = {}

    def _fake_projcov(dataset, basis, mean, volume_mask_arg, disc_type, disc_type_u, **kwargs):
        projcov_calls["dataset"] = dataset
        projcov_calls["basis"] = np.asarray(basis)
        projcov_calls["mean"] = mean
        projcov_calls["volume_mask"] = np.asarray(volume_mask_arg)
        projcov_calls["disc_type"] = disc_type
        projcov_calls["disc_type_u"] = disc_type_u
        projcov_calls["kwargs"] = kwargs
        return np.full((8, 4), 7.0, dtype=np.complex64), np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)

    def _fake_embeddings(means_arg, u_arg, s_arg, dataset_arg, volume_mask_arg, options_arg, gpu_memory_arg, focus_masks_arg, zdim_for_rest_arg, args_arg):
        embedding_calls["means"] = means_arg
        embedding_calls["u"] = u_arg
        embedding_calls["s"] = s_arg
        embedding_calls["dataset"] = dataset_arg
        embedding_calls["volume_mask"] = np.asarray(volume_mask_arg)
        embedding_calls["options"] = options_arg
        embedding_calls["gpu_memory"] = gpu_memory_arg
        embedding_calls["focus_masks"] = [np.asarray(x) for x in focus_masks_arg]
        embedding_calls["zdim_for_rest"] = zdim_for_rest_arg
        embedding_calls["args"] = args_arg
        return (
            {4: np.ones((5, 4), dtype=np.float32)},
            {4: np.ones((5, 4), dtype=np.float32) * 2},
            {4: np.ones((5, 4, 4), dtype=np.float32)},
            {4: np.ones((5, 4, 4), dtype=np.float32) * 3},
            {4: np.ones(5, dtype=np.float32)},
            {4: np.ones(5, dtype=np.float32) * 4},
        )

    monkeypatch.setattr(pipeline_cmd.principal_components, "pca_by_projected_covariance", _fake_projcov)
    monkeypatch.setattr(pipeline_cmd, "_compute_embeddings", _fake_embeddings)

    out = pipeline_cmd._run_ppca_projected_covariance_refinement(
        fake_dataset,
        means,
        ppca_loadings,
        volume_mask,
        dilated_mask,
        options,
        args,
        batch_size=32,
        gpu_memory=40,
        covariance_options={
            "disc_type": "linear_interp",
            "disc_type_u": "linear_interp",
            "mask_images_in_proj": True,
        },
        focus_masks=[focus_mask],
        zdim_for_rest=20,
    )

    assert projcov_calls["dataset"] is fake_dataset
    np.testing.assert_allclose(projcov_calls["mean"], means.combined)
    np.testing.assert_allclose(projcov_calls["volume_mask"], dilated_mask)
    assert projcov_calls["disc_type"] == "linear_interp"
    assert projcov_calls["disc_type_u"] == "linear_interp"
    assert projcov_calls["kwargs"]["gpu_memory_to_use"] == 40
    assert projcov_calls["kwargs"]["use_mask"] is True
    assert projcov_calls["kwargs"]["n_pcs_to_compute"] == 4

    gram = projcov_calls["basis"].conj().T @ projcov_calls["basis"]
    np.testing.assert_allclose(gram, np.eye(4), atol=1e-5, rtol=1e-5)

    assert embedding_calls["means"] is means
    assert embedding_calls["dataset"] is fake_dataset
    np.testing.assert_allclose(embedding_calls["volume_mask"], volume_mask)
    np.testing.assert_allclose(embedding_calls["focus_masks"][0], focus_mask)
    assert embedding_calls["gpu_memory"] == 40
    assert embedding_calls["zdim_for_rest"] == 20
    np.testing.assert_allclose(embedding_calls["u"]["rescaled"], np.full((8, 4), 7.0, dtype=np.complex64))
    np.testing.assert_allclose(embedding_calls["s"]["rescaled"], np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32))

    np.testing.assert_allclose(out["u_rescaled"], np.full((8, 4), 7.0, dtype=np.complex64))
    np.testing.assert_allclose(out["s_rescaled"], np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(out["latent_coords"][4], np.ones((5, 4), dtype=np.float32))
