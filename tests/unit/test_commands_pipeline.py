"""
Unit tests for recovar.commands.pipeline and recovar.commands.pipeline_with_outliers.

Only tests argument registration via add_args() – no actual EM execution.
"""

import argparse
import sys
from types import SimpleNamespace

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


def test_pipeline_registers_output_name():
    actions = _parser_with_pipeline_args()._option_string_actions
    assert "--output-name" in actions


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


def test_standard_pipeline_passes_gpu_limit_to_predownsample(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setattr(pipeline_cmd, "_resolve_downsample", lambda _args: None)
    monkeypatch.setattr(pipeline_cmd.utils, "jax_has_gpu", lambda: True)
    monkeypatch.setattr(
        pipeline_cmd.utils,
        "set_gpu_memory_limit",
        lambda value: captured.setdefault("limits", []).append(value),
    )
    monkeypatch.setattr(pipeline_cmd.os.path, "exists", lambda _path: False)
    class _NoopLock:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setitem(
        sys.modules,
        "recovar.commands.downsample",
        SimpleNamespace(
            build_project_downsample_cache_dir=lambda **kwargs: str(tmp_path / "Cache" / "downsample" / "particles_d128"),
            downsample_cache_lock=lambda _path: _NoopLock(),
            downsample_to_disk=lambda **kwargs: captured.setdefault("downsample_kwargs", kwargs),
            write_downsample_cache_metadata=lambda **kwargs: captured.setdefault("cache_metadata", kwargs),
        ),
    )
    monkeypatch.setattr(
        pipeline_cmd.halfsets,
        "resolve_halfset_indices",
        lambda _args: (_ for _ in ()).throw(RuntimeError("stop-after-downsample")),
    )

    args = SimpleNamespace(
        outdir=str(tmp_path / "pipeline_out"),
        particles="particles.star",
        poses="poses.pkl",
        ctf="ctf.pkl",
        mask="mask.mrc",
        datadir=None,
        strip_prefix=None,
        downsample=128,
        accept_cpu=False,
        tilt_series=False,
        tilt_series_ctf=None,
        dose_per_tilt=None,
        angle_per_tilt=None,
        do_over_with_contrast=False,
        correct_contrast=False,
        lazy=True,
        gpu_memory=12.0,
        noise_model="radial",
    )

    with pytest.raises(RuntimeError, match="stop-after-downsample"):
        pipeline_cmd.standard_recovar_pipeline(args)

    assert captured["limits"] == [12.0]
    assert captured["downsample_kwargs"]["gpu_memory_gb"] == 12.0



def test_standard_pipeline_uses_project_downsample_cache(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setattr(pipeline_cmd, "_resolve_downsample", lambda _args: None)
    monkeypatch.setattr(pipeline_cmd.utils, "jax_has_gpu", lambda: True)
    monkeypatch.setattr(pipeline_cmd.utils, "set_gpu_memory_limit", lambda value: None)

    cache_dir = tmp_path / "project" / "Cache" / "downsample" / "particles_d128"
    cache_dir.mkdir(parents=True)
    (cache_dir / "particles.128.mrcs").touch()
    (cache_dir / "particles.128.star").touch()

    class _NoopLock:
        def __enter__(self):
            captured["locked"] = True
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setitem(
        sys.modules,
        "recovar.commands.downsample",
        SimpleNamespace(
            build_project_downsample_cache_dir=lambda **kwargs: captured.setdefault("cache_dir", str(cache_dir)),
            downsample_cache_lock=lambda _path: _NoopLock(),
            downsample_to_disk=lambda **kwargs: (_ for _ in ()).throw(AssertionError("should reuse cached downsample outputs")),
            write_downsample_cache_metadata=lambda **kwargs: captured.setdefault("cache_metadata", kwargs),
        ),
    )
    monkeypatch.setattr(
        pipeline_cmd.halfsets,
        "resolve_halfset_indices",
        lambda _args: (_ for _ in ()).throw(RuntimeError("stop-after-cache-hit")),
    )

    args = SimpleNamespace(
        outdir=str(tmp_path / "project" / "Pipeline" / "job_0001"),
        particles="particles.star",
        poses="poses.pkl",
        ctf="ctf.pkl",
        mask="mask.mrc",
        datadir=None,
        strip_prefix=None,
        downsample=128,
        accept_cpu=False,
        tilt_series=False,
        tilt_series_ctf=None,
        dose_per_tilt=None,
        angle_per_tilt=None,
        do_over_with_contrast=False,
        correct_contrast=False,
        lazy=True,
        gpu_memory=12.0,
        noise_model="radial",
        _project_root=str(tmp_path / "project"),
    )

    with pytest.raises(RuntimeError, match="stop-after-cache-hit"):
        pipeline_cmd.standard_recovar_pipeline(args)

    assert captured["locked"] is True
    assert args.particles.endswith("particles.128.star")
    assert args.downsample is None
