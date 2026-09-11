"""The deferred exact-noise core and kernel calls share one keyword binding."""

import inspect

from recovar.em.dense_single_volume import local_em_engine


def test_deferred_noise_calls_share_one_keyword_binding():
    src = inspect.getsource(local_em_engine.run_local_em_exact)
    assert src.count("deferred_noise_shared_kwargs = dict(") == 1
    assert src.count("**deferred_noise_shared_kwargs,") == 2
    bind = src.index("deferred_noise_shared_kwargs = dict(")
    assert bind < src.index("if noise_stable_core_enabled:", bind) < src.index("run_deferred_local_exact_noise_core_jit(", bind) < src.index(") = run_deferred_local_exact_noise_jit(", bind)
    end = src.index("noise_scale_xa = packed_noise_scale_xa", bind)
    segment = src[bind:end]
    for key in ("image_shape=image_shape", "shell_count=n_shells", "stable_fourier_window_shapes=stable_window_active", "source_faithful_spectrum_norm=source_faithful_spectrum_norm", "unweighted_high_shell_image_power=unweighted_high_shell_image_power"):
        assert segment.count(key) == 1, key
