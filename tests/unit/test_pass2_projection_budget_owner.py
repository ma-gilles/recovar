"""Both bucketed pass-2 entry points size their projection budget through one owner."""

import inspect

from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as sp


def test_both_entry_points_use_the_owner_with_their_own_abs2_rule():
    single = inspect.getsource(sp.compute_pass2_stats_sparse_bucketed)
    fused = inspect.getsource(sp.compute_k_class_pass2_stats_sparse_fused)
    for src in (single, fused):
        assert src.count("= _pass2_projection_budget(") == 1
        assert "_projection_cache_budget_complex_dtype(" not in src and "_max_projected_rotations_per_call_for_pass(" not in src
    assert "include_abs2=not (budget_window_spec.use_window or score_only)," in single
    assert "include_abs2=not budget_window_spec.use_window," in fused


def test_owner_threads_the_budget(monkeypatch):
    seen = []
    monkeypatch.setattr(sp, "_projection_cache_budget_complex_dtype", lambda d, c, *, use_relion_projector: seen.append(("dtype", d, c, use_relion_projector)) or "CDT")
    monkeypatch.setattr(sp, "_projection_budget_pixels_for_pass", lambda n, *, use_window, use_relion_projector: seen.append(("pixels", n, use_window)) or 77)
    monkeypatch.setattr(sp, "_max_projected_rotations_per_call_for_pass", lambda **kw: seen.append(("rot", kw)) or 5)

    class Policy:
        score_complex_dtype = "c64"

    class Window:
        use_window = True

    out = sp._pass2_projection_budget("f32", Policy(), n_half=9, use_relion_projector=True, budget_window_spec=Window(), device_memory_bytes=123, include_abs2=False)
    assert out == ("CDT", 77, 5)
    assert seen[0] == ("dtype", "f32", "c64", True) and seen[1] == ("pixels", 9, True)
    assert seen[2][1] == {"device_memory_bytes": 123, "n_projection_pixels": 77, "projection_complex_dtype": "CDT", "include_abs2": False}
