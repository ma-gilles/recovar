"""The three local debug dump-request parsers share one environment-prefix owner."""

import inspect

from recovar.em.diagnostics import local_debug


def test_parsers_delegate_to_the_owner(monkeypatch, tmp_path):
    for fn, prefix in (
        (local_debug.parse_debug_score_dump_request, "RECOVAR_LOCAL_SCORE_DUMP"),
        (local_debug.parse_debug_fused_posterior_dump_request, "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP"),
        (local_debug.parse_debug_noise_component_dump_request, "RECOVAR_LOCAL_NOISE_COMPONENT_DUMP"),
    ):
        assert f'return _parse_dump_request("{prefix}")' in inspect.getsource(fn)
        for key in ("_DIR", "_GLOBAL_INDICES", "_CURRENT_SIZE", "_ITERATION"):
            monkeypatch.delenv(prefix + key, raising=False)
        assert fn() == (None, set(), None, None)
        monkeypatch.setenv(prefix + "_DIR", str(tmp_path / prefix))
        monkeypatch.setenv(prefix + "_GLOBAL_INDICES", "4,9")
        monkeypatch.setenv(prefix + "_CURRENT_SIZE", "64")
        path, targets, sizes, iterations = fn()
        assert path == tmp_path / prefix and path.is_dir() and targets == {4, 9} and sizes == {64} and iterations is None
