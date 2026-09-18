"""Label scopes preserve dump names, nesting and environment restoration."""

import os
from contextlib import nullcontext

import pytest

from recovar.em.diagnostics import local_debug

pytestmark = pytest.mark.unit
DENSE = "RECOVAR_DEBUG_PER_POSE_DUMP_LABEL"
SCORE = "RECOVAR_LOCAL_SCORE_DUMP_LABEL"
FUSED = "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_LABEL"
NAMES = (DENSE, SCORE, FUSED)


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("initial,expected", [(None, "phase"), ("", "phase"), ("arm", "arm_phase")])
@pytest.mark.parametrize("error_type", [None, RuntimeError, KeyboardInterrupt])
def test_scope_restores_absent_empty_and_existing_labels(monkeypatch, local, initial, expected, error_type):
    for name in NAMES:
        if initial is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, initial)
    original = {name: os.environ.get(name) for name in NAMES}
    changed = (SCORE, FUSED) if local else (DENSE,)
    check_error = pytest.raises(error_type, match="expected") if error_type else nullcontext()
    with check_error:
        with local_debug.score_dump_label("phase", local=local) as value:
            assert value is None
            assert {name: os.environ.get(name) for name in changed} == dict.fromkeys(changed, expected)
            for name in set(NAMES) - set(changed):
                assert os.environ.get(name) == original[name]
            if error_type:
                raise error_type("expected")
    assert {name: os.environ.get(name) for name in NAMES} == original


def test_nested_class_and_local_phase_keep_independent_prefixes(monkeypatch):
    monkeypatch.setenv(DENSE, "dense_arm")
    monkeypatch.setenv(SCORE, "score arm")
    monkeypatch.setenv(FUSED, "posterior arm")
    with local_debug.score_dump_label("fine"):
        with local_debug.score_dump_label("class003"):
            with local_debug.score_dump_label("mstep_class003", local=True):
                assert local_debug.dense_score_dump_label_suffix() == "_dense_arm_fine_class003"
                assert local_debug._local_debug_dump_label_suffix() == "_score_arm_mstep_class003"
                assert local_debug._local_fused_posterior_dump_label_suffix() == "_posterior_arm_mstep_class003"
            assert os.environ[SCORE] == "score arm"
            assert os.environ[FUSED] == "posterior arm"
        assert os.environ[DENSE] == "dense_arm_fine"
    assert os.environ[DENSE] == "dense_arm"


@pytest.mark.parametrize(
    "label,expected",
    [
        (None, ""),
        ("", ""),
        ("   ", ""),
        ("arm one", "_arm_one"),
        ("  arm.1-x/y  ", "_arm.1-x_y"),
        ("a\nb", "_a_b"),
    ],
)
def test_all_readers_share_unchanged_sanitization(monkeypatch, label, expected):
    for name in NAMES:
        if label is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, label)
    assert local_debug.dense_score_dump_label_suffix() == expected
    assert local_debug._local_debug_dump_label_suffix() == expected
    assert local_debug._local_fused_posterior_dump_label_suffix() == expected
    assert local_debug._pass_label_suffix(label) == expected


@pytest.mark.parametrize("score,fused,expected", [(None, "arm", "_arm"), ("arm", None, "_arm"), ("", "arm", "_arm")])
def test_local_readers_keep_fallback_precedence(monkeypatch, score, fused, expected):
    for name, value in [(SCORE, score), (FUSED, fused)]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert local_debug._local_debug_dump_label_suffix() == expected
    assert local_debug._local_fused_posterior_dump_label_suffix() == expected
