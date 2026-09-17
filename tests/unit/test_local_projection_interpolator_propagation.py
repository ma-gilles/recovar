"""The interpolator choice must reach the projector unchanged on both local routes.

``relion_texture_interp`` is tri-state. None means "resolve as strict parity does",
which is RELION's CUDA texture interpolator when the custom projector is available;
False forces the manual diagnostic fallback; True demands texture. The value travels
from the local engine to the projector helper along two different routes, the eager
bucket path and the big-JIT path, and it crosses a JIT static argument on the way.

A ``bool()`` on that journey silently turns None into False and moves the production
big-JIT path onto the diagnostic interpolator. These tests read the value the
projector helper actually receives, for every state and on both routes.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.unit

STATES = [None, False, True]


def _engine_signature_default():
    import inspect
    from recovar.em.local.local_em_engine import run_local_em_exact
    return inspect.signature(run_local_em_exact).parameters["projection_relion_texture_interp"].default


def test_the_engine_default_is_the_unset_tri_state():
    """Defaulting to False would force the fallback on every caller that omits it."""
    assert _engine_signature_default() is None


@pytest.mark.parametrize("state", STATES)
def test_the_eager_route_forwards_every_state_unchanged(state, monkeypatch):
    """The eager bucket path builds its projector call from projection_kwargs."""
    from recovar.em.local import local_bucket_stages

    seen = {}

    def _capture(*args, **kwargs):
        seen["relion_texture_interp"] = kwargs.get("relion_texture_interp", "absent")
        return np.zeros((1, 1), dtype=np.complex64), None

    monkeypatch.setattr(local_bucket_stages, "_compute_relion_projector_projections_block", _capture)
    monkeypatch.setattr(local_bucket_stages, "prepare_local_projector_slab", lambda slab, **kw: slab)

    class _Window:
        use_window = False
        max_r = None

    local_bucket_stages._relion_local_projector_flat(
        np.zeros((2, 2, 2), dtype=np.complex64),
        np.zeros((1, 3, 3), dtype=np.float32),
        image_shape=(4, 4),
        relion_projector_r_max=1,
        projection_padding_factor=1,
        projection_kwargs={"relion_texture_interp": state},
        window_spec=_Window(),
        projection_indices=None,
    )
    assert seen["relion_texture_interp"] is state


@pytest.mark.parametrize("state", STATES)
def test_the_big_jit_route_forwards_every_state_unchanged(state, monkeypatch):
    """The big-JIT path once dropped this argument and took the helper's default."""
    from recovar.em.local import local_big_jit

    seen = {}

    def _capture(*args, **kwargs):
        seen["relion_texture_interp"] = kwargs.get("relion_texture_interp", "absent")
        return np.zeros((1, 1), dtype=np.complex64), None

    monkeypatch.setattr(local_big_jit, "compute_relion_projector_projections_block", _capture)

    local_big_jit._project_local_half_spectrum(
        np.zeros((8,), dtype=np.complex64),
        np.zeros((2, 2, 2), dtype=np.complex64),
        np.zeros((1, 3, 3), dtype=np.float32),
        None,
        (4, 4),
        (4, 4, 4),
        "linear_interp",
        projection_half_volume=False,
        projection_max_r="auto",
        relion_projector_output_size=0,
        projection_relion_texture_interp=state,
        projection_force_jax=False,
        use_relion_projector=True,
        relion_projector_r_max=1,
        projection_padding_factor=1,
    )
    assert seen["relion_texture_interp"] is state


@pytest.mark.parametrize("state", STATES)
def test_no_propagation_site_collapses_the_tri_state(state):
    """Read the value the engine would hand to the big-JIT entry point.

    The engine reaches that entry point through a JIT static argument, so this
    inspects the argument expression rather than running the kernel: any bool() or
    truthiness coercion on the way turns None into False without failing anything
    else.
    """
    import ast
    import inspect
    from recovar.em.local import local_em_engine

    tree = ast.parse(inspect.getsource(local_em_engine))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg != "projection_relion_texture_interp":
            continue
        if not isinstance(node.value, ast.Name):
            offenders.append(ast.dump(node.value))
    assert not offenders, (
        "projection_relion_texture_interp is passed through an expression rather than "
        f"forwarded as-is at {offenders}; a bool() there collapses None to False"
    )


def test_the_big_jit_entry_point_accepts_the_unset_state():
    """The static argument must be typed for the tri-state it carries."""
    import inspect
    from recovar.em.local.local_big_jit import run_local_bucket_big_jit

    annotation = inspect.signature(run_local_bucket_big_jit).parameters[
        "projection_relion_texture_interp"
    ].annotation
    assert "None" in str(annotation), (
        f"annotated {annotation!r}; the unset state travels through this argument"
    )
    assert "projection_relion_texture_interp" in run_local_bucket_big_jit._jit_info.static_argnames
