"""Exercise the global scheduler's actual Fourier-window block without images."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from recovar.em.helpers.convergence import healpix_angular_step
from recovar.em.helpers.resolution import clamp_relion_coarse_image_size, compute_coarse_image_size

pytestmark = pytest.mark.unit


def _global_window(incoming, updated, *, current=172, sealed=None):
    """Execute the production block, not a duplicate of its sizing formula.

    The surrounding full refinement needs images and GPU scoring. Select its
    unique adaptive global-window block structurally, so this CPU regression
    still checks the real caller's choice of pre/post-update order.
    """
    path = Path(__file__).resolve().parents[2] / "recovar/em/refinement/iteration_loop.py"
    tree = ast.parse(path.read_text())
    blocks = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name) and node.test.id == "use_adaptive"
        and any(
            isinstance(child, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "effective_step_deg" for t in child.targets)
            for child in node.body
        )
    ]
    assert len(blocks) == 1, "Global Fourier-window block must be uniquely identified"
    namespace = dict(
        coarse_size_healpix_order=incoming,
        current_healpix_order=updated,
        healpix_angular_step=healpix_angular_step,
        compute_coarse_image_size=compute_coarse_image_size,
        clamp_relion_coarse_image_size=clamp_relion_coarse_image_size,
        cryo=SimpleNamespace(voxel_size=1.400011),
        optics_pixel_sizes=[1.400011],
        optics_image_sizes=[380],
        grid_size=380,
        particle_diameter_ang=250.0,
        image_current_size=current,
        cs_for_engine=current,
        cs=current,
        current_size=current,
        sealed_sampling_state=sealed,
        state=SimpleNamespace(adaptive_oversampling=1),
        logger=SimpleNamespace(info=lambda *args: None),
    )
    code = compile(ast.Module(body=blocks[0].body, type_ignores=[]), str(path), "exec")
    exec(code, namespace)
    return namespace


@pytest.mark.parametrize(
    "incoming,updated,expected",
    [(2, 3, 40), (3, 3, 80), (3, 4, 80), (4, 4, 158), (3, 2, 80)],
)
def test_global_pass1_window_uses_incoming_order(incoming, updated, expected):
    # Native10073 I7 order2 -> I8 order3: size40, not80. RELION's MPI
    # expectation sizes Fourier windows before calculateExpectedAngularErrors
    # and updateAngularSampling. The updated order still owns candidate grids.
    values = _global_window(incoming, updated)
    assert values["coarse_cs"] == expected
    assert values["current_healpix_order"] == updated
    assert values["state"].adaptive_oversampling == 1


def test_global_pass1_window_keeps_current_size_clamp():
    assert _global_window(3, 4, current=60)["coarse_cs"] == 60


def test_global_pass1_window_keeps_explicit_sealed_override():
    assert _global_window(2, 3, sealed={"coarse_size": 36})["coarse_cs"] == 36
    with pytest.raises(ValueError, match="exceeds active current_size"):
        _global_window(2, 3, sealed={"coarse_size": 174})
