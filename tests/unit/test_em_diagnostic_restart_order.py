"""Explicit one-iteration restart ordering, separate from fresh-start state."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_full_refinement as driver

pytestmark = pytest.mark.unit


def args(**overrides):
    values = dict(diagnostic_restart_particle_order=True, n_classes=1,
                  init_relion_iteration=10, max_iter=1, skip_final_iteration=True,
                  perturb_replay_relion_dir='/fixture', relion_half_sets='/fixture/data.star',
                  relion_particle_shuffle='mt19937', state_swap_variant=None,
                  state_swap_target_relion_iteration=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_restart_uses_next_physical_iteration_without_fresh_initialization():
    value = args()
    assert driver._diagnostic_restart_particle_order_iteration(value, None) == 11
    assert not driver._use_fresh_auto_refine_particle_order(value, None)


def test_default_is_inactive_even_for_old_minimal_namespaces():
    assert driver._diagnostic_restart_particle_order_iteration(SimpleNamespace(), None) is None
    assert driver._diagnostic_restart_particle_order_iteration(args(diagnostic_restart_particle_order=False), None) is None


@pytest.mark.parametrize('override', [
    dict(n_classes=4), dict(init_relion_iteration=0), dict(max_iter=2),
    dict(skip_final_iteration=False), dict(perturb_replay_relion_dir=None),
    dict(relion_half_sets=None), dict(relion_particle_shuffle='legacy'),
    dict(state_swap_variant='all_relion'), dict(state_swap_target_relion_iteration=11),
])
def test_restart_rejects_incompatible_scopes(override):
    with pytest.raises(ValueError, match='diagnostic restart particle order'):
        driver._diagnostic_restart_particle_order_iteration(args(**override), None)


def test_frozen_uninterrupted_boundary_not_a_process_restart():
    with pytest.raises(ValueError, match='diagnostic restart particle order'):
        driver._diagnostic_restart_particle_order_iteration(args(), object())


def test_cli_wires_order_iteration_trial_and_accumulation_without_fresh_noise():
    tree = ast.parse(Path(driver.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
    layout = next(n for n in calls if isinstance(n.func, ast.Name)
                  and n.func.id == '_relion_halfset_and_accuracy_layout')
    keywords = {k.arg: ast.unparse(k.value) for k in layout.keywords}
    assert keywords['random_seed'] == 'args.seed if apply_particle_order else None'
    assert keywords['first_iteration'] == 'restart_order_iteration or 1'
    parity = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'RelionParityOptions')
    assert next(ast.unparse(k.value) for k in parity.keywords
                if k.arg == 'preserve_bpref_particle_order') == 'apply_particle_order'
    fresh_noise = next(n for n in ast.walk(main) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'live_initial_noise_layout_candidate'
                               for t in n.targets))
    assert ast.unparse(fresh_noise.value).startswith('bool(use_fresh_auto_refine_order and ')
