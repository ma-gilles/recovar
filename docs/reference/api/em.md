# recovar.em

Expectation-Maximization algorithms for pose refinement and
heterogeneous reconstruction.

## states

EM state containers: EMState, SGDState, HeterogeneousEMState.

::: recovar.em.states
    options:
      members_order: source

## iterations

High-level EM loop orchestration and convergence tracking.

::: recovar.em.iterations
    options:
      members_order: source

## core

Core EM iteration logic: cross-correlation, residual computation.

::: recovar.em.core
    options:
      members_order: source

## e_step

E-step: posterior probability computation over poses and translations.

::: recovar.em.e_step
    options:
      members_order: source

## m_step

M-step: volume update via weighted backprojection.

::: recovar.em.m_step
    options:
      members_order: source
