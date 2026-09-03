# VDAM all-ten stable-shape same-state gate — H100 job 13394745

## Decision

Retain stable Fourier-window and flat-row capacities as a compile-shape
optimization, but do not promote them to defaults yet. From one shared
in-memory iteration-47 state, all four stable-off/on crossed comparisons keep
every hard scientific decision exact. The continuous accumulator changes are
small (`<=2.10e-7` normalized L2), but exceed the repeat envelope by as much as
`2.88x`; derived state reaches `4.93x` its repeat envelope while remaining
`<=5.82e-8` normalized L2. The strict atomic gate therefore remains a hold.

The result rules out a deterministic stable-shape error at the isolated
iteration-47 to 48 boundary. It does not erase the fresh-trajectory result:
one optimized repeat in job `13392701` still enters a different one-particle
pose basin at iterations 48 and 50. Full-trajectory stability remains the
promotion authority.

## Isolated transition contract

The harness ran through iteration 47 once, deep-copied that live state, and
then executed:

```text
stable off 1 -> stable on 1 -> stable on 2 -> stable off 2
```

All other optimized seams were fixed on: certified coarse hybrid, projection
cache, compact posterior, exact coarse single-translation preprocessing,
shared batched posterior primitives, flat local rows, packed projection,
deferred packed VDAM, and packed final noise. The only crossed changes were:

- stable Fourier-window shapes: off -> on with physical quantum 32;
- stable flat-row capacity: off -> on.

The stable-off arm correctly reports its inactive canonical metadata quantum
of 8; the requested environment remains quantum 32 for both arms. The
stable-on arm maps logical current size 84 to physical size 96 and replaces
strict 4,752-row local reductions with five fixed 10,752-row capacities.

| Hard gate | Result |
|---|---|
| Crossed off/on pairs | **4 / 4 pass** |
| Selected particle IDs | **exact** |
| Pose, rotation, translation, and class decisions | **exact** |
| Significant counts and coarse support hashes | **exact** |
| Complete particle state | **exact** |
| Sampling state | **exact** |
| Requested execution environment | **exact for every arm** |

## Continuous-state envelope

CUDA reconstruction atomics are nondeterministic even within one mode, so the
gate compares every crossed normalized-L2 delta with the larger observed
off/off or on/on repeat delta.

| Boundary | Maximum crossed normalized L2 | Maximum crossed / repeat envelope | Strict result |
|---|---:|---:|---|
| Halfset data/weight accumulators | `2.100e-7` | `2.881x` | hold |
| Derived final state | `5.816e-8` | `4.926x` | hold |

The largest final-state ratio is `Igrad2`; its absolute normalized L2 is only
`4.342e-9`. `Iref` reaches `1.049e-10`, and the largest final-state absolute
normalized L2 is the sigma-class reduction at `5.816e-8`. These are stable,
nondirectional floating-point effects with no hard-decision movement in this
transition, but the policy intentionally does not relabel an envelope miss as
an exact pass.

## Runtime interpretation

Only the second on/off pair is warm for both relevant executable families.
Stable physical padding is slightly slower at a fixed size; its value is
avoiding recompilation as logical sizes change, not reducing steady-state GPU
work.

| Warm iteration-48 metric | Stable off 2 | Stable on 2 | Change |
|---|---:|---:|---:|
| Transition wall | 2.158950 s | 2.217895 s | +2.73% |
| Coarse pass 1 | 0.403143 s | 0.400595 s | -0.63% |
| Fine pass 2 | 0.974924 s | 1.034145 s | +6.07% |
| Local EM | 0.966552 s | 1.025910 s | +6.14% |
| Local big JIT | 0.718374 s | 0.737417 s | +2.65% |
| New local/M-step target executables | 0 | 0 | exact |

The first off and on arms take 16.790 s and 5.408 s respectively, but those
numbers are order-dependent cold compilation and are not a valid incremental
speed comparison. The independent four-fresh-process trajectory is the
performance authority: stable capacities change median wall
`500.604 -> 274.569 s` (-45.15%) and expectation
`471.249 -> 239.133 s` (-49.26%).

The trajectory profiles identify the next high-leverage boundary. Across the
50-iteration stable-on arm, expectation totals 244.118 s. Coarse pass 1 alone
takes 5--8 s at several late logical-size changes because it still compiles
logical square windows; stable Fourier shapes currently cover only the local
pass. The next performance experiment should extend the same host-planned
physical/logical-window design to shared coarse significance while retaining
RELION's logical active-pixel mask.

## Particle 2798 margin diagnostic

The production fused posterior dump from the final warm stable-off arm has a
clear iteration-48 winner from the shared checkpoint:

| Candidate | Rotation ID | Translation (px) | Total score | Posterior |
|---|---:|---:|---:|---:|
| Winner | 34417 | (0.298079, 0.298079) | -8.141907 | 0.733036 |
| Runner-up | 34417 | (0.114652, 0.298079) | -9.856262 | 0.132005 |
| Fresh-trajectory competing translation | 34417 | (0.298079, 0.481507) | -11.313232 | 0.030749 |

The winner/runner score gap is `1.714355`, posterior gap `0.601031`, and
posterior ratio `5.553x`. The competing translation seen after divergent
fresh trajectories ranks fifth here, `3.171326` score units below the winner
with a `23.84x` posterior ratio. Its raw image score is better by `1.111783`,
but the translation prior reverses the ordering. Thus the isolated shared
state is not on an immediate numerical tie; earlier repeat-scale state drift
must accumulate before the fresh trajectory can cross this later basin
boundary.

This run exposed a diagnostic-only filename collision: K-class phase labeling
overrode the requested arm prefix, so each arm rewrote the same dump and only
the final stable-off tensor survived. Commit `0f60a14b7` gives the
fused-posterior-specific arm label precedence and adds a focused regression
test. The hard ABBA result is unaffected because its four arm artifacts and
comparisons were already distinct.

## Performance diagnosis beyond this gate

Joining registered NVTX strings in the comparable Nsight capture `13393862`
shows that the 526.111 ms optimized local range does not issue its first CUDA
call until +275.7 ms. Image fetch/collation ends at about +41.3 ms, leaving
roughly 234 ms in first executable tracing/dispatch. The entire captured
RECOVAR iteration performs only 117.986 ms of GPU kernel work. This confirms
that the remaining focused gap is executable/controller latency and changing
shape families, not a need for more arithmetic inside already-small kernels.

The concrete next sequence is:

1. stabilize the shared coarse-significance square window with a logical
   active mask and a small physical size quantum;
2. gate it with this same-state ABBA contract, including preserved arm-specific
   score dumps;
3. run a fresh `0 -> 50` repeat-controlled trajectory and require exact hard
   state plus the atomic envelope;
4. only then run the longer and expanded dataset suites.

## Provenance

- Slurm: `13394745`, `COMPLETED`, `00:06:02`, `della-h19g2`, exit `0:0`.
- Source: `d542a8a0fbb01f82fa4d73fc9a05a66d7e98c131`.
- Result root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_allten_stable_pose_margin_it47_d542a8a0f_20260903T1813Z`.
- Shared checkpoint construction wall: 269.649 s through iteration 47.
- Acceptance config SHA-256:
  `58b95bee944d7dc25c99b363c717a7148d372ade980ea5a661a701f7979b11e8`.
- Classification: `diagnostic_same_in_memory_state_one_transition_only`;
  `science_promotion_allowed=false` by construction.

Focused verification after the label fix:

```text
tests/unit/initial_model/test_vdam_hybrid_same_state_transition.py
83 passed in 18.72 s
```

No broad RECOVAR test suite was run.
