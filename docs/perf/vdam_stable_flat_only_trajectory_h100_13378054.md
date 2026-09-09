# VDAM fixed flat-row ABI-only trajectory gate — H100 job 13378054

## Decision

**HOLD default-off.** Fixing the packed fine-row capacity to `Q=B*R` is a
material but insufficient cold-runtime improvement: median fresh-process wall
through iteration 50 falls by 9.37%, expectation time by 9.58%, and pass 2 by
12.89%. It also accounts for nearly all of the combined policy's reduction in
large-JIT variants. However, it does not pass strict trajectory science. One
candidate repeat first changes a hard pose at iteration 29, the other first
changes one at iteration 35, and the first candidate later changes the search
schedule. The two ordinary repeats keep every predeclared hard field exact.

This isolates the previous combined result: fixed rows provide about nine
percentage points of its 25.90% wall improvement; stable Fourier-window classes
provide the larger remaining cold-compilation benefit. Fixed rows also expose
93% invalid projection padding near iteration 50, so they remain an ABI
experiment rather than the desired final execution layout.

The Slurm allocation is recorded as `FAILED (1:0)` because all four scientific
runs completed and the strict analyzer rejected the result. Its original
analyzer classified the iteration-46 size split as malformed setup and exited
before writing a report. Commit `d3e29dafe` makes a schedule split a recorded
science failure instead; the post-run report below was generated from the
immutable four-arm artifact.

## Setup

| Field | Value |
|---|---|
| Run source | `64d6433c41a87370a70d779415b5df95ca53a188` |
| Diagnostic analyzer | `d3e29dafe` |
| Job / node | `13378054` / `della-h19g2` |
| Physical GPU | H100 `GPU-e2c3190a-9599-15f7-a19c-7ae55e4e0a85` |
| Panel | fixed rows off / on / on / off |
| Isolation | four fresh Python processes and four fresh JAX caches |
| Trajectory | K=1 GF46, iterations `0 -> 50`, 19 logical Fourier sizes |
| Candidate-only delta | `RECOVAR_INITIAL_MODEL_STABLE_FLAT_ROW_CAPACITY=1` |
| Stable Fourier-window classes | disabled in every arm |
| CUDA artifact SHA-256 | `b4d5a24d679123faf29d438accb2a64fea6da60840be07c1e920928f72dec21d` |
| RELION binding SHA-256 | `9bbb1fb0ce6fa7ac816598ec521453515d163221642b916e5715bb2850798980` |
| Post-run analyzer SHA-256 | `6c7fbe117384842bd67bdd81cdbfc3db0c9459ce190287a288a219535c341630` |
| Analyzer report SHA-256 | `265166b28c0268bdec081e5ba27c1e3954e22ee0541bbab276a44d45d3541f54` |

Ordinary execution uses data-dependent packed row counts in 152 of 202 calls
per trajectory. Both candidate arms use exactly `Q=B*R` for every call.

## Performance

| Metric | Ordinary median | Fixed-row median | Change |
|---|---:|---:|---:|
| End-to-end wall | 512.628 s | 464.613 s | **-9.37%** |
| Expectation stage | 479.372 s | 433.451 s | **-9.58%** |
| Peak GPU memory | 17,591 MiB | 17,590 MiB | -0.01% |
| Profiled local EM | 357.886 s | 311.274 s | -13.02% |
| Local big-JIT buckets | 134.299 s | 100.028 s | **-25.52%** |
| Deferred local noise | 121.069 s | 110.277 s | -8.91% |
| Local packing | 29.990 s | 29.195 s | -2.65% |
| Coarse pass 1 | 101.262 s | 102.138 s | +0.86% |
| Fine pass 2 | 360.293 s | 313.848 s | **-12.89%** |

The ordinary caches each contain 4,774 payloads and 87
`run_local_bucket_big_jit` variants. Candidate caches contain 4,375/4,483
payloads and 48/49 big-JIT variants. Thus fixed rows remove about 44% of the
large-JIT variants, but only 7% of total cache payloads; the remaining generic
primitive and projector-shape churn is the larger target.

## Science chronology

| Pair | First hard difference | Read |
|---|---:|---|
| ordinary 1 vs ordinary 2 | none through 50 | all predeclared hard metadata exact |
| candidate 1 vs candidate 2 | iteration 29 | one near-tied pose takes different rotation/translation support |
| ordinary vs candidate 1 | iteration 29 | alternate candidate basin later changes sampling |
| ordinary vs candidate 2 | iteration 35 | later and much closer alternate basin |

At iteration 29, particle 22 in ordinary 1, ordinary 2, and candidate 2 chooses
rotation 34,878 / pose 5,162,014, while candidate 1 chooses rotation 34,882 /
pose 5,162,638. Candidate 1 alone changes its offset range and step at iteration
40, its resolution shell at iteration 45, and its current size at iteration 46.
This is not evidence of a missing candidate or changed mathematical objective:
the exact same-state transition gate passed, and the first split is a near-tie
after many float32/atomic updates. It is nevertheless a strict RELION-trajectory
failure and cannot be promoted under the current goal.

The complete report records 71 exact-field and 380 fixed-bound failures after
the basin split. Whole-trajectory map RMS normalized L2 is `1.92e-6` for the
ordinary repeat, `8.62e-3` for the split candidate repeat, and about `9.05e-5`
between the ordinary arms and candidate 2. The maximum checkpoint map distance
is `2.34e-2` because candidate 1 has already entered the different schedule.

## Next action

Do not spend more effort making `Q=B*R` denser. The next runtime gates remove
compile boundaries without changing reduction order: JIT the shared flat-row
scatter helper, then stabilize the shared RELION projector storage/radius ABI
while retaining logical support as a runtime mask. After those gates, rerun the
complete optimized `0 -> 200` repeat-controlled sentinel.

No broad RECOVAR suite was run.

Disposable artifact root:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_stable_flat_only_64d6433c4_20260903T065158Z`
