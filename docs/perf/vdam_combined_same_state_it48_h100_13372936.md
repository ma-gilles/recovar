# VDAM combined-backend iteration-48 causal gate — H100 job 13372936

## Decision

The combined certified-coarse plus packed-deferred backend does **not** make a
different discrete decision at iteration 48 when it starts from the exact same
iteration-47 state as the direct backend. All pose, translation, class,
posterior, significance, particle, sampling, and support-audit fields agree
exactly. Cross-backend continuous deltas are no larger than ordinary repeated
CUDA accumulation deltas.

This rules out a deterministic combined-backend error at the first hard split
seen in the two-arm `0 -> 200` sentinel. The trajectory split is consistent
with accumulated roundoff eventually crossing a near tie. This diagnostic is
one transition only, so it cannot promote the backend or change frozen scores.

## Qualification

| Field | Value |
|---|---|
| Source | `10cd188edc4d81d038e582c277c65d60396368d1` |
| Slurm | `13372936` (`COMPLETED`, exit `0:0`, elapsed `00:09:07`) |
| Hardware | `della-h19g2`, NVIDIA H100 80GB HBM3, `GPU-e2c3190a-9599-15f7-a19c-7ae55e4e0a85` |
| Boundary | One exact in-memory GF46 iteration-47 state, iteration `47 -> 48` |
| Panel | direct / combined / combined / direct |
| Classification | `diagnostic_same_in_memory_state_one_transition_only` |

The harness verifies that every arm starts with byte-identical model,
particle, and sampling-state manifests. Support-ID auditing is enabled in all
four transition arms.

## Science result

- Every tracked E-step decision, complete particle state, sampling state, and
  support-audit digest is exact in all six pairwise comparisons.
- Direct/direct accumulator normalized L2 reaches `9.16e-8`; candidate/candidate
  reaches `9.00e-8`; the largest cross-backend value is `9.63e-8`.
- Direct/direct final-state normalized L2 reaches `2.08e-8`; the largest
  cross-backend value is `5.59e-8`. Final-map cross deltas are only
  `5.53e-11` to `1.07e-10`.
- Cross-backend final `sigma2_noise` normalized L2 is `3.30e-10` to
  `4.42e-10`, comparable to the `3.17e-10` direct repeat.

The sentinel's iteration-48 particle-2798 translation split therefore does
not reproduce from a common live state. It requires the small continuous
differences accumulated over earlier transitions.

## Runtime result

Second-arm warmed measurements are compared. The combined backend remains a
large material win at this later and larger search state.

| Metric | Warm direct | Warm combined | Change | Speedup |
|---|---:|---:|---:|---:|
| Whole transition | 8.142330 s | 3.229593 s | **-60.34%** | **2.521x** |
| Coarse pass 1 | 5.423809 s | 1.213586 s | **-77.62%** | **4.469x** |
| Fine pass 2 | 1.347488 s | 1.068328 s | **-20.72%** | **1.261x** |
| Shared local EM | 1.334431 s | 1.055514 s | **-20.90%** | **1.264x** |
| Local big JIT | 1.174755 s | 0.753273 s | **-35.88%** | **1.560x** |

The packed path scores 19,008 rows instead of 55,296 while retaining the same
5,608 logical union rows and 511 reconstruction rows. The next performance
gate tests the existing shared-EM projection cache because these packed rows
still repeat the same global rotations across many images.

## Provenance

- Report JSON SHA-256:
  `e734d48748cc6ae41f3851df5878b8928456d3b32db31fd39511b0927c5cec5e`
- Qualified CUDA checksum-file SHA-256:
  `e01b781885ea9e4a1be1ccdb2b5276d16064f9c8647ac3c884491093849e5411`
- Disposable artifact root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_combined_same_state_it47_10cd188ed_20260903T0020Z`
