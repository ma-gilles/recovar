# VDAM combined hybrid + packed-deferred gate — H100 job 13368042

## Decision

Pass the one-transition combined quality and runtime boundary and advance to a
two-arm full `0 -> 200` trajectory sentinel.  The certified GEMM/direct-rescore
coarse path and exact scoring-projection-reuse fine path compose cleanly: all
decision-bearing state is exact, continuous state remains at CUDA repeat scale,
and warm whole-iteration time improves by 36.46%.

This remains default-off and does not change the frozen v3 scores.  A shared
in-memory one-transition panel cannot establish accumulated basin stability or
full-trajectory runtime.

## Qualification

| Field | Value |
|---|---|
| Source | `209aae4593f0a040090e150482f9d44dfa57e79f` |
| Slurm | `13368042` (`COMPLETED`, exit `0:0`, elapsed `00:05:46`) |
| Hardware | `della-h19g1`, NVIDIA H100 80GB HBM3, `GPU-0d7b80c7-fef8-e346-6332-de36ae1af518` |
| Boundary | One exact in-memory GF46 iteration-34 state, iteration `34 -> 35` |
| Panel | direct / hybrid packed deferred / hybrid packed deferred / direct |
| Diagnostics | disabled; clean timing panel |
| Peak monitored GPU memory | 17,579 MiB |

## Execution proof

The candidate's profile records both optimized seams:

- `coarse_gaussian_gemm_hybrid.enabled=true`;
- published scores are only `exact_relion_source16_or_full_rectangular` and
  expanded GEMM scores are never published;
- 516,224 of 34,099,200 coarse candidates (1.5139%) were directly rescored in
  872 source-16 blocks, with at most 18 blocks per image and zero fallbacks;
- fine scoring projects 38,016 packed rows instead of 62,208 padded rows;
- both arms project the same 1,868-pixel score/reconstruction union and use the
  same 489 nonzero-posterior reconstruction rows;
- `packed_vdam_reuses_flat_score_projection=true` in the candidate profile.

The arm artifact's legacy top-level `hybrid` convenience field is false because
the first combined-mode harness revision recognized only the standalone
`hybrid` label.  The production profile above proves execution; the harness
marker is corrected immediately after this evidence commit.

## Science result

- Every tracked pose, translation, class, posterior, significance, particle-
  state, sampling-state, and full support-audit digest is exactly equal.
- Cross-backend final `sigma2_noise` normalized L2 is `2.65e-8` and `2.51e-8`,
  with maximum absolute delta `1.86e-10`.  Direct/direct is `4.50e-9` and
  candidate/candidate is `7.53e-9`.
- Cross-backend `Igrad2` normalized L2 is `2.19e-8` and `4.69e-8`, versus
  `4.36e-8` direct/direct.  Reconstructed-map deltas are likewise at repeat
  scale (`2.69e-10` and `3.56e-10`).
- The same support SHA-256 is retained across all four arms.  No combined-path
  decision, support, or one-step basin split is observed.

The final-noise scalar amplifies cancellation of much larger float32 operands,
so its cross/repeat ratio alone is not a stability gate.  Its absolute scale,
the previously measured raw A2/XA repeat envelope, exact decisions/support, and
repeat-scale maps jointly classify this as mathematically equivalent numerical
noise.  The full trajectory is still required to test growth.

## Runtime result

Only second-arm warmed measurements are compared.

| Metric | Warm direct | Warm combined | Change | Speedup |
|---|---:|---:|---:|---:|
| Whole iteration | 2.603267 s | 1.654054 s | **-36.46%** | **1.574x** |
| Expectation | 2.377310 s | 1.429947 s | **-39.85%** | **1.663x** |
| Coarse pass 1 | 0.489487 s | 0.267612 s | **-45.33%** | **1.829x** |
| Fine pass 2 | 1.700851 s | 0.974550 s | **-42.70%** | **1.745x** |
| Shared local EM | 1.689493 s | 0.963290 s | **-42.98%** | **1.754x** |
| Local big JIT | 1.558703 s | 0.754884 s | **-51.57%** | **2.065x** |
| M-step | 0.162836 s | 0.161716 s | -0.69% | 1.007x |

Unlike the packed-deferred-only clean panel, this clears the predeclared 5%
material whole-iteration threshold by a wide margin.  It also improves both
passes, directly addressing the earlier profile where the coarse hybrid sped
up pass 1 while pass 2 stayed flat.

## Provenance

- Report JSON SHA-256:
  `302b222ebc223d580314012e6cc423f467981d96f24d2bd6bc0aff278ae5ebe1`
- Qualified CUDA checksum-file SHA-256:
  `f5faf97629f8bb9ea57b6f841f86e7c9e61d93575b972e1f562aa56b8e7be8d2`
- Artifact manifest SHA-256:
  `99ce1aaa69f4b23e0d467fab5cbd7ae06c93f5867b37673c717dfdf60aa53af8`
- Static-input checksum-file SHA-256:
  `8fa457994f5b453a150644c181102d876a5d4f4b8d4a7d1ec439e30ce646a4c0`
- Disposable artifact root:
  `/scratch/gpfs/GILLES/mg6942/vdam_runs/vdam_hybrid_packed_deferred_same_state_nodiag_it34_209aae459_20260903T020318Z`
