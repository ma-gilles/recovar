# VDAM GF46 integrated hybrid H100 qualification: job 13341618

## Decision

**PASS for the bounded selected/fallback execution gate.** The integrated,
default-off K=1 hybrid executed at the unreduced GF46 coarse geometry, selected
exact source-16 rescoring for every active row, retained the full RELION
posterior layout, passed the mature RELION-f32 posterior, and physically
executed the whole-batch rectangular fallback when the selected-block capacity
was deliberately too small.

This is not an end-to-end correctness or runtime promotion. The next required
gate is the frozen GF46 checkpoint transition from iteration 180 to 181 against
the direct scorer.

## Exact scope and provenance

| Field | Value |
|---|---|
| Slurm job | `13341618` (`COMPLETED`, 73 s) |
| Source commit | `b611aeff15004a07d6d2a1ec590bc5b97180cb5b` |
| Source tree | `39f07f5b1665e864c5112fe5da04e4f1aa07ef08` |
| Source manifest | `c3c0f7cee688f3dba949ecb9d76d9f0ddd3c7ffc91a17aa121b3e38de46414cf` |
| Repository status | clean; empty HEAD diff |
| Node / GPU | `della-h19g1`; NVIDIA H100 80GB HBM3 |
| Physical GPU UUID | `GPU-2ee3da91-970a-6714-84df-530aefe04a08` |
| Custom CUDA | rebuilt from the sealed source for `sm_90` |
| Artifact manifest | `fec2ec40f98057a428799a472ea36908725c24ed54041d21ceacd99cf0ef325a` |
| Disposable result root | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_hybrid_batch_h100_13341618_b611aeff1` |

The first attempt, job `13341559`, failed during JAX initialization because
cuSPARSE was not preloaded; it did not execute hybrid code. Commit `b611aeff1`
reused the established GF46 runtime contract by pinning and preloading the
CUDA 12.6 cuSPARSE library. Job `13341618` supersedes that setup-only failure.

## Unreduced workload

| Dimension | Value |
|---|---:|
| Image batch | 500 |
| Source rotations | 36,864 |
| Translations | 29 |
| Compact Fourier pixels | 5,100 |
| Certificate chunk | 4,608 rotations |
| Exact rescore granularity | 16 source rotations |
| Full logical candidates | 534,528,000 |

Two deliberately near source-16 blocks were embedded in an otherwise distant
projection table. The capacity-eight arm therefore had to select block IDs
`0,1` for all 500 rows. A separate capacity-one arm had to reject that same
selection and execute the physical full-direct fallback.

## Result

| Boundary | Observed | Status |
|---|---:|---|
| Selected source blocks per row | exactly 2 (`0,1`) | PASS |
| Exact selected candidates | 464,000 / 534,528,000 (`0.0868056%`) | PASS |
| Published selected score | exactly `-0.125f` | PASS |
| Raw maximum / best pose | exactly `-0.125f` / pose 0 | PASS |
| Selected output source | exact source-16 CUDA rescore; priors included once | PASS |
| Omitted candidates | full-layout `-inf` padding | PASS |
| Repeated selected summaries | byte-identical over all timed runs | PASS |
| Mature posterior support | exactly 928 per row | PASS |
| Posterior probability sum | within `2e-5` of one | PASS |
| Forced fallback reason | `block_capacity_overflow` | PASS |
| Fallback finite candidates | exactly 8,552,448 for eight rows | PASS |
| Fallback raw maximum | exactly `-0.125f` | PASS |

Every semantic check serialized by the harness is true. The focused CPU gate
covering the dispatch and qualification harness is also clean at **206 / 206**.

## Timing and memory

| Measurement | Seconds |
|---|---:|
| Operand allocation | 0.478084 |
| First selected compile + execute | 5.203693 |
| Selected warm runs | 0.471513, 0.506790, 0.470382 |
| Selected warm median | **0.471513** |
| Mature RELION-f32 posterior | **0.813419** |
| Selected score + posterior, summed component time | **1.284932** |
| Forced capacity-overflow path, certificate + full direct for B=8 | 3.121482 |

The measured device peak was **19,820,147,456 bytes (18.459 GiB)**; the XLA
pool reached 34,361,835,520 bytes (32.002 GiB), below the H100 allocation.
The selected helper median is only 40.3 ms above the separately qualified
eight-chunk certificate projection of 0.431204 s, but that difference is
context only, not an independently isolated rescore measurement.

The historical same-shape GF46 transition measured about 21.36 s warm wall for
the full direct iteration and about 4.74 s for the unsafe standalone GEMM arm.
Those numbers motivate the next paired transition, but they are not combined
with this synthetic qualification to claim an end-to-end speedup.

## Reproduce

From clean commit `b611aeff15004a07d6d2a1ec590bc5b97180cb5b`, compute the
source-manifest digest and submit
`scripts/run_vdam_gf46_hybrid_batch_h100.sbatch` with the exact commit, tree,
manifest, pixi interpreter, and repository root exported. The runner selects
the single cgroup-visible H100, rebuilds the custom CUDA library externally,
pins cuSPARSE, verifies both required CUDA symbols, and seals every artifact.
