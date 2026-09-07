# Real 10076: fixed-input accumulation repeatability

The unchanged PR158 production accumulation function produces different
float32 results when given identical inputs repeatedly on the same physical
H100. This establishes a repeatability limitation at the accumulation boundary.
It does not establish the cause of later autonomous trajectory divergence,
qualify the cleanup, or change any accuracy or performance gate.

## Captured operands

[Operand comparison](operand_comparison.json) records two shortened,
first-iteration runs at PR158 commit
`44d770de3f9336ab2f3f6a34203394bae8d1aeed`. Each contains all 5,000 half-1
particles and 40,000 valid rotation rows, in 23 packets. All 20 recorded
identity/operand fields agree exactly in every packet (460 comparisons).
Pre-join accumulators differ despite those identical captured operands:

| Field | Changed elements | Maximum absolute difference |
| --- | ---: | ---: |
| Data, complex64 | 3,686 | 1.5359765386647212e-8 |
| Weight, float32 | 550 | 2.3283064365386963e-10 |

This extends the earlier [saved-array and half-join diagnostics](../real10076-repeatability-20260907/README.md).
The recorded fields establish this boundary's inputs, not equality of every
intermediate inside the E-step.

## Repeated production function

[Fixed-input comparison](fixed_input_comparison.json) replays the first
captured stream, selected before execution, three times from fresh zero
accumulators. Each trial calls
`recovar.cuda_backproject.relion_fused_x_half_backproject_indexed` once per
particle, preserving all eight valid rows, including zero-weight rows.
Inputs retain their captured dtypes: complex64 weighted data, float32 weights
and rotations, and int32 window indices. The image shape is 256×256, the BPref
volume shape is 99³, and the captured support radius is 24.0.

The native outputs already differ, before x=0 enforcement or conversion to
RECOVAR's public layout. Each has 490,050 elements.

| Trials | Data elements changed | Data maximum gap | Weight elements changed | Weight maximum gap |
| --- | ---: | ---: | ---: | ---: |
| 0 vs 1 | 2,137 | 1.666000468656264e-8 | 281 | 1.1641532182693481e-10 |
| 0 vs 2 | 2,168 | 1.666000468656264e-8 | 295 | 1.1641532182693481e-10 |
| 1 vs 2 | 2,177 | 1.666000468656264e-8 | 274 | 1.1641532182693481e-10 |

Trial 0 uses a fresh persistent compilation cache; trials 1 and 2 reuse the
same process's compiled functions. Thus the warm repeats also differ. The
report preserves all 18 comparisons across native, post-x=0 and public outputs,
plus each trial's comparison with the original captured pre-join buffers.

The function includes preparation of native operands and the fused kernel.
These observations do not isolate one low-level instruction as the cause.
Per-packet synchronization and replay transfers differ from full scoring;
recorded replay timings are diagnostics, not ordinary execution benchmarks.

## Provenance and reproduction

[Run record](run_record.json) identifies the source, all 23 selected packet
hashes, manifests, scripts, library and archived reports. The complete operand
report also retains the identities of both captured streams. Large inputs and
18 output arrays remain outside git, with their SHA-256 digests in the reports.

- Capture: Slurm13579420; operand audit: CPU Slurm13579421.
- Fixed-input replay: Slurm13585254.
- All three jobs use unchanged PR158. Capture and replay share H100
  `GPU-75c2d200-95d1-ef57-fb52-1698386c756c`.
- CUDA library SHA-256:
  `00cd5880d601c1f170e585722aabc8b94a8e3d83b143ae9e184992213bee8a05`.
- Replay checks 1,774 source/input/instrumentation identities before execution,
  after loading the library and after execution, and records the loaded path.

Preparation and exact Slurm/Python commands are preserved under:

```text
/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/real10076_fixed_scatter/
/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/real10076_prescatter_operands/
```

The corresponding output roots are:

```text
/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/hia_real10076_fixed_scatter_20260907/
/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/hia_real10076_prescatter_operands_20260907/
```

To reproduce, retain the recorded source and fixture identities and prepare a
new output root, manifest and immutable CUDA copy. The sealed scripts reject
input changes and output overwrites; do not resubmit into their completed
roots. Saved-array comparisons can be audited without rerunning GPU work.
