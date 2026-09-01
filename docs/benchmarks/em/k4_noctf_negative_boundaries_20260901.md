# K=4 no-CTF negative fixture boundaries (2026-09-01)

## Status

These nine completed synthetic runs are **excluded negative diagnostics**, not
accepted RECOVAR-versus-RELION benchmark results. In every replicate, RELION
assigned exactly zero class and orientation mass to class 3 at the first
numbered iteration. The fail-closed launcher therefore stopped before starting
RECOVAR.

Consequently, these runs define a useful fixture boundary, but they do not
measure RECOVAR quality. Cross-engine FSC, ground-truth FSC-AUC, assignment
agreement, RECOVAR runtime, and an engine speed ratio are all undefined.

The complete machine-readable evidence is
[`diagnostics/k4-noctf-collapse-cases30-35-36-h100.json`](diagnostics/k4-noctf-collapse-cases30-35-36-h100.json).
It validates against
[`diagnostic_schema_v1.json`](diagnostic_schema_v1.json), which fixes
`registry_disposition=EXCLUDED_FROM_ACCEPTED_RESULTS`,
`accepted_result=false`, and `recovar_was_evaluated=false`.

## Hypotheses tested

All cases use the first four CryoBench2 Ribosembly structures, C1 symmetry,
box size 128, no CTF variation, three frozen seeds (41001--41003), and a
five-iteration cap.

| Case | Intended discriminator | Particles | White-noise scale | RELION initial low-pass / Fourier init radius |
| ---: | --- | ---: | ---: | --- |
| 30 | Remove CTF variation from the small positive-control candidate | 3,000 | 1.0 | 60 A / 10 px |
| 35 | Raise signal and particle count | 10,000 | 0.2 | 60 A / 10 px |
| 36 | Also give RELION a more resolved and broader initial reference | 10,000 | 0.2 | 20 A / 27 px |

Each escalation leaves the same class-3 zero-mass boundary. More particles,
fivefold lower noise, and a more resolved/broader Fourier initialization
therefore do not make these particular no-CTF fixtures suitable paired K=4
controls.

## Observations

The Slurm case status is `FAILED` because the launcher intentionally exits 1
when its RELION class-population precondition fails. It is not an application
crash. All jobs requested and received exactly one NVIDIA H100 80GB HBM3 GPU.
HBM is the peak MiB observed during the sealed RELION wall-time window; MaxRSS
is fresh `sacct` job accounting.

The original `slurm_case_accounting.json` files are retained and hashed as
historical artifacts, but some contained stale states. The job fields in the
machine record were queried afresh from `sacct` when the record was generated;
those fields, rather than the historical snapshots, produce the table below.

| Case | Seed | Job | First zero | Final RELION class distribution | RELION wall (s) | Peak HBM (MiB) | Job wall (s) | MaxRSS (GiB) |
| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 30 | 41001 | 13301411 | iter 1, class 3 | (0.259025, 0.658587, 0, 0.082387) | 39 | 79587 | 272 | 3.027 |
| 30 | 41002 | 13301412 | iter 1, class 3 | (0.127151, 0.872849, 0, 0) | 38 | 79577 | 272 | 3.028 |
| 30 | 41003 | 13301413 | iter 1, class 3 | (0.263554, 0.605769, 0, 0.130677) | 39 | 79585 | 272 | 3.036 |
| 35 | 41001 | 13302462 | iter 1, class 3 | (0.208990, 0.739835, 0, 0.051176) | 285 | 79577 | 619 | 6.491 |
| 35 | 41002 | 13302463 | iter 1, class 3 | (0.210119, 0.789881, 0, 0) | 106 | 79585 | 346 | 6.483 |
| 35 | 41003 | 13302464 | iter 1, class 3 | (0.501342, 0.394605, 0, 0.104053) | 108 | 79577 | 348 | 6.482 |
| 36 | 41001 | 13303134 | iter 1, class 3 | (0.173721, 0.779760, 0, 0.046519) | 112 | 79629 | 397 | 6.485 |
| 36 | 41002 | 13303135 | iter 1, class 3 | (0.475540, 0.524460, 0, 0) | 183 | 79601 | 535 | 6.487 |
| 36 | 41003 | 13303136 | iter 1, class 3 | (0.159879, 0.585123, 0, 0.254999) | 166 | 79629 | 515 | 6.485 |

Seed 41002 also has zero class-4 mass at the final iteration in all three
cases. The machine record retains every per-iteration zero-mass event rather
than only the final summary.

## Provenance and reproduction

The three disposable run roots, each marked `SAFE_TO_DELETE`, are:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_symmetry_multiseed_c75cbfffc_20260901`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_noctf_strong_717fa1d5a_20260901`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_resolvedinit20_43ea66e1e_20260901`

The corresponding clean RECOVAR commits are
`c75cbfffc930e4dac7bcc2785f9220ad8472a5a5`,
`717fa1d5a195bcf94e36cc647e775797950b4eb0`, and
`43ea66e1e0187c19abbc24732b1b85708d671e85`. All use RELION commit
`d476e6f6a4f1f37627c06ace5227fc374c0c2b05` plus the sealed dynamic-dispatch
source diff and executable recorded by SHA-256 in the machine record.

For each case, `runs[].reproduction.command` is a copy-safe full command. It
requires a fresh absent output root, freezes the three seeds, binds the exact
RECOVAR checkout, pixi Python, RELION source, and RELION executable, and runs
the same matrix launcher with `--watch`. The record also hashes every particle
stack, pose/CTF file, generation config, class manifest, launcher, audit,
population table, RELION optimiser file, monitor, and log used in the
conclusion.

Regenerate the compact diagnostic JSON from those sealed roots with:

```bash
pixi run python scripts/seal_em_kclass_negative_diagnostic.py \
  --run 30:/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_symmetry_multiseed_c75cbfffc_20260901 \
  --run 35:/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_noctf_strong_717fa1d5a_20260901 \
  --run 36:/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_resolvedinit20_43ea66e1e_20260901 \
  --output docs/benchmarks/em/diagnostics/k4-noctf-collapse-cases30-35-36-h100.json
```

Then validate all checked-in records and rehash every external file:

```bash
pixi run python scripts/validate_em_benchmark_registry.py --verify-files
```

## Qualification consequence

Do not reuse cases 30, 35, or 36 as claims about RECOVAR-versus-RELION K=4
quality. A replacement paired fixture must first pass the same RELION
population precondition on every frozen seed. Only then should RECOVAR run and
the ordinary accepted campaign schema evaluate matched trajectories, FSC,
assignment agreement, and performance.
