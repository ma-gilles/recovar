# Shared SPA/ET checkpoint, 7 September 2026

This archive records existing 50,000-image, 128-grid SPA and cryo-ET regression
runs at `6ebaf5fad115817a3ce074076b09d1bc20bcd446`. Each workload ran once on an
allocated H100 against the committed historical baselines. It is a historical
checkpoint, not qualification of a later source or a replacement baseline.

- [Run record](run_record.json): exact commands, environment, source, lock and
  CUDA identities, GPU UUIDs and instrumentation hashes.
- [Quality and performance](quality_and_performance.json) and
  [comparison tables](comparison_tables.md): original tests and the corrected
  read-only inventory audit, including all historical performance warnings.
- [Generated inputs and FSC curves](fixtures_and_fsc.json): 58 fixture-file
  identities and 54 consumed product hashes per workload; six complete
  63-shell curves recovered from saved outputs on CPU.

The GPU tests pass in Slurm jobs 13569618 and 13569619. The first external
inventory audit mistakenly required ten retired aliases as additional missing
measurements. Historical source establishes those aliases; audit-only job
13570324 verifies all 16 canonical required metric keys pass in each workload.
That count includes the retained `variance_fsc` alias of `variance_spatial_fsc`;
it is not a count of independent measurements. Four additional local-resolution
values are finite but have no historical baseline in these reports.

Artifact audit 13573526 seals the generated inputs and recovers mean, spatial
variance and Fourier variance FSC curves without rerunning the pipeline.
All six curves remain above 0.5 beyond DC, so their threshold-frequency summaries
saturate at the same upper endpoint. The full curves retain differences that
those scalar summaries cannot show. The CPU reevaluation records its own
frequency values and deltas; it does not establish same-device arithmetic
parity with the original GPU evaluation.

The first artifact audit, 13573499, was cancelled after review identified that
the consumed eigenvector maps also needed before/after hashes. Its script and
output root remain preserved; the second audit includes those maps. No source,
fixture, completed GPU output, tolerance or baseline was modified.

Historical performance warnings remain visible, including dataset generation,
metrics, cryo-ET state computation and stage GPU memory. These memory values
come from the existing endpoint/cumulative method. They are not independently
sampled process peaks. One historical comparison does not qualify paired or
ordinary execution performance.

## Reproduction

Use separate checkouts at the recorded commit and the committed locked pixi
environment. Verify imported package paths and the native-library identity.
Reproduce each recorded test command and environment with a fresh output root,
assigned GPU and independent cache. Fixture generation uses seed 42 and the
recorded PDB-trajectory parameters; compare generated inputs with their hashes
before treating a regenerated dataset as identical.

For a read-only artifact audit, run the recorded audit script on the saved
products using the source identified in the run record, or adapt it to freshly
reproduced outputs. Preserve the complete expected fixture/product inventory.
The full scalar definition and FSC arithmetic remain in
`recovar.commands.run_test_all_metrics` and `recovar.output.plot_utils` at the
recorded source. Large particle stacks, generated volumes and pipeline products
remain in scratch; the archive retains numeric curves, commands and identities.
