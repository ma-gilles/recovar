# K1 case25 source comparison, 7 September 2026

PR158 (`44d770de3`) and structural cleanup (`159059131`) passed the scoped
K1 case25 comparison: 1,000 particles at grid 128, eight numbered iterations
and finalization. Slurm13567539 ran both cold processes on the same A100 80 GB;
audit13567540 compared their maps and recorded controller fields.

| Measurement | Control | Candidate | Change |
| --- | --- | --- | --- |
| Process wall time, seconds | 548.2234 | 548.6691 | +0.081% |
| Sampled process-tree GPU peak, bytes | 34881929216 | 34881929216 | 0% |
| Sampled process-tree host RSS peak, bytes | 9275678720 | 9420484608 | +1.561% |

All eight compared controller fields agree. All 27 direct map FSC-AUC values
pass the existing 0.995 gate; the minimum is 0.999999998203. Both runs pass
the scoped GT and historical RELION map gates. Neither process had a sampling
error or an observed unrelated GPU process in its memory trace.

This is one cold pair with full intermediate capture. It does not qualify
ordinary execution performance, repeated timing, full particle-state parity,
other fixtures or later commits. An earlier three-pair comparison at source
`430f46325` retains a +11.772% host-RSS warning. This later pair does not resolve
that warning. These measurements do not replace any established baseline.

## Archived evidence

- [Run record](https://github.com/ma-gilles/recovar/blob/5c07fc3169d636dc3775b2bb41ae4c76d6829611/docs/development/evidence/k1-case25-20260907/run_record.json): commands, environment, source and lock hashes,
  native binary identity, all 84 fixture file identities, process exits,
  output hashes and original artifact locations.
- [Comparison](https://github.com/ma-gilles/recovar/blob/5c07fc3169d636dc3775b2bb41ae4c76d6829611/docs/development/evidence/k1-case25-20260907/comparison.json): the original report, preserved byte for byte,
  including every direct map score and the individual timing/memory results.
- [FSC curves](https://github.com/ma-gilles/recovar/blob/5c07fc3169d636dc3775b2bb41ae4c76d6829611/docs/development/evidence/k1-case25-20260907/fsc_curves.json): all 189 float64 curves, each with 63 shells,
  converted losslessly from the three original NPZ files. The run record
  identifies those files and the archived JSON by SHA256.

The direct control/candidate keys `it000` through `it007` use RECOVAR's
zero-based output iteration. The GT/RELION groups use `it001` through `it008`
for the corresponding numbered RELION iterations. Each group also contains
final products. Preserve this offset when joining the groups. Curve samples
retain their original order and values; no smoothing or interpolation was used.

An [additional particle-field audit](https://github.com/ma-gilles/recovar/blob/5c07fc3169d636dc3775b2bb41ae4c76d6829611/docs/development/evidence/k1-case25-20260907/particle_fields_audit.json) compares all
37 explicitly image-ordered fields in the two result archives. Recorded
pose/support decisions agree exactly. Pmax differences have maximum 0.000155002
and largest per-field p95 0.0000503063. This extends the recorded-field evidence;
it does not compare all candidate scores or accumulator states.

## Reproduction

Create separate checkouts at the two full commits in `run_record.json`, install
each locked pixi environment, and verify imported RECOVAR/JAX paths. Verify all
fixture and native-library hashes before running. Use the recorded command
arrays and scientific environment settings, replacing checkout, output,
library-copy and runtime-cache paths with fresh isolated locations. Never rerun
into the archived output paths. Give each process an empty compilation cache
and a private verified CUDA-library copy; record loaded library hashes before
and after execution. Run the pair sequentially on one allocated GPU.

The original orchestration and audit scripts are identified by path and hash
in the record. They remain in the Della review directory. The source-level FSC
implementations are `scripts/summarize_em_completion_bench.py` and
`scripts/audit_k1_fsc_trajectory.py` at the recorded source. Preserve the
recorded gate and iteration inventory when deriving a fresh comparison.

Large maps, particle captures, logs and raw memory samples remain in scratch;
they are not embedded here. The archive preserves map curves, summary
measurements and identities, so it can support later review and reproduction.
Investigating an individual particle or re-auditing the raw memory trace still
requires those original outputs or a fresh run.
