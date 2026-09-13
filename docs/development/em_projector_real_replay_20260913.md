# Real10073 projector-consumer replay — 13 September 2026

## Result and boundary

Frozen production repair `5e841111c116a07806ed153c922507cfceea9fce`,
parent `37faa499886c3a08742f4d1a6d4e5de0484e99bc`. No half-staging
optimization or shared-source edits are included. The parent class dispatcher
was loaded from its immutable Git object; unchanged numerical dependencies
were shared with the candidate within one process.

One difficult EMPIAR-10073 particle (original ID76678, resolved half-local
row38342), box380, saved iteration10→11 replay. The adaptive engine uses
coarse80/fine202 windows and the actual native x-half accumulation route.
Original host projector: complex128, shape1×403×403×202. All four explicit
double scoring/projection flags are false. GEMM scoring is unchanged.

| Arm | First synchronized call | Warm repeat | Warm sparse-pass stage |
|---|---:|---:|---:|
| Parent, inherited complex128 |57.7341 s|34.4841 s|34.2922 s|
| Parent, explicit complex64 |4.9345 s|3.0991 s|2.8014 s|
| Repaired consumer, original host input |3.0237 s|3.0112 s|2.6690 s|

Warm whole-call ratio is **11.45×** for this selected particle. Almost all
of the measured difference lies inside the sparse-pass wrapper.
Its misleadingly named `sparse_adaptive_mstep_s` includes fine projection,
scoring, posterior work, backprojection and accumulator return—not just
numerical reconstruction. This is not a pure-kernel attribution.

The six calls are interleaved in the table's arm order, twice, on the same
immediately idle physical A100 GPU1. The process completed0 in169.8335s,
within175s TERM/5s KILL supervision, with no survivors/errors or changed
predeclared pins. All calls save complete returned numeric fields.
First calls have unequal compilation/cache reuse and must not be ratioed.
One warm sample per arm is descriptive, not a representative speed estimate.

## Numerical findings — quality remains open

Against explicit-C64 parent, the candidate matches35/41 returned fields
exactly in both repeats, including scores, Pmax, poses and significant counts.
The six differing fields are BPref data/weight and two noise summaries,
duplicated in per-class/aggregate output. Same-path repeats differ in the same
six fields. This is not a bitwise equivalence claim or tolerance waiver:
cross-arm repeat0 data maximum5.2063e-10 and noise maximum0.0122434 exceed
the corresponding two-repeat empirical maxima. All differences are retained.

Against inherited-C128 parent, both repeats retain the winning pose but change
Pmax from0.2139879167 to0.2140051425 (delta1.72257e-5), best log-score by
0.001953125, and significant count **9184→9194**. Only15/41 returned fields
are exact. Full candidate surfaces and competing cutoff margins were not
captured; these changes are not dismissed as harmless numerical noise.
No FSC, autonomous trajectory, full support, exactly-K4 real-data or
whole-dataset performance admission follows.

Sparse accumulation is403×403×202 with C64/F32 storage. At sparse-pass return,
`relion_x_half_accumulators_to_public_layout` expands to the existing public
layout; returned Ft_y/Ft_ctf are1×65450827 C64/F32. The return conversion is
outside the per-bucket accumulation loop. Higher-precision host statistics
remain; this does not establish an entirely float32 reconstruction controller.

163 nominal1s nvidia-smi samples recorded a whole-device peak17371MiB and
mean utilization46.69%. These include startup, serialization and host gaps,
not just scoring. They are not stage peaks or a utilization improvement ratio.

## Reproduction and limits

[Audited values, comparisons and hashes](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_projector_consumer_real_replay_20260913_v2/audit.json).
[Exact command, environment and declared input pins](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_projector_consumer_real_replay_20260913_v2/supervisor/preflight.json).
[Complete worker log](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_projector_consumer_real_replay_20260913_v2/supervisor/process/worker.log).
[CPU source/helper preflight](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_projector_consumer_real_replay_20260913_v2/cpu_final_result.json).

The immutable packet contains `run.py`, `replay.py`, `constants.py`,
`driver_argv.json`, `cpu_check.py` and `audit.py`; launch recipes are in
preflight/terminal JSON. Fresh run roots are mandatory. Existing output roots
fail closed. Regenerate a report with the pinned pixi Python and
`CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu python -B audit.py`, using a fresh
report target; do not overwrite the frozen report.

The driver supports fresh mt19937 ordering only at iteration0. This late
replay explicitly retains legacy/saved ordering and resolves original IDs:
it is a RECOVAR self-comparison, not a modern RELION oracle comparison.
The unchanged real loader is made lazy and the selected row is read before
and after timing; its image/metadata bytes match. This warms OS cache.
The same mean, mean_variance, noise, priors and candidates are inventoried
before the six copied-input calls; no fitted offset or native-operand substitution.

The driver also loads subsequent it011 replay metadata, not included in the
supervisor's predeclared file pins. Do not claim exhaustive controller file-read
provenance. Engine operands remain matched in memory. Broader qualification
must close this input inventory explicitly.

Initial packet stopped before scoring in4.09s: the old harness required
`_existing_lib_path()`, which rejects timestamp-stale builds. The no-build
guard prevented compilation. V2 uses explicit library path/content pins;
CUDA source is unchanged since the sealed build. CPU rehearsal verifies
stale discovery acceptance only for the correct pinned hash and corrupt-hash
rejection. Earlier artifacts are preserved. No library was rebuilt or edited.

Next: integrator review, then unchanged-base/candidate GPU fast tier and
matched K1/exactly-K4 trajectory/FSC qualification before admission. Keep
broader batching, compilation, I/O, half-staging and RELION profiling work
active; this one-particle result does not close the general speed objective.
