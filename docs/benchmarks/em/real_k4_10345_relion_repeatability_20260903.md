# EMPIAR-10345 native-grid K=4 RELION repeatability control

This control repeats RELION half 1 from the native10k seed-42001 K=4
experiment with the same 5,000 particles, four starting maps, seed, binary,
patched source, MPI shape, module environment, and all science arguments. The
only command-array difference is the required output prefix. It calibrates
ordinary same-engine timing/reduction-order variability before interpreting
the larger RECOVAR-versus-RELION endpoint differences.

## Result

Job `13376810` completed `0:0` in 4m29s. RELION itself took 150.422 s versus
147.483 s in the original run. The sorted logical particle stream was exact,
while MPI follower ownership changed on 31,104 of 40,000 particle/iteration
visits. Despite that deliberately exercised execution-order variation:

- all 40,000 class assignments were exact;
- all 40,000 raw Euler and translation rows were exact;
- five significant-support rows changed by one sample;
- typical Pmax drift stayed small (per-iteration p95 at most 3e-5), with one
  iteration-5 outlier at 0.006006 and no class or pose change;
- every iteration had the unique identity class permutation; and
- the minimum signed non-DC FSC-AUC across all 32 numbered maps was
  0.999999995094, with maximum relative L2 1.02e-5.

Final class-map FSC-AUC values were 0.999999998597, 0.999999998288,
0.999999999125, and 0.999999997631. Thus same-engine dispatch/reduction-order
noise is roughly a 1e-9 FSC-AUC-deficit effect here. It cannot explain the
seed-42001 RECOVAR/RELION half-1 gap of 184/5,000 assignments or the worst
direct final-map FSC-AUC of 0.9732088221. Combined with the three-seed result,
the dominant scientific scale is a shared seed-sensitive basin rather than
ordinary late-trajectory numerical nondeterminism.

## Reproduction and provenance

The complete frozen control lives at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_relion_half1_repeat_10345_native10k_seed42001_20260903T055128Z`.
Its `jobs/run_relion_half1_repeat.sbatch` launcher fails closed on input/source
hashes, exact command equivalence except `--o`, H100 count, and requested
versus allocated Slurm resources. The exact command is in
`provenance/repeat_command.json`; the analysis is in
`jobs/analyze_repeatability.py` and `jobs/analyze_dispatch.py`. Use a fresh
output root and output prefix for any further repeat.

The job requested and received exactly
`cpu=24,mem=256G,node=1,billing=24,gres/gpu=1`, used three MPI tasks with eight
CPUs per task, had `OverSubscribe=OK`, and did not request exclusive access.
It ran on an H100 80GB HBM3. Both run and runtime roots contain
`SAFE_TO_DELETE`.

Sealed outputs:

- full audit JSON: SHA-256
  `af7ef651f3eefe6c043f3c938c92dd0d701e673e60d87b9cd96eb37b6cdde313`;
- FSC curves: SHA-256
  `7bdb6fc0caf992cef331bd8b3732177b790ef4eb4c919742f8991b924b776460`;
- dispatch audit: SHA-256
  `f446a2411e469bf1fd4bb9720f529d200ffc328721af27d26a42761be0269379`;
- compact report: SHA-256
  `2bb944613b1860f73e09851f73df2f3208b459e3b309e71930d40c2b4763ccae`;
- sealed-artifact manifest: SHA-256
  `408a55a8545e4e13b43fd1296608977c57c831bb7ec2235acea886c78647967d`.

The pre-run manifest covered all 40 original numbered STAR/map outputs, and
its post-run check passed 40/40, proving that the repeat did not modify the
original evidence.
