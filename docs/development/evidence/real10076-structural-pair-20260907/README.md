# Real 10076 structural checkpoint: failed comparison

The unchanged PR158 control (`44d770de3`) and frozen structural candidate
(`be1f2913e`) both execute 16 numbered iterations and final all-data
reconstruction. All eight compared controller fields agree. The pair fails
the standing map and particle-field contracts; it does not qualify cleanup
equivalence or ordinary performance. Later structural commits are not covered
by this run.

## Map and particle results

[The map/controller report](map_and_controller.json) compares 51 products:
two half-maps and their merged map for every iteration, plus three final
products. All 51 [127-shell FSC curves](shellwise_fsc.json) are preserved with a
lossless numeric roundtrip from the original NPZ. The unchanged FSC-AUC floor
is 0.995. The first failing map is half 1 in iteration 8 (0.9933899201);
27 of 51 products fail.

| Final product | Direct control/candidate FSC-AUC | Result |
| --- | ---: | --- |
| Half 1 | 0.9621420920 | Failed |
| Half 2 | 0.9762480498 | Failed |
| Merged | 0.9723503638 | Failed |

[The particle report](particle_fields.json) checks all 69 required image-ordered
fields, including final outputs. Sixty fail the existing recorded-field
contract. In the first numbered iteration, support, Pmax, Euler coordinates
and translations are exact. Support counts first differ in iteration 2:
14 images differ by one retained sample. Iteration-2 Pmax remains within the
existing limits (maximum absolute difference 0.0006858557).

In iteration 3, image 8690 changes translation by 0.5 pixel. Pmax's maximum
difference is 0.0152229667 and its 95th percentile is 0.0001603410, exceeding
the existing limits. Euler-coordinate differences are reported as coordinates;
they do not alone quantify physical angular separation. No near-tie exception
is claimed without the competing candidate scores.

The [unchanged-source repeatability study](../real10076-repeatability-20260907/README.md)
already fails a same-GPU trajectory comparison and finds different
first-iteration accumulators before the half join. This negative control means
the present pair cannot attribute divergence to cleanup alone. It does not
waive this pair's failed gates.

## Run identity and reproduction

[The run record](run_record.json) preserves complete commands, environment
overrides, source/native-library identities, 174 fixture identities, independent
cache directories and archive hashes. Both runs used H100
`GPU-202f2d43-7a0a-bec1-135f-bb496ab059a7`. Source, fixture, native-library and
output checks passed before and after the comparisons.

- GPU execution: Slurm13569949, completed in 2:58:41.
- Map/controller audit: Slurm13569953, completed comparison with exit 2
  for failed gates; 693.75 seconds.
- Particle audit: Slurm13577514, completed comparison with exit 2
  for failed fields; 4.10 seconds of audit work.
- Exact manifests and scripts:
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/structural_real10076_pair/`.
- Original outputs:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/recovar_structural_real10076_pair_20260907/`.

Recreate each recorded commit and environment in a separate checkout, retain
the same fixture/oracle inputs, and use fresh output and cache directories.
The directory name of a newer checkout is not the recorded candidate source.
Recorded wall times are observations from one instrumented pair with failed
quality gates, not a performance qualification. These archived measurements
are not new expected baselines.
