# VDAM shared local-projection cache at iteration 48 — H100 job 13373715

## Decision

Reject the current local-projection cache for InitialModel.  The mature shared
EM cache is fast and repeatable within a mode, but its approximate rotation-ID
key aliases distinct oversampled matrices in this workload.  From the exact
same iteration-47 state it changes fine-pose decisions and escapes the CUDA
repeat envelope.  It remains disabled and is not part of the combined backend.

## Qualification

| Field | Value |
|---|---|
| Source | `8816d487e76540690e8e38cfa12bd014ba5d856f` |
| Slurm | `13373715` (`COMPLETED`, exit `0:0`, `00:08:46`) |
| Hardware | `della-h19g1`, H100, `GPU-0d7b80c7-fef8-e346-6332-de36ae1af518` |
| Boundary | One exact in-memory GF46 state, iteration `47 -> 48` |
| Panel | direct / combined+cache / combined+cache / direct |
| Cache | 696 projection rows, 0.015785 GB, 1 GB candidate cap |

Every arm starts from deep copies with identical model, particle, and sampling
state manifests.  Candidate arms add the shared local projection cache to the
already qualified hybrid+packed+deferred bundle.  Controls force the cache and
the complete candidate bundle off.

## Science result

- Direct/direct and cache/cache repeats are internally discrete-identical.
- Direct versus cache changes 21/200 fine pose assignments, including nine
  best rotation IDs and 33 translation-coordinate values.  Pmax changes for
  198/200 particles.
- Coarse significant counts and the complete coarse-support digest remain
  exact, localizing the failure after coarse selection.
- Maximum cross accumulator normalized L2 is `0.18379525` for data and
  `0.00537855` for weights.  Direct/direct maxima are only `8.60e-8` and
  `4.59e-8`; cache/cache maxima are `9.28e-8` and `5.01e-8`.
- Maximum cross final-state normalized L2 is `0.00375666`, versus
  `1.13e-8` direct/direct and `1.94e-8` cache/cache.

This is a deterministic cache-specific displacement, not ordinary atomic
noise.  It agrees with the earlier warning that one approximate rotation ID
may represent multiple oversampled exact matrices.  A future cache would need
an exact matrix identity/key and must preserve the source execution order.

## Runtime result

The warm bundled candidate wall is `2.562403 s` versus `6.100077 s` for the
warm direct arm, but that comparison includes the already-qualified coarse
hybrid and packed/deferred changes and therefore cannot attribute the gain to
the cache.  Against the prior same-state combined/no-cache job, pass 2 changes
only `1.068328 -> 1.037490 s` and big-JIT time `0.753273 -> 0.731390 s` across
different jobs.  That roughly 3% cross-job signal is neither an admissible
timing comparison nor worth the correctness failure.

## Provenance

- Report SHA-256:
  `0ec47650576126f977380235595402e5766cf43671c856e665fa86ce7256c557`
- CUDA SHA-256:
  `491006306c1c24486e0360002a25ffec2755ee8cdb0c116f36f1884905307eca`
- Disposable root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_local_projection_cache_it47_8816d487e_20260903T043106Z`
