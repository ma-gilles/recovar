# Projector dtype regression: source history

The source transition is commit `7e9e5c3c99936795038bdeb026f3897e4e962125`,
“em: preserve double precision sampling and RELION projector data” (committed
September 8). It removed both explicit complex64 casts from
`reference_to_relion_projector_half_maps` in the then-current
`recovar/em/initial_model/dense_adapter.py`. The native producer's complex128
output consequently survived this wrapper. This commit is an ancestor of
candidate `5e841111c`, but not of historical timing source `9216a1b8f`.

An executable CPU check loads the actual function bodies from `9216a1b8f`,
the change's parent, the change itself, and the current shared producer.
With identical explicitly stubbed native complex128 output, the first two
return complex64 and the latter two return complex128. The unchanged texture
eligibility function rejects complex128. The repaired production consumer
returns exactly the old complex64 values and restores texture eligibility.
Native FFT execution and CUDA availability are stubbed in this CPU check;
it is a dtype/dispatch test, not a GPU arithmetic or speed measurement.

This identifies a source cause of the dispatch regression. It does **not**
establish how much of the historical whole-iteration slowdown it caused.
The separately [measured real one-particle replay](em_projector_real_replay_20260913.md)
observes warm parent34.484s versus repaired3.011s, but has changed significant
support and no final FSC qualification. Other historical timing confounds,
transfers, batching, compilation and I/O still require measurement.

The narrow repair keeps native host preparation and double diagnostics intact;
it restores the production conversion at the consumer. Reverting the whole
historical commit would also alter unrelated sampling/metadata behavior and
is not proposed. The later projector-owner extraction `5e8369be8` retained
the already-existing dtype behavior; it did not introduce this transition.

- [Executable source-history check](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_projector_dtype_history_20260913/check.py)
- [Actual-source result and hashes](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_projector_dtype_history_20260913/result.json)
- [Matched K1 pair plan, job13829869](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_projector_consumer_k1_pair_20260913/plan.json)
- [Existing K4 oracle verification with current loader](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_projector_k4_oracle_check_20260913/result.json)
- [K4 identity and follower mapping: 40,000 exact rows](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_projector_k4_oracle_check_20260913/ordering.json)

Job13829869 is a selected three-iteration fast-tier K1 regression on unmodified
parent37faa and repaired5e, sequentially on one H100, with matched native
startup noise/mt19937 and separate cold caches. It is not full convergence,
the whole fast tier or general speed acceptance. Canonical FSC/state review
is required after termination. Existing K4 dispatch data passed the current
CPU loader; K4 GPU qualification has not been executed for this pair.
The current driver maps captured particle IDs and follower assignments directly;
it does not use the fresh K1 shuffle option for K4. Its actual identity helpers
and schedule lookup reproduce all 5,000 rows across eight iterations. This
does not establish matching physical accumulation order or trajectory quality.
