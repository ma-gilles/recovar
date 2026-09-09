# Late hidden-variable counter: effect in this InitialModel run

**The differing counter cannot control sampling or termination in the executed fixed200, gradient InitialModel mode. It remains a real strict-state discrepancy.** This conclusion is limited to the counter and this configuration, not an assertion that all accompanying state differences are harmless or numerical noise.

## Observed boundary

The completed analysis records the first integer mismatch in sampling_nr_iter_wo_large_hidden_variable_changes at187: candidate0, native5. Nearby185→189 counters are candidate3,4,0,1,2 and native3,4,5,6,7. The resolution-stall counter is equal6,7,8,9,10; both saved current sizes are74 and sampling_updated=False over those checkpoints. No class differences are inferred from this counter; the parent's separate class audit is authoritative.

Saved native run_it187_optimiser.star explicitly records DoGradientRefine1, DoStochasticGradientDescent1, GradEmIters0, GradHasConverged0, DoAutoRefine0, DoAutoSampling1, HasConverged0. Candidate saved options retain nr_iter200/no diagnostic stop; the completed producer reached its natural200 boundary. This is not auto-refine termination.

## Why the counter changes

Native ml_optimiser.cpp:11195–11216 and RECOVAR driver.py:1518–1559 evaluate the same small-change predicate before updating sticky minima. The offset clause passes when offset_change/effective_translation_step<.40 **or** 1.03×current_offset_change≥previous_smallest_offset_change. The counter increments only when class, offset and orientation clauses all pass.

At187, candidate current offset change is0.4893768643805873Å; its previous minimum from186 is0.5045204426339308Å. Effective translation step is0.6629999999999995Å, so the ratio isabout.738, above.40. The 3% clause margin is **-0.00046227232192586065Å**, belowzero; the counter therefore resets. Native serialized current offset change is.498122Å, previous minimum.503053Å and step.663Å; its 3% clause margin is **0.010012660000000007Å**, abovezero, so that clause passes. Native orientation/class clauses also pass, consistent with4→5. These margins are far larger than one last serialized decimal; no hidden high-precision offset equality is assumed. This validates the observed branch using published values, but does not identify which upstream pose/offset update caused the different inputs.

## Actual control flow

- **Native initialiseGeneral, ml_optimiser.cpp:2741–2755:** every gradient_refine run sets auto_ignore_angle_changes=true. This is not restricted to iterations where do_grad happens to be true.
- **Native updateAngularSampling,11711–11720:** resolution readiness is combined with `(auto_ignore_angle_changes || counter >= threshold)`. The first operand istrue here, so replacing counter0 by5 would not alter this condition. Other inputs—resolution, angular accuracy, offset-change-derived range—can still matter independently.
- **Native iterate,3651–3688:** loop bound is iter≤nr_iter; checkConvergence is called only inside `if(do_auto_refine)`. That flag isfalse in this InitialModel run. Native checkConvergence12063–12085 also bypasses hidden changes under auto_ignore_angle_changes and uses a separate resolution-only gradient convergence predicate, but this routine is not invoked here.
- **RECOVAR driver.py:1118–1129,1253–1274:** InitialModel sampling cadence is separate; _prepare_native_sampling_for_iteration checks resolution stalls and calls the update routine without consulting nr_iter_wo_large_hidden_variable_changes. The counter is maintained/reset/published, not used for decisions on this path.
- **RECOVAR iteration_loop.py:831–853,914–955:** final_iteration comes from state.nr_iter (or explicit diagnostic override, absent here), the fixed loop executes through that bound, and no hidden-variable counter read or counter-driven break exists. The inherited InitialModel state convergence flags are distinct; this monitor does not set them.

The source search covered all mentions of this field in InitialModel and the pinned native optimiser; no second reachable InitialModel reader was found. The native optimiser bytes exactly match the recorded source_before manifest of the protected binary (SHA9ce583a2270f8477cd9f8750de1d6a2966c01eedc6506573ae32d353cedf9c6f). RECOVAR source4f9a194923b084c649c7d9ce929eec7ae9f78902 remains clean.

## Auto-refine and unresolved scope

In ordinary auto-refine without auto_ignore_angle_changes, this counter is load-bearing: it gates angular-sampling refinement and the EM convergence predicate. A continuation imported into such a configuration could consume the saved value. This finding must not be generalized to auto-refine, other schedules or exactlyK4.

The state equality requirement still fails at187. No competing scores, same-input arithmetic replay or float32 error-bound evidence was examined here, so no roundoff/tie waiver is made. Current offset/orientation changes and their minima already differ; those underlying quantities can affect other controller formulas even when this counter is bypassed. The parent's full-map conditions and matching200 finalization remain separate evidence, not a substitute for strict-state closure.

Only this Markdown artifact was written. Read-only CPU/source inspection; no test, source edit, build, GPU or job. Input/source files were hashed; relevant pins:

```json
{
  "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_integrated_full200_4f9a19492_20260909/analysis/results_v1/adaptive_state_cross_engine.json": "352ad294a510f0b8decac9dd626e5065da5521f361ddc6af7d658b0c81091014",
  "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_native_residual_capture_plan_20260909/attempts/v2/source_before.json": "452c5288ea904a3c573a457a741c60af17ea7bd83175c47f535922020ab3ffc1",
  "/scratch/gpfs/GILLES/mg6942/relion/src/ml_optimiser.cpp": "9ce583a2270f8477cd9f8750de1d6a2966c01eedc6506573ae32d353cedf9c6f",
  "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_quality_prefix_integrated_20260909/recovar/em/initial_model/driver.py": "c86d99b568c1ddf95c56979cd73aa98feda8014d1905890812b0d175c785b05a",
  "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_quality_prefix_integrated_20260909/recovar/em/initial_model/iteration_loop.py": "676e3a307d9d04f5b0146771945708162dff8aa20655f02afea9147aab640707",
  "/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_vdam_quality_prefix_integrated_20260909/recovar/em/dense_single_volume/helpers/convergence.py": "a59883e8acc8cd4515b6e165af0178f232910c6b1b0bac39770828157bbfad2b",
  "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_integrated_full200_4f9a19492_20260909/runs/candidate/output/run_native_options.json": "f938346b52d16a9da25ae19cd0414ba627bb5071019b0d1b142a90bccc8e2829",
  "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_integrated_full200_4f9a19492_20260909/runs/candidate/output/run_it185_recovar_meta.json": "443ea179af2872960db11e2812a2bd91c1f89138b340aa3597adb97114cc88ba",
  "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_integrated_full200_4f9a19492_20260909/runs/native/output/run_it185_optimiser.star": "ce5d4170b3b61e8b2de2889f1fdb111d896f4af145101818a6e17d5d325c637f",
  "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_integrated_full200_4f9a19492_20260909/runs/candidate/output/run_it186_recovar_meta.json": "6fe40527ef5051fc9e1aabc83a2d558e72c28cdfdc50dcb9ad80f21f383c11cf",
  "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_integrated_full200_4f9a19492_20260909/runs/native/output/run_it186_optimiser.star": "289ac8e4d6fcdece8bc28d172eef78b09b23527bdcbcb4cf430b59cbed8273e4",
  "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_integrated_full200_4f9a19492_20260909/runs/candidate/output/run_it187_recovar_meta.json": "df3e7cf26c6fe82473a83f1dda92ae0d9f2f2323fbd73080a129fcbecaab14d1",
  "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_integrated_full200_4f9a19492_20260909/runs/native/output/run_it187_optimiser.star": "a9f3405c462aa80b52fbe39066060eed9e511420a9162ab20449c1c76fc815c1",
  "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_integrated_full200_4f9a19492_20260909/runs/candidate/output/run_it188_recovar_meta.json": "fb6f6ffc6de65f760cd3bd26070758e302b81b6017140bdd040cdd7c984a79a1",
  "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_integrated_full200_4f9a19492_20260909/runs/native/output/run_it188_optimiser.star": "f889d131cd2a3326daa4c745cf9b8b7e4403e7c642a518c34cf5829e28cc866e",
  "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_integrated_full200_4f9a19492_20260909/runs/candidate/output/run_it189_recovar_meta.json": "7d365c80473b969b72d5104ae3e14d647e710b1e9cc51800a75cda5a448dfc46",
  "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_integrated_full200_4f9a19492_20260909/runs/native/output/run_it189_optimiser.star": "0fe4a933cb20c99155472bd61c9bbf1e7e963dae8c777178e565063a3a881b63"
}
```
