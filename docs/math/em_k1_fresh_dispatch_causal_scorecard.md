# K=1 fresh physical-dispatch causal scorecard

This fixed-denominator diagnostic distinguishes completed causal
evaluations from successful treatment rescues. It is non-scoring.
Map gates use signed FSC/FSC-AUC; correlation is forbidden.

Evaluated: **2 / 2**.
Standalone rescues: **0 / 2**.

| Checked | Case | Control final FSC-AUC | Dispatch final FSC-AUC | Delta | Strict rescue | Topology | Conclusion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| [x] | `k1-22` | 0.826070514 | 0.826083923 | +0.000013408 | fail | fail | not supported |
| [x] | `k1-26` | 0.963328057 | 0.963274717 | -0.000053340 | fail | pass | not supported |

Classification: `fresh_dispatch_order_not_supported_as_standalone_fix_in_case22_or_case26`.

Both A/Bs verified the order transform and retained mixed latent
movement, but neither closed the fixed strict FSC gate. Case 22
also retained its topology failure; case 26's final FSC-AUC
decreased. Particle order remains a structural invariant and
possible mediator, not an accepted standalone production fix.

Both A/Bs now explicitly verify the complete treatment physical
order, the ordered first 100 expected-accuracy identities and local
gathers, and the runtime float64 expected-accuracy CTF rows. This
closes the alignment audit, but it does not establish physical-vs-
internal execution equivalence or accept production output
restoration.

The producer Slurm states are preserved exactly: case 22 ended
after a post-science analysis exception and was reanalyzed from
sealed arm outputs; case 26 intentionally exited nonzero after
recording a complete negative causal result.

To validate and regenerate:

```bash
pixi run python scripts/summarize_em_k1_fresh_dispatch_causal_scorecard.py --check
```
