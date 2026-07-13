# Portable K=1 supplied-map parity smoke

This is a one-iteration clone-and-test entry point for a known RELION K=1
state. It validates and compares the RECOVAR replay with RELION using FSC
curves and normalized FSC-AUC. Correlation is reported only as an auxiliary
diagnostic and is never a pass/fail metric.

The smoke is deliberately narrow. Passing it does **not** prove full-trajectory,
convergence, K=4, quality-matrix, or performance parity.

## Setup

From the repository root:

```bash
pixi install
pixi run install-recovar
```

The replay also needs the RECOVAR RELION bindings. If they are not already
built, point `--relion-src-dir` at the `src/` directory of a compatible RELION
source checkout and build with:

```bash
RELION_SRC_DIR=/path/to/relion/src pixi run python recovar/relion_bind/build.py
```

## Portable fixture convention

`--fixture-dir FIXTURE` resolves:

```text
FIXTURE/
  particles.star
  reference_gt.mrc
  relion/
    run_it003_{half1_model,half2_model,data,sampling,optimiser}.star
    run_it003_{half1,half2}_class001.mrc
    run_it004_{half1_model,half2_model,data,sampling,optimiser}.star
    run_it004_{half1,half2}_class001.mrc
```

Every particle stack referenced by `particles.star` must also exist. Relative
stack paths are resolved from the STAR directory by default; use
`--particle-root` to select another base. Explicit `--data-star`, `--gt-volume`,
and `--relion-dir` override the convention. No dataset or RELION path is
hardcoded in the launcher.

The default RELION prefix is `run`; use `--relion-run-prefix` for a differently
named RELION job. The prefix is propagated through numbered replay state,
unnumbered final state, perturbation seed recovery, and oracle-map lookup.

Validate immediately after setup, without selecting or initializing a GPU:

```bash
pixi run python scripts/run_k1_parity_smoke.py \
  --fixture-dir /path/to/fixture \
  --output-dir /path/to/output \
  --validate-only
```

Run on a conservatively idle local GPU, or submit to Slurm when none is idle:

```bash
pixi run python scripts/run_k1_parity_smoke.py \
  --fixture-dir /path/to/fixture \
  --output-dir /path/to/output \
  --mode auto
```

Use `--mode local` or `--mode slurm` to make the choice explicit. Slurm account
and partition are optional and configurable; the generated script and logs are
stored under `OUTPUT/logs/`. `--dry-run` validates inputs and renders the plan
without running or submitting it.

## Quality contract and outputs

The launcher requires finite GT FSC curves/FSC-AUC for RECOVAR and RELION half
maps and merged maps. By default, RECOVAR's merged FSC-AUC may trail RELION by
at most `1e-4`, matching the existing completion parity tolerance, and the
direct RECOVAR-vs-RELION merged FSC-AUC must be at least `0.995`, the program's
immutable K=1 direct-map gate. These smoke
thresholds are explicit CLI options, not hidden environment settings.

Outputs include:

- `provenance.json`: commit, dirty diff hash, exact command, hostname, input
  hashes, and quality policy.
- `k1_parity_smoke.log`: complete replay output.
- `refinement_results.npz`: raw curves, scores, state, and diagnostics.
- `quality_summary.json`: pass/fail FSC metrics and auxiliary correlations.

The runner creates `SAFE_TO_DELETE` in its output directory. Keep the fixture
and output on storage with enough space for maps and replay intermediates.
