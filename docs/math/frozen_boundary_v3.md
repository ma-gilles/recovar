# Frozen EM boundary v3

`recovar.em.frozen_boundary.v3` is the diagnostic, fail-closed fixed arm
`real10076.k1.physical_it2.reconstructed_projector.v1`. It is not a general
iteration checkpoint, does not claim identity to RELION's full in-memory
physical iteration, and is not a production checkpoint format.

Within that fixed arm, v3 seals the incoming half maps, per-half
tau2/noise/corrections/poses/priors,
the exact captured base sampling arrays and scalars, schedule and convergence
state, CTF parameters, runtime controls, declared command/build/source
provenance, and
all referenced input stack bytes. Completed-iteration STARs are validation
sources. Consumer-iteration optimiser/sampling/data/model STARs and their K=1
map bytes are post-consumer validation sources and never override incoming
state. Captured RELION Iref bytes and transformed RECOVAR complex64 means have
separate hashes and an explicit transform identifier.

The arm is deliberately labelled `reconstructed-projector
boundary`: it rebuilds the RECOVAR projector from the captured Iref-derived
mean. It must not claim exact captured-projector parity. A future exact
projector arm must directly consume the schema-4 captured projector arrays.

## Finalization

Use `scripts/finalize_frozen_boundary_v3.py` with CPU JAX. It upgrades a
validated v2 compact state bundle using a validated schema-4 live capture:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu pixi run python \
  scripts/finalize_frozen_boundary_v3.py \
  --base-v2-dir /absolute/path/to/v2-boundary \
  --capture-manifest /absolute/path/to/iterN_VALIDATED_SHA256SUMS \
  --source-paths-json /absolute/path/to/source_paths.json \
  --runtime-config-json /absolute/path/to/runtime_config.json \
  --consumer-iteration N \
  --output-dir /absolute/new/output/path
```

`source_paths.json` maps semantic names to absolute files. Required fixed
names are exported as `V3_REQUIRED_FIXED_SOURCE_NAMES`; additionally include
contiguous `particle_stack:0...` entries and
`consumer_map:half1:class1`/`consumer_map:half2:class1`. The finalizer verifies
that the numerically ordered stack entries equal the particle STAR's unique
resolved stack paths in lexical order, that consumer model STAR references
equal those sealed map paths, and that every entry in the live-capture
manifest verifies before writing. The RECOVAR source manifest must exactly
name the sealed clean commit.

`runtime_config.json` must exactly cover all `config_*` schema fields. Notable
provenance fields are `declared_relion_command_line`,
`declared_relion_base_git_commit`, `recovar_git_commit`,
`declared_relion_build_id`, `replay_prefix`, and `projector_boundary_kind`
(currently `reconstructed-projector boundary`). The source and runtime
hardware/toolchain identities are not sealed by this arm. Consequently its
scope is explicitly cross-device-unverified, and results from it must not be
classified as same-device equivalence or numerical noise.

Per-half scorer tau2 is an explicit property of this fixed arm. The ordinary
K=1 path retains its historical shared tau2 before and after every update,
including state-swap diagnostics; v3 cannot silently change default EM
behavior.

The runtime requires explicit matching provenance flags plus the environment,
source-tree, and live-capture manifests. External STARs are byte-validated but
cannot supply or overwrite v3 incoming state. The fixed arm reads mask,
maximum-significant-pose, CTF, seed, and ini-high metadata only from its sealed
completed optimiser; explicit alternate optimiser input is rejected.
Intermediate gates use exact
array/posterior diagnostics; map quality gates use shellwise FSC and FSC-AUC,
never correlation.
