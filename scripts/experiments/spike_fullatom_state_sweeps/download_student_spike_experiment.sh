#!/usr/bin/env bash
# Set up this RECOVAR branch and build the pixi environment for the spike
# full-atom experiments on Della.
#
# Fresh-start usage:
#   STUDENT_ROOT=/scratch/gpfs/CRYOEM/gilleslab/tmp/$USER/spike_fullatom_student
#   mkdir -p "$STUDENT_ROOT/clone"
#   git clone --branch codex/kernel-bandwidth-student-clean \
#     git@github.com:ma-gilles/recovar.git "$STUDENT_ROOT/clone/recovar"
#   "$STUDENT_ROOT/clone/recovar/scripts/experiments/spike_fullatom_state_sweeps/download_student_spike_experiment.sh" \
#     "$STUDENT_ROOT"

set -euo pipefail

STUDENT_ROOT="${1:-/scratch/gpfs/CRYOEM/gilleslab/tmp/${USER}/spike_fullatom_student}"
REPO_URL="${REPO_URL:-git@github.com:ma-gilles/recovar.git}"
BRANCH="${BRANCH:-codex/kernel-bandwidth-student-clean}"
CHECKOUT="$STUDENT_ROOT/clone/recovar"

# Shared read-only inputs used by the current full-atom spike experiment.
PDB_DIR="${PDB_DIR:-/projects/CRYOEM/singerlab/mg6942/spike_morph_pdbs}"
DEFAULT_MASK="${DEFAULT_MASK:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc}"

mkdir -p "$STUDENT_ROOT"/{clone,slurmo,tmp,pixi_home,rattler_cache,inputs}

if [[ ! -d "$CHECKOUT/.git" ]]; then
  if ! git clone "$REPO_URL" "$CHECKOUT"; then
    echo "SSH clone failed; retrying with HTTPS." >&2
    git clone "https://github.com/ma-gilles/recovar.git" "$CHECKOUT"
  fi
else
  git -C "$CHECKOUT" fetch origin
fi

git -C "$CHECKOUT" checkout "$BRANCH"
git -C "$CHECKOUT" pull --ff-only origin "$BRANCH"

cd "$CHECKOUT"

AGENT_ID="student_setup_$(date +%Y%m%d_%H%M%S)"
export TMPDIR="$STUDENT_ROOT/tmp/$AGENT_ID"
export PIXI_HOME="$STUDENT_ROOT/pixi_home/$AGENT_ID"
export RATTLER_CACHE_DIR="$STUDENT_ROOT/rattler_cache/$AGENT_ID"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1

pixi install
pixi run install-recovar

PIXI_PY="$(pixi run which python)"
"$PIXI_PY" -m pip uninstall -y recovar || true
"$PIXI_PY" -m pip install -e . --no-deps --no-build-isolation --ignore-installed

module load cudatoolkit/12.8 >/dev/null 2>&1 || true
PYTHON="$PIXI_PY" make -C recovar/cuda clean all
module unload cudatoolkit/12.8 >/dev/null 2>&1 || true

pixi run smoke-import-recovar
"$PIXI_PY" - <<'PY'
import pathlib
import jax
import recovar

repo = pathlib.Path.cwd().resolve()
recovar_file = pathlib.Path(recovar.__file__).resolve()
jax_file = pathlib.Path(jax.__file__).resolve()
print("recovar", recovar_file)
print("jax", jax_file)
assert str(recovar_file).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(jax_file)
PY

for path in \
  "$PDB_DIR/morph_001.pdb" \
  "$PDB_DIR/morph_050.pdb" \
  "$PDB_DIR/morph_100.pdb" \
  "$DEFAULT_MASK"; do
  if [[ ! -r "$path" ]]; then
    echo "Missing or unreadable shared input: $path" >&2
    echo "Copy it somewhere readable and override PDB_DIR or DEFAULT_MASK." >&2
    exit 2
  fi
done

cat > "$STUDENT_ROOT/student_spike_env.sh" <<EOF
export RECOVAR_STUDENT_ROOT="$STUDENT_ROOT"
export RECOVAR_CHECKOUT="$CHECKOUT"
export PDB_DIR="$PDB_DIR"
export DEFAULT_MASK="$DEFAULT_MASK"
EOF

echo
echo "Ready."
echo "Checkout: $CHECKOUT"
echo "Env file: $STUDENT_ROOT/student_spike_env.sh"
echo
echo "Next:"
echo "  source $STUDENT_ROOT/student_spike_env.sh"
echo "  $CHECKOUT/scripts/experiments/spike_fullatom_state_sweeps/run_student_spike_experiment.sh smoke"
