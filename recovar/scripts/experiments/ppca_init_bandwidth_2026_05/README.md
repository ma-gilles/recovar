# PPCA init / W-bandwidth experiment (May 2026)

This directory archives the scripts and findings of a one-off
experiment that explored whether band-limiting the principal-component
loadings (``W``) below the resolution of the mean volume (``μ``) helps
PPCA refinement bootstrap from a coarse RELION K-class ab-initio
reconstruction.

## TL;DR — the experiment failed

Band-limiting ``W`` coarser than ``μ`` **destroys class separation**.
Concretely, on ribo_n01 (Ribosembly K=4, 20k images, noise=0.1)
and igg_n01 (IgG-1D continuous, same shape):

| Dataset | Arm | μCC | W ⟨cos⟩ | **discrim** | iter-5 med pose |
|---|---|---|---|---|---|
| ribo | baseline (no μ low-pass) | 0.760 | 0.186 | **0.910** | 7.9 ° |
| ribo | lpmu_R24 only | 0.744 | 0.089 | 0.205 | 133 ° |
| ribo | lpmu_R24 + Wlim_R12 | 0.714 | 0.250 | **0.000** | 142 ° |
| igg | baseline (no μ low-pass) | 0.760 | 0.269 | **0.760** | 3.7 ° |
| igg | lpmu_R24 only | 0.269 | 0.085 | 0.137 | 131 ° |
| igg | lpmu_R24 + Wlim_R12 | 0.267 | 0.000 | **0.000** | 125 ° |

* ``lpmu_R24``: ``μ_init`` low-passed at frequency-grid radius 24
  (≈ 22 Å on Ribo, ≈ 16 Å on IgG).
* ``Wlim_R12``: every M-step output is low-passed at radius 12 so
  ``W`` is coarser than ``μ``.

See ``SUMMARY.md`` and ``W_bandlimit_comparison.png`` for the
per-iter trajectories.

**Mechanism**: applying a hard Fourier low-pass to ``W`` after each
M-step strips the W shells that PPCA had just learned. From iter 2
onward ``W ≡ 0``, ``Pmax → 0``, ``E[z] / GT → NaN``. The companion
finding — that GT-μ low-pass alone (``lpmu_R24``) stalls PPCA pose
recovery at the random initialisation — also rules out coarse-μ
bootstraps as a near-term path for PPCA refinement.

This experiment fed no library-side improvements other than the
``make_localized_moving_gt_mask`` helper, which graduated to
``recovar.core.mask`` on a separate branch.

## Scripts

| Script | Purpose |
|---|---|
| ``simulate_dataset.py``  | Wraps ``recovar.simulation.simulator.generate_synthetic_dataset`` for the experiment-fixture parameters (Ribo K=4 / IgG continuous, 20k images, noise=0.1, box 128, voxel auto). |
| ``build_init_npz.py``    | Reads a RELION ``--grad --denovo_3dref`` run and builds an init NPZ for PPCA: ``μ = anchor or weighted mean``, ``W = SVD basis of (K vols − μ)`` zero-padded to ``q``. No GT alignment. |
| ``run_ppca_trajectory.py`` | The runner with the ``--w-bandlimit-R`` and ``--gt-mu-lowpass-R`` knobs that drive this experiment, plus the per-iter ``W_shell_diagnostic.png`` plot and ``trajectory.json`` capture. |
| ``evaluate_state.py``    | Per-cell eval: μCC, W ⟨cos⟩, embedding scatter at GT poses, ``prior_shell_ratio`` diagnostic. Writes ``eval/metrics.json`` + ``eval/embedding_grid.png``. |

Each script has its own ``--help``. The driving flags for the
bandwidth-experiment cells are:

```text
# Baseline cell (no μ low-pass, free W)
python run_ppca_trajectory.py --init gt_mu_random_w --q 4 --n-iters 5 --healpix-order 3 \
    --dataset-dir <DATASET> --gt-vol-glob '<DATASET>/../pdb_volumes/vol*.mrc' \
    --out-dir runs/baseline --label baseline --prior-source init

# μ low-pass arm
python run_ppca_trajectory.py ... --gt-mu-lowpass-R 24 --out-dir runs/lpmu_R24 ...

# μ low-pass + W bandlimit arm
python run_ppca_trajectory.py ... --gt-mu-lowpass-R 24 --w-bandlimit-R 12 --out-dir runs/lpmu_R24_Wlim_R12 ...
```

## Hardcoded paths

Each Python script has ``--workdir`` defaulting to the experiment
worktree

```text
/scratch/gpfs/GILLES/mg6942/recovar_wt_ppca_postmerge_followup_20260510_110827
```

so they can be invoked without ``pixi run`` overhead from the
companion Slurm wrappers. Pass ``--workdir <your_repo_path>`` to use
them from any other worktree.

## Related artefacts not committed here

* Bootstraps ``bootstraps/{ribo,igg}_relionK{1,4}`` and synthetic
  datasets ``datasets/{ribo,igg}_n01`` lived in
  ``/scratch/gpfs/GILLES/mg6942/ppca_init_2026_05_11/`` at the time
  the experiment ran (≈ 44 GB total, regeneratable from these scripts).
* The Slurm fan-out scripts (``slurm_*.sh``, ``submit_*.sh``) and
  workspace-specific helper scripts were intentionally left in the
  workspace; only the load-bearing Python and the conclusion sit
  here.

## Memory cross-references

* ``[[project_initialmodel_session_2026_05_09_summary]]`` —
  preceding InitialModel parity work that fed this experiment.
* ``[[feedback_dataset_simulator_defaults]]`` — simulator
  ``disc_type="cubic"`` and ``contrast_std=0.0`` discipline used by
  this experiment.
* ``[[project_relion_normalize_required_2026_05_11]]`` — a
  side-finding from the same workspace: ``image_dtype=np.float32``
  + ``relion_normalize=True`` are required for RELION ab-initio to
  read recovar simulator output correctly.
