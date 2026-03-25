# Changelog

## [Unreleased] (dev branch)

### Added
- Project directory system: `recovar init_project`, `recovar project_status`, `--project` flag for auto-numbered job directories
- Job metadata: every job creates `job.json`, `command.txt`, `run.log`, `README.txt`
- Per-zdim embedding directories (`model/zdim_4/latent_coords.npy`) instead of monolithic `embeddings.pkl`
- Organized output layout: `plots/`, `data/`, `kmeans/`, `traj000/` subdirectories in analyze output
- Per-volume diagnostics in `diagnostics/{stem}/` subdirectories
- `--save-all-plots` flag for junk/outlier detection commands
- CLAUDE.md infrastructure for AI-assisted development
- CI pipeline: ruff lint/format + unit tests on every push/PR
- `@claude` GitHub integration for PR review and issue handling
- Pre-commit hooks (ruff format + lint)
- CONTRIBUTING.md, REVIEW.md, issue/PR templates
- Claude Code hooks: baseline protection, tolerance guard, auto-formatting
- Git worktree-based development workflow
- Performance regression tracking (per-GPU baselines)

### Changed
- Repository migrated from `heterogeneity_dev` to `recovar`
- Documentation cleaned up for public use (no hardcoded paths)
- Development branch is `dev` (not `main`)
- Renamed job types: `ComputeState` -> `ReconstructState`, `ComputeTrajectory` -> `ReconstructTrajectory` (directory names in project mode)
- Volume naming: `center000.mrc`/`state000.mrc` (zero-padded, with half-maps alongside)
- K-means results saved to `data/kmeans_result.pkl` (was `centers.pkl`)
- Removed `filtered_noB.mrc` from volume outputs

### Fixed
- CUDA coordinate swap in `_rot_to_compact` (commit ba75416)
- Half-image catastrophic cancellation in `compute_projected_covariance`

## [0.4.5] - 2025

Initial public release on PyPI.

- Regularized covariance estimation and kernel regression
- Cryo-EM and cryo-ET support
- Focus mask support
- Outlier detection
- Web GUI
- Docker/Singularity container support
