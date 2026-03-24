# Changelog

## [Unreleased] (dev branch)

### Added
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
