# GUI v2 — Issue Tracker

Check off items as they are fixed. Add new items as they are discovered.

## Bugs

- [ ] **Volume viewer: isosurface controls not discoverable** — Sigma slider and Slice/3D toggle exist but users can't find them. Make 3D view default, make slider more prominent.
- [ ] **Volume viewer: local resolution files shown as volumes** — locres files appear in "other" category. They should be hidden or shown as a shading option, not standalone volumes.
- [ ] **Volume viewer: analyze job volumes poorly organized** — kmeans centers, halfmaps, trajectories all mixed. Need clearer subfolder/category labels.
- [x] **Failed job logs empty after server restart** — Fixed: SLURM log path fallback + REST `/api/jobs/{id}/logs` endpoint
- [x] **Project context lost on direct URL navigation** — Fixed: auto-restore from `project_id` in job response
- [x] **SPA catch-all returns HTML for unmatched /api/* paths** — Fixed: guard returns JSON 404
- [x] **Generic "Job failed" error message** — Fixed: SLURM failure reason detection (TIMEOUT, OOM, NODE_FAIL)

## Missing Features

- [ ] **Interactive diagnostic charts (Plotly)** — PHASE1.md lists Plotly.js as a dependency for diagnostic charts. Currently only showing static PNGs. Consider: are interactive charts actually needed for Phase 1, or are PNGs sufficient?
- [ ] **Compute State/Trajectory from latent explorer clicks** — Backend works. Frontend compute dialog appears on click. Need to verify UX is clear to users.
- [ ] **Lasso selection UX** — Shift+drag works but is not discoverable. Add clearer hint or toolbar toggle.
- [ ] **Subset export confirmation** — After exporting .ind file, show file path and particle count more prominently.
- [ ] **WebSocket reconnect banners** — Spec requires yellow "Reconnecting..." and red "Cannot reach server" banners after 5 min. Not verified.
- [ ] **Queued > 1 hour warning** — Spec requires yellow warning when SLURM job queued > 1 hour.
- [ ] **Disk space < 5 GB red banner** — Spec requires red banner when disk critically low.

## UX Improvements

- [ ] **Volume viewer: default to 3D isosurface view** instead of slice view
- [ ] **Volume viewer: hide halfmaps by default** (already done, but verify toggle is clear)
- [ ] **Analyze volumes: group by subfolder** (kmeans/, trajectories/, etc.)
- [ ] **Related job click should prefill form** — Clicking a child job type in sidebar or overview should navigate with prefilled params (partially works via suggested-next buttons)
- [ ] **Job form: show validation errors inline** — Currently server returns 400 but frontend may not display the error message clearly

## Validated & Working

- [x] Dashboard with project tree, system info, disk usage
- [x] All 8 job type forms render correctly
- [x] Pipeline form validation (submit disabled without particles)
- [x] Server-side validation (particles, mask, result_dir)
- [x] Analyze form pre-fill from suggested next steps
- [x] Volume viewer: slice view + slider
- [x] Volume viewer: 3D vtk.js isosurface rendering
- [x] Latent explorer: PCA + UMAP scatter plots with k-means
- [x] Latent explorer: click k-means center -> Compute State dialog
- [x] Job cancel via API and UI
- [x] Job clone -> pre-filled form
- [x] CLI command display
- [x] Logs tab (WebSocket streaming + REST fallback)
- [x] Plots tab (PNG grid display)
- [x] File browser with allowlist security
- [x] SLURM + local mode auto-detection
- [x] Job submission to SLURM through GUI
