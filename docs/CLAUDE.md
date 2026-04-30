# Documentation Development Guide

## Vision

The docs should be the **best documentation among cryo-EM heterogeneity tools** (cryoDRGN, RELION multi-body, 3DFlex, etc.). Inspired by **cryoSPARC docs**: clean, GUI-focused, professional, good screenshots.

### Audience
Tool users — structural biologists who want to process their cryo-EM data. Not methods developers.

### First Impression
When a user lands on the homepage, they should immediately see:
- A short feature list: CryoBench resolution, conformational density estimation, cryo-ET support, GUI with sub-particle selection
- Prominent example output images (eigenvalue plots, UMAP scatter, 3D volumes)
- A clear path to get started

### Tone
**Friendly and approachable** — like napari or scikit-learn. Welcoming, explains concepts, encourages exploration. No marketing hype ("revolutionary", "cutting-edge", "state-of-the-art").

### GUI vs CLI
**GUI-first.** The default tab should be GUI with screenshots. CLI is the alternative tab. Most users coming to the docs will prefer the visual approach.

### User Pain Points (prioritize in docs)
1. **Installation/GPU setup** — make this foolproof with clear troubleshooting
2. **Choosing parameters** — explain what zdim, downsample, n-clusters mean and when to change them
3. **Interpreting results** — what do eigenvalues mean? How to pick the right zdim? What's a good FSC?

### Do NOT include
- Marketing language or hype
- Too many screenshots (1-2 per page max, not a gallery)
- Math/equations (link to paper for theory)

## Build & Preview

```bash
pip install mkdocs mkdocs-material mkdocstrings mkdocstrings-python
mkdocs serve    # preview at http://localhost:8000/recovar/
mkdocs build    # build to site/
```

GitHub Pages deploys from the `main` branch automatically.

## Structure

```
docs/
  index.md                    # Homepage — hero, features, example outputs
  getting-started/            # Install tab: installation, quickstart, docker, testing
  guide/                      # Processing tab: tutorials, workflow, advanced topics
  reference/                  # CLI & API tab: CLI commands, file formats, Python API
  troubleshooting.md          # Common issues
  _static/gui/                # Annotated GUI screenshots
  _static/examples/           # Example output images (plots, volumes)
  stylesheets/extra.css       # Custom CSS overrides
```

## Design Principles

1. **GUI-first.** Default tab is always GUI with screenshots. CLI is the alternative.
2. **Goal-oriented.** Lead with what the user wants to achieve, not what the tool does.
3. **Show results early.** Every guide page should show what the output looks like before explaining how to get there.
4. **Help with decisions.** Explain parameter choices (zdim, downsample, n-clusters) with concrete guidance: "For a first run, use zdim=4. For publication, try 10 or 20."
5. **Progressive disclosure.** Essential content first, advanced/optional behind collapsible sections.
6. **1-2 screenshots per page max.** Don't make pages into screenshot galleries. Each screenshot should earn its place.

## Content Tabs Pattern (GUI-first)

```markdown
=== ":material-monitor: GUI"

    ![Screenshot](../../_static/gui/06_new_job_pipeline.png)

    Click **+ New Job** > **Pipeline**, browse to your particles file.

=== ":octicons-terminal-16: CLI"

    ```bash
    recovar pipeline particles.star --mask mask.mrc
    ```
```

Note: GUI tab comes FIRST (it's the default).

## Navigation Hierarchy

Top-level tabs: **Home**, **Install**, **Processing**, **Reference**, **Troubleshooting**

Processing section is grouped into:
- **First Steps** — GUI, Tutorial (essential)
- **Basics** — Input Data, Pipeline, Analysis (the main steps)
- **Advanced** — Masks, Downsampling, Density, Subsets, Outliers, Cryo-ET, External Embeddings

## Admonition Styles

- `!!! tip` — helpful advice, parameter recommendations
- `!!! warning` — things that can go wrong
- `!!! info` — background context
- `!!! example "Choose your workflow"` — CLI/GUI mode selector banner
- `??? note "Advanced"` — collapsible advanced content

## Screenshots

- Capture with Playwright (headless Chromium)
- Annotate with Pillow: colored rectangles + text labels
- Save as PNG in `docs/_static/gui/`
- Max 1-2 per page — make each one count
- Reference: `![Alt text](../../_static/gui/filename.png)`
