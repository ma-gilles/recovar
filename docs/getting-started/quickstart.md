# Quick Start

## Interactive wizard

The easiest way to set up your first run:

```bash
recovar quickstart
```

The wizard walks you through selecting your input files, mask, downsampling, and other options, then generates and optionally executes the pipeline command. Works over SSH with no extra dependencies.

## Manual quick start

### Recommended: project workflow

```bash
recovar init_project my_project
cd my_project

# RELION
recovar pipeline particles.star --mask mask.mrc --project .

# cryoSPARC
recovar pipeline particles.cs --mask mask.mrc --datadir /path/to/cryosparc/project --project .

# Analyze the latest completed Pipeline job
recovar analyze --zdim=10 --project .
```

In project mode, RECOVAR keeps machine-stable numbered directories on disk (for example `Pipeline/job_0001/`) and stores readable job aliases in project metadata for the CLI and GUI.

### Optional: standalone output directories

If you prefer the older explicit-path style, it still works:

```bash
recovar pipeline particles.star -o output --mask mask.mrc
recovar analyze output --zdim=10
```

### With downsampling

If your images are larger than ~256 pixels, downsample for faster processing:

```bash
recovar pipeline particles.star --mask mask.mrc --downsample 128 --project .
```

This automatically pre-downsamples images into the shared project cache on the first run and reuses that cache across matching project runs.


### View volumes

Open the generated `.mrc` files in ChimeraX, Chimera, or any MRC viewer:

```
output/analysis_10/kmeans/center000.mrc
output/analysis_10/kmeans/center001.mrc
...
```

## Project system

For multi-step workflows, use the project system. It is the standard RECOVAR workflow: numbered job directories stay stable on disk (e.g. `Pipeline/job_0001/`, `Analyze/job_0001/`), while the CLI and GUI show human-readable job names from project metadata.

```bash
# Initialize a project directory
recovar init_project my_project
cd my_project

# Run pipeline (auto-creates Pipeline/job_0001/)
recovar pipeline particles.star --mask mask.mrc --project .

# Analyze the latest completed pipeline (auto-creates Analyze/job_0001/)
recovar analyze --zdim=10 --project .

# Check status of all jobs and aliases
recovar project_status
```

All commands accept `--project <dir>` to enable project mode. If you run from within a project directory, it is auto-detected. Downstream commands can omit `result_dir`; RECOVAR then uses the latest completed Pipeline job in that project.

## Web GUI

For a visual interface, launch the RECOVAR web GUI:

```bash
recovar gui
```

The GUI lets you configure and launch jobs, interactively explore the latent space, and view 3D volumes — all from your browser. See the [GUI Guide](../guide/gui.md) for details.

## What's next

- [Tutorial](../guide/tutorial.md) — full worked example with plots on EMPIAR-10076
- [Web GUI](../guide/gui.md) — browser-based interface for job management and interactive analysis
- [Input Data](../guide/input-data.md) — understand supported formats and data preparation
- [Downsampling](../guide/downsampling.md) — when and how to downsample
- [Running the Pipeline](../guide/pipeline.md) — all pipeline options explained
- [Analyzing Results](../guide/analysis.md) — interpreting and visualizing output
