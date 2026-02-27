# Masks

A real-space mask is important to boost SNR by focusing the analysis on the region of interest.

## Solvent mask

Most consensus reconstruction software outputs a solvent mask. Use it directly:

```bash
recovar pipeline particles.star -o output --mask mask.mrc
```

!!! warning
    Make sure the mask is not too tight. Use `--mask-dilate-iter` to expand it if needed.

### Auto-generated masks

If you don't have a mask:

| Option | Description |
|--------|-------------|
| `--mask=from_halfmaps` | Estimate mask from the two half-maps of the mean reconstruction |
| `--mask=sphere` | Use a loose spherical mask |
| `--mask=none` | No mask (not recommended) |

A good approach is to first run with `--mask=sphere`, inspect the variance map to see which regions have heterogeneity, then create a focused mask around those regions.

## Focus mask

A focus mask restricts the heterogeneity analysis to a specific region of the molecule. This is useful when you're interested in a particular domain or binding site.

```bash
recovar pipeline particles.star -o output \
    --mask mask.mrc --focus-mask focus_mask.mrc
```

If you only have a focus mask:

```bash
recovar pipeline particles.star -o output \
    --mask=sphere --focus-mask focus_mask.mrc
```

### Creating a focus mask

You can create a focus mask in UCSF Chimera or ChimeraX:

1. Open your consensus map
2. Select the region of interest
3. Create a mask around that region
4. Save as `.mrc`

See [cryoSPARC's guide on mask generation](https://guide.cryosparc.com/processing-data/tutorials-and-case-studies/mask-selection-and-generation-in-ucsf-chimera) for step-by-step instructions.

## Mask dilation

To expand a mask by a few pixels:

```bash
recovar pipeline particles.star -o output \
    --mask mask.mrc --mask-dilate-iter 5
```

The `--mask-dilate-iter` flag applies to both the solvent mask and the focus mask.
