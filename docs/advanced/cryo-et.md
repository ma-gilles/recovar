# Cryo-ET / Tomography

RECOVAR supports tilt-series data for cryo-ET heterogeneity analysis. One practical advantage over cryoDRGN-ET and tomodrgn is that a focus mask can be used.

!!! warning "Experimental"
    Cryo-ET support is newer than SPA support and may be less stable. No paper has been published on this feature yet.

## Usage

```bash
recovar pipeline particles.star -o output \
    --mask mask.mrc --tilt-series
```

The input format is the same as cryoDRGN-ET: a STAR file with tilt-series metadata.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--tilt-series` | False | Enable tilt-series mode |
| `--tilt-series-ctf` | Auto | CTF model: `cryoem`, `relion5`, `warp` |
| `--dose-per-tilt` | From file | Dose per tilt in e/A^2 |
| `--angle-per-tilt` | From file | Tilt angle increment |
| `--ntilts` | All | Maximum number of tilts to use |

### CTF models

| Model | Description |
|-------|-------------|
| `cryoem` | Standard cryo-EM CTF (for subtomogram averaging) |
| `relion5` | RELION 5 tilt-series CTF with dose weighting |
| `warp` | Warp-style CTF |

The default is `relion5` for tilt-series data and `cryoem` otherwise.

## With focus mask

A key advantage of RECOVAR for cryo-ET is focus mask support:

```bash
recovar pipeline particles.star -o output \
    --mask mask.mrc --focus-mask binding_site.mrc --tilt-series
```

## Tips

- For cryo-ET data, the `--maskrad-fraction` default (20) may need adjustment. Lower-resolution data may benefit from increasing this value.
- The `--n-min-particles` default (100) may need to be reduced for smaller tomography datasets.
- Use `--ntilts` to limit the number of tilts if some have poor quality.
