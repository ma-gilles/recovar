# External Embeddings

You can use RECOVAR's volume generation (kernel regression) with latent spaces produced by other heterogeneity methods. This is useful for:

- Improving resolution of cryoDRGN or other method's reconstructions
- Validating results from neural network methods (RECOVAR's kernel regression is transparent and produces no hallucinations)
- Combining strengths of different methods

## Usage

```bash
recovar reconstruct_from_external_embedding particles.star \
    --poses poses.pkl --ctf ctf.pkl \
    -o external_output \
    --embedding z.pkl \
    --target coords.txt
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `particles` | Required | Input particles (`.mrcs`, `.star`, `.cs`, or `.txt`) |
| `--poses` | Required | Poses file (`.pkl`) |
| `--ctf` | Required | CTF parameters (`.pkl`) |
| `-o`, `--outdir` | Required | Output directory |
| `--embedding` | Required | External latent coordinates (`.pkl`, shape N x zdim) |
| `--target` | Required | Points at which to generate volumes (`.txt`) |
| `--Bfactor` | 0 | B-factor sharpening |
| `--n-bins` | 50 | Bins for kernel regression |
| `--zdim1` | False | Enable for 1D latent space |
| `--tilt-series` | False | Use tilt-series data |

!!! note "Poses and CTF must be `.pkl` files"
    Unlike `recovar pipeline`, this command does **not** auto-extract poses and CTF from a `.star`/`.cs` file. Even when `particles` is a `.star` or `.cs`, you must pass `--poses` and `--ctf` as separate `.pkl` files (for example, the `poses.pkl` / `ctf.pkl` that cryoDRGN writes during preprocessing).

## Example: using cryoDRGN embeddings

1. Run cryoDRGN to get latent coordinates (`z.pkl`)
2. Pick target points (e.g., k-means centers): `np.savetxt("coords.txt", centers)`
3. Generate volumes using cryoDRGN's latent space with RECOVAR's kernel regression:

```bash
recovar reconstruct_from_external_embedding particles.mrcs \
    --poses poses.pkl --ctf ctf.pkl \
    -o cryodrgn_recovar \
    --embedding z.24.pkl \
    --target coords.txt --Bfactor=50
```

The resulting volumes use RECOVAR's transparent kernel regression for volume generation but follow cryoDRGN's latent space structure.
