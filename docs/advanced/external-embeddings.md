# External Embeddings

You can use RECOVAR's volume generation (kernel regression) with latent spaces produced by other heterogeneity methods. This is useful for:

- Improving resolution of cryoDRGN or other method's reconstructions
- Validating results from neural network methods (RECOVAR's kernel regression is transparent and produces no hallucinations)
- Combining strengths of different methods

## Usage

```bash
recovar reconstruct_from_external_embedding output \
    -o external_output \
    --external-embedding external_z.txt \
    --latent-points coords.txt
```

### Arguments

| Flag | Description |
|------|-------------|
| `--external-embedding` | Path to external latent coordinates (`.txt`, shape N x zdim) |
| `--latent-points` | Points at which to generate volumes |
| `--Bfactor` | B-factor sharpening |
| `--n-bins` | Bins for kernel regression |
| `--maskrad-fraction` | Kernel radius parameter |

## Example: using cryoDRGN embeddings

1. Run cryoDRGN to get latent coordinates (`z.pkl`)
2. Convert to text: `np.savetxt("z.txt", z)`
3. Run RECOVAR pipeline to get the covariance model
4. Generate volumes using cryoDRGN's latent space:

```bash
recovar reconstruct_from_external_embedding pipeline_output \
    -o cryodrgn_recovar \
    --external-embedding z.txt \
    --latent-points coords.txt --Bfactor=50
```

The resulting volumes use RECOVAR's transparent kernel regression for volume generation but follow cryoDRGN's latent space structure.
