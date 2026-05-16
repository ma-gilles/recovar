# Translation handling in the dense E-step and M-step

## 1. The core computation

For image $i$, rotation $r$, translation $t$, the E-step score is:

$$d_{i,r,t} = \|y_i\|^2_{\Lambda^{-1}} - 2\,\mathrm{Re}\langle S_t w_i,\; p_r \rangle + \langle c_i,\; |p_r|^2 \rangle$$

where $w_i = C_i y_i / \sigma^2$, $p_r = P_r \mu$, $c_i = C_i^2 / \sigma^2$, $S_t$ is the Fourier phase shift.

Only the **cross-term** depends on translation $t$:

$$\text{cross}_{i,r,t} = -2\,\mathrm{Re}\sum_k \overline{w_i(k)}\, p_r(k)\, e^{2\pi i k \cdot t}$$

## 2. What the EM iteration needs

The full $\gamma_{i,r,t}$ tensor is never needed. The EM needs only:

1. $\gamma_{i,r} = \sum_t \gamma_{i,r,t}$ — marginal posterior (for normalization, hard assignment).
2. $a_{i,r} = \sum_t \frac{\gamma_{i,r,t}}{\gamma_{i,r}} S_t^* w_i$ — translation-weighted image (for M-step).
3. $\gamma_{i,r}$ again — for Ft_ctf (no translation dependence).

Hard assignment: best rotation from $\gamma_{i,r}$; best translation recomputed cheaply (one cross-correlation).

## 3. Two approaches

### 3a. GEMM (current, recommended)

Create $n_\text{trans}$ shifted copies of each image, stack, one matmul:

**E-step:**
```
shifted_flat = (n_img · n_trans, N)          ← phase-shifted copies
cross = conj(shifted_flat) @ projections.T   ← ONE GEMM → (n_img·n_trans, n_rot)
```

**M-step:**
```
P = (n_rot, n_img · n_trans)                 ← probability matrix
summed = P @ shifted_flat                    ← ONE GEMM → (n_rot, N)
```

Cost: $O(n_\text{img} \cdot n_\text{trans} \cdot N \cdot n_\text{rot})$ per step.

### 3b. FFT (cross-correlation)

For each $(i, r)$ pair: element-wise product + iFFT gives cross-term at ALL pixel translations:

$$f_{i,r}(t) = -2\,\mathrm{Re}\;\mathcal{F}^{-1}\!\big[\overline{w_i} \odot p_r\big](t)$$

Cost: $O(n_\text{img} \cdot n_\text{rot} \cdot N \log N)$ — independent of $n_\text{trans}$.

## 4. Why GEMM wins for batched rotations

Both approaches compute the same quantity. The difference is **data reuse**.

### Memory traffic

The GEMM reads two matrices and reuses them across all output elements:
- Left matrix (shifted images): shared across all $n_\text{rot}$ columns.
- Right matrix (projections): shared across all $n_\text{img} \cdot n_\text{trans}$ rows.
- **Total read: ~1.5 GB** for 500 images × 5000 rotations × 128² pixels.

The FFT forms a unique $(N,)$ intermediate product for every $(i, r)$ pair:
- $n_\text{img} \times n_\text{rot}$ products, each of size $N$.
- **Total read+write: ~327 GB** — each pair is independent, no sharing.

**200× more memory traffic for the FFT approach.**

### Measured performance (500 images, 5000 rotations, 128×128, n_trans=13)

| Approach | Time | Throughput | Bottleneck |
|---|---|---|---|
| GEMM | 45 ms | 47 TFLOPS | Compute-bound |
| FFT (any sub-batch) | 1500 ms | 0.7 TFLOPS | Memory-bound |

The ratio is **33×** in favor of GEMM. Tile size and sub-batch size have no effect — the total memory traffic is the same regardless of how the work is partitioned.

### This applies equally to E-step and M-step

Both steps have the same structure: an inner product $\sum_k f_i(k) \cdot g_r(k) \cdot \text{phase}(k, t)$ summed over pixels $k$. The GEMM factorizes this as a matrix multiply (reusing $f$ across rotations and $g$ across images). The FFT computes each $(i, r)$ pair independently. The 200× memory traffic gap is identical in both steps.

### Per single rotation: FFT wins

When processing ONE rotation against many images, there is no cross-rotation reuse to exploit:

| $n_\text{img}$ | GEMM (shifts+dot) | FFT (iFFT+pick) |
|---|---|---|
| 500 | 0.82 ms | **0.46 ms** |
| 5000 | 7.35 ms | **3.39 ms** |

FFT is 2× faster per rotation because it avoids creating the $n_\text{trans}$ shifted copies. This is relevant for per-image refinement (few rotations per image), but not for the dense grid (many rotations batched).

### FFT's unique advantage: dense translation marginalization

The FFT evaluates cross-terms at ALL $N$ pixel translations for free. Picking 13 or summing all $N$ costs the same. This enables marginalizing over a continuous translation grid without increasing cost. The GEMM cost grows linearly with $n_\text{trans}$.

## 5. Conclusion

**For the dense grid path** ($n_\text{rot} \gg 1$, $n_\text{trans} \ll N$): use GEMM. The $n_\text{trans}$ inflation in the matrix dimensions is the price of 200× better data reuse.

**For per-image refinement** ($n_\text{rot} = 1$ per image, or dense translations): use FFT.

Both options should be available in the codebase. The GEMM is the default for the dense grid. The FFT path exists at `core.py:33` (`crosscorr_from_ft`) and `m_step.py:18` (`sum_up_translate_one_image`, `translation_fn="fft"`).
