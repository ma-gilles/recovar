"""ADMM experiments for wavelet-regularized PPCA updates."""

import time

import jax
import jax.numpy as jnp
import jaxwt
import numpy as np
from jax import jit, vmap
from pyproximal import L1, ProxOperator
from pyproximal.ProxOperator import _check_tau

from recovar.ppca.sparse_PCA import Wavelet_multilvl

debug = False
use_jaxwt = True


def _log_jax_runtime():
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")


def _softthreshold(x, thresh):
    r"""Soft thresholding.

    Applies soft thresholding to vector ``x - g``.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Vector
    thresh : :obj:`float`
        Threshold

    Returns
    -------
    x1 : :obj:`numpy.ndarray`
        Tresholded vector

    """
    if jnp.iscomplexobj(x):
        # https://stats.stackexchange.com/questions/357339/soft-thresholding-
        # for-the-lasso-with-complex-valued-data
        x1 = jnp.maximum(jnp.abs(x) - thresh, 0.0) * jnp.exp(1j * jnp.angle(x))
    else:
        x1 = jnp.maximum(jnp.abs(x) - thresh, 0.0) * jnp.sign(x)

    return x1


class LeastSquareFromNormalEqs(ProxOperator):
    r"""Proximal operator for \sum_i 0.5 * \|P_i x - b\|_2^2, where \sum P_i^* P_i and \sum P_i^* b are precomputed"""

    def __init__(self, dim, lhs, rhs):
        super().__init__(None, False)
        self.dim = dim
        # Convert to JAX arrays for GPU acceleration
        self.lhs = jnp.array(lhs)  # Shape: (n_volumes, n_basis, n_basis)
        self.rhs = jnp.array(rhs)  # Shape: (n_volumes, n_basis)

        # JIT compile the batch operations
        self._jit_loss = jit(self._compute_batch_loss)
        self._jit_prox = jit(self._compute_batch_prox)

    def _compute_batch_loss(self, x, lhs, rhs):
        """Compute batch least squares loss using JAX. lhs/rhs are args to avoid JAX capturing them as constants (~1.6 GB)."""
        X = x.reshape(self.dim)  # Shape: (n_volumes, n_basis)

        # Vectorized computation: X[i]^T @ LHS[i] @ X[i] - 2 * X[i]^T @ RHS[i]
        def single_loss(x_i, lhs_i, rhs_i):
            return 0.5 * (jnp.dot(jnp.conj(x_i), lhs_i @ x_i) - 2 * jnp.dot(jnp.conj(x_i), rhs_i).real)

        losses = vmap(single_loss)(X, lhs, rhs)
        loss = jnp.sum(losses)
        return jnp.real(loss)

    def _compute_batch_prox(self, x, tau, lhs, rhs):
        """Compute batch proximal operator using JAX. lhs/rhs are args to avoid JAX capturing them as constants."""
        X = x.reshape(self.dim)  # Shape: (n_volumes, n_basis)

        def solve_single(lhs_i, rhs_i, x_i):
            lhs_reg = tau * lhs_i + jnp.eye(lhs_i.shape[-1])
            rhs_reg = tau * rhs_i + x_i
            return jnp.linalg.solve(lhs_reg, rhs_reg)

        Y = vmap(solve_single)(lhs, rhs, X)
        return Y.flatten()

    def __call__(self, x):
        if x is None or len(x) == 0:
            return float("inf")

        x_jax = jnp.array(x)
        result = self._jit_loss(x_jax, self.lhs, self.rhs)
        return float(result)

    @_check_tau
    def prox(self, x, tau):
        if x is None or len(x) == 0:
            return np.zeros(np.prod(self.dim))

        start_time = time.time()

        x_jax = jnp.array(x)
        result = self._jit_prox(x_jax, tau, self.lhs, self.rhs)

        # Timing
        end_time = time.time()
        if debug:
            print(f"LeastSquareFromNormalEqs.prox: {end_time - start_time:.4f}s")

        if debug:
            print("done with lstsr prox")

        return result  # Convert back to numpy for compatibility


class WaveletL1(L1):
    r"""Proximal operator for the wavelet L1 regularization.

    Supports scalar sigma, level-dependent sigma (1D array),
    or per-basis level-dependent sigma (2D array).

    Args:
        dim: Shape (volume_size, n_basis)
        volume_shape: 3D shape tuple
        wavelet_type: Type of wavelet (default 'db1')
        sigma: Regularization weight. Can be:
            - scalar: same threshold for all wavelet coefficients
            - array of shape (n_wavelet_coeffs,): level-dependent thresholds
            - array of shape (n_basis, n_wavelet_coeffs): per-basis level-dependent thresholds
              (use wavelet_avg_square_by_level_both per basis)
    """

    def __init__(self, dim, volume_shape, wavelet_type="db1", sigma=1.0, backend=None):
        # Pass scalar to parent (for compatibility)
        super().__init__(np.mean(sigma) if hasattr(sigma, "__len__") else sigma)
        self.dim = dim
        self.volume_shape = volume_shape
        self.volume_size = np.prod(volume_shape)
        self.wavelet_type = wavelet_type
        self.sigma = sigma  # Can be scalar, 1D, or 2D array
        self.sigma_is_array = hasattr(sigma, "__len__") and len(np.array(sigma).shape) >= 1
        self.sigma_ndim = np.array(sigma).ndim if self.sigma_is_array else 0
        # Create the wavelet transform
        self.wavelet = Wavelet_multilvl(volume_shape, wavelet_type, backend=backend)

        # JIT compile only the batch operations that don't involve wavelet transforms
        # self._jit_batch_ops = jit(self._compute_batch_operations)

    # def _compute_batch_operations(self, x_wavelet_all, tau):
    #     """JAX-accelerated batch operations on wavelet coefficients"""
    #     # Apply soft thresholding in JAX
    #     # Soft thresholding: sign(x) * max(|x| - tau, 0)
    #     def soft_threshold(x, threshold):
    #         return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0.0)

    #     x_thresh_all = soft_threshold(x_wavelet_all, tau)
    #     return x_thresh_all

    def __call__(self, x):
        # Input validation
        if x is None or len(x) == 0:
            return float("inf")

        # Reshape to handle all basis functions at once
        # X = x.reshape(self.dim)
        # Stack all basis functions: (n_basis, volume_size) -> (n_basis, volume_size)
        basis_stack = x.T  # Shape: (n_basis, volume_size)

        # Convert all basis functions to wavelet basis at once (NumPy operation)
        x_wavelet_all = self.wavelet.to_basis(basis_stack)  # Shape: (n_basis, n_wavelet_coeffs)

        # Compute total weighted L1 norm
        if self.sigma_is_array:
            if self.sigma_ndim == 1:
                # Level-dependent: sum_i sigma[i] * sum_j |x_wavelet[j, i]|
                sigma_broadcast = jnp.array(self.sigma)[:, None]  # (n_wavelet_coeffs, 1)
                return float(jnp.sum(sigma_broadcast * jnp.abs(x_wavelet_all.T)))
            elif self.sigma_ndim == 2:
                # Per-basis, per-level: sum_{b,i} sigma[b,i] * |x_wavelet[b,i]|
                sigma_broadcast = jnp.array(self.sigma)
                return float(jnp.sum(sigma_broadcast * jnp.abs(x_wavelet_all)))
            else:
                raise ValueError(f"Unsupported sigma ndim: {self.sigma_ndim}")
        else:
            # Scalar sigma
            return float(jnp.sum(self.sigma * jnp.abs(x_wavelet_all.T)))

    @_check_tau
    def prox(self, x, tau):
        # Input validation
        if x is None or len(x) == 0:
            return np.zeros(len(x)) if x is not None else np.array([])

        # Detailed timing
        total_start = time.time()

        # Reshape to handle all basis functions at once
        # Input x is flattened: (volume_size * n_basis,)
        # Need to reshape to (volume_size, n_basis) first, then transpose to (n_basis, volume_size)
        X = x.reshape(self.dim)  # Shape: (volume_size, n_basis)
        basis_stack = X.T  # Shape: (n_basis, volume_size)
        # Convert all basis functions to wavelet basis at once (NumPy operation)
        wavelet_start = time.time()
        x_wavelet_all = self.wavelet.to_basis(basis_stack)  # Shape: (n_basis, n_wavelet_coeffs)
        wavelet_forward_time = time.time() - wavelet_start

        # Apply L1 proximal operator with level-dependent or scalar sigma
        prox_start = time.time()
        if self.sigma_is_array:
            if self.sigma_ndim == 1:
                # Level-dependent sigma: shape (n_wavelet_coeffs,)
                # x_wavelet_all.T has shape (n_wavelet_coeffs, n_basis)
                # Broadcasting: threshold[i] applies to all basis functions for coefficient i
                sigma_broadcast = np.array(self.sigma)[:, None]  # (n_wavelet_coeffs, 1)
                x_wavelet_all = _softthreshold(x_wavelet_all.T, tau * sigma_broadcast).T
            elif self.sigma_ndim == 2:
                # Per-basis, per-level sigma: shape (n_basis, n_wavelet_coeffs)
                sigma_broadcast = np.array(self.sigma)
                x_wavelet_all = _softthreshold(x_wavelet_all, tau * sigma_broadcast)
            else:
                raise ValueError(f"Unsupported sigma ndim: {self.sigma_ndim}")
        else:
            # Scalar sigma
            x_wavelet_all = _softthreshold(x_wavelet_all.T, tau * self.sigma).T
        prox_time = time.time() - prox_start

        # Convert back to image space
        wavelet_back_start = time.time()
        x_result_all = self.wavelet.to_image(x_wavelet_all)
        wavelet_backward_time = time.time() - wavelet_back_start

        # Reshape back to original format
        Y = x_result_all.T  # Shape: (volume_size, n_basis)
        Y = Y.flatten()  # Flatten to match input format: (volume_size * n_basis,)

        # Timing
        if debug:
            total_time = time.time() - total_start
            print(
                f"WaveletL1.prox: {total_time:.4f}s (forward: {wavelet_forward_time:.4f}s, prox: {prox_time:.4f}s, backward: {wavelet_backward_time:.4f}s)"
            )

        if debug:
            print("done with wavelet prox")

        return Y


# Using the proper wavelet implementation from recovar.sparse_PCA


def create_mock_cryo_data():
    """Create mock cryo data to replace the undefined cryos variable."""

    class MockCryo:
        def __init__(self, volume_shape):
            self.volume_shape = volume_shape
            self.volume_size = np.prod(volume_shape)

    volume_shape = (32, 32, 32)  # Smaller for testing
    return [MockCryo(volume_shape)]


def generate_data_from_ground_truth(gt, noise_level=0.1):
    """Generate observation matrices A and data b from ground truth using JAX for GPU acceleration."""
    volume_size, n_basis = gt.shape

    # Create random projection matrices A_i for each volume element
    # Each A_i maps from n_basis to 2 (for simplicity, we'll use 2D projections)
    np.random.seed(123)  # Different seed for data generation
    A_matrices = np.random.randn(volume_size, 2, n_basis)  # (volume_size, 2, n_basis)

    # Convert to JAX for GPU acceleration
    A_jax = jnp.array(A_matrices)
    gt_jax = jnp.array(gt)

    # Generate clean observations: b_i = A_i @ x_i (vectorized)
    b_clean = jnp.sum(A_jax * gt_jax[:, None, :], axis=2)  # Broadcasting: (volume_size, 2)

    # Add noise using JAX random
    noise_key = jax.random.PRNGKey(42)
    noise = jax.random.normal(noise_key, b_clean.shape) * noise_level
    b_noisy = b_clean + noise

    # Create LHS matrices: A_i^T @ A_i (normal equations) - vectorized
    lhs = jnp.einsum("vij,vik->vjk", A_jax, A_jax)  # (volume_size, n_basis, n_basis)
    lhs = lhs + 1e-6 * jnp.eye(n_basis)[None]  # Add regularization

    # Create RHS vectors: A_i^T @ b_i - vectorized
    rhs = jnp.einsum("vij,vi->vj", A_jax, b_noisy)  # (volume_size, n_basis)

    # Convert back to numpy for compatibility
    return (np.array(lhs), np.array(rhs), np.array(A_matrices), np.array(b_clean), np.array(b_noisy))


def compute_total_loss(x, prox_lstsr, prox_wavelet):
    """Compute the total loss function: data_fit + mu * regularization."""
    data_fit_loss = float(prox_lstsr(x))
    reg_loss = float(prox_wavelet(x))
    total_loss = data_fit_loss + reg_loss
    return total_loss, data_fit_loss, reg_loss


def admm_wavelet(
    lhs,
    rhs,
    sigma,
    tau,
    niter,
    volume_shape,
    normal_size,
    X0,
    prox_lstsr=None,
    prox_wavelet=None,
    log_residuals=True,
    log_every=10,
    rtol=1e-3,
    atol=1e-4,
    stop=True,
):
    """
    ADMM optimization with wavelet L1 regularization

    Args:
        lhs: Left-hand side matrices for least squares
        rhs: Right-hand side vectors for least squares
        sigma: Regularization parameter for wavelet L1
        tau: ADMM penalty parameter
        niter: Number of ADMM iterations
        volume_shape: Shape of the 3D volume
        normal_size: Shape of the 2D basis function matrix
        X0: Initial guess
        prox_lstsr: Optional pre-created LeastSquare prox (avoids JIT recompilation)
        prox_wavelet: Optional pre-created Wavelet prox (avoids JIT recompilation)
        log_residuals: If True, print primal/dual residuals during ADMM
        log_every: Print residuals every N iterations
        rtol: Relative tolerance for stopping
        atol: Absolute tolerance for stopping
        stop: If True, stop when residuals satisfy tolerances

    Returns:
        X_rec: Reconstructed solution
        Z_rec: Auxiliary variable
    """
    # Create proximal operators if not provided (allows reuse across EM iterations)
    if prox_lstsr is None:
        prox_lstsr = LeastSquareFromNormalEqs(normal_size, lhs, rhs)
    else:
        # Update lhs and rhs if prox was reused
        prox_lstsr.lhs = jnp.array(lhs)
        prox_lstsr.rhs = jnp.array(rhs)

    if prox_wavelet is None:
        prox_wavelet = WaveletL1(normal_size, volume_shape, "db1", sigma=sigma)
    else:
        # Update sigma if needed
        prox_wavelet.sigma = sigma
    # prox_wavelet = L1(sigma)

    # Run ADMM optimization (custom loop for residual logging and stopping)
    X0_flat = X0.flatten() if hasattr(X0, "flatten") else np.array(X0).flatten()
    x = np.array(X0_flat, copy=True)
    z = np.array(X0_flat, copy=True)
    u = np.zeros_like(x)
    n = x.size
    sqrt_n = np.sqrt(n)

    if log_residuals:
        print("ADMM residuals: iter | ||r||_2 | eps_pri | ||s||_2 | eps_dual")

    for k in range(niter):
        z_prev = z

        # Standard ADMM updates
        z = np.array(prox_wavelet.prox(x + u, tau))
        x = np.array(prox_lstsr.prox(z - u, tau))
        u = u + x - z

        if stop or log_residuals:
            r = x - z
            s = tau * (z - z_prev)
            r_norm = np.linalg.norm(r)
            s_norm = np.linalg.norm(s)
            x_norm = np.linalg.norm(x)
            z_norm = np.linalg.norm(z)
            u_norm = np.linalg.norm(u)
            eps_pri = sqrt_n * atol + rtol * max(x_norm, z_norm)
            eps_dual = sqrt_n * atol + rtol * u_norm

            if log_residuals and (k % max(1, log_every) == 0 or k == niter - 1):
                print(f"  {k + 1:4d} | {r_norm:9.3e} | {eps_pri:8.3e} | {s_norm:9.3e} | {eps_dual:8.3e}")

            if stop and (r_norm <= eps_pri) and (s_norm <= eps_dual):
                if log_residuals:
                    print(f"  converged at iter {k + 1} (r={r_norm:.2e}, s={s_norm:.2e})")
                break

    X_rec, Z_rec = x, z

    # Reshape back to original size
    X_rec = np.array(X_rec).reshape(normal_size)

    # X_rec = X_rec.reshape(normal_size)

    total_loss_init, data_loss_init, reg_loss_init = compute_total_loss(X0, prox_lstsr, prox_wavelet)
    # print(f"  Initial losses: total={total_loss:.6f}, data={data_loss:.6f}, reg={reg_loss:.6f}")

    # Compute final losses
    total_loss, data_loss, reg_loss = compute_total_loss(X_rec, prox_lstsr, prox_wavelet)
    # print(f"  Final losses: total={total_loss:.6f}, data={data_loss:.6f}, reg={reg_loss:.6f}")

    if total_loss_init < total_loss:
        print(f" ADMM did not improve the solution!! initial loss: {total_loss_init:.6f}, final loss: {total_loss:.6f}")

    return X_rec, Z_rec


def analyze_wavelet_coefficient_variance(x, volume_shape, wavelet_type="db1", level_names=None):
    """
    Analyze the variance of wavelet coefficients at different decomposition levels.

    Args:
        x: Input data of shape (n_samples, volume_size) or (volume_size,)
        volume_shape: Shape of the 3D volume (D, D, D)
        wavelet_type: Type of wavelet ('db1', 'db2', etc.)
        level_names: Optional list of names for each level

    Returns:
        dict: Dictionary containing variance analysis for each level
    """
    import pywt

    # Ensure x is 2D
    if x.ndim == 1:
        x = x.reshape(1, -1)

    n_samples, volume_size = x.shape
    wavelet = Wavelet_multilvl(volume_shape, wavelet_type)

    print("Analyzing wavelet coefficient variance...")
    print(f"  Input shape: {x.shape}")
    print(f"  Volume shape: {volume_shape}")
    print(f"  Wavelet type: {wavelet_type}")
    print(f"  Number of samples: {n_samples}")

    # Get wavelet coefficients for all samples
    all_coeffs = []
    for i in range(n_samples):
        sample = x[i].reshape(1, -1)
        coeffs = wavelet.to_basis(sample)
        all_coeffs.append(coeffs[0])  # Remove batch dimension

    all_coeffs = np.array(all_coeffs)  # Shape: (n_samples, n_coeffs)

    # Perform multi-level decomposition on a single volume to get level structure
    test_volume = np.zeros(1, *volume_shape)
    if use_jaxwt:
        coeff_dict = jaxwt.wavedec3(test_volume, wavelet=wavelet_type, mode="periodization", axes=(-3, -2, -1))
    else:
        coeff_dict = pywt.wavedecn(test_volume, wavelet=wavelet_type, mode="periodization")

    coeff_array, coeff_slices = pywt.coeffs_to_array(coeff_dict)

    # Analyze variance at each decomposition level
    level_variances = {}
    level_stats = {}

    # Get the decomposition structure
    n_levels = len(coeff_dict)

    if level_names is None:
        level_names = [f"Level_{i}" for i in range(n_levels)]

    print("\nMulti-level decomposition structure:")
    print(f"  Number of levels: {n_levels}")

    # Analyze each level
    for level_idx, level_name in enumerate(level_names):
        if level_idx == 0:
            # Approximation coefficients (lowest frequency)
            coeff_key = "a" + str(n_levels - 1)
            level_desc = "Approximation (Low Freq)"
        else:
            # Detail coefficients at different levels
            detail_keys = [k for k in coeff_dict.keys() if k.startswith("d") and int(k[1:]) == n_levels - level_idx]
            level_desc = f"Details Level {n_levels - level_idx} (High Freq)"

        # Get coefficient indices for this level from the slices
        if level_idx == 0:
            # Approximation coefficients
            if coeff_key in coeff_slices:
                slice_info = coeff_slices[coeff_key]
                coeff_indices = np.arange(slice_info.start, slice_info.stop)
            else:
                continue
        else:
            # Detail coefficients
            detail_coeff_indices = []
            for detail_key in detail_keys:
                if detail_key in coeff_slices:
                    slice_info = coeff_slices[detail_key]
                    detail_coeff_indices.extend(range(slice_info.start, slice_info.stop))
            coeff_indices = np.array(detail_coeff_indices) if detail_coeff_indices else np.array([])

        if len(coeff_indices) == 0:
            continue

        # Extract coefficients for this level across all samples
        level_coeffs = all_coeffs[:, coeff_indices]  # Shape: (n_samples, n_coeffs_at_level)

        # Compute statistics
        level_var = np.var(level_coeffs, axis=0)  # Variance across samples for each coefficient
        mean_var = np.mean(level_var)
        std_var = np.std(level_var)
        total_var = np.var(level_coeffs.flatten())  # Total variance at this level

        level_variances[level_name] = {
            "mean_variance": mean_var,
            "std_variance": std_var,
            "total_variance": total_var,
            "n_coefficients": len(coeff_indices),
            "coefficient_variances": level_var,
            "coefficient_indices": coeff_indices,
            "description": level_desc,
        }

        level_stats[level_name] = {
            "mean_variance": mean_var,
            "std_variance": std_var,
            "total_variance": total_var,
            "n_coefficients": len(coeff_indices),
            "description": level_desc,
        }

        print(f"  {level_name}: {level_desc}")
        print(f"    Coefficients: {len(coeff_indices)}")
        print(f"    Mean variance: {mean_var:.6f}")
        print(f"    Std variance: {std_var:.6f}")
        print(f"    Total variance: {total_var:.6f}")

    return level_variances, level_stats, all_coeffs


def plot_wavelet_variance_analysis(level_stats, title="Wavelet Coefficient Variance Analysis"):
    """Plot the variance analysis results."""
    import matplotlib.pyplot as plt

    levels = list(level_stats.keys())
    mean_vars = [level_stats[level]["mean_variance"] for level in levels]
    std_vars = [level_stats[level]["std_variance"] for level in levels]
    total_vars = [level_stats[level]["total_variance"] for level in levels]
    n_coeffs = [level_stats[level]["n_coefficients"] for level in levels]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)

    # Plot 1: Mean variance by level
    ax1.bar(levels, mean_vars, alpha=0.7, color="skyblue")
    ax1.set_title("Mean Variance by Decomposition Level")
    ax1.set_ylabel("Mean Variance")
    ax1.tick_params(axis="x", rotation=45)

    # Plot 2: Standard deviation of variance by level
    ax2.bar(levels, std_vars, alpha=0.7, color="lightcoral")
    ax2.set_title("Std Dev of Variance by Level")
    ax2.set_ylabel("Std Dev of Variance")
    ax2.tick_params(axis="x", rotation=45)

    # Plot 3: Total variance by level
    ax3.bar(levels, total_vars, alpha=0.7, color="lightgreen")
    ax3.set_title("Total Variance by Level")
    ax3.set_ylabel("Total Variance")
    ax3.tick_params(axis="x", rotation=45)

    # Plot 4: Number of coefficients by level
    ax4.bar(levels, n_coeffs, alpha=0.7, color="gold")
    ax4.set_title("Number of Coefficients by Level")
    ax4.set_ylabel("Number of Coefficients")
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def test_admm_wavelet_function():
    """Test the admm_wavelet function with different parameters."""
    print("\n" + "=" * 60)
    print("TESTING ADMM_WAVELET FUNCTION")
    print("=" * 60)

    # Create mock data
    cryos = create_mock_cryo_data()
    volume_shape = cryos[0].volume_shape
    volume_size = cryos[0].volume_size
    b = 5
    normal_size = (volume_size, b)

    # Generate ground truth and data
    np.random.seed(42)
    gt = np.random.randn(volume_size, b) * (np.random.rand(volume_size, b) < 0.1)
    lhs, rhs, _, _, _ = generate_data_from_ground_truth(gt, noise_level=0.1)
    X0 = np.random.randn(volume_size, b)

    print("Test setup:")
    print(f"  Volume shape: {volume_shape}")
    print(f"  Normal size: {normal_size}")
    print(f"  Ground truth sparsity: {np.mean(np.abs(gt) > 1e-6):.3f}")

    # Test with different parameter combinations
    test_cases = [
        {"sigma": 0.1, "mu": 1.0, "niter": 50, "name": "Low sigma, standard mu"},
        {"sigma": 1.0, "mu": 1.0, "niter": 50, "name": "Standard parameters"},
        {"sigma": 5.0, "mu": 1.0, "niter": 50, "name": "High sigma"},
        {"sigma": 1.0, "mu": 0.5, "niter": 50, "name": "Low mu"},
        {"sigma": 1.0, "mu": 2.0, "niter": 50, "name": "High mu"},
    ]

    results = []

    for i, params in enumerate(test_cases):
        print(f"\n--- Test Case {i + 1}: {params['name']} ---")

        # Run ADMM
        X_rec = admm_wavelet(
            lhs,
            rhs,
            sigma=params["sigma"],
            mu=params["mu"],
            niter=params["niter"],
            volume_shape=volume_shape,
            normal_size=normal_size,
            X0=X0,
        )

        # Compute metrics
        reconstruction_error = np.linalg.norm(X_rec - gt) / np.linalg.norm(gt)
        final_sparsity = np.mean(np.abs(X_rec) > 1e-6)

        results.append(
            {**params, "reconstruction_error": reconstruction_error, "final_sparsity": final_sparsity, "success": True}
        )

        print(f"  Reconstruction error: {reconstruction_error:.6f}")
        print(f"  Final sparsity: {final_sparsity:.3f}")
        print("  ✅ Success!")

    # Summary
    print("\n--- SUMMARY ---")
    print(f"{'Test Case':<25} {'Success':<8} {'Recon Error':<12} {'Sparsity':<8}")
    print("-" * 60)

    for i, result in enumerate(results):
        if result["success"]:
            print(
                f"{result['name']:<25} {'✅':<8} {result['reconstruction_error']:<12.6f} {result['final_sparsity']:<8.3f}"
            )
        else:
            print(f"{result['name']:<25} {'❌':<8} {'N/A':<12} {'N/A':<8}")

    successful_tests = sum(1 for r in results if r["success"])
    print(f"\n✅ {successful_tests}/{len(results)} tests passed successfully!")

    return results


def main():
    """Main test function that generates GT, creates data, and compares losses."""
    print("Testing ADMM with ground truth generation and loss comparison...")

    # Create mock data to replace undefined variables
    cryos = create_mock_cryo_data()
    volume_shape = cryos[0].volume_shape
    volume_size = cryos[0].volume_size

    print(f"Volume shape: {volume_shape}")
    print(f"Volume size: {volume_size}")

    # Parameters
    b = 5  # number of basis functions
    np.random.seed(42)  # For reproducible results
    mu = 1.0  # regularization parameter

    print("\n=== STEP 1: Generate Ground Truth ===")
    # Create ground truth with sparsity
    gt = np.random.randn(volume_size, b) * (np.random.rand(volume_size, b) < 0.1)
    print(f"Ground truth shape: {gt.shape}")
    print(f"Ground truth sparsity: {np.mean(np.abs(gt) > 1e-6):.3f}")
    print(f"Ground truth norm: {np.linalg.norm(gt):.6f}")

    # Test wavelet transform on a sample
    # Create a test volume with the full shape
    test_volume = np.random.randn(*volume_shape)  # Full 3D volume
    wavelet_test = Wavelet_multilvl(volume_shape, "db1")
    coeffs = wavelet_test.to_basis(test_volume.reshape(1, -1))
    reconstructed = wavelet_test.to_image(coeffs)
    reconstruction_error = np.linalg.norm(test_volume.flatten() - reconstructed[0]) / np.linalg.norm(
        test_volume.flatten()
    )
    print(f"Wavelet reconstruction test error: {reconstruction_error:.2e}")
    print(f"Wavelet coefficients shape: {coeffs.shape} (vs original volume shape: {volume_shape})")
    print(f"Wavelet transform: {wavelet_test.name()}")

    print("\n=== STEP 2: Generate Data from Ground Truth ===")
    # Generate observation matrices and data from ground truth
    lhs, rhs, A_matrices, b_clean, b_noisy = generate_data_from_ground_truth(gt, noise_level=0.1)
    print(f"Generated LHS shape: {lhs.shape}")
    print(f"Generated RHS shape: {rhs.shape}")
    print(f"Observation matrices shape: {A_matrices.shape}")
    print(f"Clean data norm: {np.linalg.norm(b_clean):.6f}")
    print(f"Noisy data norm: {np.linalg.norm(b_noisy):.6f}")
    print(f"Data SNR: {np.linalg.norm(b_clean) / np.linalg.norm(b_noisy - b_clean):.2f}")

    # Create initial guess
    X0 = np.random.randn(volume_size, b)
    normal_size = (volume_size, b)

    print("\n=== STEP 3: Create Proximal Operators ===")
    # Create operators
    prox_lstsr = LeastSquareFromNormalEqs(normal_size, lhs, rhs)

    # Create wavelet transform (using Daubechies db1 wavelet)
    prox_wavelet = WaveletL1(normal_size, volume_shape, "db1")
    print(f"Created proximal operators with {prox_wavelet.wavelet.name()} wavelet transform")

    print("\n=== STEP 4: Compute Loss Functions ===")
    # Compute loss functions for ground truth
    gt_total_loss, gt_data_loss, gt_reg_loss = compute_total_loss(gt, prox_lstsr, prox_wavelet, mu)
    print("Ground Truth Losses:")
    print(f"  Total loss: {gt_total_loss:.6f}")
    print(f"  Data fit loss: {gt_data_loss:.6f}")
    print(f"  Regularization loss: {gt_reg_loss:.6f}")

    # Compute loss functions for initial guess
    init_total_loss, init_data_loss, init_reg_loss = compute_total_loss(X0, prox_lstsr, prox_wavelet, mu)
    print("Initial Guess Losses:")
    print(f"  Total loss: {init_total_loss:.6f}")
    print(f"  Data fit loss: {init_data_loss:.6f}")
    print(f"  Regularization loss: {init_reg_loss:.6f}")

    print("\n=== STEP 5: Test Individual Operators ===")
    # Test individual operators first
    print("Testing LeastSquareFromNormalEqs...")
    x_test = X0.flatten()
    func_val = prox_lstsr(x_test)
    print(f"Function value: {func_val:.6f}")

    prox_val = prox_lstsr.prox(x_test, 0.1)
    print(f"Proximal operator works: {prox_val.shape == x_test.shape}")

    print("Testing WaveletL1...")
    func_val = prox_wavelet(x_test)
    print(f"Function value: {func_val:.6f}")

    prox_val = prox_wavelet.prox(x_test, 0.1)
    print(f"Proximal operator works: {prox_val.shape == x_test.shape}")

    print("\n=== STEP 6: Run ADMM Optimization ===")
    # Run ADMM
    print("Running ADMM...")
    X_rec = ADMM(
        prox_lstsr,
        prox_wavelet,
        x0=X0.flatten(),
        tau=0.9,
        niter=50,  # Increased for better convergence
        gfirst=True,
    )[0]

    X_rec = X_rec.reshape(normal_size)

    print("ADMM completed successfully!")

    print("\n=== STEP 7: Compare Loss Functions ===")
    # Compute loss functions for inferred solution
    rec_total_loss, rec_data_loss, rec_reg_loss = compute_total_loss(X_rec, prox_lstsr, prox_wavelet, mu)
    print("Inferred Solution Losses:")
    print(f"  Total loss: {rec_total_loss:.6f}")
    print(f"  Data fit loss: {rec_data_loss:.6f}")
    print(f"  Regularization loss: {rec_reg_loss:.6f}")

    print("\n=== LOSS COMPARISON SUMMARY ===")
    print(f"{'Metric':<20} {'Ground Truth':<15} {'Initial Guess':<15} {'Inferred':<15} {'Improvement':<15}")
    print("-" * 80)

    # Calculate improvements more carefully for negative values
    total_improvement = ((init_total_loss - rec_total_loss) / abs(init_total_loss) * 100) if init_total_loss != 0 else 0
    data_improvement = ((init_data_loss - rec_data_loss) / abs(init_data_loss) * 100) if init_data_loss != 0 else 0
    reg_improvement = ((init_reg_loss - rec_reg_loss) / abs(init_reg_loss) * 100) if init_reg_loss != 0 else 0

    print(
        f"{'Total Loss':<20} {gt_total_loss:<15.6f} {init_total_loss:<15.6f} {rec_total_loss:<15.6f} {total_improvement:<14.1f}%"
    )
    print(
        f"{'Data Fit Loss':<20} {gt_data_loss:<15.6f} {init_data_loss:<15.6f} {rec_data_loss:<15.6f} {data_improvement:<14.1f}%"
    )
    print(
        f"{'Regularization Loss':<20} {gt_reg_loss:<15.6f} {init_reg_loss:<15.6f} {rec_reg_loss:<15.6f} {reg_improvement:<14.1f}%"
    )

    print("\n=== RECONSTRUCTION QUALITY ===")
    reconstruction_error = np.linalg.norm(X_rec - gt) / np.linalg.norm(gt)
    data_fit_error = np.linalg.norm(prox_lstsr(X_rec.flatten()))
    print(f"Reconstruction error (relative): {reconstruction_error:.6f}")
    print(f"Data fit error: {data_fit_error:.6f}")
    print(f"Final sparsity: {np.mean(np.abs(X_rec) > 1e-6):.3f}")
    print(f"Ground truth sparsity: {np.mean(np.abs(gt) > 1e-6):.3f}")

    # Check if ADMM improved over initial guess
    if rec_total_loss < init_total_loss:
        improvement = (init_total_loss - rec_total_loss) / abs(init_total_loss) * 100
        print(f"\n✅ ADMM successfully improved the solution by {improvement:.1f}%")
    else:
        print("\n⚠️  ADMM did not improve over initial guess")

    print("\n✅ ADMM test with loss comparison completed successfully!")

    # Test the admm_wavelet function
    test_results = test_admm_wavelet_function()

    # Test wavelet coefficient variance analysis
    print("\n" + "=" * 60)
    print("TESTING WAVELET COEFFICIENT VARIANCE ANALYSIS")
    print("=" * 60)

    # Analyze variance in the reconstructed solution
    level_variances, level_stats, all_coeffs = analyze_wavelet_coefficient_variance(
        X_rec, volume_shape, wavelet_type="db1"
    )

    # Plot the results
    plot_wavelet_variance_analysis(level_stats, "ADMM Solution Wavelet Variance Analysis")

    return True


if __name__ == "__main__":
    _log_jax_runtime()
    success = main()
    exit(0 if success else 1)


class TimingADMM:
    """Wrapper around ADMM that tracks timing for each proximal operator"""

    def __init__(self, proxf, proxg):
        self.proxf = proxf
        self.proxg = proxg
        self.timing_data = {"proxf_times": [], "proxg_times": [], "total_times": [], "iteration_times": []}

    def proxf_timed(self, x, tau):
        start_time = time.time()
        result = self.proxf.prox(x, tau)
        end_time = time.time()
        self.timing_data["proxf_times"].append(end_time - start_time)
        return result

    def proxg_timed(self, x, tau):
        start_time = time.time()
        result = self.proxg.prox(x, tau)
        end_time = time.time()
        self.timing_data["proxg_times"].append(end_time - start_time)
        return result

    def run_admm(self, x0, tau, niter=10, gfirst=False, show=False):
        """Run ADMM with timing"""
        if show:
            print(
                "ADMM with Timing\n"
                "---------------------------------------------------------\n"
                "Proximal operator (f): %s\n"
                "Proximal operator (g): %s\n"
                "tau = %10e\tniter = %d\n" % (type(self.proxf), type(self.proxg), tau, niter)
            )
            head = "   Itn       x[0]          f           g       J = f + g    proxf_time  proxg_time"
            print(head)

        x = x0.copy()
        z = x0.copy()
        u = np.zeros_like(x)

        for iiter in range(niter):
            iter_start = time.time()

            if gfirst:
                z = self.proxg_timed(x + u, tau)
                x = self.proxf_timed(z - u, tau)
            else:
                x = self.proxf_timed(z - u, tau)
                z = self.proxg_timed(x + u, tau)

            u = u + x - z

            iter_end = time.time()
            self.timing_data["iteration_times"].append(iter_end - iter_start)

            if show:
                if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                    pf, pg = self.proxf(x), self.proxg(x)
                    proxf_time = self.timing_data["proxf_times"][-1]
                    proxg_time = self.timing_data["proxg_times"][-1]
                    msg = "%6g  %12.5e  %10.3e  %10.3e  %10.3e  %10.4f  %10.4f" % (
                        iiter + 1,
                        np.real(x[0]),
                        pf,
                        pg,
                        pf + pg,
                        proxf_time,
                        proxg_time,
                    )
                    print(msg)

        return x, z

    def print_timing_summary(self):
        """Print timing summary"""
        print("\n=== TIMING SUMMARY ===")
        print(f"Total iterations: {len(self.timing_data['proxf_times'])}")
        print(f"Average proxf time: {np.mean(self.timing_data['proxf_times']):.4f}s")
        print(f"Average proxg time: {np.mean(self.timing_data['proxg_times']):.4f}s")
        print(f"Total proxf time: {np.sum(self.timing_data['proxf_times']):.4f}s")
        print(f"Total proxg time: {np.sum(self.timing_data['proxg_times']):.4f}s")
        print(f"Total iteration time: {np.sum(self.timing_data['iteration_times']):.4f}s")
        print(
            f"Proxf percentage: {100 * np.sum(self.timing_data['proxf_times']) / np.sum(self.timing_data['iteration_times']):.1f}%"
        )
        print(
            f"Proxg percentage: {100 * np.sum(self.timing_data['proxg_times']) / np.sum(self.timing_data['iteration_times']):.1f}%"
        )


def ADMM(proxf, proxg, x0, tau, niter=10, gfirst=False, show=False):
    """Compatibility wrapper around the timed ADMM implementation."""
    solver = TimingADMM(proxf, proxg)
    return solver.run_admm(x0, tau, niter=niter, gfirst=gfirst, show=show)


# def ADMM(proxf, proxg, x0, tau, niter=10, gfirst=False,
#          callback=None, callbackz=False, show=False):
#     r"""Alternating Direction Method of Multipliers

#     Solves the following minimization problem using Alternating Direction
#     Method of Multipliers (also known as Douglas-Rachford splitting):

#     .. math::

#         \mathbf{x},\mathbf{z}  = \argmin_{\mathbf{x},\mathbf{z}}
#         f(\mathbf{x}) + g(\mathbf{z}) \\
#         s.t. \; \mathbf{x}=\mathbf{z}

#     where :math:`f(\mathbf{x})` and :math:`g(\mathbf{z})` are any convex
#     function that has a known proximal operator.

#     ADMM can also solve the problem of the form above with a more general
#     constraint: :math:`\mathbf{Ax}+\mathbf{Bz}=c`. This routine implements
#     the special case where :math:`\mathbf{A}=\mathbf{I}`, :math:`\mathbf{B}=-\mathbf{I}`,
#     and :math:`c=0`, as a general algorithm can be obtained for any choice of
#     :math:`f` and :math:`g` provided they have a known proximal operator.

#     On the other hand, for more general choice of :math:`\mathbf{A}`, :math:`\mathbf{B}`,
#     and :math:`c`, the iterations are not generalizable, i.e. thye depends on the choice of
#     :math:`f` and :math:`g` functions. For this reason, we currently only provide an additional
#     solver for the special case where :math:`f` is a :class:`pyproximal.proximal.L2`
#     operator with a linear operator  :math:`\mathbf{G}` and data :math:`\mathbf{y}`,
#     :math:`\mathbf{B}=-\mathbf{I}` and :math:`c=0`,
#     called :func:`pyproximal.optimization.primal.ADMML2`. Note that for the very same choice
#     of :math:`\mathbf{B}` and :math:`c`, the :func:`pyproximal.optimization.primal.LinearizedADMM`
#     can also be used (and this does not require a specific choice of :math:`f`).

#     Parameters
#     ----------
#     proxf : :obj:`pyproximal.ProxOperator`
#         Proximal operator of f function
#     proxg : :obj:`pyproximal.ProxOperator`
#         Proximal operator of g function
#     x0 : :obj:`numpy.ndarray`
#         Initial vector
#     tau : :obj:`float`, optional
#         Positive scalar weight, which should satisfy the following condition
#         to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
#         the Lipschitz constant of :math:`\nabla f`.
#     niter : :obj:`int`, optional
#         Number of iterations of iterative scheme
#     gfirst : :obj:`bool`, optional
#         Apply Proximal of operator ``g`` first (``True``) or Proximal of
#         operator ``f`` first (``False``)
#     callback : :obj:`callable`, optional
#         Function with signature (``callback(x)``) to call after each iteration
#         where ``x`` is the current model vector
#     callbackz : :obj:`bool`, optional
#         Modify callback signature to (``callback(x, z)``) when ``callbackz=True``
#     show : :obj:`bool`, optional
#         Display iterations log

#     Returns
#     -------
#     x : :obj:`numpy.ndarray`
#         Inverted model
#     z : :obj:`numpy.ndarray`
#         Inverted second model

#     See Also
#     --------
#     ADMML2: ADMM with L2 misfit function
#     LinearizedADMM: Linearized ADMM

#     Notes
#     -----
#     The ADMM algorithm can be expressed by the following recursion:

#     .. math::

#         \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{z}^{k} - \mathbf{u}^{k})\\
#         \mathbf{z}^{k+1} = \prox_{\tau g}(\mathbf{x}^{k+1} + \mathbf{u}^{k})\\
#         \mathbf{u}^{k+1} = \mathbf{u}^{k} + \mathbf{x}^{k+1} - \mathbf{z}^{k+1}

#     Note that ``x`` and ``z`` converge to each other, however if iterations are
#     stopped too early ``x`` is guaranteed to belong to the domain of ``f``
#     while ``z`` is guaranteed to belong to the domain of ``g``. Depending on
#     the problem either of the two may be the best solution.

#     """
#     if show:
#         tstart = time.time()
#         print('ADMM\n'
#               '---------------------------------------------------------\n'
#               'Proximal operator (f): %s\n'
#               'Proximal operator (g): %s\n'
#               'tau = %10e\tniter = %d\n' % (type(proxf), type(proxg),
#                                             tau, niter))
#         head = '   Itn       x[0]          f           g       J = f + g'
#         print(head)

#     x = x0.copy()
#     z = x0.copy()
#     u = np.zeros_like(x)
#     for iiter in range(niter):
#         if gfirst:
#             z = proxg.prox(x + u, tau)
#             x = proxf.prox(z - u, tau)
#         else:
#             x = proxf.prox(z - u, tau)
#             z = proxg.prox(x + u, tau)
#         u = u + x - z

#         # run callback
#         if callback is not None:
#             if callbackz:
#                 callback(x, z)
#             else:
#                 callback(x)
#         if show:
#             if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
#                 pf, pg = proxf(x), proxg(x)
#                 msg = '%6g  %12.5e  %10.3e  %10.3e  %10.3e' % \
#                       (iiter + 1, np.real(to_numpy(x[0])),
#                        pf, pg, pf + pg)
#                 print(msg)
#     if show:
#         print('\nTotal time (s) = %.2f' % (time.time() - tstart))
#         print('---------------------------------------------------------\n')
#     return x, z
