"""Ground-truth controls for spike full-atom diagnostics.

These helpers deliberately recompute GT latent coordinates and latent-space
distances from the current pipeline output.  Do not read cached files such as
``gtpc_iid_zdim*_true_by_state.npy`` here; those are scratch artifacts and can
be in different coordinate systems.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.output import output as output_mod
from recovar.simulation import synthetic_dataset


@dataclass(frozen=True)
class ComputeStateInputs:
    pipeline: Path
    target_path: Path
    target: np.ndarray
    zdim: int
    coords_entry: str
    precision_entry: str
    embedding_option: str


def raw_gt_pc_coordinates(run_dir: Path, pipeline_dir: Path, zdim: int) -> np.ndarray:
    """Return ``U^H (V_state - mean)`` in the pipeline's saved PC basis."""
    run_dir = Path(run_dir)
    pipeline_dir = Path(pipeline_dir)
    sim_info = utils.pickle_load(run_dir / "03_dataset" / "simulation_info.pkl")
    hvd = synthetic_dataset.load_heterogeneous_reconstruction(sim_info)
    pipeline_output = output_mod.PipelineOutput(str(pipeline_dir))
    volume_shape = tuple(int(v) for v in pipeline_output.params["volume_shape"])
    mean_real = utils.load_mrc(pipeline_dir / "output" / "volumes" / "mean.mrc")
    mean_ft = np.asarray(ftu.get_dft3(mean_real.reshape(volume_shape)), dtype=np.complex64).reshape(-1)
    u = np.asarray(pipeline_output.get_u(n_pcs=zdim), dtype=np.complex64)
    centered = np.asarray(hvd.volumes, dtype=np.complex64) - mean_ft[None, :]
    coords = np.empty((centered.shape[0], zdim), dtype=np.float64)
    for k in range(zdim):
        coords[:, k] = np.asarray(centered @ np.conj(u[k]), dtype=np.complex128).real
    return coords


def parse_compute_state_inputs(compute_root: Path) -> ComputeStateInputs:
    """Parse the actual compute_state command used for an output directory."""
    compute_root = Path(compute_root)
    command_path = compute_root / "command.txt"
    if not command_path.exists():
        raise FileNotFoundError(f"Missing compute_state command file: {command_path}")
    tokens = shlex.split(command_path.read_text().strip().splitlines()[0])
    try:
        script_idx = next(i for i, token in enumerate(tokens) if token.endswith("compute_state.py"))
    except StopIteration as exc:
        raise ValueError(f"Could not locate compute_state.py in {command_path}") from exc
    if script_idx + 1 >= len(tokens):
        raise ValueError(f"Missing pipeline argument in {command_path}")
    pipeline = Path(tokens[script_idx + 1])

    def option_value(name: str, default: str | None = None) -> str | None:
        if name not in tokens:
            return default
        idx = tokens.index(name)
        if idx + 1 >= len(tokens):
            raise ValueError(f"{name} has no value in {command_path}")
        return tokens[idx + 1]

    target_value = option_value("--latent-points")
    if target_value is None:
        raise ValueError(f"Missing --latent-points in {command_path}")
    target_path = Path(target_value)
    target = np.asarray(np.loadtxt(target_path), dtype=np.float64).reshape(-1)
    no_z_regularization = "--no-z-regularization" in tokens
    coords_entry = "latent_coords_noreg" if no_z_regularization else "latent_coords"
    precision_entry = "latent_precision_noreg" if no_z_regularization else "latent_precision"
    embedding_option = str(option_value("--embedding-option", "cov_dist"))
    return ComputeStateInputs(
        pipeline=pipeline,
        target_path=target_path,
        target=target,
        zdim=int(target.size),
        coords_entry=coords_entry,
        precision_entry=precision_entry,
        embedding_option=embedding_option,
    )


def _load_unsorted_embedding(pipeline_dir: Path, entry: str, zdim: int) -> np.ndarray:
    pipeline_output = output_mod.PipelineOutput(str(pipeline_dir))
    return np.asarray(pipeline_output.get_unsorted_embedding_component(entry, zdim), dtype=np.float64)


def recompute_latent_distances_from_compute_state(compute_root: Path) -> tuple[np.ndarray, ComputeStateInputs]:
    """Recompute per-image heterogeneity distances in original dataset order."""
    info = parse_compute_state_inputs(compute_root)
    z = _load_unsorted_embedding(info.pipeline, info.coords_entry, info.zdim)
    z = z[:, : info.zdim]
    diff = z - info.target[None, :]

    if info.embedding_option == "dist":
        distances = 0.5 * np.sum(diff * diff, axis=1)
    else:
        precision = _load_unsorted_embedding(info.pipeline, info.precision_entry, info.zdim)
        precision = precision[:, : info.zdim, : info.zdim]
        quad = np.einsum("ni,nij,nj->n", diff, precision, diff, optimize=True)
        if info.embedding_option == "cov_dist":
            distances = quad
        elif info.embedding_option == "llh":
            sign, logdet = np.linalg.slogdet(precision)
            if not np.all(sign > 0):
                raise ValueError(f"Non-positive precision determinant while recomputing {compute_root}")
            distances = 0.5 * (quad - logdet)
            distances = distances - np.nanmin(distances)
        else:
            raise ValueError(f"Unsupported embedding_option={info.embedding_option!r}")

    return np.asarray(distances, dtype=np.float64), info


def state_weights_from_nearest_distances(
    run_dir: Path,
    distances: np.ndarray,
    n_nearest: int,
    *,
    n_states: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(np.load(Path(run_dir) / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)
    if distances.shape[0] != states.size:
        raise ValueError(f"distance/state length mismatch: {distances.shape[0]} vs {states.size}")
    n = min(int(n_nearest), distances.size)
    nearest = np.argpartition(distances, n - 1)[:n] if n < distances.size else np.arange(distances.size)
    counts = np.bincount(states[nearest], minlength=n_states).astype(np.float64)
    weights = counts / counts.sum()
    return weights, nearest


def state_weights_from_compute_state_nearest(
    run_dir: Path,
    compute_root: Path,
    n_nearest: int,
    *,
    n_states: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ComputeStateInputs]:
    distances, info = recompute_latent_distances_from_compute_state(compute_root)
    weights, nearest = state_weights_from_nearest_distances(run_dir, distances, n_nearest, n_states=n_states)
    return weights, nearest, distances, info
