"""
Per-iteration trajectory bookkeeping for ``refine_single_volume``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def add_significant_count_artifacts(save_dict, significant_counts, half_indices, n_images):
    """Save significant-count history in both half and original image order."""
    half_order_indices = np.concatenate(
        [np.asarray(indices, dtype=np.int64) for indices in half_indices],
    )
    for iteration, counts in enumerate(significant_counts):
        if counts is None:
            continue
        counts_half_order = np.asarray(counts)
        save_dict[f"sig_counts_half_order_iter_{iteration:03d}"] = counts_half_order
        flat_counts = counts_half_order.reshape(-1)
        if flat_counts.shape[0] != half_order_indices.shape[0]:
            continue
        counts_by_image = np.full(int(n_images), -1, dtype=flat_counts.dtype)
        counts_by_image[half_order_indices] = flat_counts
        save_dict[f"sig_counts_by_image_iter_{iteration:03d}"] = counts_by_image



@dataclass(frozen=True)
class RefinementHistory:
    """Accumulates per-iteration trajectories for one RELION refinement run."""

    current_sizes: list = field(default_factory=list)
    state_swap_probe_applied_relion_iterations: list = field(default_factory=list)
    fsc_history: list = field(default_factory=list)
    fsc_for_growth_history: list = field(default_factory=list)
    pixel_resolutions: list = field(default_factory=list)
    wall_times: list = field(default_factory=list)
    significant_counts: list = field(default_factory=list)
    data_vs_prior_trajectory: list = field(default_factory=list)
    healpix_order_trajectory: list = field(default_factory=list)
    ave_Pmax_trajectory: list = field(default_factory=list)
    ave_Pmax_denominator_trajectory: list = field(default_factory=list)
    pmax_per_image_history: list = field(default_factory=list)
    noise_radial_trajectory: list = field(default_factory=list)
    noise_radial_per_half_trajectory: list = field(default_factory=list)
    tau2_radial_trajectory: list = field(default_factory=list)
    tau2_sigma2_trajectory: list = field(default_factory=list)
    tau2_avg_weight_trajectory: list = field(default_factory=list)
    tau2_shell_sum_trajectory: list = field(default_factory=list)
    tau2_shell_count_trajectory: list = field(default_factory=list)
    tau2_fsc_used_trajectory: list = field(default_factory=list)
    tau2_ssnr_trajectory: list = field(default_factory=list)
    sigma_offset_used_trajectory: list = field(default_factory=list)
    sigma_offset_used_per_half_trajectory: list = field(default_factory=list)
    sigma_offset_trajectory: list = field(default_factory=list)
    sigma_offset_per_half_trajectory: list = field(default_factory=list)
    per_class_sigma_offset_trajectory: list = field(default_factory=list)
    direction_prior_trajectory_per_half: list = field(default_factory=list)
    rotation_posterior_trajectory_per_half: list = field(default_factory=list)
    frac_changed_trajectory: list = field(default_factory=list)
    acc_rot_trajectory: list = field(default_factory=list)
    acc_trans_trajectory: list = field(default_factory=list)
    acc_rot_per_class_trajectory: list = field(default_factory=list)
    acc_trans_per_class_trajectory: list = field(default_factory=list)
    expected_accuracy_class_counts_trajectory: list = field(default_factory=list)
    expected_accuracy_status_trajectory: list = field(default_factory=list)
    smallest_change_angles_trajectory: list = field(default_factory=list)
    smallest_change_offsets_trajectory: list = field(default_factory=list)
    best_rotation_eulers_history: list = field(default_factory=list)
    best_translations_history: list = field(default_factory=list)
    class_mstep_weight_trajectory: list = field(default_factory=list)
    class_full_posterior_weight_trajectory: list = field(default_factory=list)
    class_assignment_history: list = field(default_factory=list)
    local_profile_history: list = field(default_factory=list)
    global_profile_history: list = field(default_factory=list)
    relion_follower_owners_half1_trajectory: list = field(default_factory=list)
    relion_scale_follower_scales_numbered_pre_score_trajectory: list = field(default_factory=list)
    relion_scale_follower_scales_numbered_post_mstep_trajectory: list = field(default_factory=list)
    relion_follower_scale_replay_applied_iterations: list = field(default_factory=list)

    # -- scheduling / sampling grid ------------------------------------

    def record_scheduling(self, cs, healpix_order, sigma_offset_used, sigma_offset_used_per_half) -> None:
        self.current_sizes.append(cs)
        self.healpix_order_trajectory.append(healpix_order)
        self.sigma_offset_used_trajectory.append(sigma_offset_used)
        self.sigma_offset_used_per_half_trajectory.append(sigma_offset_used_per_half)

    # -- RELION follower-scale replay/dispatch bookkeeping -------------

    def record_follower_scale_pre_score(self, pre_score_scales, owners_half1) -> None:
        self.relion_scale_follower_scales_numbered_pre_score_trajectory.append(pre_score_scales)
        self.relion_follower_owners_half1_trajectory.append(owners_half1)

    # -- per-iteration E-step / M-step outputs --------------------------

    def record_direction_prior(
        self, class_direction_prior_per_half, global_direction_prior_per_half, *, k_class_enabled: bool
    ) -> None:
        """Record float64 copies of each half's learned direction prior.

        K-class runs record class 0 of each half's ``(n_classes, n_pixels)``
        prior; K=1 records the half's global prior. Missing priors stay ``None``.
        """
        source = class_direction_prior_per_half if k_class_enabled else global_direction_prior_per_half
        self.direction_prior_trajectory_per_half.append(
            [
                None
                if prior_k is None
                else np.asarray(prior_k[0] if k_class_enabled else prior_k, dtype=np.float64).copy()
                for prior_k in source
            ]
        )

    def record_rotation_posterior(self, rotation_posterior_per_half) -> None:
        """Record float64 copies of the pre-collapse orientation posterior.

        Kept separate from ``record_direction_prior`` so a direction-prior
        mismatch can be localized to posterior aggregation versus collapse.
        """
        self.rotation_posterior_trajectory_per_half.append(
            [None if value is None else np.asarray(value, dtype=np.float64).copy() for value in rotation_posterior_per_half]
        )

    def record_fsc(self, fsc, fsc_for_growth) -> None:
        self.fsc_history.append(fsc)
        self.fsc_for_growth_history.append(fsc_for_growth)

    def record_pmax(self, ave_pmax, ave_pmax_denominator, per_image_pmax) -> None:
        self.ave_Pmax_trajectory.append(ave_pmax)
        self.ave_Pmax_denominator_trajectory.append(ave_pmax_denominator)
        self.pmax_per_image_history.append(per_image_pmax)

    def record_class_weights(self, mstep_weights, posterior_weights) -> None:
        """Snapshot the M-step and full-posterior class-weight definitions."""
        self.class_mstep_weight_trajectory.append(mstep_weights.copy())
        self.class_full_posterior_weight_trajectory.append(posterior_weights.copy())

    def record_pose_history(self, euler_snapshot_per_half, translation_snapshot_per_half) -> None:
        self.best_rotation_eulers_history.append(euler_snapshot_per_half)
        self.best_translations_history.append(translation_snapshot_per_half)

    def record_noise_and_tau2(self, noise_radial, noise_radial_per_half, tau2_details, *, k_class_enabled: bool) -> None:
        """Format shell diagnostics before appending a complete iteration.

        Float64 input shell arrays remain shared; stacking the halves creates
        a fresh array. Prepare every field before mutating history so malformed
        input cannot leave the trajectory lists at different lengths.
        """
        noise_radial = np.asarray(noise_radial, dtype=np.float64)
        noise_radial_per_half = np.stack(
            [np.asarray(noise_k, dtype=np.float64) for noise_k in noise_radial_per_half],
            axis=0,
        )
        tau2_details = (
            {}
            if tau2_details is None
            else {
                "prior_shells": np.asarray(tau2_details["prior_shells"], dtype=np.float64),
                "sigma2_shells": np.asarray(tau2_details["sigma2_shells"], dtype=np.float64),
                "avg_weight_shells": np.asarray(tau2_details["avg_weight_shells"], dtype=np.float64),
                "shell_sum": np.asarray(tau2_details["shell_sum"], dtype=np.float64),
                "shell_count": np.asarray(tau2_details["shell_count"], dtype=np.float64),
                "fsc_shells": None if k_class_enabled else np.asarray(tau2_details["fsc_shells"], dtype=np.float64),
                "ssnr_shells": np.asarray(tau2_details["ssnr_shells"], dtype=np.float64),
            }
        )
        self.noise_radial_trajectory.append(noise_radial)
        self.noise_radial_per_half_trajectory.append(noise_radial_per_half)
        self.tau2_radial_trajectory.append(tau2_details.get("prior_shells"))
        self.tau2_sigma2_trajectory.append(tau2_details.get("sigma2_shells"))
        self.tau2_avg_weight_trajectory.append(tau2_details.get("avg_weight_shells"))
        self.tau2_shell_sum_trajectory.append(tau2_details.get("shell_sum"))
        self.tau2_shell_count_trajectory.append(tau2_details.get("shell_count"))
        self.tau2_ssnr_trajectory.append(tau2_details.get("ssnr_shells"))
        self.tau2_fsc_used_trajectory.append(tau2_details.get("fsc_shells"))

    def record_sigma_offset_update(self, sigma_offset, sigma_offset_per_half, per_class_sigma_offset) -> None:
        self.sigma_offset_trajectory.append(sigma_offset)
        self.sigma_offset_per_half_trajectory.append(sigma_offset_per_half)
        self.per_class_sigma_offset_trajectory.append(per_class_sigma_offset)

    def record_pose_accuracy_diagnostics(
        self,
        acc_rot,
        acc_trans,
        acc_rot_per_class,
        acc_trans_per_class,
        expected_accuracy_class_counts,
        expected_accuracy_status,
        smallest_change_angles,
        smallest_change_offsets,
    ) -> None:
        self.acc_rot_trajectory.append(acc_rot)
        self.acc_trans_trajectory.append(acc_trans)
        self.acc_rot_per_class_trajectory.append(acc_rot_per_class)
        self.acc_trans_per_class_trajectory.append(acc_trans_per_class)
        self.expected_accuracy_class_counts_trajectory.append(expected_accuracy_class_counts)
        self.expected_accuracy_status_trajectory.append(expected_accuracy_status)
        self.smallest_change_angles_trajectory.append(smallest_change_angles)
        self.smallest_change_offsets_trajectory.append(smallest_change_offsets)

    def to_dict(self) -> dict:
        """Return the trajectory entries of the function's result dict."""
        return {
            "fsc": self.fsc_history[-1] if self.fsc_history else None,
            "current_sizes": self.current_sizes,
            "fsc_history": self.fsc_history,
            "pixel_resolutions": self.pixel_resolutions,
            "wall_times": self.wall_times,
            "significant_counts": self.significant_counts,
            "data_vs_prior_trajectory": self.data_vs_prior_trajectory,
            "healpix_order_trajectory": self.healpix_order_trajectory,
            "ave_Pmax_trajectory": self.ave_Pmax_trajectory,
            "ave_Pmax_denominator_trajectory": self.ave_Pmax_denominator_trajectory,
            "pmax_per_image_history": self.pmax_per_image_history,
            "noise_radial_trajectory": self.noise_radial_trajectory,
            "noise_radial_per_half_trajectory": self.noise_radial_per_half_trajectory,
            "tau2_radial_trajectory": self.tau2_radial_trajectory,
            "tau2_sigma2_trajectory": self.tau2_sigma2_trajectory,
            "tau2_avg_weight_trajectory": self.tau2_avg_weight_trajectory,
            "tau2_shell_sum_trajectory": self.tau2_shell_sum_trajectory,
            "tau2_shell_count_trajectory": self.tau2_shell_count_trajectory,
            "tau2_fsc_used_trajectory": self.tau2_fsc_used_trajectory,
            "tau2_ssnr_trajectory": self.tau2_ssnr_trajectory,
            "sigma_offset_used_trajectory": self.sigma_offset_used_trajectory,
            "sigma_offset_used_per_half_trajectory": self.sigma_offset_used_per_half_trajectory,
            "sigma_offset_trajectory": self.sigma_offset_trajectory,
            "sigma_offset_per_half_trajectory": self.sigma_offset_per_half_trajectory,
            "per_class_sigma_offset_trajectory": self.per_class_sigma_offset_trajectory,
            "direction_prior_trajectory_per_half": self.direction_prior_trajectory_per_half,
            "rotation_posterior_trajectory_per_half": self.rotation_posterior_trajectory_per_half,
            "frac_changed_trajectory": self.frac_changed_trajectory,
            "acc_rot_trajectory": self.acc_rot_trajectory,
            "acc_trans_trajectory": self.acc_trans_trajectory,
            "acc_rot_per_class_trajectory": self.acc_rot_per_class_trajectory,
            "acc_trans_per_class_trajectory": self.acc_trans_per_class_trajectory,
            "expected_accuracy_class_counts_trajectory": self.expected_accuracy_class_counts_trajectory,
            "expected_accuracy_status_trajectory": self.expected_accuracy_status_trajectory,
            "smallest_change_angles_trajectory": self.smallest_change_angles_trajectory,
            "smallest_change_offsets_trajectory": self.smallest_change_offsets_trajectory,
            "best_rotation_eulers_history": self.best_rotation_eulers_history,
            "best_translations_history": self.best_translations_history,
            "class_mstep_weight_trajectory": self.class_mstep_weight_trajectory,
            "class_full_posterior_weight_trajectory": self.class_full_posterior_weight_trajectory,
            "class_assignment_history": self.class_assignment_history,
            "state_swap_probe_applied_relion_iterations": list(self.state_swap_probe_applied_relion_iterations),
            "local_profile_history": self.local_profile_history,
            "global_profile_history": self.global_profile_history,
        }


def _pose_history_half_arrays(iter_entry, *, dtype=np.float32):
    if iter_entry is None:
        return None
    if not isinstance(iter_entry, (list, tuple)):
        return [np.asarray(iter_entry, dtype=dtype)]
    return [None if arr is None else np.asarray(arr, dtype=dtype) for arr in iter_entry]



def _pose_history_by_image(iter_entry, half_indices, n_images, trailing_shape, *, dtype=np.float32):
    half_arrays = _pose_history_half_arrays(iter_entry, dtype=dtype)
    if half_arrays is None or all(arr is None for arr in half_arrays):
        return None
    out = np.full((int(n_images), *trailing_shape), np.nan, dtype=dtype)
    for half_idx, arr in zip(half_indices, half_arrays):
        if arr is None:
            continue
        half_idx = np.asarray(half_idx, dtype=np.int64)
        if arr.shape[0] != half_idx.shape[0]:
            raise ValueError(
                f"Pose history length {arr.shape[0]} does not match half-set index length {half_idx.shape[0]}"
            )
        out[half_idx] = arr
    return out


def add_class_history_artifacts(save_dict, result, half1_idx, half2_idx, n_images):
    """Append refinement history in its existing NPZ layout and precision."""
    # Save K-class metadata when available (n_classes>1).
    for key in (
        "class_weights",
        "class_mstep_weight_trajectory",
        "class_full_posterior_weight_trajectory",
    ):
        if result.get(key) is not None:
            save_dict[key] = np.asarray(result[key], dtype=np.float64)
    if result.get("class_assignments") is not None and any(c is not None for c in result["class_assignments"]):
        for k, ca in enumerate(result["class_assignments"]):
            if ca is not None:
                save_dict[f"class_assignments_half{k}"] = np.asarray(ca, dtype=np.int32)
    if result.get("class_assignment_history") is not None:
        class_half_order_indices = np.concatenate(
            [np.asarray(half1_idx, dtype=np.int64), np.asarray(half2_idx, dtype=np.int64)],
        )
        for i, classes in enumerate(result["class_assignment_history"]):
            classes_half_order = np.asarray(classes, dtype=np.int32).reshape(-1)
            save_dict[f"class_assignments_iter_{i:03d}"] = classes_half_order
            save_dict[f"class_assignments_half_order_iter_{i:03d}"] = classes_half_order
            if classes_half_order.shape[0] == class_half_order_indices.shape[0]:
                classes_by_image = np.full(int(n_images), -1, dtype=np.int32)
                classes_by_image[class_half_order_indices] = classes_half_order
                save_dict[f"class_assignments_by_image_iter_{i:03d}"] = classes_by_image
    if result.get("per_class_sigma_offset_trajectory") is not None:
        # Per-iter K-vector or None; serialize as object array via dtype=object.
        save_dict["per_class_sigma_offset_trajectory"] = np.asarray(
            result["per_class_sigma_offset_trajectory"], dtype=object
        )


def add_refinement_history_artifacts(save_dict, result, half1_idx, half2_idx, n_images):
    """Append refinement history in its existing NPZ layout and precision."""
    # Save FSC curves per iteration
    for i, fsc in enumerate(result["fsc_history"]):
        save_dict[f"fsc_iter_{i:03d}"] = np.asarray(fsc)

    # Save significant counts per iteration (if available). The refinement
    # loop concatenates half 1 then half 2, which is not generally image order.
    add_significant_count_artifacts(
        save_dict,
        result["significant_counts"],
        [half1_idx, half2_idx],
        n_images,
    )

    if "data_vs_prior_trajectory" in result:
        for i, dvp in enumerate(result["data_vs_prior_trajectory"]):
            save_dict[f"data_vs_prior_iter_{i:03d}"] = np.asarray(dvp)

    # Per-iteration shell profiles share the same float64 artifact format.
    for result_key, prefix in [
        ("noise_radial_trajectory", "noise_radial_iter"),
        ("noise_radial_per_half_trajectory", "noise_radial_per_half_iter"),
        ("tau2_radial_trajectory", "tau2_radial_iter"),
        ("tau2_sigma2_trajectory", "tau2_sigma2_iter"),
        ("tau2_avg_weight_trajectory", "tau2_avg_weight_iter"),
        ("tau2_shell_sum_trajectory", "tau2_shell_sum_iter"),
        ("tau2_shell_count_trajectory", "tau2_shell_count_iter"),
        ("tau2_fsc_used_trajectory", "tau2_fsc_used_iter"),
        ("tau2_ssnr_trajectory", "tau2_ssnr_iter"),
    ]:
        if result_key in result:
            for i, arr in enumerate(result[result_key]):
                if arr is not None:
                    save_dict[f"{prefix}_{i:03d}"] = np.asarray(arr, dtype=np.float64)

    # Save per-image Pmax per iteration (if available)
    if "pmax_per_image_history" in result:
        pmax_half_order_indices = np.concatenate(
            [np.asarray(half1_idx, dtype=np.int64), np.asarray(half2_idx, dtype=np.int64)],
        )
        for i, pmax in enumerate(result["pmax_per_image_history"]):
            pmax_half_order = np.asarray(pmax, dtype=np.float32).reshape(-1)
            save_dict[f"pmax_per_image_iter_{i:03d}"] = pmax_half_order
            save_dict[f"pmax_per_half_order_iter_{i:03d}"] = pmax_half_order
            if pmax_half_order.shape[0] == pmax_half_order_indices.shape[0]:
                pmax_by_image = np.full(int(n_images), np.nan, dtype=np.float32)
                pmax_by_image[pmax_half_order_indices] = pmax_half_order
                save_dict[f"pmax_per_image_by_image_iter_{i:03d}"] = pmax_by_image
    if "ave_Pmax_denominator_trajectory" in result:
        save_dict["ave_Pmax_denominator_trajectory"] = np.asarray(
            result["ave_Pmax_denominator_trajectory"],
            dtype=np.float64,
        )
    if result.get("final_all_data_fsc") is not None:
        save_dict["fsc_final_all_data"] = np.asarray(result["final_all_data_fsc"], dtype=np.float32)
    if "final_all_data_ran" in result:
        save_dict["final_all_data_ran"] = np.asarray(result["final_all_data_ran"], dtype=np.bool_)
    for key in (
        "tau2_radial_final_all_data",
        "tau2_fsc_used_final_all_data",
        "tau2_ssnr_final_all_data",
    ):
        if result.get(key) is not None:
            save_dict[key] = np.asarray(result[key], dtype=np.float64)
    for key, dtype in (
        ("final_all_data_sampling_perturbation", np.float32),
        ("final_all_data_sampling_perturbation_applied", np.bool_),
        ("final_all_data_sampling_relion_iteration", np.int32),
    ):
        if key in result:
            save_dict[key] = np.asarray(result[key], dtype=dtype)
    if result.get("final_all_data_sampling_star") is not None:
        save_dict["final_all_data_sampling_star"] = np.asarray(str(result["final_all_data_sampling_star"]))
    if result.get("final_all_data_sampling_star_source") is not None:
        save_dict["final_all_data_sampling_star_source"] = np.asarray(
            str(result["final_all_data_sampling_star_source"])
        )
    for key, dtype in (
        ("final_all_data_sampling_offset_range", np.float32),
        ("final_all_data_sampling_offset_step", np.float32),
        ("final_all_data_grid_correct", np.bool_),
    ):
        if key in result:
            save_dict[key] = np.asarray(result[key], dtype=dtype)
    if result.get("final_all_data_gridding_correct") is not None:
        save_dict["final_all_data_gridding_correct"] = np.asarray(
            str(result["final_all_data_gridding_correct"])
        )
    if result.get("tau2_weight_combination_final_all_data") is not None:
        save_dict["tau2_weight_combination_final_all_data"] = np.asarray(
            str(result["tau2_weight_combination_final_all_data"])
        )

    half_indices = [
        np.asarray(half1_idx, dtype=np.int64),
        np.asarray(half2_idx, dtype=np.int64),
    ]
    for prefix, trailing_shape in (
        ("best_rotation_eulers", (3,)),
        ("best_translations", (2,)),
    ):
        for i, iter_poses in enumerate(result.get(f"{prefix}_history", [])):
            half_arrays = _pose_history_half_arrays(iter_poses, dtype=np.float32)
            if half_arrays is None or all(arr is None for arr in half_arrays):
                continue
            compact = []
            for k, arr in enumerate(half_arrays):
                if arr is None:
                    continue
                save_dict[f"{prefix}_iter_{i:03d}_half{k}"] = arr
                compact.append(arr)
            if compact:
                save_dict[f"{prefix}_iter_{i:03d}"] = np.concatenate(compact, axis=0)
            by_image = _pose_history_by_image(iter_poses, half_indices, n_images, trailing_shape, dtype=np.float32)
            if by_image is not None:
                save_dict[f"{prefix}_by_image_iter_{i:03d}"] = by_image
                save_dict[f"{prefix}_final_by_image"] = by_image

    for result_key, prefix, trailing_shape in (
        ("final_all_data_best_rotation_eulers", "best_rotation_eulers", (3,)),
        ("final_all_data_best_translations", "best_translations", (2,)),
        ("final_all_data_max_posterior", "pmax", ()),
    ):
        final_values = result.get(result_key)
        half_arrays = _pose_history_half_arrays(final_values, dtype=np.float32)
        if half_arrays is None or all(arr is None for arr in half_arrays):
            continue
        compact = []
        for k, arr in enumerate(half_arrays):
            if arr is None:
                continue
            save_dict[f"{prefix}_final_all_data_half{k}"] = arr
            compact.append(arr)
        if compact:
            save_dict[f"{prefix}_final_all_data"] = np.concatenate(compact, axis=0)
        by_image = _pose_history_by_image(final_values, half_indices, n_images, trailing_shape, dtype=np.float32)
        if by_image is not None:
            save_dict[f"{prefix}_final_all_data_by_image"] = by_image


def _load_init_previous_best_poses_npz(path, pose_iter="last"):
    """Load previous best poses from a RECOVAR refinement_results.npz file.

    This is a diagnostic/debugging hook for starting directly in the local
    search branch. It does not affect the default GUI/CLI path.
    """

    pose_path = Path(path)
    with np.load(pose_path, allow_pickle=False) as npz:
        if str(pose_iter).lower() in {"last", "latest"}:
            pattern = re.compile(r"^best_rotation_eulers_iter_(\d{3})_half0$")
            available = sorted(
                int(match.group(1))
                for key in npz.files
                if (match := pattern.match(key)) is not None
                and f"best_rotation_eulers_iter_{match.group(1)}_half1" in npz.files
                and f"best_translations_iter_{match.group(1)}_half0" in npz.files
                and f"best_translations_iter_{match.group(1)}_half1" in npz.files
            )
            if not available:
                raise ValueError(f"No numbered per-half best-pose arrays found in {pose_path}")
            iter_label = f"{available[-1]:03d}"
        elif str(pose_iter).lower() in {"final_all_data", "final-all-data"}:
            euler_keys = [
                "best_rotation_eulers_final_all_data_half0",
                "best_rotation_eulers_final_all_data_half1",
            ]
            trans_keys = [
                "best_translations_final_all_data_half0",
                "best_translations_final_all_data_half1",
            ]
            missing = [key for key in euler_keys + trans_keys if key not in npz.files]
            if missing:
                raise ValueError(f"Missing final-all-data pose arrays in {pose_path}: {missing}")
            eulers = [np.asarray(npz[key], dtype=np.float32) for key in euler_keys]
            translations = [np.asarray(npz[key], dtype=np.float32) for key in trans_keys]
            return {
                "iteration": "final_all_data",
                "previous_best_rotation_eulers": eulers,
                "previous_best_translations": translations,
            }
        else:
            iter_label = f"{int(pose_iter):03d}"

        euler_keys = [
            f"best_rotation_eulers_iter_{iter_label}_half0",
            f"best_rotation_eulers_iter_{iter_label}_half1",
        ]
        trans_keys = [
            f"best_translations_iter_{iter_label}_half0",
            f"best_translations_iter_{iter_label}_half1",
        ]
        missing = [key for key in euler_keys + trans_keys if key not in npz.files]
        if missing:
            raise ValueError(f"Missing pose arrays for iter {iter_label} in {pose_path}: {missing}")
        eulers = [np.asarray(npz[key], dtype=np.float32) for key in euler_keys]
        translations = [np.asarray(npz[key], dtype=np.float32) for key in trans_keys]

    for half, (euler, translation) in enumerate(zip(eulers, translations), start=1):
        if euler.ndim != 2 or euler.shape[1] != 3:
            raise ValueError(f"half-{half} Euler array must have shape (N, 3), got {euler.shape}")
        if translation.ndim != 2 or translation.shape[1] != 2:
            raise ValueError(f"half-{half} translation array must have shape (N, 2), got {translation.shape}")
        if euler.shape[0] != translation.shape[0]:
            raise ValueError(
                f"half-{half} Euler/translation row mismatch: {euler.shape[0]} vs {translation.shape[0]}",
            )

    return {
        "iteration": iter_label,
        "previous_best_rotation_eulers": eulers,
        "previous_best_translations": translations,
    }


def _load_init_noise_radial_npz(path, noise_iter="last"):
    """Load a diagnostic initial noise spectrum from refinement_results.npz."""

    noise_path = Path(path)
    with np.load(noise_path, allow_pickle=False) as npz:
        if str(noise_iter).lower() in {"last", "latest"}:
            pattern = re.compile(r"^noise_radial_iter_(\d{3})$")
            available = sorted(
                int(match.group(1)) for key in npz.files if (match := pattern.match(key)) is not None
            )
            if not available:
                raise ValueError(f"No numbered noise_radial_iter arrays found in {noise_path}")
            iter_label = f"{available[-1]:03d}"
        else:
            iter_label = f"{int(noise_iter):03d}"
        key = f"noise_radial_iter_{iter_label}"
        if key not in npz.files:
            raise ValueError(f"Missing {key} in {noise_path}")
        noise_radial = np.asarray(npz[key], dtype=np.float64)

    if noise_radial.ndim != 1:
        raise ValueError(f"{key} must be a 1D radial spectrum, got shape {noise_radial.shape}")
    if not np.all(np.isfinite(noise_radial)):
        raise ValueError(f"{key} contains non-finite values")
    if np.any(noise_radial <= 0.0):
        raise ValueError(f"{key} must be strictly positive")
    return {
        "iteration": iter_label,
        "noise_radial": noise_radial,
    }
