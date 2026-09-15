"""Reconstruction diagnostic serialization; scheduling stays in the controller.

Callers retain their environment gates. These writers preserve historical NPZ
field names, casts and optional-field behavior; they do not update EM state.
"""

import os

import jax.numpy as jnp
import numpy as np

from recovar.em.helpers.resolution import shell_index_to_resolution_angstrom


def write_kclass_current_size(
    *,
    output_dir,
    computed_cs,
    data_vs_prior_prev,
    data_vs_prior_prev_raw,
    grid_size,
    iteration,
    per_class_res_shell,
    prev_cs,
    raw_cs,
    relion_has_high_fsc_at_limit,
    relion_incr_size,
    res_shell,
    state,
):
    """Write the existing kclass current size NPZ schema."""
    import pathlib

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        pathlib.Path(output_dir) / f"recovar_kclass_current_size_it{iteration + 1:03d}.npz",
        iteration=np.int32(iteration + 1),
        previous_current_size=np.int32(prev_cs),
        grid_size=np.int32(grid_size),
        resolution_shell=np.int32(res_shell),
        per_class_resolution_shells=np.asarray(per_class_res_shell, dtype=np.int32),
        ave_Pmax=np.float64(float(state.ave_Pmax)),
        state_current_resolution=np.float64(float(state.current_resolution)),
        state_previous_resolution=np.float64(float(state.previous_resolution)),
        relion_incr_size=np.int32(relion_incr_size),
        relion_has_high_fsc_at_limit=np.int32(int(relion_has_high_fsc_at_limit)),
        data_vs_prior_prev_raw=np.asarray(data_vs_prior_prev_raw, dtype=np.float32),
        data_vs_prior_prev=np.asarray(data_vs_prior_prev, dtype=np.float32),
        raw_current_size=np.int32(raw_cs),
        quantized_current_size=np.int32(computed_cs),
    )


def write_kclass_mstep(
    *,
    Ft_ctf_0,
    Ft_ctf_1,
    Ft_ctf_combined,
    Ft_y_combined,
    PADDING_FACTOR,
    output_dir,
    class_idx,
    current_size,
    data_vs_prior_k,
    grid_size,
    iteration,
    kclass_tau2_frame_scale,
    kclass_tau2_source,
    mstep_accumulator_shape,
    mstep_full_half_axis,
    previous_means,
    reconstruct_floor_stats_k,
    shell_stats_k,
    tau2_fudge,
    tau2_shells_recovar_frame_k,
    tau2_shells_relion_frame_k,
):
    """Write the existing kclass mstep NPZ schema."""
    import pathlib

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    _preserve_kclass_dump_dtype = os.environ.get("RECOVAR_KCLASS_DUMP_PRESERVE_DTYPE", "").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }
    np.savez(
        pathlib.Path(output_dir) / f"recovar_kclass_mstep_it{iteration + 1:03d}_c{class_idx + 1:02d}.npz",
        iteration=np.int32(iteration + 1),
        class_index=np.int32(class_idx + 1),
        current_size=np.int32(current_size),
        padding_factor=np.int32(PADDING_FACTOR),
        grid_size=np.int32(grid_size),
        mstep_accumulator_shape=np.asarray(mstep_accumulator_shape, dtype=np.int32),
        mstep_full_half_axis=np.int32(mstep_full_half_axis),
        tau2_fudge=np.float64(tau2_fudge),
        tau2_frame_scale=np.float64(kclass_tau2_frame_scale),
        previous_mean=np.asarray(previous_means[0][class_idx], dtype=np.complex64),
        previous_mean_half0=np.asarray(previous_means[0][class_idx], dtype=np.complex64),
        previous_mean_half1=np.asarray(previous_means[1][class_idx], dtype=np.complex64),
        Ft_y_combined=(
            np.asarray(Ft_y_combined[class_idx])
            if _preserve_kclass_dump_dtype
            else np.asarray(Ft_y_combined[class_idx], dtype=np.complex64)
        ),
        Ft_ctf_0=(
            (
                np.asarray(Ft_ctf_0[class_idx])
                if _preserve_kclass_dump_dtype
                else np.asarray(Ft_ctf_0[class_idx], dtype=np.complex64)
            )
            if Ft_ctf_0 is not None
            else np.empty(0, dtype=np.complex64)
        ),
        Ft_ctf_1=(
            (
                np.asarray(Ft_ctf_1[class_idx])
                if _preserve_kclass_dump_dtype
                else np.asarray(Ft_ctf_1[class_idx], dtype=np.complex64)
            )
            if Ft_ctf_1 is not None
            else np.empty(0, dtype=np.complex64)
        ),
        Ft_ctf_combined=(
            np.asarray(Ft_ctf_combined[class_idx])
            if _preserve_kclass_dump_dtype
            else np.asarray(Ft_ctf_combined[class_idx], dtype=np.complex64)
        ),
        dump_preserve_dtype=np.int32(int(_preserve_kclass_dump_dtype)),
        tau2_shells=np.asarray(tau2_shells_recovar_frame_k, dtype=np.float64),
        tau2_shells_relion=np.asarray(tau2_shells_relion_frame_k, dtype=np.float64),
        tau2_source=np.asarray(kclass_tau2_source),
        sigma2_shells=np.asarray(
            jnp.where(
                shell_stats_k["avg_weight_shells"] > 0,
                1.0 / (PADDING_FACTOR**3 * shell_stats_k["avg_weight_shells"]),
                0.0,
            ),
            dtype=np.float64,
        ),
        avg_weight_shells=np.asarray(shell_stats_k["avg_weight_shells"], dtype=np.float64),
        shell_sum=np.asarray(shell_stats_k["shell_sum"], dtype=np.float64),
        shell_count=np.asarray(shell_stats_k["shell_count"], dtype=np.float64),
        reconstruct_floor_avg_weight_shells=np.asarray(
            reconstruct_floor_stats_k["avg_weight_shells"],
            dtype=np.float64,
        ),
        reconstruct_floor_shell_count=np.asarray(
            reconstruct_floor_stats_k["shell_count"],
            dtype=np.float64,
        ),
        data_vs_prior=np.asarray(data_vs_prior_k, dtype=np.float64),
    )


def write_tau2_update(
    *,
    _replay_meta,
    output_dir,
    voxel_size,
    current_size,
    dvp_iter,
    fsc,
    grid_size,
    iteration,
    mstep_accumulator_shape,
    perturb_replay_relion_dir,
    perturb_replay_relion_prefix,
    pixel_res,
    sealed_sampling_state,
    tau2_update_details,
    tau2_update_details_per_half,
    logger,
):
    """Write the existing tau2 update NPZ schema."""
    import pathlib

    _tau2_dump = {
        "iteration": np.int32(iteration + 1),
        "relion_iteration": np.int32(iteration + 1),
        "current_size": np.int32(current_size),
        "grid_size": np.int32(grid_size),
        "voxel_size": np.float64(voxel_size),
        "pixel_res": np.float64(pixel_res),
        "res_angstrom": np.float64(
            shell_index_to_resolution_angstrom(pixel_res, grid_size, voxel_size) if pixel_res > 0.0 else np.inf
        ),
        "dvp_iter": np.asarray(dvp_iter, dtype=np.float64),
        "mstep_accumulator_shape": np.asarray(mstep_accumulator_shape, dtype=np.int32),
    }
    if tau2_update_details is not None:
        for _key in (
            "fsc_shells",
            "ssnr_shells",
            "prior_shells",
            "sigma2_shells",
            "avg_weight_shells",
            "shell_sum",
            "shell_count",
        ):
            if _key in tau2_update_details and tau2_update_details[_key] is not None:
                _tau2_dump[f"tau2_{_key}"] = np.asarray(tau2_update_details[_key], dtype=np.float64)
    if tau2_update_details_per_half is not None:
        for _half_idx, _detail in enumerate(tau2_update_details_per_half):
            if _detail is None:
                continue
            for _key in ("fsc_shells", "ssnr_shells", "prior_shells", "sigma2_shells"):
                if _key in _detail and _detail[_key] is not None:
                    _tau2_dump[f"half{_half_idx + 1}_{_key}"] = np.asarray(
                        _detail[_key],
                        dtype=np.float64,
                    )
    if fsc is not None:
        _tau2_dump["current_iter_fsc"] = np.asarray(fsc, dtype=np.float64)
    if perturb_replay_relion_dir is not None and sealed_sampling_state is None:
        _model_path = os.path.join(
            str(perturb_replay_relion_dir),
            f"{perturb_replay_relion_prefix}_it{iteration + 1:03d}_half1_model.star",
        )
        _tau2_dump["relion_model_path"] = np.asarray(_model_path)
        _tau2_dump["relion_model_exists"] = np.bool_(os.path.exists(_model_path))
    if _replay_meta is not None:
        for _key, _value in _replay_meta.items():
            try:
                _tau2_dump[f"replay_meta_{_key}"] = np.asarray(_value)
            except Exception:
                _tau2_dump[f"replay_meta_{_key}"] = np.asarray(str(_value))
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    _tau2_dump_path = pathlib.Path(output_dir) / f"recovar_tau2_debug_it{iteration + 1:03d}.npz"
    np.savez(_tau2_dump_path, **_tau2_dump)
    logger.info("RELION tau2 debug dump written: %s", _tau2_dump_path)


def write_final_bpref_accumulators(
    *,
    PADDING_FACTOR,
    PROJECTION_PADDING_FACTOR,
    output_dir,
    voxel_size,
    final_Ft_ctf_0,
    final_Ft_ctf_1,
    final_Ft_y_0,
    final_Ft_y_1,
    final_current_size,
    final_ft_ctf,
    final_ft_y,
    final_grid_correct,
    final_iter_fsc,
    final_mstep_accumulator_shape,
    final_mstep_full_half_axis,
    final_tau2_update_details,
    final_unfiltered_Ft_ctf_0,
    final_unfiltered_Ft_ctf_1,
    final_unfiltered_Ft_y_0,
    final_unfiltered_Ft_y_1,
    grid_size,
    k_class_enabled,
    tau2_fudge,
    volume_shape,
    logger,
):
    """Write the existing final bpref accumulators NPZ schema."""
    import pathlib

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    _final_dump = {
        "current_size": np.int32(final_current_size),
        "padding_factor": np.int32(PADDING_FACTOR),
        "projection_padding_factor": np.int32(PROJECTION_PADDING_FACTOR),
        "grid_size": np.int32(grid_size),
        "voxel_size": np.float32(voxel_size),
        "volume_shape": np.asarray(volume_shape, dtype=np.int32),
        "mstep_accumulator_shape": np.asarray(final_mstep_accumulator_shape, dtype=np.int32),
        "tau2_fudge": np.float64(tau2_fudge),
        "k_class_enabled": np.bool_(k_class_enabled),
        "grid_correct": np.bool_(final_grid_correct),
        "tau2_weight_combination": np.asarray("class_iref" if k_class_enabled else "sum"),
        "mstep_full_half_axis": np.int32(final_mstep_full_half_axis),
        "Ft_y_0": np.asarray(final_Ft_y_0),
        "Ft_y_1": np.asarray(final_Ft_y_1),
        "Ft_ctf_0": np.asarray(final_Ft_ctf_0).real,
        "Ft_ctf_1": np.asarray(final_Ft_ctf_1).real,
        "Ft_y_0_pre_lowres_join": np.asarray(final_unfiltered_Ft_y_0),
        "Ft_y_1_pre_lowres_join": np.asarray(final_unfiltered_Ft_y_1),
        "Ft_ctf_0_pre_lowres_join": np.asarray(final_unfiltered_Ft_ctf_0).real,
        "Ft_ctf_1_pre_lowres_join": np.asarray(final_unfiltered_Ft_ctf_1).real,
        "Ft_y": np.asarray(final_ft_y),
        "Ft_ctf": np.asarray(final_ft_ctf).real,
    }
    if final_iter_fsc is not None:
        _final_dump["fsc_shells"] = np.asarray(final_iter_fsc, dtype=np.float64)
    if final_tau2_update_details is not None:
        for _key in (
            "prior_shells",
            "sigma2_shells",
            "avg_weight_shells",
            "shell_sum",
            "shell_count",
            "fsc_shells",
            "ssnr_shells",
        ):
            if _key in final_tau2_update_details and final_tau2_update_details[_key] is not None:
                _final_dump[f"tau2_{_key}"] = np.asarray(final_tau2_update_details[_key], dtype=np.float64)
    _final_dump_path = pathlib.Path(output_dir) / "recovar_final_bpref_accum.npz"
    np.savez(_final_dump_path, **_final_dump)
    logger.info("Final all-data BPref accumulators dumped: %s", _final_dump_path)
