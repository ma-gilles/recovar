import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import half_volume_mstep
from scripts import validate_bpref_production_capture as validator


def _write_stage(path, *, stage, data, weight, run_id="fixture"):
    np.savez_compressed(
        path,
        magic=np.asarray(validator.MAGIC),
        schema=np.asarray(validator.SCHEMA),
        schema_version=np.int32(1),
        dump_index=np.int64(0),
        iteration=np.int32(1),
        half=np.int32(2),
        run_id=np.asarray(run_id),
        Ft_y=np.asarray(data),
        Ft_ctf=np.asarray(weight),
        current_size=np.int32(2),
        n_images=np.int32(4),
        recon_volume_shape=np.asarray((5, 5, 5), dtype=np.int32),
        stage=np.asarray(stage),
        topology_claim=np.asarray(validator.PRODUCTION_TOPOLOGY),
        accumulator_layout=np.asarray(
            "public-full-c-order"
            if stage == "post_public_layout"
            else "relion-x-half-flat-c-order"
        ),
        arithmetic_mutated=np.bool_(False),
    )


def test_array_metrics_and_repeat_envelope_are_exact_array_diagnostics():
    lhs = np.asarray([1 + 2j, 3 - 4j], dtype=np.complex64)
    rhs = np.asarray([1 + 2j, 3 - 3j], dtype=np.complex64)
    metrics = validator.array_metrics(lhs, rhs)
    assert metrics["array_equal"] is False
    assert metrics["mismatch_count"] == 1
    assert metrics["delta_rms_abs"] == pytest.approx(np.sqrt(0.5))
    assert validator.calibrated_envelope(metrics) == pytest.approx(
        5 * metrics["delta_rms_over_lhs_rms"]
    )


def test_load_stage_rejects_nonproduction_topology(tmp_path):
    path = tmp_path / "pre.npz"
    values = np.zeros(75, dtype=np.complex64)
    _write_stage(path, stage="pre_x0", data=values, weight=values.real)
    with np.load(path, allow_pickle=False) as archive:
        copied = {key: archive[key] for key in archive.files}
    copied["topology_claim"] = np.asarray("diagnostic-shadow")
    np.savez_compressed(path, **copied)
    with pytest.raises(ValueError, match="ordinary flattened production adjoint"):
        validator.load_stage(path, "pre_x0")


def test_stage_fixture_replays_x0_and_public_layout_exactly(tmp_path):
    shape = (5, 5, 5)
    half_size = shape[0] * shape[1] * (shape[2] // 2 + 1)
    rng = np.random.default_rng(123)
    pre_data = (
        rng.standard_normal(half_size) + 1j * rng.standard_normal(half_size)
    ).astype(np.complex64)
    pre_weight = rng.standard_normal(half_size).astype(np.float32)
    post_data = np.asarray(
        half_volume_mstep.enforce_relion_half_volume_x0_hermitian_host(
            pre_data, shape
        )
    )
    post_weight = np.asarray(
        half_volume_mstep.enforce_relion_half_volume_x0_hermitian_host(
            pre_weight, shape
        )
    )
    public_data = np.asarray(
        half_volume_mstep.relion_x_half_volume_to_full(post_data, shape)
    ).reshape(shape)
    public_weight = np.asarray(
        half_volume_mstep.relion_x_half_volume_to_full(post_weight, shape)
    ).reshape(shape).real

    paths = {stage: tmp_path / f"{stage}.npz" for stage in (
        "pre_x0", "post_x0", "post_public_layout"
    )}
    _write_stage(
        paths["pre_x0"], stage="pre_x0", data=pre_data, weight=pre_weight
    )
    _write_stage(
        paths["post_x0"], stage="post_x0", data=post_data, weight=post_weight
    )
    _write_stage(
        paths["post_public_layout"],
        stage="post_public_layout",
        data=public_data,
        weight=public_weight,
    )
    loaded = {
        stage: validator.load_stage(path, stage) for stage, path in paths.items()
    }
    assert validator.array_metrics(
        post_data, loaded["post_x0"]["Ft_y"]
    )["array_equal"]
    assert validator.array_metrics(
        public_data, loaded["post_public_layout"]["Ft_y"]
    )["array_equal"]


def test_companion_requires_authoritative_production_operands(tmp_path):
    panel = tmp_path / "panel.npz"
    contributions = tmp_path / "contributions"
    contributions.mkdir()
    np.savez(
        panel,
        operand_source=np.asarray(validator.PRODUCTION_OPERANDS),
        production_adjoint_topology=np.asarray(validator.PRODUCTION_TOPOLOGY),
        topology_claim=np.asarray(validator.PANEL_TOPOLOGY),
    )
    np.savez(
        contributions / "part.npz",
        operand_source=np.asarray(validator.PRODUCTION_OPERANDS),
        production_adjoint_topology=np.asarray(validator.PRODUCTION_TOPOLOGY),
        original_indices=np.arange(4, dtype=np.int64),
    )
    result = validator.validate_companion(panel, contributions)
    assert result["particle_count"] == 4
    assert result["contribution_shards"] == 1

    with np.load(panel, allow_pickle=False) as archive:
        copied = {key: archive[key] for key in archive.files}
    copied["operand_source"] = np.asarray("sequential-shadow-reduction")
    np.savez(panel, **copied)
    with pytest.raises(ValueError, match="production-reduced operands"):
        validator.validate_companion(panel, contributions)
