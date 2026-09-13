from scripts import run_vdam_schedule_representation_panel as panel


def test_representation_panel_is_mirrored_and_balanced() -> None:
    labels = tuple(label for label, _capacity in panel.ARM_SPECS)
    capacities = tuple(capacity for _label, capacity in panel.ARM_SPECS)

    assert labels == (
        "dynamic_1",
        "oracle_1",
        "oracle_2",
        "dynamic_2",
        "oracle_3",
        "dynamic_3",
        "dynamic_4",
        "oracle_4",
    )
    assert capacities.count(64) == capacities.count(288) == 4
    assert panel.PREWARM_SPECS == (
        ("prewarm_dynamic", 64),
        ("prewarm_oracle", 288),
    )
