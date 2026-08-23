from scripts.analyze_k1_firstiter_ppref_sources import classify


def test_classify_serialized_replay_source_mismatch():
    assert classify(fresh_relative_l2=1e-9, serialized_relative_l2=1e-5) == (
        "serialized_it000_replay_is_the_ppref_source_mismatch"
    )


def test_classify_fresh_boundary_open():
    assert classify(fresh_relative_l2=2e-6, serialized_relative_l2=1e-5) == (
        "fresh_initial_reference_to_ppref_boundary_remains_open"
    )
