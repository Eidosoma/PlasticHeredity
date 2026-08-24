import numpy as np

from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.mechanistic_features import (
    BETA_ONLY_FEATURE_NAMES,
    H8_FEATURE_NAMES,
    INTERACTION_FEATURE_NAMES,
    STATE_ONLY_FEATURE_NAMES,
)
from plastic_heredity.simulator import (
    Snapshot,
    generate_beta,
    generate_initial_composition,
    simulate_future_absorbing,
    simulate_lineage,
)


def test_registered_feature_blocks_have_intended_semantics():
    assert "normalized_fissions_since_break" not in H8_FEATURE_NAMES
    assert len(H8_FEATURE_NAMES) == len(set(H8_FEATURE_NAMES)) == 8
    assert len(STATE_ONLY_FEATURE_NAMES) == 26
    assert all(
        name.split("__", 1)[0] in {"composition_fraction", "present"}
        for name in STATE_ONLY_FEATURE_NAMES
    )
    assert len(BETA_ONLY_FEATURE_NAMES) == 195
    assert len(INTERACTION_FEATURE_NAMES) == 52
    assert all(
        name.split("__", 1)[0]
        in {
            "log_in_catalysis",
            "log_out_catalysis",
            "log_active_in_catalysis",
            "log_active_out_catalysis",
        }
        for name in INTERACTION_FEATURE_NAMES
    )
    assert not any(
        token in name
        for name in INTERACTION_FEATURE_NAMES
        for token in ("mass", "generation", "step", "phase", "landmark", "fission")
    )


def test_clock_capture_is_cumulative_and_does_not_drive_future_rng():
    config = GardConfig(generations=3)
    beta = generate_beta(config, np.random.default_rng(220))
    initial = generate_initial_composition(config, np.random.default_rng(221))
    lineage = simulate_lineage(
        initial, beta, config, CANDIDATES["02"], np.random.default_rng(222)
    )
    assert all(snapshot.previous_growth_steps > 0 for snapshot in lineage)
    assert [snapshot.cumulative_growth_steps for snapshot in lineage] == list(
        np.cumsum([snapshot.previous_growth_steps for snapshot in lineage])
    )

    original = lineage[-1]
    changed_clocks = Snapshot(
        original.composition,
        original.generation,
        original.inheritance,
        original.boundary_h,
        previous_growth_steps=999_999,
        cumulative_growth_steps=999_999_999,
    )
    left, left_complete = simulate_future_absorbing(
        original,
        beta,
        config,
        CANDIDATES["02"],
        2,
        np.random.default_rng(223),
    )
    right, right_complete = simulate_future_absorbing(
        changed_clocks,
        beta,
        config,
        CANDIDATES["02"],
        2,
        np.random.default_rng(223),
    )
    assert left_complete == right_complete
    for a, b in zip(left, right):
        np.testing.assert_array_equal(a.parent, b.parent)
        np.testing.assert_array_equal(a.daughter, b.daughter)
        assert a.h == b.h
        assert a.growth_steps == b.growth_steps
