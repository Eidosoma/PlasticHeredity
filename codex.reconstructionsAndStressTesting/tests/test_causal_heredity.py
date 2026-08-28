from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from plastic_ca.causal_heredity import (
    CAUSAL_PROFILES,
    CausalContract,
    _exact_random_state,
    _intervention_state,
    _site_mask,
    _strict_event,
    build_rule_panel,
)
from plastic_ca.e19 import E19Contract, _simulate_seed, capture_e19_seed
from plastic_ca.life_family import (
    LifeFamilyContract,
    _simulate_launch,
    capture_life_family_launch,
    launch_library,
    parse_life_rule,
)


class CaptureTests(unittest.TestCase):
    def test_eca_capture_is_observational(self) -> None:
        contract = E19Contract(
            n_seeds=1,
            futures_per_seed=2,
            horizon=3,
            flip_noise=0.0,
            copy_error=0.0,
        )
        ordinary = _simulate_seed(35, 0, contract)
        captured = capture_e19_seed(35, 0, contract)
        traces = captured.pop("captures")
        self.assertEqual(ordinary, captured)
        self.assertEqual(len(traces), 2)
        self.assertTrue(all("terminal_row_hex" in generation for trace in traces for generation in trace))

    def test_life_capture_is_observational(self) -> None:
        contract = LifeFamilyContract(futures_per_launch=2, horizon=3)
        initial = launch_library(contract)[0]
        ordinary = _simulate_launch(parse_life_rule("B3/S23"), 0, initial, contract)
        captured = capture_life_family_launch(parse_life_rule("B3/S23"), 0, initial, contract)
        traces = captured.pop("captures")
        self.assertEqual(ordinary, captured)
        self.assertEqual(len(traces), 2)


class InterventionTests(unittest.TestCase):
    def test_masks_have_exact_registered_area(self) -> None:
        for shape, geometries in (
            ((64,), ("one_interval", "two_interval", "dispersed")),
            ((16, 16), ("square", "strip", "two_lobe", "dispersed")),
        ):
            for fraction in (0.25, 0.5, 0.75):
                for geometry in geometries:
                    with self.subTest(shape=shape, fraction=fraction, geometry=geometry):
                        mask = _site_mask(shape, fraction, geometry, "fixture")
                        self.assertEqual(int(mask.sum()), round(np.prod(shape) * fraction))

    def test_complementary_halves_partition_state(self) -> None:
        state = np.asarray([(index % 3) == 0 for index in range(64)], dtype=np.bool_)
        mask = _site_mask((64,), 0.5, "one_interval", "partition")
        left, right = state & mask, state & ~mask
        self.assertFalse((left & right).any())
        np.testing.assert_array_equal(left | right, state)

    def test_shuffles_and_density_controls_are_deterministic(self) -> None:
        state = np.arange(256).reshape(16, 16) % 5 == 0
        first = _intervention_state(state, 0.5, "shuffled", "same-key")
        second = _intervention_state(state, 0.5, "shuffled", "same-key")
        np.testing.assert_array_equal(first, second)
        random_state = _exact_random_state((16, 16), 23, "density")
        self.assertEqual(int(random_state.sum()), 23)


class DetectionAndPanelTests(unittest.TestCase):
    def test_historical_strict_boundary_fixture(self) -> None:
        old = np.asarray((1.0, 0.0))
        anchor = np.asarray((0.0, 1.0))
        acquired = np.asarray((1.0, 1.0))
        # One extra post-break transition is required because E19 compares
        # the coherent daughters with the break-causing daughter itself.
        compositions = np.stack([old, old, anchor] + [acquired] * 9)
        self.assertEqual(_strict_event(compositions, CausalContract().thresholds), (1, 3))

    def test_reference_panel_is_frozen_and_deduplicated(self) -> None:
        atlas = Path("results/ca-campaign-round-1/life-family/frozen-b48/family.csv")
        panel = build_rule_panel(atlas, CAUSAL_PROFILES["reference"])
        self.assertEqual(len(panel["eca"]), 20)
        self.assertEqual(len(panel["life"]), 24)
        self.assertEqual(len({row["rule"] for row in panel["life"]}), 24)
        for notation in ("B3/S23", "B36/S23", "B2/S"):
            self.assertIn(parse_life_rule(notation), {row["rule"] for row in panel["life"]})


if __name__ == "__main__":
    unittest.main()
