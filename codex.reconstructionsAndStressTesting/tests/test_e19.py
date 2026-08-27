from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np

from plastic_ca.e19 import (
    E19Contract,
    LAUNCH_HEX,
    PINNED_NUMPY,
    _hex_to_row,
    _integer_step,
    _row_to_hex,
    _simulate_seed,
    e19_step,
    final4_counts,
    load_golden_fixture,
    trajectory_seed,
    validate_golden_fixture,
)


class E19GoldenTests(unittest.TestCase):
    def test_environment_is_pinned(self) -> None:
        self.assertEqual(np.__version__, PINNED_NUMPY)

    def test_complete_code_free_trace_replays(self) -> None:
        report = validate_golden_fixture()
        self.assertTrue(report["passed"], report["errors"][:3])
        self.assertEqual(report["launch_checks"], 16)
        self.assertEqual(report["sweep_checks"], 907)
        self.assertEqual(report["spectrum_checks"], 15)

    def test_vector_rule_map_matches_independent_scalar_map(self) -> None:
        for row_hex in LAUNCH_HEX:
            row = _hex_to_row(row_hex)
            for rule in (0, 8, 13, 35, 110, 172, 204, 255):
                actual = _row_to_hex(e19_step(row[None, :], rule)[0])
                expected = f"{_integer_step(int(row_hex, 16), rule):016x}"
                self.assertEqual(actual, expected)

    def test_disclosed_rng_recipe_recreates_golden_process_masks(self) -> None:
        fixture = load_golden_fixture()
        for rule in (8, 13, 35, 110, 172):
            seed = trajectory_seed(rule, 1, "eca-golden-trace-v1")
            rng = np.random.default_rng(seed)
            generation = fixture["traces"][str(rule)]["standard_noise"]["generations"][0]
            for sweep in range(1, int(generation["sweep_count"]) + 1):
                mask = _row_to_hex(rng.random((1, 64))[0] < 0.01)
                self.assertEqual(
                    mask,
                    generation["process_masks_nonzero"].get(str(sweep), "0000000000000000"),
                )

    def test_final4_uses_disclosed_msb_first_bins(self) -> None:
        fixture = load_golden_fixture()
        generation = fixture["traces"]["35"]["no_noise"]["generations"][0]
        row = _hex_to_row(generation["terminal_row_hex"])
        self.assertEqual(
            final4_counts(row[None, :])[0].astype(int).tolist(),
            generation["final4_counts_of_64"],
        )

    def test_timeout_is_death_even_when_terminal_is_not_monochrome(self) -> None:
        seed = _simulate_seed(
            13,
            9,
            E19Contract(
                flip_noise=0.0,
                copy_error=0.0,
                n_seeds=10,
                futures_per_seed=1,
            ),
        )
        self.assertEqual(seed["first_generation_times"], [128])
        self.assertEqual(seed["survival_sum"], 0)
        self.assertEqual(seed["death_counts"], {"timeout": 1})

    def test_rule35_seed0_pins_batch_rng_and_historical_indices(self) -> None:
        seed = _simulate_seed(35, 0, E19Contract())
        self.assertEqual(seed["strict_count"], 96)
        self.assertEqual(seed["break_by_8_count"], 128)
        self.assertEqual(seed["survival_sum"], 3108)
        self.assertEqual(seed["form_support"], 532)
        self.assertEqual(seed["form_n_futures"], 128)
        self.assertEqual(seed["total_sweeps"], 17528)


if __name__ == "__main__":
    unittest.main()
