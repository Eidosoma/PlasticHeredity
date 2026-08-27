from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from plastic_ca.life_carrier import (
    CARRIER_PROFILES,
    CarrierContract,
    _assignment,
    _holm,
    _live_neighbor_hist,
    _morphology_surrogate,
    _pair_donors,
    _replace_mask,
    _tile_mask,
    _transform_board,
    build_carrier_panel,
    select_holdout_candidates,
)


class PanelAndPairingTests(unittest.TestCase):
    def test_reference_panel_is_frozen_by_atlas_fields(self) -> None:
        atlas = Path("results/ca-campaign-round-1/life-family/frozen-b48/family.csv")
        panel = build_carrier_panel(atlas, CARRIER_PROFILES["reference"])
        self.assertEqual(len(panel), 24)
        self.assertEqual(panel[0]["rule"], 26353)
        self.assertTrue(all(row["development_library_size"] >= 2 for row in panel))
        self.assertTrue(all(0.005 <= row["development_strict"] <= 0.5 for row in panel))

    def test_pairing_is_same_launch_density_matched_and_non_reusing(self) -> None:
        donors = []
        for form, densities in ((3, (0.20, 0.30)), (7, (0.21, 0.29))):
            for index, density in enumerate(densities):
                donors.append(
                    {
                        "form_id": form,
                        "density": density,
                        "launch_index": index,
                        "donor_id": f"{form}-{index}",
                        "rule": 1,
                        "target_compositions": {
                            "primary": [1.0, 0.0] if form == 3 else [0.0, 1.0]
                        },
                    }
                )
        result = _pair_donors(donors, 0.05)
        self.assertEqual(result["forms"], [3, 7])
        self.assertEqual(len(result["pairs"]), 2)
        self.assertEqual(len({row["donor_a"]["donor_id"] for row in result["pairs"]}), 2)
        self.assertTrue(all(row["density_delta"] <= 0.05 for row in result["pairs"]))
        self.assertTrue(all(row["target_similarity"] <= 0.80 for row in result["pairs"]))

    def test_pairing_rejects_discrete_ids_with_same_continuous_form(self) -> None:
        donors = [
            {
                "form_id": form,
                "density": 0.2,
                "launch_index": 0,
                "donor_id": str(form),
                "rule": 1,
                "target_compositions": {"primary": [1.0, 0.01 * form]},
            }
            for form in (3, 7)
        ]
        self.assertEqual(_pair_donors(donors, 0.05)["pairs"], [])


class InterventionTests(unittest.TestCase):
    def test_replace_mask_copies_both_live_and_dead_bits(self) -> None:
        recipient = np.ones((4, 4), dtype=np.bool_)
        source = np.zeros((4, 4), dtype=np.bool_)
        source[0, 0] = True
        mask = np.zeros((4, 4), dtype=np.bool_)
        mask[:2, :2] = True
        result = _replace_mask(recipient, source, mask)
        self.assertTrue(result[0, 0])
        self.assertFalse(result[0, 1])
        self.assertTrue(result[3, 3])

    def test_transforms_preserve_mass_and_are_deterministic(self) -> None:
        board = np.zeros((16, 16), dtype=np.bool_)
        board[1, 2] = True
        for operation in ("identity", "translate", "rotate90", "reflect"):
            first = _transform_board(board, operation)
            second = _transform_board(board, operation)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(int(first.sum()), 1)

    def test_tile_masks_partition_board(self) -> None:
        masks = [_tile_mask((16, 16), tile) for tile in range(16)]
        self.assertTrue(all(int(mask.sum()) == 16 for mask in masks))
        np.testing.assert_array_equal(np.sum(masks, axis=0), np.ones((16, 16), dtype=int))

    def test_saturated_morphology_surrogate_is_exact(self) -> None:
        contract = CarrierContract()
        source = np.ones((16, 16), dtype=np.bool_)
        mask = _tile_mask((16, 16), 0)
        surrogate, metadata = _morphology_surrogate(source, mask, "fixture", contract, 10)
        self.assertIsNotNone(surrogate)
        np.testing.assert_array_equal(surrogate, mask)
        self.assertEqual(metadata["neighbor_error"], 0.0)

    def test_neighbor_histogram_is_normalized(self) -> None:
        board = np.zeros((8, 8), dtype=np.bool_)
        board[2:4, 2:4] = True
        self.assertAlmostEqual(float(_live_neighbor_hist(board).sum()), 1.0)


class InferenceTests(unittest.TestCase):
    def test_assignment_requires_similarity_and_margin(self) -> None:
        contract = CarrierContract()
        a = np.asarray((1.0, 0.0))
        b = np.asarray((0.0, 1.0))
        self.assertEqual(_assignment(a, a, b, contract), "A")
        self.assertEqual(_assignment(b, a, b, contract), "B")
        self.assertIsNone(_assignment(np.asarray((1.0, 1.0)), a, b, contract))

    def test_holm_adjustment_is_monotone_in_rank(self) -> None:
        adjusted = _holm((0.01, 0.03, 0.04))
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_candidate_seal_is_effect_then_rule_order(self) -> None:
        screen = {
            "9": {"eligible_for_holdout": True, "mean_crossover": 0.2},
            "3": {"eligible_for_holdout": True, "mean_crossover": 0.2},
            "7": {"eligible_for_holdout": False, "mean_crossover": 0.8},
        }
        self.assertEqual(select_holdout_candidates(screen, 2), [3, 9])


if __name__ == "__main__":
    unittest.main()
