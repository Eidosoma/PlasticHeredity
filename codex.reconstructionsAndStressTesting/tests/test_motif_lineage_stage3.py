from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plastic_ca.motif_lineage import MotifContract, motif3_codes
from plastic_ca.motif_lineage_stage3 import (
    STAGE3_PROFILES,
    Stage3Contract,
    _apply_entry_intervention,
    load_frozen_stage2,
    motif_counts_batch,
    run_motif_lineage_stage3,
    select_stage3_pairs,
    simulate_lineage,
    write_energy_from_counts,
)


class Stage3PrimitiveTests(unittest.TestCase):
    def test_batched_motif_counts_match_individual_bincounts(self) -> None:
        rng = np.random.default_rng(303)
        codes = rng.integers(0, 512, size=(5, 16, 16), dtype=np.uint16)
        observed = motif_counts_batch(codes)
        expected = np.stack(
            [np.bincount(row.ravel(), minlength=512) for row in codes]
        ).astype(np.float64)
        np.testing.assert_array_equal(observed, expected)

    def test_daughter_writer_matches_registered_smoothed_log_frequency(self) -> None:
        rng = np.random.default_rng(304)
        counts = rng.integers(0, 100, size=(3, 512)).astype(np.float64)
        reference = rng.random(512)
        reference /= reference.sum()
        contract = MotifContract()
        observed = write_energy_from_counts(counts, reference, contract)
        probability = (counts + contract.jeffreys_alpha) / (
            counts.sum(axis=1, keepdims=True)
            + 512.0 * contract.jeffreys_alpha
        )
        expected = np.clip(
            np.log(probability) - np.log(reference[None, :]),
            -contract.energy_clip,
            contract.energy_clip,
        ).astype(np.float32)
        np.testing.assert_allclose(observed, expected)

    def test_ablation_and_rescues_modify_only_the_registered_boundary(self) -> None:
        contract = Stage3Contract()
        replicates = 2
        carrier = np.arange(4 * 512, dtype=np.float32).reshape(4, 512)
        sources = [carrier + 10.0, carrier + 20.0, carrier + 30.0]
        ablated = _apply_entry_intervention(
            carrier, "ablate_after_g2", 3, "pair", replicates, contract, None
        )
        np.testing.assert_array_equal(ablated, np.zeros_like(carrier))
        same = _apply_entry_intervention(
            carrier, "rescue_same_enter_g4", 4, "pair", replicates, contract, sources
        )
        opposite = _apply_entry_intervention(
            carrier, "rescue_opposite_enter_g4", 4, "pair", replicates, contract, sources
        )
        np.testing.assert_array_equal(same, sources[2])
        np.testing.assert_array_equal(opposite[:replicates], sources[2][replicates:])
        np.testing.assert_array_equal(opposite[replicates:], sources[2][:replicates])

    def test_corruption_mask_is_paired_across_histories(self) -> None:
        contract = Stage3Contract()
        carrier = np.ones((6, 512), dtype=np.float32)
        corrupted = _apply_entry_intervention(
            carrier,
            "carrier_corruption_1",
            5,
            "pair",
            3,
            contract,
            None,
        )
        np.testing.assert_array_equal(corrupted[:3] < 0.0, corrupted[3:] < 0.0)


class FrozenStage3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = load_frozen_stage2()
        cls.contract = Stage3Contract()
        cls.writer_contract = MotifContract()
        cls.smoke_pairs = select_stage3_pairs(
            STAGE3_PROFILES["smoke"], cls.frozen, cls.contract
        )

    def test_reader_is_exact_frozen_stage2_winner(self) -> None:
        configuration = self.frozen["configuration"]
        self.assertEqual(configuration.id, "motif_energy512-w32-s025-d32")
        self.assertEqual(configuration.strength, 0.25)
        self.assertEqual(configuration.read_duration, 32)
        self.assertEqual(configuration.write_window, 32)

    def test_all_stage3_cohorts_are_fresh_and_disjoint(self) -> None:
        smoke = {row["pair_id"] for row in self.smoke_pairs}
        reference = {
            row["pair_id"]
            for row in select_stage3_pairs(
                STAGE3_PROFILES["reference"], self.frozen, self.contract
            )
        }
        pilot = {
            row["pair_id"]
            for row in select_stage3_pairs(
                STAGE3_PROFILES["pilot"], self.frozen, self.contract
            )
        }
        self.assertFalse(smoke & reference)
        self.assertFalse(smoke & pilot)
        self.assertFalse(reference & pilot)
        self.assertFalse((smoke | reference | pilot) & self.frozen["used_pair_ids"])

    def test_simulator_is_label_and_prototype_blind(self) -> None:
        parameters = inspect.signature(simulate_lineage).parameters
        self.assertNotIn("label", parameters)
        self.assertNotIn("prototype", parameters)

    def test_no_rewrite_carrier_decays_exactly_by_half(self) -> None:
        result, _ = simulate_lineage(
            self.smoke_pairs[0],
            self.frozen["configuration"],
            "no_rewrite",
            2,
            4,
            self.frozen["reference"],
            self.writer_contract,
            self.contract,
        )
        history = result["carrier_history"]
        founder = result["founder_carrier"]["mean_abs"]
        self.assertAlmostEqual(history["1"]["entry"]["mean_abs"], founder, places=7)
        self.assertAlmostEqual(history["2"]["entry"]["mean_abs"], founder * 0.5, places=7)
        self.assertAlmostEqual(history["4"]["entry"]["mean_abs"], founder * 0.125, places=7)
        self.assertTrue(result["reset_asserted_before_every_generation"])

    def test_native_reset_is_shared_between_histories(self) -> None:
        pair = self.smoke_pairs[0]
        self.assertEqual(
            pair["donor_a"]["initial_state_hex"],
            pair["donor_b"]["initial_state_hex"],
        )

    def test_smoke_campaign_completes_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage3"
            first = run_motif_lineage_stage3(
                output,
                profile_name="smoke",
                workers=2,
                max_hours=0.10,
            )
            self.assertEqual(first["state"], "complete")
            self.assertEqual(
                first["adjudication"]["verdict"], "NOT_ADJUDICATED_PROFILE"
            )
            self.assertTrue((output / "COMPLETE").exists())
            self.assertTrue((output / "STAGE_DECISION.json").exists())
            second = run_motif_lineage_stage3(
                output,
                profile_name="smoke",
                workers=1,
                max_hours=0.10,
                resume=True,
            )
            self.assertEqual(first["adjudication"], second["adjudication"])

    def test_maximum_wall_time_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_motif_lineage_stage3(
                    Path(directory), profile_name="smoke", max_hours=8.01
                )


if __name__ == "__main__":
    unittest.main()
