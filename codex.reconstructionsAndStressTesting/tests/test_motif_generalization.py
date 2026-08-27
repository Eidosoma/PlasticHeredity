from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plastic_ca.motif_generalization import (
    GENERALIZATION_PROFILES,
    GeneralizationContract,
    _forward_board,
    environment_reset,
    launch_reset_bank,
    load_frozen_stage1,
    mix_history_carriers,
    motif_code_permutation,
    run_motif_generalization,
    select_stage2_pairs,
    simulate_generalization_condition,
    transform_energy_carrier,
    writer_audit,
)
from plastic_ca.motif_lineage import (
    MotifContract,
    _founders,
    motif3_codes,
    write_parent_carriers,
)


class SymmetryAndCarrierTests(unittest.TestCase):
    def test_code_permutations_match_transformed_boards(self) -> None:
        rng = np.random.default_rng(44)
        state = rng.random((2, 16, 16)) < 0.47
        codes = motif3_codes(state)
        for environment in ("native_rot90", "native_reflect_x"):
            permutation = motif_code_permutation(environment)
            expected = _forward_board(permutation[codes], environment)
            observed = motif3_codes(_forward_board(state, environment))
            np.testing.assert_array_equal(observed, expected)

    def test_transformed_carrier_preserves_total_energy(self) -> None:
        rng = np.random.default_rng(45)
        state = rng.random((2, 16, 16)) < 0.5
        carrier = rng.normal(size=(2, 512)).astype(np.float32)
        original = np.take_along_axis(carrier, motif3_codes(state).reshape(2, -1), axis=1).sum(axis=1)
        for environment in ("native_rot90", "native_reflect_x"):
            transformed_state = _forward_board(state, environment)
            transformed_carrier = transform_energy_carrier(carrier, environment)
            transformed = np.take_along_axis(
                transformed_carrier, motif3_codes(transformed_state).reshape(2, -1), axis=1
            ).sum(axis=1)
            np.testing.assert_allclose(transformed, original, rtol=1e-6, atol=1e-5)

    def test_carrier_mixture_has_registered_endpoints(self) -> None:
        rng = np.random.default_rng(46)
        carrier = rng.normal(size=(2, 512)).astype(np.float32)
        midpoint = mix_history_carriers(carrier, 0.0)
        intact = mix_history_carriers(carrier, 1.0)
        np.testing.assert_allclose(midpoint[0], midpoint[1])
        np.testing.assert_allclose(intact, carrier)


class FrozenGeneralizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = load_frozen_stage1()
        cls.contract = GeneralizationContract()
        cls.writer_contract = MotifContract()
        cls.profile = GENERALIZATION_PROFILES["smoke"]
        cls.pairs = select_stage2_pairs(cls.profile, cls.frozen, cls.contract)
        cls.bank = launch_reset_bank()

    def test_reader_is_exact_frozen_stage1_winner(self) -> None:
        configuration = self.frozen["configuration"]
        self.assertEqual(configuration.id, "motif_energy512-w32-s025-d32")
        self.assertEqual(configuration.strength, 0.25)
        self.assertEqual(configuration.read_duration, 32)
        self.assertEqual(configuration.write_window, 32)

    def test_stage2_pairs_exclude_every_stage1_pair(self) -> None:
        self.assertFalse(
            {pair["pair_id"] for pair in self.pairs} & self.frozen["used_pair_ids"]
        )

    def test_reset_bank_and_density_reset_are_exact(self) -> None:
        self.assertEqual(set(self.bank), {0, 1, 2, 3})
        pair = self.pairs[0]
        first = environment_reset(pair, "random_density_30", self.bank, self.contract)
        second = environment_reset(pair, "random_density_30", self.bank, self.contract)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(int(first.sum()), round(256 * 0.30))

    def test_writer_audit_is_equivariant_and_label_separating(self) -> None:
        reference_profile = GENERALIZATION_PROFILES["reference"]
        reference_pairs = select_stage2_pairs(reference_profile, self.frozen, self.contract)
        audit = writer_audit(
            reference_pairs[: reference_profile.audit_pairs],
            self.frozen["configuration"],
            self.frozen["reference"],
            self.writer_contract,
            self.contract,
        )
        self.assertTrue(audit["writer_gate"])
        self.assertGreaterEqual(audit["leave_one_pair_out_accuracy"], 0.80)
        self.assertEqual(
            audit["raw_motif_symmetry_max_abs_error"],
            {"translation": 0.0, "rotation90": 0.0, "reflection_x": 0.0},
        )

    def test_midpoint_gives_identical_paired_histories(self) -> None:
        pair = self.pairs[0]
        source = self.pairs[1]
        configuration = self.frozen["configuration"]
        written = write_parent_carriers(
            _founders(pair), (configuration.write_window,), self.frozen["reference"], self.writer_contract
        )[configuration.write_window]
        source_written = write_parent_carriers(
            _founders(source), (configuration.write_window,), self.frozen["reference"], self.writer_contract
        )[configuration.write_window]
        result = simulate_generalization_condition(
            pair,
            configuration,
            written[configuration.family],
            written["terminal"],
            source_written[configuration.family],
            self.bank,
            "native",
            "midpoint",
            2,
            self.writer_contract,
            self.contract,
        )
        for outcome in result["outcomes"].values():
            self.assertEqual(outcome["primary"]["crossover"], 0.0)

    def test_simulator_has_no_label_or_prototype_parameters(self) -> None:
        parameters = inspect.signature(simulate_generalization_condition).parameters
        self.assertNotIn("label", parameters)
        self.assertNotIn("prototype", parameters)

    def test_smoke_campaign_completes_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage2"
            first = run_motif_generalization(
                output,
                profile_name="smoke",
                workers=2,
                max_hours=0.10,
            )
            self.assertEqual(first["state"], "complete")
            self.assertTrue((output / "COMPLETE").exists())
            self.assertTrue((output / "STAGE_DECISION.json").exists())
            second = run_motif_generalization(
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
                run_motif_generalization(Path(directory), profile_name="smoke", max_hours=8.1)


if __name__ == "__main__":
    unittest.main()
