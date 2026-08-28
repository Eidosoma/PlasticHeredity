from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plastic_ca.motif_lineage import (
    MOTIF_PROFILES,
    MotifContract,
    ReaderConfiguration,
    _paired_uniforms,
    _step,
    apply_contextual_reader,
    apply_energy_reader,
    build_reference,
    context8_codes,
    motif3_codes,
    motif_energy_advantage,
    run_motif_lineage_stage1,
    select_cohorts,
    write_parent_carriers,
)
from plastic_ca.causal_heredity import _state_from_hex


class MotifPrimitiveTests(unittest.TestCase):
    def test_motif_addresses_are_translation_equivariant(self) -> None:
        rng = np.random.default_rng(17)
        state = rng.random((3, 16, 16)) < 0.43
        shift = (4, -3)
        np.testing.assert_array_equal(
            motif3_codes(np.roll(state, shift, axis=(1, 2))),
            np.roll(motif3_codes(state), shift, axis=(1, 2)),
        )
        np.testing.assert_array_equal(
            context8_codes(np.roll(state, shift, axis=(1, 2))),
            np.roll(context8_codes(state), shift, axis=(1, 2)),
        )

    def test_context_code_excludes_centre(self) -> None:
        state = np.zeros((1, 16, 16), dtype=np.bool_)
        before = context8_codes(state)
        state[0, 7, 7] = True
        after = context8_codes(state)
        self.assertEqual(int(after[0, 7, 7]), int(before[0, 7, 7]))
        self.assertNotEqual(int(motif3_codes(state)[0, 7, 7]), 0)

    def test_zero_carriers_are_exactly_inert(self) -> None:
        rng = np.random.default_rng(9)
        predicted = rng.random((2, 16, 16)) < 0.5
        uniforms = rng.random(predicted.shape)
        contextual = apply_contextual_reader(
            predicted,
            context8_codes(predicted),
            np.zeros((2, 256), dtype=np.float32),
            uniforms,
            1.0,
        )
        energy = apply_energy_reader(
            predicted,
            np.zeros((2, 512), dtype=np.float32),
            uniforms,
            1.0,
        )
        np.testing.assert_array_equal(contextual, predicted)
        np.testing.assert_array_equal(energy, predicted)

    def test_energy_advantage_matches_brute_force(self) -> None:
        rng = np.random.default_rng(42)
        state = rng.random((1, 16, 16)) < 0.5
        carrier = rng.normal(size=(1, 512)).astype(np.float32)
        advantage = motif_energy_advantage(state, carrier)
        y, x = 6, 11

        def energy(board: np.ndarray) -> float:
            codes = motif3_codes(board)[0]
            return float(carrier[0, codes].sum())

        flipped = state.copy()
        flipped[0, y, x] ^= True
        self.assertAlmostEqual(float(advantage[0, y, x]), energy(flipped) - energy(state), places=5)

    def test_paired_rng_repeats_across_histories(self) -> None:
        values = _paired_uniforms("pair", "read", 3, 5)
        np.testing.assert_array_equal(values[:5], values[5:])

    def test_readers_have_no_label_or_prototype_inputs(self) -> None:
        for function in (apply_contextual_reader, apply_energy_reader):
            parameters = inspect.signature(function).parameters
            self.assertNotIn("label", parameters)
            self.assertNotIn("prototype", parameters)


class MotifWriterAndCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = MotifContract()
        cls.profile = MOTIF_PROFILES["smoke"]
        cls.cohorts = select_cohorts(cls.profile, cls.contract)

    def test_cohorts_are_disjoint(self) -> None:
        groups = [set(row["pair_id"] for row in self.cohorts[name]) for name in ("calibration", "discovery", "validation")]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])

    def test_writer_is_translation_invariant_and_label_blind(self) -> None:
        reference = build_reference(self.cohorts["calibration"], (16,), self.contract)
        pair = self.cohorts["discovery"][0]
        founders = np.stack(
            (
                _state_from_hex("life", pair["donor_a"]["donor_state_hex"]),
                _state_from_hex("life", pair["donor_b"]["donor_state_hex"]),
            )
        )
        first = write_parent_carriers(founders, (16,), reference, self.contract)[16]
        shifted = write_parent_carriers(
            np.roll(founders, (3, 5), axis=(1, 2)), (16,), reference, self.contract
        )[16]
        np.testing.assert_allclose(first["contextual256"], shifted["contextual256"])
        np.testing.assert_allclose(first["motif_energy512"], shifted["motif_energy512"])
        self.assertNotIn("label", inspect.signature(write_parent_carriers).parameters)
        self.assertNotIn("prototype", inspect.signature(write_parent_carriers).parameters)

    def test_visible_reset_mismatch_is_rejected_by_campaign_inputs(self) -> None:
        pair = self.cohorts["discovery"][0]
        self.assertEqual(
            pair["donor_a"]["initial_state_hex"], pair["donor_b"]["initial_state_hex"]
        )

    def test_smoke_campaign_is_complete_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage1"
            first = run_motif_lineage_stage1(
                output,
                profile_name="smoke",
                workers=2,
                max_hours=0.10,
            )
            self.assertEqual(first["state"], "complete")
            self.assertTrue((output / "COMPLETE").exists())
            self.assertTrue((output / "STAGE_DECISION.json").exists())
            self.assertTrue((output / "QUEUE.json").exists())
            second = run_motif_lineage_stage1(
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
                run_motif_lineage_stage1(
                    Path(directory), profile_name="smoke", max_hours=8.01
                )


if __name__ == "__main__":
    unittest.main()
