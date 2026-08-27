from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

import numpy as np

from plastic_ca.lineage_field import (
    FIELD_PROFILES,
    LineageFieldContract,
    MechanismParameters,
    _simulate_condition,
    apply_field_reader,
    block_compress,
    calibrate_mechanism,
    choose_timing_profile,
    diffuse_write_step,
    latch_write,
    load_round3_pairs,
    random_retain,
)


class FieldPrimitiveTests(unittest.TestCase):
    def test_signed_reader_is_generic_and_directional(self) -> None:
        predicted = np.asarray([[[False, True], [False, True]]])
        carrier = np.asarray([[[1.0, -1.0], [-1.0, 1.0]]], dtype=np.float32)
        uniforms = np.zeros_like(carrier)
        result = apply_field_reader(predicted, carrier, uniforms, 1.0)
        expected = np.asarray([[[True, False], [False, True]]])
        np.testing.assert_array_equal(result, expected)
        self.assertNotIn("label", inspect.signature(apply_field_reader).parameters)
        self.assertNotIn("prototype", inspect.signature(apply_field_reader).parameters)

    def test_latch_has_hysteretic_middle_region(self) -> None:
        old = np.asarray([[[-0.4, 0.3, 0.2]]], dtype=np.float32)
        occupancy = np.asarray([[[0.2, 0.5, 0.8]]], dtype=np.float32)
        result = latch_write(old, occupancy, 0.6, 0.4)
        np.testing.assert_allclose(result, [[[-1.0, 0.3, 1.0]]])

    def test_diffusion_is_translation_equivariant(self) -> None:
        carrier = np.zeros((1, 16, 16), dtype=np.float32)
        visible = np.zeros((1, 16, 16), dtype=np.bool_)
        carrier[0, 3, 5] = 0.75
        visible[0, 7, 2] = True
        first = diffuse_write_step(carrier, visible, 0.08, 0.12)
        shifted = diffuse_write_step(
            np.roll(carrier, (4, 6), axis=(1, 2)),
            np.roll(visible, (4, 6), axis=(1, 2)),
            0.08,
            0.12,
        )
        np.testing.assert_allclose(np.roll(first, (4, 6), axis=(1, 2)), shifted)

    def test_block_compression_has_registered_degrees_of_freedom(self) -> None:
        carrier = np.arange(256, dtype=np.float32).reshape(1, 16, 16)
        for block, degrees in ((2, 64), (4, 16), (8, 4), (16, 1)):
            compressed = block_compress(carrier, block)
            self.assertEqual(len(np.unique(compressed)), degrees)

    def test_random_retention_is_exact_and_deterministic(self) -> None:
        carrier = np.ones((2, 16, 16), dtype=np.float32)
        first = random_retain(carrier, 16, 123)
        second = random_retain(carrier, 16, 123)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(np.count_nonzero(first[0]), 16)
        self.assertEqual(np.count_nonzero(first[1]), 16)


class CalibrationAndLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pair = load_round3_pairs()[31649][0]

    def test_calibration_is_label_blind_and_selects_each_mechanism(self) -> None:
        self.assertNotIn("label", inspect.signature(calibrate_mechanism).parameters)
        for mechanism in ("latch", "diffuse"):
            calibration = calibrate_mechanism(mechanism, LineageFieldContract())
            self.assertEqual(calibration["mechanism"], mechanism)
            self.assertEqual(calibration["selected"]["mechanism"], mechanism)
            self.assertGreater(calibration["candidate_count"], 1)

    def test_visible_reset_mismatch_is_rejected(self) -> None:
        pair = dict(self.pair)
        pair["donor_a"] = dict(pair["donor_a"])
        pair["donor_b"] = dict(pair["donor_b"])
        pair["donor_b"]["initial_state_hex"] = "1" + pair["donor_b"]["initial_state_hex"][1:]
        with self.assertRaises(AssertionError):
            _simulate_condition(
                pair,
                MechanismParameters("latch", 0.05, 0.55),
                "intact",
                1,
                1,
                LineageFieldContract(process_noise=0.0),
            )

    def test_one_generation_is_deterministic_and_records_reset_hash(self) -> None:
        contract = replace(LineageFieldContract(), process_noise=0.0)
        parameters = MechanismParameters("latch", 0.05, 0.55)
        first, _ = _simulate_condition(self.pair, parameters, "intact", 1, 1, contract)
        second, _ = _simulate_condition(self.pair, parameters, "intact", 1, 1, contract)
        self.assertEqual(first, second)
        self.assertEqual(len(first["reset_sha256"]), 64)
        self.assertIn("1", first["outcomes"])

    def test_timing_selector_is_symmetric_and_bounded(self) -> None:
        name, projections = choose_timing_profile(1e-6, 8.0, 20)
        self.assertEqual(name, "reference")
        name, projections = choose_timing_profile(10.0, 8.0, 20)
        self.assertEqual(name, "floor")
        self.assertEqual(FIELD_PROFILES[name].generations, 16)


if __name__ == "__main__":
    unittest.main()
