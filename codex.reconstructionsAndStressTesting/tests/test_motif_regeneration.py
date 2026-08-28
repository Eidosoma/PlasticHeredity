from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plastic_ca.motif_compression import load_stage3r_fit_matrix
from plastic_ca.motif_regeneration import (
    REGENERATION_PROFILES,
    RegenerationContract,
    build_regenerative_candidates,
    calibrate_regenerative_dynamics,
    fit_regenerative_writers,
    germinate_payload,
    load_frozen_stage5,
    regenerative_mechanism_audit,
    ring_reduce_exact,
    run_motif_regeneration,
    select_regeneration_cohorts,
    simulate_regenerative_lineage,
)


class RegenerationPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = load_frozen_stage5()
        cls.stage4 = cls.frozen["stage4"]
        cls.model = cls.stage4["winner_model"]

    def test_runtime_api_has_no_label_parent_prototype_or_target(self) -> None:
        parameters = inspect.signature(simulate_regenerative_lineage).parameters
        for forbidden in ("label", "prototype", "target", "parent_carrier"):
            self.assertNotIn(forbidden, parameters)

    def test_one_site_seed_fills_torus_exactly_in_eight_local_steps(self) -> None:
        rng = np.random.default_rng(551)
        payload = rng.normal(size=(2, 1, 1, 16)).astype(np.float32)
        origins = np.asarray([[0, 0], [9, 5]], dtype=np.int16)
        field, occupied, trace = germinate_payload(payload, origins, "flood-retain", 8)
        expected = np.broadcast_to(payload, field.shape)
        np.testing.assert_array_equal(field, expected)
        self.assertTrue(np.all(occupied))
        self.assertEqual(trace[-1]["occupied_fraction"], 1.0)

    def test_wave_respects_light_cone_and_exact_zero(self) -> None:
        payload = np.ones((1, 1, 1, 1), dtype=np.float32)
        origin = np.asarray([[8, 8]], dtype=np.int16)
        field, _, _ = germinate_payload(payload, origin, "flood-consensus", 3)
        for y, x in np.argwhere(field[0, ..., 0] != 0.0):
            dy = min(abs(int(y) - 8), 16 - abs(int(y) - 8))
            dx = min(abs(int(x) - 8), 16 - abs(int(x) - 8))
            self.assertLessEqual(max(dy, dx), 3)
        zero, _, _ = germinate_payload(np.zeros_like(payload), origin, "flood-consensus", 8)
        np.testing.assert_array_equal(zero, np.zeros_like(zero))

    def test_thirty_step_ring_reduction_is_spatial_mean(self) -> None:
        rng = np.random.default_rng(552)
        values = rng.normal(size=(3, 16, 16, 9))
        reduced = ring_reduce_exact(values)
        expected = np.broadcast_to(values.mean(axis=(1, 2), keepdims=True), values.shape)
        np.testing.assert_allclose(reduced, expected, atol=1e-12)

    def test_registered_writer_and_candidate_atlas(self) -> None:
        fit_matrix, _, _ = load_stage3r_fit_matrix(self.stage4["stage3r"])
        writers, audit = fit_regenerative_writers(
            fit_matrix,
            self.stage4["reference"][32]["motif_probability"],
            self.model,
        )
        calibration = calibrate_regenerative_dynamics(
            self.model["quantizer_scale"], RegenerationContract()
        )
        candidates = build_regenerative_candidates(writers, calibration)
        self.assertEqual(len(writers), 3)
        self.assertEqual(len(candidates), 12)
        self.assertGreater(audit["full_ridge_r2"], 0.9)
        self.assertEqual(
            {candidate["payload_bits"] for candidate in candidates}, {64, 256}
        )
        self.assertEqual(len(calibration["selected_propagators"]), 2)

    def test_mechanism_audit_passes(self) -> None:
        calibration = calibrate_regenerative_dynamics(
            self.model["quantizer_scale"], RegenerationContract()
        )
        self.assertTrue(regenerative_mechanism_audit(self.model, calibration)["passed"])


class FrozenRegenerationCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = load_frozen_stage5()

    def test_reference_split_uses_96_and_leaves_62(self) -> None:
        cohorts = select_regeneration_cohorts(
            REGENERATION_PROFILES["reference"], self.frozen, profile_name="reference"
        )
        confirmation = {pair["pair_id"] for pair in cohorts["confirmation"]}
        later = {pair["pair_id"] for pair in cohorts["later_audit"]}
        exposed = {
            *self.frozen["cohorts"]["anatomy_pair_ids"],
            *self.frozen["cohorts"]["screen_pair_ids"],
            *self.frozen["cohorts"]["qualification_pair_ids"],
        }
        self.assertEqual((len(confirmation), len(later)), (96, 62))
        self.assertFalse(confirmation & later)
        self.assertFalse(confirmation & exposed)
        self.assertFalse(later & exposed)

    def test_confirmation_requires_separate_explicit_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_motif_regeneration(
                    Path(directory), profile_name="smoke", phases=("confirm",)
                )
            with self.assertRaises(ValueError):
                run_motif_regeneration(
                    Path(directory),
                    profile_name="smoke",
                    phases=("fit", "confirm"),
                    resume=True,
                    authorize_confirmation=True,
                )

    def test_eight_hour_wall_limit_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_motif_regeneration(
                    Path(directory), profile_name="smoke", max_hours=8.01
                )

    def test_audit_only_keeps_confirmation_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "stage5r"
            result = run_motif_regeneration(
                root,
                profile_name="smoke",
                phases=("audit",),
                workers=1,
                max_hours=0.05,
            )
            self.assertEqual(result["state"], "phases_complete")
            self.assertTrue((root / "CLEANROOM_AUDIT.json").exists())
            self.assertFalse((root / "confirmation").exists())


if __name__ == "__main__":
    unittest.main()
