from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plastic_ca.motif_repair import (
    REPAIR_PROFILES,
    RepairContract,
    _apply_boundary_intervention,
    _fit_diagonal,
    _fit_reduced_rank,
    _fit_scalar,
    _fold_assignments,
    _predict_fit,
    apply_repair,
    build_simple_models,
    carrier_transition_metrics,
    cross_validate_repair,
    heldout_lineage_accuracy,
    load_frozen_stage3,
    load_repair_models,
    run_motif_repair,
    save_repair_models,
    select_repair_cohorts,
    simulate_repair_lineage,
)


class RepairPrimitiveTests(unittest.TestCase):
    def test_runtime_repair_api_has_no_parent_label_or_target(self) -> None:
        parameters = inspect.signature(apply_repair).parameters
        self.assertEqual(tuple(parameters), ("raw_carrier", "model"))
        simulator = inspect.signature(simulate_repair_lineage).parameters
        for forbidden in ("label", "prototype", "target", "parent_carrier"):
            self.assertNotIn(forbidden, simulator)

    def test_simple_gain_and_gauge_norm_are_exact(self) -> None:
        raw = np.asarray([[1.0, -1.0] * 256], dtype=np.float32)
        gain = {
            "kind": "gain-200",
            "gain": 2.0,
        }
        np.testing.assert_array_equal(apply_repair(raw, gain), raw * 2.0)
        gauge = {"kind": "gauge-norm", "norm_target": 3.0}
        repaired = apply_repair(raw, gauge)
        self.assertAlmostEqual(float(repaired.mean()), 0.0, places=7)
        self.assertAlmostEqual(float(np.linalg.norm(repaired)), 3.0, places=6)

    def test_learned_fitters_recover_a_synthetic_channel(self) -> None:
        rng = np.random.default_rng(701)
        x = rng.normal(size=(40, 512))
        y = 1.7 * x - 0.2
        scalar = _fit_scalar(x, y, 1e-8)
        np.testing.assert_allclose(_predict_fit(x, "scalar-affine", scalar), y, atol=1e-7)
        diagonal = _fit_diagonal(x, y, 1e-8)
        np.testing.assert_allclose(_predict_fit(x, "diagonal-ridge", diagonal), y, atol=1e-6)
        low_rank_y = np.outer(rng.normal(size=40), rng.normal(size=512))
        reduced = _fit_reduced_rank(x, low_rank_y, 1e-4, 8)
        prediction = _predict_fit(x, "reduced-rank-ridge", reduced)
        self.assertEqual(prediction.shape, low_rank_y.shape)
        self.assertTrue(np.isfinite(prediction).all())

    def test_cross_validation_keeps_whole_pairs_in_one_fold(self) -> None:
        groups = [f"pair-{index // 8}" for index in range(64)]
        assignments = _fold_assignments(groups, 8)
        for group in set(groups):
            observed = {int(assignments[index]) for index, value in enumerate(groups) if value == group}
            self.assertEqual(len(observed), 1)
        rng = np.random.default_rng(702)
        x = rng.normal(size=(64, 512))
        y = 0.8 * x + 0.1
        score = cross_validate_repair(x, y, groups, "scalar-affine", 1e-4)
        self.assertLess(score["normalized_error"], 0.01)

    def test_lineage_decoder_uses_disjoint_replicate_halves(self) -> None:
        rng = np.random.default_rng(703)
        first = rng.normal(-2.0, 0.1, size=(16, 8))
        second = rng.normal(2.0, 0.1, size=(16, 8))
        accuracy = heldout_lineage_accuracy(np.concatenate((first, second)), 16, 44)
        self.assertEqual(accuracy, 1.0)

    def test_transition_metrics_detect_identity_and_reversal(self) -> None:
        rng = np.random.default_rng(704)
        entry = rng.normal(size=(8, 512)).astype(np.float32)
        founder_delta = entry[:4].mean(axis=0) - entry[4:].mean(axis=0)
        identity = carrier_transition_metrics(entry, entry, founder_delta, 4)
        self.assertAlmostEqual(identity["normalized_rmse"], 0.0)
        self.assertAlmostEqual(identity["parent_child_delta_cosine"], 1.0, places=6)
        reversed_carrier = np.concatenate((entry[4:], entry[:4]))
        reversed_metrics = carrier_transition_metrics(
            entry, reversed_carrier, founder_delta, 4
        )
        self.assertAlmostEqual(
            reversed_metrics["parent_child_delta_cosine"], -1.0, places=6
        )

    def test_boundary_ablation_and_rescue_are_exact(self) -> None:
        contract = RepairContract()
        carrier = np.arange(4 * 512, dtype=np.float32).reshape(4, 512)
        sources = [carrier + 1.0, carrier + 2.0, carrier + 3.0]
        ablated = _apply_boundary_intervention(
            carrier, "ablate_after_g2", 3, "pair", 2, contract, None
        )
        np.testing.assert_array_equal(ablated, np.zeros_like(carrier))
        same = _apply_boundary_intervention(
            carrier, "rescue_same_enter_g4", 4, "pair", 2, contract, sources
        )
        opposite = _apply_boundary_intervention(
            carrier, "rescue_opposite_enter_g4", 4, "pair", 2, contract, sources
        )
        np.testing.assert_array_equal(same, sources[2])
        np.testing.assert_array_equal(opposite[:2], sources[2][2:])
        np.testing.assert_array_equal(opposite[2:], sources[2][:2])

    def test_model_archive_is_non_pickled_and_hash_checked(self) -> None:
        models = build_simple_models(("strict-33-64",), 4.0)
        learned = {
            "candidate_id": "learned--strict-33-64--diagonal-ridge",
            "mechanism_class": "learned",
            "window_id": "strict-33-64",
            "tier": "strict",
            "kind": "diagonal-ridge",
            "complexity": 1024,
            "slope": np.ones(512, dtype=np.float32),
            "intercept": np.zeros(512, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_repair_models(
                root,
                [models[0], learned],
                design_digest="design",
                diagnostic_digest="diagnostic",
            )
            loaded = load_repair_models(root, "design")
            self.assertEqual(len(loaded), 2)
            np.testing.assert_array_equal(loaded[1]["slope"], learned["slope"])
            with np.load(root / "REPAIR_MODELS.npz", allow_pickle=False) as archive:
                self.assertTrue(all(archive[key].dtype != object for key in archive.files))


class FrozenRepairCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = load_frozen_stage3()
        cls.contract = RepairContract()

    def test_reference_cohorts_are_fresh_disjoint_and_leave_382_pairs(self) -> None:
        cohorts = select_repair_cohorts(
            REPAIR_PROFILES["reference"], self.frozen, self.contract
        )
        selection = {pair["pair_id"] for pair in cohorts["selection"]}
        confirmation = {pair["pair_id"] for pair in cohorts["confirmation"]}
        self.assertEqual(len(selection), 64)
        self.assertEqual(len(confirmation), 96)
        self.assertFalse(selection & confirmation)
        self.assertFalse((selection | confirmation) & self.frozen["used_pair_ids"])
        untouched = len(self.frozen["all_pairs"]) - len(self.frozen["used_pair_ids"])
        self.assertEqual(untouched - len(selection) - len(confirmation), 382)

    def test_smoke_reuses_exposed_engineering_pairs(self) -> None:
        cohorts = select_repair_cohorts(
            REPAIR_PROFILES["smoke"], self.frozen, self.contract
        )
        ids = {pair["pair_id"] for pair in cohorts["selection"]}
        self.assertTrue(ids <= self.frozen["used_pair_ids"])

    def test_confirmation_requires_separate_explicit_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_motif_repair(
                    Path(directory), profile_name="smoke", phases=("confirm",)
                )
            with self.assertRaises(ValueError):
                run_motif_repair(
                    Path(directory),
                    profile_name="smoke",
                    phases=("diagnose", "confirm"),
                    authorize_confirmation=True,
                    resume=True,
                )

    def test_maximum_wall_time_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_motif_repair(Path(directory), profile_name="smoke", max_hours=8.01)

    def test_smoke_campaign_stops_before_confirmation_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage3r"
            first = run_motif_repair(
                output,
                profile_name="smoke",
                workers=2,
                max_hours=0.20,
            )
            self.assertEqual(first["state"], "awaiting_human_review")
            self.assertTrue((output / "PRECONFIRMATION_COMPLETE").exists())
            self.assertTrue((output / "CONFIRMATION_DESIGN.json").exists())
            self.assertFalse((output / "confirmation").exists())
            second = run_motif_repair(
                output,
                profile_name="smoke",
                workers=1,
                max_hours=0.20,
                resume=True,
            )
            self.assertEqual(
                first["selection_decision"], second["selection_decision"]
            )


if __name__ == "__main__":
    unittest.main()
