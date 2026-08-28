from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plastic_ca.motif_lineage import MotifContract
from plastic_ca.motif_minimality import MINIMALITY_PROFILES, select_minimality_cohorts
from plastic_ca.motif_minimality_repair import (
    FULL_BRIDGE_ID,
    MinimalityRepairContract,
    REPAIR_PROFILES,
    build_repair_candidates,
    corrected_window_moments,
    correction_audit,
    heldout_lineage_diagnostics,
    legacy_window_moments,
    load_frozen_stage6a,
    repair_origins,
    run_motif_minimality_repair,
    simulate_repaired_lineage,
    toroidal_chebyshev_distance,
)
from plastic_ca.motif_minimality import load_frozen_stage5r


class MinimalityRepairPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = load_frozen_stage5r()
        cls.frozen6a = load_frozen_stage6a()

    def test_registered_matrix_and_promotion_bounds(self) -> None:
        candidates, diagnostics = build_repair_candidates(self.frozen)
        self.assertEqual(len(candidates), 60)
        self.assertEqual(len(diagnostics), 3)
        self.assertEqual(len({row["candidate_id"] for row in candidates}), 60)
        self.assertEqual(sum(bool(row["promotable"]) for row in candidates), 24)
        full = next(row for row in candidates if row["candidate_id"] == FULL_BRIDGE_ID)
        self.assertEqual(full["germination_hops"], 8)
        self.assertEqual(full["consolidation_span"], 15)
        self.assertFalse(full["promotable"])

    def test_corrected_writer_matches_explicit_counts_for_all_spans(self) -> None:
        rng = np.random.default_rng(6201)
        signs = rng.choice((-1.0, 1.0), size=(512, 6))
        write_sweeps = 16
        alpha = 0.5
        site_sum = rng.choice(
            (-1.0, 1.0), size=(2, write_sweeps, 16, 16, 6)
        ).sum(axis=1)
        origins = np.asarray([[0, 0], [9, 5]], dtype=np.int16)
        from plastic_ca.motif_minimality import bounded_reduce_endpoint

        for span in (0, 2, 4, 7, 15):
            endpoint = bounded_reduce_endpoint(site_sum, span, origins)
            actual = corrected_window_moments(
                endpoint, signs, span, write_sweeps, alpha
            )
            expected = []
            for sample, (y, x) in enumerate(origins):
                ys = (int(y) - np.arange(span + 1)) % 16
                xs = (int(x) - np.arange(span + 1)) % 16
                count = site_sum[sample][np.ix_(ys, xs)].sum(axis=(0, 1))
                expected.append(
                    (count + alpha * signs.sum(axis=0))
                    / (write_sweeps * (span + 1) ** 2 + 512.0 * alpha)
                )
            np.testing.assert_allclose(actual, np.stack(expected), atol=1e-12)

    def test_legacy_full_span_attenuates_empirical_term_by_256(self) -> None:
        endpoint = np.asarray([[2.0, -3.0, 4.0]])
        signs = np.zeros((512, 3), dtype=np.float64)
        corrected = corrected_window_moments(endpoint, signs, 15, 16, 0.5)
        legacy = legacy_window_moments(endpoint, signs, 15, 16, 0.5)
        np.testing.assert_allclose(legacy, corrected / 256.0, atol=1e-15)

    def test_origin_policies_have_registered_geometry(self) -> None:
        co1 = repair_origins("pair", 1, 12, 16, "co-located")
        co9 = repair_origins("pair", 9, 12, 16, "co-located")
        adjacent1 = repair_origins("pair", 1, 12, 16, "adjacent")
        adjacent2 = repair_origins("pair", 2, 12, 16, "adjacent")
        np.testing.assert_array_equal(co1, co9)
        np.testing.assert_array_equal(
            toroidal_chebyshev_distance(adjacent1, adjacent2, 16),
            np.ones(24, dtype=np.int64),
        )
        translated = repair_origins(
            "pair", 3, 12, 16, "co-located", translated=True
        )
        np.testing.assert_array_equal(
            translated, (co1.astype(np.int64) + np.asarray((5, 7))) % 16
        )

    def test_decoder_ties_are_chance_not_zero(self) -> None:
        result = heldout_lineage_diagnostics(
            np.zeros((16, 4), dtype=np.float64), 8, 6202
        )
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertEqual(result["tie_fraction"], 1.0)

    def test_runtime_api_has_no_label_parent_or_target(self) -> None:
        parameters = inspect.signature(simulate_repaired_lineage).parameters
        for forbidden in ("label", "prototype", "target", "parent_carrier"):
            self.assertNotIn(forbidden, parameters)

    def test_semantic_streams_do_not_depend_on_candidate_label(self) -> None:
        candidates, _ = build_repair_candidates(self.frozen)
        candidate = next(
            row for row in candidates if row["candidate_id"] == FULL_BRIDGE_ID
        )
        renamed = {**candidate, "candidate_id": "same-mechanism-different-label"}
        cohorts = select_minimality_cohorts(
            MINIMALITY_PROFILES["smoke"],
            self.frozen,
            profile_name="smoke",
            open_audit=False,
        )
        arguments = (
            cohorts["locality_screen"][0],
            self.frozen["stage4"]["configuration"],
        )
        first, first_exits = simulate_repaired_lineage(
            *arguments,
            candidate,
            "intact",
            2,
            2,
            self.frozen["stage4"]["reference"],
            MotifContract(),
            MinimalityRepairContract(),
            retain_exits=True,
        )
        second, second_exits = simulate_repaired_lineage(
            *arguments,
            renamed,
            "intact",
            2,
            2,
            self.frozen["stage4"]["reference"],
            MotifContract(),
            MinimalityRepairContract(),
            retain_exits=True,
        )
        self.assertEqual(first["outcomes"], second["outcomes"])
        self.assertEqual(first["carrier_history"], second["carrier_history"])
        for left, right in zip(first_exits, second_exits):
            np.testing.assert_array_equal(left, right)

    def test_real_correction_audit_passes_without_opening_reserve(self) -> None:
        cohorts = select_minimality_cohorts(
            MINIMALITY_PROFILES["smoke"],
            self.frozen,
            profile_name="smoke",
            open_audit=False,
        )
        cohorts["audit"] = []
        result = correction_audit(
            self.frozen,
            self.frozen6a,
            MotifContract(),
            MinimalityRepairContract(),
            cohorts,
        )
        self.assertTrue(result["gate"])
        self.assertTrue(result["full_span_quantized_payload_exact_match"])
        self.assertTrue(result["legacy_checkpoint_collapse_reproduced"])
        self.assertTrue(result["reserve_untouched"])


class MinimalityRepairWorkflowTests(unittest.TestCase):
    def test_worker_and_wall_caps_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_motif_minimality_repair(
                    Path(directory), profile_name="smoke", workers=5
                )
            with self.assertRaises(ValueError):
                run_motif_minimality_repair(
                    Path(directory), profile_name="smoke", max_hours=4.01
                )

    def test_successors_require_resume_and_qualification_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_motif_minimality_repair(
                    Path(directory), phase="bridge", profile_name="smoke"
                )
            with self.assertRaises(ValueError):
                run_motif_minimality_repair(
                    Path(directory),
                    phase="qualify",
                    profile_name="smoke",
                    resume=True,
                )

    def test_audit_phase_writes_review_gate_and_no_successor_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = run_motif_minimality_repair(
                output,
                phase="audit",
                profile_name="smoke",
                workers=1,
                max_hours=1.0,
            )
            self.assertTrue(result["stage_gate"])
            queue = __import__("json").loads(
                (output / "QUEUE.json").read_text(encoding="utf-8")
            )
            self.assertEqual(queue["next_phase"], "bridge")
            self.assertFalse(queue["automatic_launch"])
            cohorts = __import__("json").loads(
                (output / "COHORTS.json").read_text(encoding="utf-8")
            )
            self.assertEqual(cohorts["final_audit_trajectory_state"], "untouched")
            self.assertTrue(cohorts["final_audit_trajectories_not_loaded"])


if __name__ == "__main__":
    unittest.main()
