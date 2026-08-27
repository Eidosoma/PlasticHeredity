from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plastic_ca.motif_localization import apply_local_reader
from plastic_ca.motif_minimality import (
    MINIMALITY_PROFILES,
    MinimalityContract,
    apply_masked_payload_reader,
    bounded_reduce_endpoint,
    bounded_shift_reduce,
    build_locality_candidates,
    embed_dynamic_seed,
    load_frozen_stage5r,
    propagate_bounded,
    run_motif_minimality,
    select_minimality_cohorts,
    simulate_bounded_lineage,
    stage6_mechanism_audit,
)


class MinimalityPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = load_frozen_stage5r()

    def test_registered_factorial_and_bounds(self) -> None:
        candidates, anchors = build_locality_candidates(self.frozen)
        self.assertEqual(len(candidates), 20)
        self.assertEqual(
            {(row["germination_hops"], row["consolidation_span"]) for row in candidates},
            {(hops, span) for hops in (2, 4, 5, 8) for span in (0, 2, 4, 7, 15)},
        )
        self.assertEqual(sum(bool(row["bounded"]) for row in candidates), 12)
        self.assertEqual(anchors["compact"]["germination_hops"], 8)
        self.assertEqual(anchors["compact"]["consolidation_steps"], 30)

    def test_finite_wave_and_zero_stability(self) -> None:
        payload = np.ones((1, 4), dtype=np.float32)
        origins = np.asarray([[8, 8]], dtype=np.int16)
        field, occupied = embed_dynamic_seed(payload, origins, 16)
        field, occupied, _ = propagate_bounded(field, occupied, 3)
        for y, x in np.argwhere(occupied[0]):
            dy = min(abs(int(y) - 8), 16 - abs(int(y) - 8))
            dx = min(abs(int(x) - 8), 16 - abs(int(x) - 8))
            self.assertLessEqual(max(dy, dx), 3)
        zero, zero_occupied = embed_dynamic_seed(np.zeros_like(payload), origins, 16)
        zero, _, _ = propagate_bounded(zero, zero_occupied, 5)
        np.testing.assert_array_equal(zero, np.zeros_like(zero))

    def test_bounded_endpoint_matches_explicit_local_routing(self) -> None:
        rng = np.random.default_rng(6101)
        values = rng.normal(size=(3, 16, 16, 5))
        origins = np.asarray([[0, 0], [5, 7], [11, 3]], dtype=np.int16)
        explicit = bounded_shift_reduce(values, 7)
        expected = explicit[np.arange(3), origins[:, 0], origins[:, 1]]
        actual = bounded_reduce_endpoint(values, 7, origins)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_masked_reader_is_exact_local_reader(self) -> None:
        rng = np.random.default_rng(6102)
        states = rng.random((3, 16, 16)) < 0.30
        occupied = rng.random((3, 16, 16)) < 0.45
        payload = rng.normal(size=(3, 4)).astype(np.float32)
        basis = rng.normal(size=(512, 4)).astype(np.float32)
        field = occupied[..., None] * payload[:, None, None]
        uniforms = rng.random((3, 16, 16))
        expected = apply_local_reader(states, field, basis, uniforms, 0.25)
        actual = apply_masked_payload_reader(
            states, payload @ basis.T, occupied, uniforms, 0.25
        )
        np.testing.assert_array_equal(actual, expected)

    def test_runtime_api_has_no_label_parent_prototype_or_target(self) -> None:
        parameters = inspect.signature(simulate_bounded_lineage).parameters
        for forbidden in ("label", "prototype", "target", "parent_carrier"):
            self.assertNotIn(forbidden, parameters)

    def test_mechanism_audit_passes(self) -> None:
        self.assertTrue(stage6_mechanism_audit(MinimalityContract())["passed"])


class MinimalitySealingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = load_frozen_stage5r()

    def test_reference_reserve_stays_closed_until_requested(self) -> None:
        profile = MINIMALITY_PROFILES["reference"]
        closed = select_minimality_cohorts(
            profile, self.frozen, profile_name="reference"
        )
        opened = select_minimality_cohorts(
            profile, self.frozen, profile_name="reference", open_audit=True
        )
        self.assertEqual(closed["audit"], [])
        self.assertEqual(len(opened["audit"]), 62)
        development = {
            pair["pair_id"]
            for name, rows in closed.items()
            if name != "audit"
            for pair in rows
        }
        self.assertFalse(development & {pair["pair_id"] for pair in opened["audit"]})

    def test_final_audit_requires_explicit_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_motif_minimality(
                    Path(directory), round_name="audit", profile_name="smoke"
                )

    def test_four_hour_wall_limit_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_motif_minimality(
                    Path(directory), profile_name="smoke", max_hours=4.01
                )


if __name__ == "__main__":
    unittest.main()
