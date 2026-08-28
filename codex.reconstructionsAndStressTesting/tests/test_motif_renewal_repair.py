from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from plastic_ca.motif_lineage import MotifContract
from plastic_ca.motif_minimality import (
    MINIMALITY_PROFILES,
    load_frozen_stage5r,
    select_minimality_cohorts,
)
from plastic_ca.motif_renewal_repair import (
    RENEWAL_PROFILES,
    RenewalContract,
    _embed_seed_payloads,
    _full_anchor,
    _outside_light_cone_fraction,
    _qualification_candidates,
    _run_adjudicate,
    build_renewal_candidates,
    build_scale_variants,
    centered_reduce_endpoint,
    centred_causal_overlap,
    coded_payload_roundtrip,
    hamming84_decode,
    hamming84_encode,
    latch_update,
    renewal_seed_origins,
    run_motif_renewal_repair,
    seed_offsets,
    seed_ablation_activity,
    seedify_payload,
    select_renewal_cohorts,
    simulate_renewal_lineage,
    summarize_scale,
)


class RenewalPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = load_frozen_stage5r()

    def test_centered_endpoint_matches_explicit_square(self) -> None:
        rng = np.random.default_rng(6301)
        values = rng.normal(size=(3, 16, 16, 5))
        origins = np.asarray(((0, 0), (5, 7), (11, 3)), dtype=np.int16)
        for radius in (0, 2, 4, 5):
            actual = centered_reduce_endpoint(values, radius, origins)
            expected = []
            offsets = np.arange(-radius, radius + 1)
            for sample, (y, x) in enumerate(origins):
                ys = (int(y) + offsets) % 16
                xs = (int(x) + offsets) % 16
                expected.append(values[sample][np.ix_(ys, xs)].mean((0, 1)))
            np.testing.assert_allclose(actual, np.stack(expected), atol=1e-12)

    def test_centered_writer_is_inside_radius_five_cone(self) -> None:
        origins = np.asarray(((0, 0), (8, 8), (4, 12)), dtype=np.int16)
        self.assertEqual(
            centred_causal_overlap(origins, origins, 16, hops=5, radius=5),
            1.0,
        )

    def test_latch_requires_writing_when_rho_is_zero(self) -> None:
        codec = {
            "bits": 4,
            "quantizer_scale": np.ones(3, dtype=np.float32),
        }
        parent = np.asarray(((0.5, -0.5, 0.25),), dtype=np.float32)
        proposal = np.asarray(((-0.25, 0.75, 0.5),), dtype=np.float32)
        written, _ = latch_update(parent, proposal, 0.0, codec)
        erased, _ = latch_update(
            parent, proposal, 0.0, codec, write_enabled=False
        )
        self.assertGreater(float(np.max(np.abs(written))), 0.0)
        np.testing.assert_array_equal(erased, np.zeros_like(erased))

    def test_hamming84_corrects_every_single_bit_and_flags_doubles(self) -> None:
        values = np.arange(16, dtype=np.uint8)
        encoded = hamming84_encode(values)
        decoded, flags = hamming84_decode(encoded)
        np.testing.assert_array_equal(decoded, values)
        self.assertFalse(np.any(flags))
        for bit in range(8):
            corrupted = encoded.copy()
            corrupted[:, bit] ^= np.uint8(1)
            decoded, flags = hamming84_decode(corrupted)
            np.testing.assert_array_equal(decoded, values)
            self.assertFalse(np.any(flags))
        doubled = encoded.copy()
        doubled[:, 0] ^= np.uint8(1)
        doubled[:, 1] ^= np.uint8(1)
        _, flags = hamming84_decode(doubled)
        self.assertTrue(np.all(flags))

    def test_coded_payload_single_error_roundtrip(self) -> None:
        codec = {
            "bits": 4,
            "quantizer_scale": np.ones(4, dtype=np.float32),
        }
        payload = np.asarray(((0.1, -0.4, 0.8, -0.9),), dtype=np.float32)
        baseline, _ = coded_payload_roundtrip(payload, codec, "hamming84")
        flips = np.zeros((1, 4, 8), dtype=np.uint8)
        flips[..., 3] = 1
        corrected, uncorrectable = coded_payload_roundtrip(
            payload, codec, "hamming84", flips
        )
        np.testing.assert_array_equal(corrected, baseline)
        self.assertEqual(uncorrectable, 0.0)

    def test_seed_constellation_and_partition_have_fixed_semantics(self) -> None:
        np.testing.assert_array_equal(
            seed_offsets(16, 4),
            np.asarray(((0, 0), (0, 8), (8, 0), (8, 8)), dtype=np.int16),
        )
        payload = np.arange(16, dtype=np.float32)[None, :]
        seeded, masks = seedify_payload(payload, 4, "partitioned")
        self.assertEqual(seeded.shape, (1, 4, 16))
        np.testing.assert_array_equal(masks.sum(axis=0), np.ones(16))
        self.assertEqual(int(np.count_nonzero(masks)), 16)
        base = renewal_seed_origins("pair", 3, 2, 16, 4, "co-located")
        translated = renewal_seed_origins(
            "pair", 3, 2, 16, 4, "co-located", translated=True
        )
        np.testing.assert_array_equal(
            translated, (base.astype(np.int64) + np.asarray((5, 7))) % 16
        )

    def test_zero_valued_partition_channels_remain_physical(self) -> None:
        payload = np.zeros((1, 4, 8), dtype=np.float32)
        _, masks = seedify_payload(payload[:, 0], 4, "partitioned")
        origins = seed_offsets(16, 4)[None, :, :]
        field, occupied = _embed_seed_payloads(
            payload,
            origins,
            16,
            np.broadcast_to(masks[None, :, :], payload.shape),
        )
        self.assertEqual(int(np.count_nonzero(occupied)), 8)
        self.assertEqual(float(np.max(np.abs(field))), 0.0)
        self.assertEqual(
            _outside_light_cone_fraction(
                np.any(occupied, axis=-1), origins, hops=0
            ),
            0.0,
        )

    def test_two_seed_ablation_has_exact_aggregate_dose_and_pairing(self) -> None:
        quarter = seed_ablation_activity("pair", 8, 2, 0.25, "test")
        half = seed_ablation_activity("pair", 8, 2, 0.50, "test")
        self.assertEqual(float(np.mean(~quarter)), 0.25)
        self.assertEqual(float(np.mean(~half)), 0.50)
        np.testing.assert_array_equal(quarter[:8], quarter[8:])
        np.testing.assert_array_equal(half[:8], half[8:])

    def test_registered_grid_has_sixty_label_blind_candidates(self) -> None:
        candidates = build_renewal_candidates(self.frozen)
        self.assertEqual(len(candidates), 60)
        self.assertEqual(len({row["candidate_id"] for row in candidates}), 60)
        self.assertEqual(
            {row["window_id"] for row in candidates},
            {"directed-5", "directed-8", "centered-5", "centered-9", "centered-11"},
        )
        for candidate in candidates:
            self.assertNotIn("runtime_label", candidate)

    def test_runtime_api_does_not_receive_family_labels_or_targets(self) -> None:
        parameters = inspect.signature(simulate_renewal_lineage).parameters
        for forbidden in ("label", "prototype", "target", "parent_carrier"):
            self.assertNotIn(forbidden, parameters)

    def test_scale_gate_uses_measured_not_asserted_light_cone(self) -> None:
        source = build_renewal_candidates(self.frozen)[0]
        variants = [
            row
            for row in build_scale_variants((source,))
            if row["scale_mode"] == "fixed"
        ]
        rows = []
        for candidate in variants:
            outside = 0.01 if int(candidate["extent"]) == 64 else 0.0
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "conditions": {
                        "intact": {
                            "outcomes": {
                                "8": {
                                    "primary": {"crossover": 0.5},
                                    "distance_bands": {
                                        "inside-cone": {"crossover": 0.5}
                                    },
                                }
                            },
                            "carrier_history": {
                                "8": {
                                    "outside_light_cone_occupied_fraction": outside
                                }
                            },
                        }
                    },
                }
            )
        result = summarize_scale(
            rows, variants, RENEWAL_PROFILES["smoke"], RenewalContract()
        )
        source_id = source["candidate_id"]
        self.assertFalse(result["class_pass"][source_id]["fixed"])
        self.assertFalse(result["fixed_budget_exact_light_cone_pass"])

    def test_semantic_streams_ignore_candidate_name(self) -> None:
        candidates = build_renewal_candidates(self.frozen)
        candidate = candidates[0]
        renamed = {**candidate, "candidate_id": "same-mechanism-renamed"}
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
        first, first_exits = simulate_renewal_lineage(
            *arguments,
            candidate,
            "intact",
            2,
            1,
            self.frozen["stage4"]["reference"],
            MotifContract(),
            RenewalContract(),
            retain_exits=True,
        )
        second, second_exits = simulate_renewal_lineage(
            *arguments,
            renamed,
            "intact",
            2,
            1,
            self.frozen["stage4"]["reference"],
            MotifContract(),
            RenewalContract(),
            retain_exits=True,
        )
        self.assertEqual(first["outcomes"], second["outcomes"])
        self.assertEqual(first["carrier_history"], second["carrier_history"])
        for left, right in zip(first_exits, second_exits):
            np.testing.assert_array_equal(left, right)


class RenewalWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = load_frozen_stage5r()

    def test_worker_wall_and_reference_authorization_caps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_motif_renewal_repair(
                    Path(directory), profile_name="smoke", workers=5
                )
            with self.assertRaises(ValueError):
                run_motif_renewal_repair(
                    Path(directory), profile_name="smoke", max_hours=8.01
                )
            with self.assertRaises(ValueError):
                run_motif_renewal_repair(
                    Path(directory), profile_name="reference"
                )

    def test_positive_branch_seals_before_automatic_final_opening(self) -> None:
        profile = RENEWAL_PROFILES["smoke"]
        base = select_minimality_cohorts(
            MINIMALITY_PROFILES["smoke"],
            self.frozen,
            profile_name="smoke",
            open_audit=False,
        )
        cohorts = select_renewal_cohorts(
            base, self.frozen, profile, open_final=False
        )
        candidate = build_renewal_candidates(self.frozen)[0]
        anchor = _full_anchor(self.frozen)
        positive = {
            "crossover": {"mean": 0.5, "ci": [0.4, 0.6]},
            "active_rewrite_advantage": {"mean": 0.2, "ci": [0.1, 0.3]},
            "direction_a_mean": 0.5,
            "direction_b_mean": 0.5,
            "fraction_pairs_positive": 1.0,
            "survival_mean": 1.0,
            "translation_retention": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "transient").mkdir()
            (output / "repair").mkdir()
            (output / "adjudicate").mkdir()
            (output / "transient/RESULTS.json").write_text(
                '{"stage_gate": true}\n', encoding="utf-8"
            )
            (output / "repair/RESULTS.json").write_text(
                '{"anchor_generation8_crossover_mean": 0.5}\n',
                encoding="utf-8",
            )
            (output / "RENEWAL_MODELS.json").write_text("{}\n", encoding="utf-8")
            (output / "RENEWAL_MODELS.npz").write_bytes(b"sealed-model")
            (output / "COHORTS.json").write_text(
                '{"final_audit_trajectory_state": "untouched", '
                '"final_audit_trajectories_not_loaded": true}\n',
                encoding="utf-8",
            )
            with (
                patch(
                    "plastic_ca.motif_renewal_repair._qualification_candidates",
                    return_value=([candidate], anchor),
                ),
                patch(
                    "plastic_ca.motif_renewal_repair._run_json_checkpoints",
                    return_value=([], True),
                ),
                patch(
                    "plastic_ca.motif_renewal_repair.summarize_candidate",
                    side_effect=lambda *args, **kwargs: dict(positive),
                ),
                patch(
                    "plastic_ca.motif_renewal_repair._causal_gate",
                    return_value=(True, {"causal_test_double": True}),
                ),
            ):
                result = _run_adjudicate(
                    output,
                    self.frozen,
                    cohorts,
                    profile,
                    "smoke",
                    RenewalContract(),
                    MotifContract(),
                    "d" * 64,
                    workers=1,
                    resume=False,
                    deadline=float("inf"),
                    status=lambda *args, **kwargs: None,
                    scientific=False,
                    auto_final_audit=True,
                )
            self.assertTrue(result["stage_gate"])
            self.assertTrue((output / "FINAL_AUDIT_DESIGN.json").exists())
            cohorts_after = __import__("json").loads(
                (output / "COHORTS.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                cohorts_after["final_audit_trajectory_state"], "complete"
            )

    def test_failed_repair_candidate_cannot_reenter_at_qualification(self) -> None:
        candidate = {
            **build_renewal_candidates(self.frozen)[0],
            "candidate_id": "coverage-diagnostic",
            "source_candidate_id": "failed-repair-source",
        }
        anchor = _full_anchor(self.frozen)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "repair").mkdir()
            (output / "coverage").mkdir()
            (output / "scale").mkdir()
            (output / "repair/RESULTS.json").write_text(
                '{"eligible_candidate_ids": []}\n', encoding="utf-8"
            )
            (output / "coverage/RESULTS.json").write_text(
                '{"eligible_candidate_ids": ["coverage-diagnostic"], '
                '"candidate_summaries": {"coverage-diagnostic": '
                '{"crossover": {"mean": 1.0}}}}\n',
                encoding="utf-8",
            )
            (output / "scale/RESULTS.json").write_text(
                '{"passing_source_candidate_ids": ["coverage-diagnostic"]}\n',
                encoding="utf-8",
            )
            with patch(
                "plastic_ca.motif_renewal_repair._rebuild_coverage_candidates",
                return_value=([candidate], anchor),
            ):
                selected, returned_anchor = _qualification_candidates(
                    output, "d" * 64
                )
            self.assertEqual(selected, [])
            self.assertEqual(returned_anchor["candidate_id"], anchor["candidate_id"])


if __name__ == "__main__":
    unittest.main()
