from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plastic_ca.motif_compression import (
    ANCHOR_ID,
    COMPRESSION_PROFILES,
    CompressionContract,
    _apply_payload_intervention,
    _count_groups,
    _d4_orbits,
    _pool_basis,
    decode_payload,
    encode_payload,
    load_codec_models,
    load_frozen_stage3r,
    quantize_payload,
    run_motif_compression,
    save_codec_models,
    select_compression_cohorts,
    simulate_compressed_lineage,
    stress_scenarios,
)


class CompressionPrimitiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = {
            "candidate_id": ANCHOR_ID,
            "family": "identity",
            "rank": 512,
            "bits": 32,
            "payload_bits": 16384,
            "codebook_bits": 0,
            "interpretable": True,
        }

    def test_runtime_api_has_no_parent_label_prototype_or_target(self) -> None:
        parameters = inspect.signature(simulate_compressed_lineage).parameters
        for forbidden in ("label", "prototype", "target", "parent_carrier"):
            self.assertNotIn(forbidden, parameters)

    def test_identity_float32_is_bit_exact_and_zero_preserving(self) -> None:
        values = np.random.default_rng(401).normal(size=(5, 512)).astype(np.float32)
        payload, clipping = encode_payload(values, self.identity)
        decoded = decode_payload(payload, self.identity)
        np.testing.assert_array_equal(decoded, values)
        self.assertEqual(clipping, 0.0)
        zero, _ = encode_payload(np.zeros_like(values), self.identity)
        np.testing.assert_array_equal(zero, np.zeros_like(values))

    def test_signed_quantizer_has_registered_levels_and_preserves_zero(self) -> None:
        model = {
            "family": "identity",
            "bits": 2,
            "quantizer_scale": np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
        }
        values = np.asarray([[-3.0, 0.0, 1.1]], dtype=np.float32)
        quantized, clipping = quantize_payload(values, model)
        np.testing.assert_array_equal(quantized, np.asarray([[-2.0, 0.0, 2.0]], dtype=np.float32))
        self.assertAlmostEqual(clipping, 1.0 / 3.0)

    def test_structural_partitions_have_expected_dimensions(self) -> None:
        d4 = _d4_orbits()
        counts = _count_groups()
        self.assertEqual(len(d4), 102)
        self.assertEqual(len(counts), 18)
        self.assertEqual(sorted(value for group in d4 for value in group), list(range(512)))
        self.assertEqual(sorted(value for group in counts for value in group), list(range(512)))
        np.testing.assert_allclose(_pool_basis(d4).T @ _pool_basis(d4), np.eye(102), atol=1e-6)

    def test_payload_ablation_and_rescue_are_exact(self) -> None:
        payload = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
        sources = [payload + 1.0, payload + 2.0, payload + 3.0]
        model = {**self.identity, "rank": 8}
        contract = CompressionContract()
        ablated, _ = _apply_payload_intervention(
            payload, model, "ablate_after_g2", 3, "pair", 2, contract, None
        )
        same, _ = _apply_payload_intervention(
            payload, model, "rescue_same_enter_g4", 4, "pair", 2, contract, sources
        )
        opposite, _ = _apply_payload_intervention(
            payload, model, "rescue_opposite_enter_g4", 4, "pair", 2, contract, sources
        )
        np.testing.assert_array_equal(ablated, np.zeros_like(payload))
        np.testing.assert_array_equal(same, sources[2])
        np.testing.assert_array_equal(opposite[:2], sources[2][2:])
        np.testing.assert_array_equal(opposite[2:], sources[2][:2])

    def test_model_archive_is_non_pickled_and_hash_bound(self) -> None:
        quantized = {
            **self.identity,
            "candidate_id": "identity-r512-q08",
            "bits": 8,
            "quantizer_scale": np.ones(512, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_codec_models(root, [self.identity, quantized], design_digest="d", fit_trace_digest="t")
            loaded = load_codec_models(root, "d")
            np.testing.assert_array_equal(loaded[1]["quantizer_scale"], np.ones(512))
            with np.load(root / "CODEC_MODELS.npz", allow_pickle=False) as archive:
                self.assertTrue(all(archive[key].dtype != object for key in archive.files))

    def test_registered_stress_grid_is_complete(self) -> None:
        scenarios = stress_scenarios()
        self.assertEqual(len(scenarios), 25)
        self.assertEqual(scenarios["moderate_joint"]["process_noise"], 0.004)
        self.assertEqual(scenarios["harsh_joint"]["erasure"], 0.25)


class FrozenCompressionCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = load_frozen_stage3r()
        cls.contract = CompressionContract()

    def test_reference_cohorts_are_fresh_disjoint_and_leave_stage5_reserve(self) -> None:
        cohorts = select_compression_cohorts(
            COMPRESSION_PROFILES["reference"], self.frozen, self.contract, profile_name="reference"
        )
        selection = {pair["pair_id"] for pair in cohorts["selection"]}
        confirmation = {pair["pair_id"] for pair in cohorts["confirmation"]}
        reserve = {pair["pair_id"] for pair in cohorts["reserve"]}
        self.assertEqual((len(selection), len(confirmation), len(reserve)), (96, 128, 158))
        self.assertFalse(selection & confirmation)
        self.assertFalse(selection & reserve)
        self.assertFalse(confirmation & reserve)
        self.assertFalse((selection | confirmation | reserve) & self.frozen["used_pair_ids"])

    def test_smoke_reuses_only_exposed_pairs(self) -> None:
        cohorts = select_compression_cohorts(
            COMPRESSION_PROFILES["smoke"], self.frozen, self.contract, profile_name="smoke"
        )
        self.assertTrue({pair["pair_id"] for pair in cohorts["selection"]} <= self.frozen["used_pair_ids"])

    def test_confirmation_requires_explicit_separate_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_motif_compression(Path(directory), profile_name="smoke", phases=("confirm",))
            with self.assertRaises(ValueError):
                run_motif_compression(
                    Path(directory), profile_name="smoke", phases=("fit", "confirm"),
                    resume=True, authorize_confirmation=True
                )

    def test_eight_hour_wall_limit_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_motif_compression(Path(directory), profile_name="smoke", max_hours=8.01)

    def test_audit_only_does_not_open_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "stage4"
            result = run_motif_compression(
                root, profile_name="smoke", phases=("audit",), workers=1, max_hours=0.05
            )
            self.assertEqual(result["state"], "phases_complete")
            self.assertTrue((root / "CLEANROOM_AUDIT.json").exists())
            self.assertFalse((root / "confirmation").exists())


if __name__ == "__main__":
    unittest.main()
