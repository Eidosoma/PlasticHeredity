from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plastic_ca.motif_localization import (
    LOCALIZATION_PROFILES,
    REGISTERED_MODE_IDS,
    _apply_local_boundary_intervention,
    apply_local_reader,
    build_anatomy_models,
    embed_patch,
    extract_patch,
    load_frozen_stage4,
    local_energy_advantage,
    patch_origins,
    quantize_channels,
    run_motif_localization,
    select_localization_cohorts,
    simulate_local_lineage,
    transcode_audit,
    transport_field,
    walsh_mode_ids,
)
from plastic_ca.motif_compression import decode_payload
from plastic_ca.motif_lineage import motif_energy_advantage


class LocalizationPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = load_frozen_stage4()
        cls.model = cls.frozen["winner_model"]

    def test_runtime_api_has_no_label_parent_prototype_or_target(self) -> None:
        parameters = inspect.signature(simulate_local_lineage).parameters
        for forbidden in ("label", "prototype", "target", "parent_carrier"):
            self.assertNotIn(forbidden, parameters)

    def test_exact_archived_walsh_modes_and_anatomy_size(self) -> None:
        self.assertEqual(tuple(walsh_mode_ids(self.model)), REGISTERED_MODE_IDS)
        self.assertEqual(len(build_anatomy_models(self.model)), 74)

    def test_four_bit_channel_quantizer_preserves_zero(self) -> None:
        scale = np.asarray([2.0, 4.0], dtype=np.float32)
        values = np.asarray([[-3.0, 0.0], [1.1, 4.5]], dtype=np.float32)
        quantized, clipping = quantize_channels(values, scale)
        self.assertEqual(float(quantized[0, 1]), 0.0)
        self.assertAlmostEqual(float(quantized[1, 0]), 8.0 / 7.0, places=6)
        self.assertAlmostEqual(clipping, 0.5)

    def test_transport_is_mass_conserving_and_respects_light_cone(self) -> None:
        field = np.zeros((1, 16, 16, 1), dtype=np.float32)
        field[0, 8, 8, 0] = 1.0
        result = field.copy()
        for _ in range(3):
            result = transport_field(result, 0.2)
        self.assertAlmostEqual(float(result.sum()), 1.0, places=5)
        for y, x in np.argwhere(result[0, ..., 0] > 0.0):
            dy = min(abs(int(y) - 8), 16 - abs(int(y) - 8))
            dx = min(abs(int(x) - 8), 16 - abs(int(x) - 8))
            self.assertLessEqual(max(dy, dx), 3)

    def test_patch_boundary_roundtrip_and_paired_origins(self) -> None:
        rng = np.random.default_rng(505)
        payload = rng.normal(size=(4, 4, 4, 16)).astype(np.float32)
        origins = patch_origins("pair", 3, 2)
        np.testing.assert_array_equal(origins[:2], origins[2:])
        recovered = extract_patch(embed_patch(payload, origins), 4, origins)
        np.testing.assert_array_equal(recovered, payload)

    def test_uniform_local_field_is_frozen_global_reader(self) -> None:
        basis = np.asarray(self.model["basis"], dtype=np.float32)
        scale = np.asarray(self.model["quantizer_scale"], dtype=np.float32)
        rng = np.random.default_rng(506)
        states = rng.random((3, 16, 16)) < 0.4
        payload = quantize_channels(
            rng.uniform(-1.0, 1.0, size=(3, 16)).astype(np.float32) * scale,
            scale,
        )[0]
        field = np.broadcast_to(payload[:, None, None, :], (3, 16, 16, 16)).copy()
        local = local_energy_advantage(states, field, basis)
        global_value = motif_energy_advantage(states, decode_payload(payload, self.model))
        np.testing.assert_allclose(local, global_value, atol=2e-5)

    def test_zero_field_is_exactly_inert(self) -> None:
        states = np.zeros((2, 16, 16), dtype=np.bool_)
        field = np.zeros((2, 16, 16, 16), dtype=np.float32)
        result = apply_local_reader(
            states,
            field,
            self.model["basis"],
            np.zeros_like(states, dtype=np.float64),
            0.25,
        )
        np.testing.assert_array_equal(result, states)

    def test_ablation_and_opposite_rescue_are_exact(self) -> None:
        payload = np.arange(4 * 2 * 2 * 16, dtype=np.float32).reshape(4, 2, 2, 16)
        scale = np.full(16, 1_000.0, dtype=np.float32)
        sources = [payload + 1.0, payload + 2.0, payload + 3.0]
        ablated, _ = _apply_local_boundary_intervention(
            payload, "ablate_after_g2", 3, "pair", 2, None, scale
        )
        opposite, _ = _apply_local_boundary_intervention(
            payload, "rescue_opposite_enter_g4", 4, "pair", 2, sources, scale
        )
        np.testing.assert_array_equal(ablated, np.zeros_like(payload))
        expected = quantize_channels(np.concatenate((sources[2][2:], sources[2][:2])), scale)[0]
        np.testing.assert_array_equal(opposite, expected)

    def test_registered_transcode_audit_passes(self) -> None:
        audit = transcode_audit(self.model)
        self.assertTrue(audit["passed"])
        self.assertLessEqual(audit["uniform_field_global_equivalence_max_abs_error"], 2e-5)


class FrozenLocalizationCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = load_frozen_stage4()

    def test_reference_cohort_uses_128_and_leaves_30(self) -> None:
        cohorts = select_localization_cohorts(
            LOCALIZATION_PROFILES["reference"], self.frozen, profile_name="reference"
        )
        confirmation = {pair["pair_id"] for pair in cohorts["confirmation"]}
        later = {pair["pair_id"] for pair in cohorts["later_audit"]}
        exposed = {
            *self.frozen["cohorts"]["selection_pair_ids"],
            *self.frozen["cohorts"]["confirmation_pair_ids"],
        }
        self.assertEqual((len(confirmation), len(later)), (128, 30))
        self.assertFalse(confirmation & later)
        self.assertFalse(confirmation & exposed)
        self.assertFalse(later & exposed)

    def test_smoke_confirmation_reuses_exposed_pairs(self) -> None:
        cohorts = select_localization_cohorts(
            LOCALIZATION_PROFILES["smoke"], self.frozen, profile_name="smoke"
        )
        exposed = set(self.frozen["cohorts"]["confirmation_pair_ids"])
        self.assertTrue({pair["pair_id"] for pair in cohorts["confirmation"]} <= exposed)

    def test_confirmation_requires_explicit_separate_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_motif_localization(Path(directory), profile_name="smoke", phases=("confirm",))
            with self.assertRaises(ValueError):
                run_motif_localization(
                    Path(directory),
                    profile_name="smoke",
                    phases=("calibrate", "confirm"),
                    resume=True,
                    authorize_confirmation=True,
                )

    def test_eight_hour_wall_limit_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_motif_localization(Path(directory), profile_name="smoke", max_hours=8.01)

    def test_audit_only_does_not_open_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "stage5"
            result = run_motif_localization(
                root, profile_name="smoke", phases=("audit",), workers=1, max_hours=0.05
            )
            self.assertEqual(result["state"], "phases_complete")
            self.assertTrue((root / "CLEANROOM_AUDIT.json").exists())
            self.assertFalse((root / "confirmation").exists())
            cohorts = __import__("json").loads((root / "COHORTS.json").read_text())
            self.assertEqual(cohorts["confirmation_trajectory_state"], "untouched")


if __name__ == "__main__":
    unittest.main()
