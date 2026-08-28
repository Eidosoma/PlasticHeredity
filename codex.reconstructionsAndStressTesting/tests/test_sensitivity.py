from __future__ import annotations

import unittest

from plastic_ca.eca import canonical_rules, wolfram_class
from plastic_ca.sensitivity import (
    SensitivitySetting,
    enumerate_settings,
    score_cell,
    stratified_rule_split,
)


class SensitivityTests(unittest.TestCase):
    def test_design_has_1440_unique_canonical_cells(self) -> None:
        settings = enumerate_settings()
        self.assertEqual(len(settings), 1440)
        self.assertEqual(len({setting.setting_id for setting in settings}), 1440)

    def test_stratified_split_is_balanced_and_complete(self) -> None:
        reference = {
            rule: {
                "strict": rule / 255,
                "break_by_8": rule / 255,
                "median_gen_sweeps": 1.0,
                "mean_survival": 1.0,
                "wolfram_class": float(wolfram_class(rule)),
            }
            for rule in canonical_rules()
        }
        development, holdout = stratified_rule_split(reference)
        self.assertEqual(len(development), 44)
        self.assertEqual(len(holdout), 44)
        self.assertEqual(set(development) | set(holdout), set(canonical_rules()))
        self.assertFalse(set(development) & set(holdout))

    def test_perfect_cell_scores_zero_endpoint_error(self) -> None:
        rules = canonical_rules()[:12]
        reference = {
            rule: {
                "strict": index / 20,
                "break_by_8": index / 12,
                "median_gen_sweeps": float(index + 1),
                "mean_survival": float(index + 2),
                "wolfram_class": float(wolfram_class(rule)),
            }
            for index, rule in enumerate(rules)
        }
        setting = SensitivitySetting(
            "prepared_seed",
            "sweeps_1",
            "expected_half_hash",
            "post_rule_each_sweep",
            "realized",
            "terminal_only",
            "pre_copy_terminal",
        )
        cell = {
            "setting_id": setting.setting_id,
            "setting": setting.__dict__,
            "contract_plausible": True,
            "results": [
                {
                    "rule": rule,
                    "wolfram_class": wolfram_class(rule),
                    "disputed": False,
                    **{key: reference[rule][key] for key in ("strict", "break_by_8", "median_gen_sweeps", "mean_survival")},
                }
                for rule in rules
            ],
        }
        metrics = score_cell(cell, reference)
        self.assertAlmostEqual(metrics["strict_rho"], 1.0)
        self.assertAlmostEqual(metrics["break_rho"], 1.0)
        self.assertEqual(metrics["strict_mae"], 0.0)
        self.assertEqual(metrics["break_mae"], 0.0)


if __name__ == "__main__":
    unittest.main()

