from __future__ import annotations

import unittest

from plastic_ca.config import ECAConfig
from plastic_ca.eca import (
    canonical_rule,
    canonical_rules,
    eca_step,
    reflect_rule,
    rule_orbit,
    simulate_lineage,
)


class ECATests(unittest.TestCase):
    def test_literature_orbit_count_and_rule_110_golden(self) -> None:
        self.assertEqual(len(canonical_rules()), 88)
        self.assertEqual(rule_orbit(110), frozenset({110, 124, 137, 193}))
        self.assertEqual(canonical_rule(193), 110)

    def test_rule_zero_and_identity(self) -> None:
        width = 16
        row = 0b1011010010010110
        self.assertEqual(eca_step(row, 0, width), 0)
        self.assertEqual(eca_step(row, 204, width), row)

    def test_reflection_is_an_involution(self) -> None:
        for rule in range(256):
            self.assertEqual(reflect_rule(reflect_rule(rule)), rule)

    def test_lineage_replays_exactly(self) -> None:
        config = ECAConfig(n_seeds=1, futures_per_seed=1)
        first = simulate_lineage(35, 0, 0, config)
        second = simulate_lineage(35, 0, 0, config)
        self.assertEqual(first, second)
        self.assertLessEqual(first.survived, config.thresholds.horizon)


if __name__ == "__main__":
    unittest.main()

