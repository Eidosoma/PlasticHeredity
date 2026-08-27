from __future__ import annotations

import unittest

from plastic_ca.config import ECAConfig
from plastic_ca.particle import build_domain_dictionary, figure_mask, spacetime_codes


class ParticleTests(unittest.TestCase):
    def test_one_code_per_cell(self) -> None:
        history = (0b0011, 0b0101, 0b1110)
        self.assertEqual(len(spacetime_codes(history, 4)), 4)

    def test_rule_zero_domain_collapses(self) -> None:
        config = ECAConfig(width=16, n_seeds=1, futures_per_seed=1)
        dictionary = build_domain_dictionary(rule=0, config=config, n_seeds=2, burnin=8, collect=4)
        self.assertEqual(dictionary.n_distinct_codes, 1)
        self.assertEqual(len(dictionary.codes), 1)
        self.assertEqual(figure_mask((0, 0, 0), dictionary, 16), 0)


if __name__ == "__main__":
    unittest.main()

