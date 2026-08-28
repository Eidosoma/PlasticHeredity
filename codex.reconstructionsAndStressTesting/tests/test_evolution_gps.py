from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from plastic_ca.evolution_gps import _shortest_path, top_census_forms


class EvolutionGPSTests(unittest.TestCase):
    def test_shortest_path_uses_one_bit_edges_and_tie_break(self) -> None:
        path = _shortest_path(0, lambda rule: rule in {3, 5})
        self.assertEqual(path, (0, 1, 3))
        self.assertTrue(all((left ^ right).bit_count() == 1 for left, right in zip(path, path[1:])))

    def test_shortest_path_respects_budget(self) -> None:
        self.assertEqual(_shortest_path(0, lambda rule: rule == 255, max_edits=2), (0,))

    def test_top_forms_preserve_first_encounter_for_ties(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atlas.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("rule", "strict", "support_masks"))
                writer.writeheader()
                writer.writerow({"rule": 0, "strict": 0, "support_masks": "9|4|1"})
                writer.writerow({"rule": 1, "strict": 0, "support_masks": "9|4|2"})
            self.assertEqual(top_census_forms(path, 4), (9, 4, 1, 2))


if __name__ == "__main__":
    unittest.main()

