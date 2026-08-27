from __future__ import annotations

import unittest

from plastic_ca.config import ObserverThresholds
from plastic_ca.metrics import (
    cosine,
    cyclic_kmer_spectrum,
    jaccard_bits,
    mass_support,
    strict_coherent_event,
)


class MetricTests(unittest.TestCase):
    def test_cyclic_spectrum_has_one_window_per_cell(self) -> None:
        spectrum = cyclic_kmer_spectrum(0b10110010, 8, 4)
        self.assertEqual(len(spectrum), 16)
        self.assertEqual(sum(spectrum), 8)

    def test_cosine_and_support(self) -> None:
        self.assertAlmostEqual(cosine((1, 0), (1, 0)), 1.0)
        self.assertAlmostEqual(cosine((1, 0), (0, 1)), 0.0)
        self.assertEqual(mass_support((5, 3, 2), 0.5), 0b001)
        self.assertEqual(jaccard_bits(0b101, 0b110), 1 / 3)

    def test_strict_event_requires_break_then_coherent_distinct_run(self) -> None:
        old = (1.0, 0.0)
        new = (0.0, 1.0)
        compositions = [old, new] + [new] * 8
        event = strict_coherent_event(compositions, ObserverThresholds())
        self.assertTrue(event.occurred)
        self.assertEqual(event.first_break, 0)
        self.assertEqual(event.run_start, 1)

    def test_no_break_means_no_strict_event(self) -> None:
        same = [(1.0, 0.0)] * 12
        self.assertFalse(strict_coherent_event(same, ObserverThresholds()).occurred)


if __name__ == "__main__":
    unittest.main()

