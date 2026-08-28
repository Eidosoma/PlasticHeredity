from __future__ import annotations

from dataclasses import replace
import unittest

from plastic_ca.config import ECAConfig, ECASemantics, ObserverThresholds
from plastic_ca.eca import activity_generation, prepare_launch_state, simulate_lineage
from plastic_ca.metrics import cyclic_kmer_spectrum


class SemanticTests(unittest.TestCase):
    def test_exact_and_stratified_seed_densities(self) -> None:
        exact = ECAConfig(
            launch_burnin_sweeps=0,
            semantics=ECASemantics(seed_mode="exact_half"),
        )
        row, _ = prepare_launch_state(35, 0, exact)
        self.assertEqual(row.bit_count(), 32)

        stratified = replace(exact, semantics=ECASemantics(seed_mode="density_stratified"))
        self.assertEqual(
            [prepare_launch_state(35, index, stratified)[0].bit_count() for index in range(4)],
            [13, 26, 38, 51],
        )

    def test_realized_activity_counts_noise_but_deterministic_does_not(self) -> None:
        base = ECAConfig(
            width=8,
            activity_budget=1,
            min_sweeps=1,
            max_sweeps=3,
            flip_noise=1.0,
            launch_burnin_sweeps=0,
            semantics=ECASemantics(activity_count="realized"),
        )
        realized = activity_generation(0b10110010, 204, base, object())
        deterministic = activity_generation(
            0b10110010,
            204,
            replace(base, semantics=ECASemantics(activity_count="deterministic")),
            object(),
        )
        self.assertEqual(realized.sweeps, 1)
        self.assertEqual(deterministic.sweeps, 3)

    def test_deterministic_monochrome_death_precedes_noise(self) -> None:
        base = ECAConfig(
            width=8,
            activity_budget=64,
            min_sweeps=4,
            max_sweeps=8,
            flip_noise=1.0,
            launch_burnin_sweeps=0,
        )
        terminal_only = activity_generation(0b10110010, 0, base, object())
        deterministic_death = activity_generation(
            0b10110010,
            0,
            replace(base, semantics=ECASemantics(monochrome_death="deterministic_immediate")),
            object(),
        )
        self.assertEqual(terminal_only.terminal, 0xFF)
        self.assertEqual(terminal_only.sweeps, 8)
        self.assertEqual(deterministic_death.terminal, 0)
        self.assertEqual(deterministic_death.sweeps, 1)

    def test_pre_and_post_rule_noise_have_distinct_terminals(self) -> None:
        base = ECAConfig(
            width=8,
            activity_budget=999,
            min_sweeps=1,
            max_sweeps=1,
            flip_noise=1.0,
            launch_burnin_sweeps=0,
        )
        post = activity_generation(0b10110010, 0, base, object())
        pre = activity_generation(
            0b10110010,
            0,
            replace(base, semantics=ECASemantics(process_noise="pre_rule_each_sweep")),
            object(),
        )
        self.assertEqual(post.terminal, 0xFF)
        self.assertEqual(pre.terminal, 0)

    def test_post_copy_observer_reads_the_offspring(self) -> None:
        thresholds = ObserverThresholds(horizon=2, strict_run=2, break_horizon=2)
        base = ECAConfig(
            width=8,
            activity_budget=999,
            min_sweeps=1,
            max_sweeps=1,
            flip_noise=0.0,
            copy_error=1.0,
            launch_burnin_sweeps=0,
            thresholds=thresholds,
        )
        launch, _ = prepare_launch_state(204, 0, base)
        before = simulate_lineage(204, 0, 0, base)
        after = simulate_lineage(
            204,
            0,
            0,
            replace(base, semantics=ECASemantics(observed_daughter="post_copy_offspring")),
        )
        self.assertEqual(before.compositions[1], cyclic_kmer_spectrum(launch, 8, 4))
        self.assertEqual(after.compositions[1], cyclic_kmer_spectrum(launch ^ 0xFF, 8, 4))

    def test_both_launch_anchor_modes_supply_the_requested_horizon(self) -> None:
        thresholds = ObserverThresholds(horizon=3, strict_run=2, break_horizon=2)
        base = ECAConfig(
            width=8,
            activity_budget=1,
            min_sweeps=1,
            max_sweeps=2,
            flip_noise=0.0,
            copy_error=0.0,
            launch_burnin_sweeps=0,
            thresholds=thresholds,
        )
        prepared = simulate_lineage(204, 0, 0, base)
        first_generation = simulate_lineage(
            204,
            0,
            0,
            replace(base, semantics=ECASemantics(launch_anchor="first_completed_generation")),
        )
        self.assertEqual(len(prepared.compositions), 4)
        self.assertEqual(len(first_generation.compositions), 4)


if __name__ == "__main__":
    unittest.main()

