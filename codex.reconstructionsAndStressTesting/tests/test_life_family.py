from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from plastic_ca.life import life_like_step, live_2x2_spectrum, named_patterns
from plastic_ca.life_family import (
    LifeFamilyContract,
    _array_to_hex,
    _board_to_array,
    contract_for_condition,
    evaluate_life_family_rule,
    family_edges,
    fixed_scale_subset,
    launch_library,
    life_like_step_batch,
    life_rule_notation,
    live_2x2_counts_batch,
    load_rule_registry,
    parse_life_rule,
    rule_sets,
    run_life_family_rule_set,
)


class LifeFamilyRuleTests(unittest.TestCase):
    def test_known_encodings_and_roundtrip(self) -> None:
        expected = {"B3/S23": 2060, "B36/S23": 18444, "B2/S": 1024}
        for notation, rule in expected.items():
            with self.subTest(notation=notation):
                self.assertEqual(parse_life_rule(notation), rule)
                self.assertEqual(life_rule_notation(rule), notation)

    def test_rejects_b0(self) -> None:
        with self.assertRaises(ValueError):
            parse_life_rule("B03/S23")

    def test_retained_registry_and_edges(self) -> None:
        rules = load_rule_registry()
        self.assertEqual(len(rules), 1024)
        self.assertEqual(len(family_edges(rules)), 825)
        subset = fixed_scale_subset(rules)
        self.assertEqual(len(subset), 128)
        for notation in ("B3/S23", "B36/S23", "B2/S"):
            self.assertIn(parse_life_rule(notation), subset)


class LifeFamilyEngineTests(unittest.TestCase):
    def test_scalar_vector_step_and_spectrum_agree(self) -> None:
        rng = random.Random(19)
        for notation in ("B3/S23", "B2/S", "B36/S23", "B3678/S34678"):
            rule = parse_life_rule(notation)
            board = rng.getrandbits(256)
            array = _board_to_array(board, 16, 16)
            actual = life_like_step_batch(array[None, :, :], rule)[0]
            births, survives = rule_sets(rule)
            expected = life_like_step(board, 16, 16, birth=births, survive=survives)
            self.assertEqual(_array_to_hex(actual), f"{expected:064x}")
            self.assertEqual(
                tuple(live_2x2_counts_batch(array[None, :, :])[0]),
                live_2x2_spectrum(board, 16, 16),
            )

    def test_life_fixtures_and_seeds_block(self) -> None:
        patterns = named_patterns()
        life = parse_life_rule("B3/S23")
        glider = _board_to_array(patterns["glider"], 16, 16)
        evolved = glider[None, :, :]
        for _ in range(4):
            evolved = life_like_step_batch(evolved, life)
        scalar = patterns["glider"]
        for _ in range(4):
            scalar = life_like_step(scalar, 16, 16)
        self.assertEqual(_array_to_hex(evolved[0]), f"{scalar:064x}")

        block = _board_to_array(patterns["block_descriptive"], 16, 16)
        seeds_daughter = life_like_step_batch(block[None, :, :], parse_life_rule("B2/S"))[0]
        # Seeds has no survival clause: every original block cell dies even
        # though new cells can be born around its perimeter.
        self.assertFalse((seeds_daughter & block).any())

    def test_hash_launches_are_exact_density_and_deterministic(self) -> None:
        contract = LifeFamilyContract(futures_per_launch=1)
        left = launch_library(contract)
        right = launch_library(contract)
        self.assertEqual([_array_to_hex(row) for row in left], [_array_to_hex(row) for row in right])
        self.assertEqual([int(row.sum()) for row in left[4:]], [26, 51, 77, 102])

    def test_rule_evaluation_is_deterministic(self) -> None:
        contract = LifeFamilyContract(futures_per_launch=1, horizon=4)
        left = evaluate_life_family_rule(parse_life_rule("B3/S23"), contract)
        right = evaluate_life_family_rule(parse_life_rule("B3/S23"), contract)
        self.assertEqual(left.to_dict(), right.to_dict())
        self.assertEqual(left.n_futures, 8)
        self.assertEqual(len(left.form_supports), 8)

    def test_checkpoint_resume_and_worker_invariance(self) -> None:
        contract = LifeFamilyContract(futures_per_launch=1, horizon=3)
        rules = (parse_life_rule("B3/S23"), parse_life_rule("B2/S"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            serial = run_life_family_rule_set(contract, root / "serial", rules, workers=1, resume=False)
            resumed = run_life_family_rule_set(contract, root / "serial", rules, workers=2, resume=True)
            parallel = run_life_family_rule_set(contract, root / "parallel", rules, workers=2, resume=False)
            self.assertEqual([row.to_dict() for row in serial], [row.to_dict() for row in resumed])
            self.assertEqual([row.to_dict() for row in serial], [row.to_dict() for row in parallel])

    def test_condition_matrix(self) -> None:
        self.assertEqual(contract_for_condition("frozen-b48").activity_budget, 48)
        self.assertEqual(contract_for_condition("area-b1024").activity_budget, 1024)
        self.assertEqual(contract_for_condition("scale-32").cells, 1024)
        self.assertEqual(contract_for_condition("scale-32").activity_budget, 4096)


if __name__ == "__main__":
    unittest.main()
