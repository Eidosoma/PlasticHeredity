from __future__ import annotations

import unittest

import numpy as np

from plastic_ca.ca_carrier_v3 import (
    V3_PROFILES,
    V3Contract,
    _narrow_verdict,
    _trace_summary,
    _life_3x3_counts,
    _eca_mesoscale,
    _morphology_distances,
    _prototype_label,
    _structure_factor,
    conditional_null_ensemble,
    continuous_clusters,
    discover_continuous_candidates,
    pair_prototype_donors,
    select_mapping_tile,
    select_narrow_prototype,
)
from plastic_ca.causal_heredity import BatchTrace


def donor(identifier: str, vector: list[float], *, launch: int = 0, density: float = 0.25) -> dict:
    board = np.zeros((16, 16), dtype=np.bool_)
    board.flat[: int(round(density * board.size))] = True
    value = 0
    for bit in board.ravel():
        value = (value << 1) | int(bit)
    state_hex = f"{value:064x}"
    return {
        "donor_id": identifier,
        "substrate": "life",
        "rule": 1,
        "launch_index": launch,
        "density": density,
        "donor_state_hex": state_hex,
        "ancestor_state_hex": state_hex,
        "anchor_state_hex": state_hex,
        "initial_state_hex": "0" * 64,
        "target_compositions": {
            "primary": vector,
            "primary_terminal": vector,
            "local_secondary": vector,
            "local_aux": vector,
            "global": vector,
        },
    }


class FrozenHypothesisTests(unittest.TestCase):
    def test_prototype_uses_continuous_separation_not_support_ids(self) -> None:
        hypothesis = select_narrow_prototype()
        self.assertEqual(hypothesis["rule"], 31649)
        self.assertAlmostEqual(hypothesis["target_similarity"], 0.39311324872873266)
        self.assertGreater(hypothesis["pooled_support_id_diagnostic"]["centroid_cosine"], 0.99)
        self.assertEqual(hypothesis["historical_prototype_match_diagnostic"]["counts"]["A"], 110)
        self.assertEqual(hypothesis["historical_prototype_match_diagnostic"]["counts"]["B"], 120)
        self.assertNotIn("form_id", hypothesis["targets"])

    def test_prototype_assignment_requires_similarity_and_margin(self) -> None:
        contract = V3Contract()
        a = np.asarray((1.0, 0.0))
        b = np.asarray((0.0, 1.0))
        self.assertEqual(_prototype_label(a, a, b, contract), "A")
        self.assertEqual(_prototype_label(b, a, b, contract), "B")
        self.assertIsNone(_prototype_label((1.0, 1.0), a, b, contract))


class ObserverAndControlTests(unittest.TestCase):
    def test_eca_mesoscale_is_a_normalized_final4_vector(self) -> None:
        rows = np.zeros((2, 3, 64), dtype=np.bool_)
        rows[1, :, ::3] = True
        vectors = _eca_mesoscale(rows, 110)
        self.assertEqual(vectors.shape, (2, 16))
        self.assertTrue(all(total in (0.0, 1.0) for total in vectors.sum(axis=1)))

    def test_3x3_observer_is_per_generation_normalized_and_translation_invariant(self) -> None:
        board = np.zeros((16, 16), dtype=np.bool_)
        board[2:5, 7] = True
        first = _life_3x3_counts(board[None, ...])[0]
        shifted = _life_3x3_counts(np.roll(board, (3, 4), axis=(0, 1))[None, ...])[0]
        self.assertAlmostEqual(float(first.sum()), 1.0)
        np.testing.assert_allclose(first, shifted)

    def test_structure_factor_is_translation_invariant(self) -> None:
        board = np.zeros((16, 16), dtype=np.bool_)
        board[1:4, 3:6] = True
        first = _structure_factor(board[None, ...])[0]
        second = _structure_factor(np.roll(board, (2, 5), axis=(0, 1))[None, ...])[0]
        np.testing.assert_allclose(first, second)

    def test_conditional_null_is_deterministic_exact_count_and_never_missing(self) -> None:
        source = np.zeros((16, 16), dtype=np.bool_)
        source[2:10, 3:11] = True
        mask = np.zeros_like(source)
        mask[:12, :12] = True
        first, metadata = conditional_null_ensemble(
            "life", source, mask, "fixture", candidates=32, keep=8
        )
        second, second_metadata = conditional_null_ensemble(
            "life", source, mask, "fixture", candidates=32, keep=8
        )
        self.assertEqual(len(first), 8)
        self.assertEqual(metadata, second_metadata)
        target_mass = int(np.count_nonzero(source & mask))
        self.assertTrue(all(int(board.sum()) == target_mass for board in first))
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left, right)

    def test_morphology_identity_has_zero_distance(self) -> None:
        board = np.zeros((16, 16), dtype=np.bool_)
        board[1:3, 2:5] = True
        distances = _morphology_distances("life", board, board)
        self.assertAlmostEqual(distances["neighbor_error"], 0.0)
        self.assertAlmostEqual(distances["component_cosine"], 1.0)
        self.assertAlmostEqual(distances["structure_error"], 0.0)

    def test_invalid_checkpoint_stays_in_assignment_denominator(self) -> None:
        compositions = np.zeros((2, 8, 2), dtype=float)
        compositions[:, :, 0] = 1.0
        valid = np.ones((2, 8), dtype=np.bool_)
        valid[1, 7] = False
        terminals = np.zeros((2, 8, 16, 16), dtype=np.bool_)
        trace = BatchTrace(
            compositions=compositions,
            valid=valid,
            terminals=terminals,
            offspring=terminals.copy(),
            sweeps=np.zeros((2, 8), dtype=np.int16),
            activity=np.zeros((2, 8), dtype=np.int32),
            death=[None, "terminal"],
        )
        summary = _trace_summary(
            trace,
            "life",
            31649,
            {"primary": {"A": [1.0, 0.0], "B": [0.0, 1.0]}},
            V3Contract(),
            [8],
        )["8"]
        self.assertEqual(summary["observers"]["primary"]["p_a"], 0.5)
        self.assertEqual(summary["survival"], 0.5)
        self.assertEqual(summary["persistent_p_a"], 0.5)


class ContinuousClusteringTests(unittest.TestCase):
    def test_clustering_is_order_invariant_and_complete_linkage_constrained(self) -> None:
        donors = [
            donor("c", [0.97, 0.03]),
            donor("a", [1.0, 0.0]),
            donor("b", [0.99, 0.01]),
            donor("z", [0.0, 1.0], launch=1),
        ]
        forward = continuous_clusters(donors, "primary", threshold=0.95)
        reverse = continuous_clusters(list(reversed(donors)), "primary", threshold=0.95)
        self.assertEqual(
            [cluster["member_ids"] for cluster in forward],
            [cluster["member_ids"] for cluster in reverse],
        )
        for cluster in forward:
            members = [next(row for row in donors if row["donor_id"] == identifier) for identifier in cluster["member_ids"]]
            for left in range(len(members)):
                for right in range(left):
                    a = np.asarray(members[left]["target_compositions"]["primary"])
                    b = np.asarray(members[right]["target_compositions"]["primary"])
                    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                    self.assertGreaterEqual(cosine, 0.95)

    def test_candidate_pairing_is_density_matched_and_non_reusing(self) -> None:
        donors = []
        for index in range(16):
            donors.append(donor(f"a-{index:02d}", [1.0, 0.0], launch=index % 2, density=0.20 + index * 0.0001))
            donors.append(donor(f"b-{index:02d}", [0.0, 1.0], launch=index % 2, density=0.21 + index * 0.0001))
        result = discover_continuous_candidates(donors, V3Contract())
        candidate = result["families"]["local"]["candidates"][0]
        self.assertEqual(len(candidate["pairs"]), 16)
        self.assertEqual(len({pair["donor_a"]["donor_id"] for pair in candidate["pairs"]}), 16)
        self.assertTrue(all(pair["density_delta"] <= 0.05 for pair in candidate["pairs"]))

    def test_prototype_pairing_does_not_reuse_donors(self) -> None:
        contract = V3Contract()
        prototype = {"targets": {name: {"A": [1.0, 0.0], "B": [0.0, 1.0]} for name in ("primary", "primary_terminal", "local_secondary", "local_aux", "global")}}
        rows = []
        for index in range(3):
            left = donor(f"a-{index}", [1.0, 0.0], density=0.20 + index * 0.001)
            right = donor(f"b-{index}", [0.0, 1.0], density=0.21 + index * 0.001)
            left["prototype_label"] = "A"
            right["prototype_label"] = "B"
            rows.extend((left, right))
        pairs = pair_prototype_donors([{"donors": rows}], prototype, contract)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(len({pair["donor_a"]["donor_id"] for pair in pairs}), 3)
        self.assertEqual(len({pair["donor_b"]["donor_id"] for pair in pairs}), 3)


class SelectionTests(unittest.TestCase):
    def test_empty_mapping_selection_is_deterministic(self) -> None:
        selection = select_mapping_tile([])
        self.assertEqual(selection["selected_tile"], 0)
        self.assertEqual(len(selection["scores"]), 16)

    def test_claim_ladder_cannot_skip_pair_specific_gate(self) -> None:
        positive = {"mean": 0.3, "ci95": [0.2, 0.4]}
        replay = {
            "n_pairs": 0,
            "crossover": positive,
            "survival_mean": 1.0,
        }
        confirmation = {
            "n_pairs": V3_PROFILES["smoke"].confirmation_pairs,
            "crossover": positive,
            "direction_a_mean": 0.3,
            "direction_b_mean": 0.3,
            "survival_mean": 1.0,
            "fraction_pairs_positive": 1.0,
            "control_advantages": {
                name: positive for name in ("ancestor", "exact_random", "block2", "conditional_null")
            },
            "observers": {
                name: positive for name in ("local_secondary", "local_aux", "global")
            },
            "generation_128_pass": True,
        }
        pedigree = {"depth8_crossover": positive}
        claim = _narrow_verdict(
            replay,
            confirmation,
            pedigree,
            {"geometry_pass": True, "scale_pass": True, "moderate_noise_pass": True},
            1.0,
            V3_PROFILES["smoke"],
            V3Contract(),
        )
        self.assertFalse(claim["reusable_local_gate"])
        self.assertFalse(claim["durable_local_gate"])


if __name__ == "__main__":
    unittest.main()
