from __future__ import annotations

import unittest

from plastic_ca.life import life_like_step, named_patterns, shift_board


class LifeFixtureTests(unittest.TestCase):
    width = 16
    height = 16

    def evolve(self, board: int, steps: int) -> int:
        for _ in range(steps):
            board = life_like_step(board, self.width, self.height)
        return board

    def test_block_is_still(self) -> None:
        block = named_patterns()["block_descriptive"]
        self.assertEqual(self.evolve(block, 1), block)

    def test_blinker_and_toad_have_period_two(self) -> None:
        patterns = named_patterns()
        for name in ("blinker", "toad"):
            with self.subTest(name=name):
                self.assertNotEqual(self.evolve(patterns[name], 1), patterns[name])
                self.assertEqual(self.evolve(patterns[name], 2), patterns[name])

    def test_glider_period_four_diagonal_shift(self) -> None:
        glider = named_patterns()["glider"]
        self.assertEqual(
            self.evolve(glider, 4),
            shift_board(glider, 1, 1, self.width, self.height),
        )

    def test_lwss_moves_two_cells_west_in_four_steps(self) -> None:
        lwss = named_patterns()["lwss"]
        self.assertEqual(
            self.evolve(lwss, 4),
            shift_board(lwss, -2, 0, self.width, self.height),
        )


if __name__ == "__main__":
    unittest.main()

