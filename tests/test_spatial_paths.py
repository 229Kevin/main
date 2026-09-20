import unittest

import torch

from models.basic_spa_spe_mamba import build_spatial_orders


class SpatialPathTests(unittest.TestCase):
    def test_counts_reverses_and_round_trip(self):
        self.assertEqual(len(build_spatial_orders("snake_single", 7)), 1)
        self.assertEqual(len(build_spatial_orders("cross4", 7)), 4)
        source = torch.arange(49)
        for name in ("snake_single", "snake_bidir", "cross4", "ring_bidir"):
            for order in build_spatial_orders(name, 7):
                self.assertEqual(sorted(order.tolist()), list(range(49)))
                restored = source.index_select(0, order).index_select(
                    0, order.argsort()
                )
                self.assertTrue(torch.equal(restored, source))

    def test_ring_continuity_and_center(self):
        outside_in, inside_out = build_spatial_orders("ring_bidir", 7)
        self.assertTrue(torch.equal(inside_out, outside_in.flip(0)))
        self.assertEqual(outside_in[-1].item(), 24)
        for left, right in zip(outside_in.tolist(), outside_in.tolist()[1:]):
            left_row, left_column = divmod(left, 7)
            right_row, right_column = divmod(right, 7)
            self.assertEqual(
                abs(left_row - right_row) + abs(left_column - right_column), 1
            )

    def test_cross_paths_are_four_neighbor_continuous(self):
        for order in build_spatial_orders("cross4", 7):
            for left, right in zip(order.tolist(), order.tolist()[1:]):
                left_row, left_column = divmod(left, 7)
                right_row, right_column = divmod(right, 7)
                self.assertEqual(
                    abs(left_row - right_row) + abs(left_column - right_column), 1
                )

    def test_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "Unsupported spa_scan"):
            build_spatial_orders("diagonal", 7)
        with self.assertRaisesRegex(ValueError, "positive odd"):
            build_spatial_orders("snake_single", 6)


if __name__ == "__main__":
    unittest.main()
