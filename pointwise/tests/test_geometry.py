from __future__ import annotations

from pointwise.geometry import estimate_normal

import numpy as np
from numpy.testing import assert_array_almost_equal
import unittest


class TestGeometry(unittest.TestCase):
    def test_square_flat_grid(self: TestGeometry) -> None:
        points = np.array([
            [0., 0., 0.],
            [0., 10., 0.],
            [10., 10., 0.],
            [10., 0., 0.]
        ])
        normal, planarity = estimate_normal(points)
        assert_array_almost_equal(normal, [0., 0., 1.])
        self.assertAlmostEqual(1., planarity)

        points = np.array([
            [0., 0., 0.],
            [0., 10., 0.],
            [0., 10., 10.],
            [0., 0., 10.]
        ])
        normal, planarity = estimate_normal(points)
        assert_array_almost_equal(normal, [1., 0., 0.])
        self.assertAlmostEqual(1., planarity)

    def test_peak_grid(self: TestGeometry) -> None:
        points = np.array([
            [0., 0., 0.],
            [0., 5., 0.],
            [0., 10., 0.],

            [5., 0., 0.],
            [5., 5., -1.],
            [5., 10., 0.],

            [10., 0., 0.],
            [10., 5., 0.],
            [10., 10., 0.]
        ])
        normal, planarity1 = estimate_normal(points)
        assert_array_almost_equal(normal, [0., 0., 1.])
        self.assertLess(planarity1, 1.)

        points[4, 2] = -5.
        normal, planarity2 = estimate_normal(points)
        assert_array_almost_equal(normal, [0., 0., 1.])
        self.assertLess(planarity2, planarity1)
