from __future__ import annotations

from pointwise.euler import euler_to_matrix, matrix_to_euler
from pointwise.euler import homogeneous_matrix_euler, H_transform

import numpy as np
from numpy.testing import assert_array_almost_equal
import unittest


class TestEuler(unittest.TestCase):
    def test_euler_xyz(self: TestEuler) -> None:
        ypr = (0., 0., 0.)

        R = euler_to_matrix(ypr=ypr, axis_order='xyz')
        self.assertEqual(R.shape, (3, 3))
        assert_array_almost_equal(matrix_to_euler(R, axis_order='xyz'), ypr)

        ypr = (1.23, -0.345, -1.56)

        R = euler_to_matrix(ypr=ypr, axis_order='xyz')
        self.assertEqual(R.shape, (3, 3))
        assert_array_almost_equal(matrix_to_euler(R, axis_order='xyz'), ypr)

    def test_homogeneous_matrix(self: TestEuler) -> None:
        t = 123., 456., 789.

        H = homogeneous_matrix_euler(ypr=(0., 0., 0.), t=t)
        self.assertEqual(H.shape, (4, 4))

        # Test transforming a single point.
        X = np.array([0., 0., 0.])
        Xt = H_transform(H, X)
        self.assertEqual(Xt.shape, X.shape)
        assert_array_almost_equal(Xt, t)

        # Test transforming an array of points.
        X = np.array([
            [0., 0., 0.],
            [10., 10., 10.]
        ])
        Xt = H_transform(H, X)
        self.assertEqual(Xt.shape, X.shape)
        assert_array_almost_equal(Xt, X + t)
