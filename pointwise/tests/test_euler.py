from __future__ import annotations

from pointwise.euler import euler_to_matrix, matrix_to_euler

from numpy.testing import assert_array_almost_equal
import unittest


class TestEuler(unittest.TestCase):
    def test_euler_xyz(self: TestEuler) -> None:
        ypr = (0., 0., 0.)

        mat = euler_to_matrix(ypr=ypr, axis_order='xyz')
        self.assertEqual(mat.shape, (3, 3))
        assert_array_almost_equal(matrix_to_euler(mat, axis_order='xyz'), ypr)

        ypr = (1.23, -0.345, -1.56)

        mat = euler_to_matrix(ypr=ypr, axis_order='xyz')
        self.assertEqual(mat.shape, (3, 3))
        assert_array_almost_equal(matrix_to_euler(mat, axis_order='xyz'), ypr)
