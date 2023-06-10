from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


def euler_to_matrix(ypr: ArrayLike, axis_order: str = 'xyz') -> NDArray:
    """
    Create a 3x3 rotation matrix from Euler angles.

    Parameters:
        ypr: Euler angles (yaw, pitch, roll) in radians.
        axis: Cartesian axis order in which yaw, pitch, roll is applied.

    Returns:
        Rotation matrix.
    """
    if axis_order == 'xyz':
        return euler_xyz_to_matrix(ypr)
    else:
        raise EulerException(
            f"'{axis_order}' is not an implemented axis order")


def matrix_to_euler(mat: NDArray, axis_order: str = 'xyz') -> NDArray:
    """
    Decompose a matrix into Euler angles.

    Parameters:
        mat: Matrix which upper 3x3 part is a rotation matrix.
        axis: Cartesian axis order in which yaw, pitch, roll is applied.

    Returns:
        Array with yaw, pitch and roll in radians.
    """
    if axis_order == 'xyz':
        return matrix_to_euler_xyz(mat)
    else:
        raise EulerException(
            f"'{axis_order}' is not an implemented axis order")


def euler_xyz_to_matrix(ypr: ArrayLike) -> NDArray:
    x, y, z = ypr

    cx = math.cos(x)
    sx = math.sin(x)
    cy = math.cos(y)
    sy = math.sin(y)
    cz = math.cos(z)
    sz = math.sin(z)

    mat = np.empty((3, 3), dtype=np.float64)

    mat[0, 0] = cy * cz
    mat[0, 1] = -cy * sz
    mat[0, 2] = sy

    mat[1, 0] = cz * sx * sy + cx * sz
    mat[1, 1] = cx * cz - sx * sy * sz
    mat[1, 2] = -cy * sx

    mat[2, 0] = -cx * cz * sy + sx * sz
    mat[2, 1] = cz * sx + cx * sy * sz
    mat[2, 2] = cx * cy

    return mat


def matrix_to_euler_xyz(mat: NDArray) -> NDArray:
    theta_x = math.atan2(-mat[1, 2], mat[2, 2])
    theta_y = math.asin(mat[0, 2])
    theta_z = math.atan2(-mat[0, 1], mat[0, 0])

    return np.array((theta_x, theta_y, theta_z))


class EulerException(Exception):
    """Euler exception"""
