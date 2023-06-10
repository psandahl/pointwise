from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


def euler_to_matrix(ypr: ArrayLike, axis_order: str = 'xyz') -> NDArray:
    """
    Create a 3x3 rotation matrix from Euler angles.

    Parameters:
        ypr: Euler angles (yaw, pitch, roll) in radians.
        axis_order: Cartesian axis order in which yaw, pitch, roll is applied.

    Returns:
        Rotation matrix.
    """
    if axis_order == 'xyz':
        return euler_xyz_to_matrix(ypr)
    else:
        raise EulerException(
            f"'{axis_order}' is not an implemented axis order")


def matrix_to_euler(R: NDArray, axis_order: str = 'xyz') -> NDArray:
    """
    Decompose a matrix into Euler angles.

    Parameters:
        mat: Matrix which upper 3x3 part is a rotation matrix.
        axis_order: Cartesian axis order in which yaw, pitch, roll is applied.

    Returns:
        Array with yaw, pitch and roll in radians.
    """
    if axis_order == 'xyz':
        return matrix_to_euler_xyz(R)
    else:
        raise EulerException(
            f"'{axis_order}' is not an implemented axis order")


def homogeneous_matrix_euler(ypr: ArrayLike,
                             t: ArrayLike,
                             axis_order: str = 'xyz') -> NDArray:
    """
    Create a 4x4 homogeneous matrix from Euler angles and
    translation.

    Parameters:
        ypr: Euler angles (yaw, pitch, roll) in radians.
        t: Translation vector.
        axis_order: Cartesian axis order in which yaw, pitch, roll is applied.

    Returns:
        Homogeneous matrix.
    """
    return homogeneous_matrix(R=euler_to_matrix(ypr, axis_order=axis_order),
                              t=t)


def homogeneous_matrix(R: NDArray, t: ArrayLike) -> NDArray:
    """
    Create a 4x4 homogeneous matrix from R and t.

    Parameters:
        R: Rotation matrix.
        t: Translation vector.        

    Returns:
        Homogeneous matrix.
    """
    H = np.hstack((R, np.array(t).reshape(3, 1)))
    H = np.vstack((H, (0., 0., 0., 1.)))

    return H


def H_transform(H: NDArray, X: ArrayLike) -> NDArray:
    """
    Transform a coordinate using the honogeneous matrix.

    Parameters:
        H: Homogeneous matrix.
        X: 3D coordinate.

    Returns:
        Transformed 3D coordinate.
    """
    Xt = H @ np.append(X, 1.)
    Xt /= Xt[3]

    return Xt[:3]


def euler_xyz_to_matrix(ypr: ArrayLike) -> NDArray:
    x, y, z = ypr

    cx = math.cos(x)
    sx = math.sin(x)
    cy = math.cos(y)
    sy = math.sin(y)
    cz = math.cos(z)
    sz = math.sin(z)

    R = np.empty((3, 3), dtype=np.float64)

    R[0, 0] = cy * cz
    R[0, 1] = -cy * sz
    R[0, 2] = sy

    R[1, 0] = cz * sx * sy + cx * sz
    R[1, 1] = cx * cz - sx * sy * sz
    R[1, 2] = -cy * sx

    R[2, 0] = -cx * cz * sy + sx * sz
    R[2, 1] = cz * sx + cx * sy * sz
    R[2, 2] = cx * cy

    return R


def matrix_to_euler_xyz(R: NDArray) -> NDArray:
    theta_x = math.atan2(-R[1, 2], R[2, 2])
    theta_y = math.asin(R[0, 2])
    theta_z = math.atan2(-R[0, 1], R[0, 0])

    return np.array((theta_x, theta_y, theta_z))


class EulerException(Exception):
    """Euler exception"""
