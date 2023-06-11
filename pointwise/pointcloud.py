from __future__ import annotations

from .transform import H_transform

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pathlib
from typing import List


class PointCloud(pd.DataFrame):
    """
    Class providing an abstraction for working with points clouds.
    """

    def __init__(self: PointCloud, data: NDArray, columns: List[str]) -> None:
        """
        Create a PointCloud from the NDArray.

        Parameters:
            data: NDArray with points data to load.
            columns: List with labels for each of the data's colums. Must
                     include x, y and z.
        """
        _, col = data.shape
        assert col == len(columns)

        for column in ('x', 'y', 'z'):
            if column not in columns:
                raise PointCloudException(
                    f"The required column '{column}' is not set")

        super().__init__(data=data, columns=columns)

        self._num_points = len(self)

    @staticmethod
    def from_xyz(path: pathlib.Path,
                 columns: List[str] = ['x', 'y', 'z']) -> PointCloud:
        """
        Create a PointCloud from an xyz file.
        """
        return PointCloud(data=np.loadtxt(path,
                                          dtype=np.float64), columns=columns)

    @staticmethod
    def from_npy(path: pathlib.Path,
                 columns: List[str] = ['x', 'y', 'z']) -> PointCloud:
        """
        Create a PointCloud from an npy file.
        """
        return PointCloud(data=np.load(path), columns=columns)

    @staticmethod
    def from_file(path: pathlib.Path,
                  columns: List[str] = ['x', 'y', 'z']) -> PointCloud:
        """
        Create a PointCloud from file.
        """
        if path.suffix == '.xyz':
            return PointCloud.from_xyz(path=path, columns=columns)
        elif path.suffix == '.npy':
            return PointCloud.from_npy(path=path, columns=columns)
        else:
            raise PointCloud(f"File suffix must be '.xyz' or '.npy'")

    def X(self: PointCloud) -> NDArray:
        """
        Get all points x, y and z.
        """
        return self[['x', 'y', 'z']].to_numpy()

    def rigid_body_transform(self: PointCloud, H: NDArray) -> None:
        """
        Perform a rigid body transform of the PointCloud using the
        homogenous matrix.        
        """
        Xt = H_transform(H, self.X())
        self['x'] = Xt[:, 0]
        self['y'] = Xt[:, 1]
        self['z'] = Xt[:, 2]

    def save_xyz(self: PointCloud, path: pathlib.Path,
                 columns: List[str] = ['x', 'y', 'z']) -> None:
        """
        Save the PointCloud to an xyz file using the provided columns.
        """
        np.savetxt(path, self[columns].to_numpy(), fmt='%.6f')

    def save_npy(self: PointCloud, path: pathlib.Path,
                 columns: List[str] = ['x', 'y', 'z']) -> None:
        """
        Save the PointCloud to an npy file using the provided columns.
        """
        np.save(path, self[columns].to_numpy())

    def num_points(self: PointCloud) -> int:
        """
        Get the number of of points in the PointCloud.
        """
        return self._num_points


class PointCloudException(Exception):
    """PointCloud exception"""
