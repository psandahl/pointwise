from __future__ import annotations

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
        self._initial_columns = columns

    @staticmethod
    def from_xyz(path: pathlib.Path, columns: List[str]) -> PointCloud:
        """
        Create a PointCloud from an xyz file.
        """
        return PointCloud(data=np.loadtxt(path,
                                          dtype=np.float64), columns=columns)

    @staticmethod
    def from_npy(path: pathlib.Path, columns: List[str]) -> PointCloud:
        """
        Create a PointCloud from an npy file.
        """
        return PointCloud(data=np.load(path), columns=columns)

    def save_xyz(self: PointCloud, path: pathlib.Path) -> None:
        """
        Save the PointCloud to an xyz file using same columns as created with.
        """
        np.savetxt(path, self[self._initial_columns].to_numpy(), fmt='%.6f')

    def save_npy(self: PointCloud, path: pathlib.Path) -> None:
        """
        Save the PointCloud to an npy file using same columns as created with.
        """
        np.save(path, self[self._initial_columns].to_numpy())


class PointCloudException(Exception):
    """PointCloud exception"""
