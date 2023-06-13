from __future__ import annotations

from .geometry import estimate_normal
from .transform import H_transform

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pathlib
from scipy.spatial import KDTree
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

        # All points are initially marked as selected
        if 'selected' not in self:
            self['selected'] = np.ones(self._num_points, dtype=bool)
        else:
            self['selected'].values[:] = True

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

    def has_normals(self: PointCloud) -> bool:
        """
        Check if the PointCloud has normals.
        """
        return {'nx', 'ny', 'nz', 'planarity'}.issubset(self)

    def select_all_points(self: PointCloud) -> None:
        """
        Mark all points as selected.
        """
        self['selected'].values[:] = True

    def unselect_all_points(self: PointCloud) -> None:
        """
        Mark all points as unselected.
        """
        self['selected'].values[:] = False

    def num_selected_points(self: PointCloud) -> int:
        """
        Count the number of selected points.
        """
        return sum(self['selected'])

    def selected_indices(self: PointCloud) -> NDArray:
        """
        Get the indices for the selected points.
        """
        return np.where(self['selected'])[0]

    def select_n_points(self: PointCloud, n: int) -> None:
        """
        Select n points evenly across the selected points.
        """
        num_selected = self.num_selected_points()
        if num_selected > n:
            idx_subset_n_points = np.round(
                np.linspace(0, num_selected - 1, n)).astype(int)

            idx_selected_new = self.selected_indices()[idx_subset_n_points]
            self.unselect_all_points()
            self.loc[idx_selected_new, 'selected'] = True

    def estimate_normals(self: PointCloud, num_neighbors: int) -> None:
        nx = np.full(self.num_points(), np.nan, dtype=np.float32)
        ny = np.full(self.num_points(), np.nan, dtype=np.float32)
        nz = np.full(self.num_points(), np.nan, dtype=np.float32)
        py = np.full(self.num_points(), np.nan, dtype=np.float32)

        # Pour in all data in a kdtree.
        X = self.X()
        kdtree = KDTree(data=X)

        X_selected = self.X_selected()
        _, neighbor_indices = kdtree.query(
            x=X_selected, k=num_neighbors, p=2, workers=-1)

        for i, indices in enumerate(neighbor_indices):
            neighbors = X[indices]
            normal, planarity = estimate_normal(data=neighbors)

            nx[i] = normal[0]
            ny[i] = normal[1]
            nz[i] = normal[2]
            py[i] = planarity

        self['nx'] = pd.arrays.SparseArray(nx)
        self['ny'] = pd.arrays.SparseArray(ny)
        self['nz'] = pd.arrays.SparseArray(nz)
        self['planarity'] = pd.arrays.SparseArray(py)

    def X(self: PointCloud) -> NDArray:
        """
        Get all points x, y and z.
        """
        return self[['x', 'y', 'z']].to_numpy()

    def X_selected(self: PointCloud) -> NDArray:
        """
        Get all points x, y and z for the selected rows.
        """
        return self.loc[self['selected'], ['x', 'y', 'z']].to_numpy()

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
