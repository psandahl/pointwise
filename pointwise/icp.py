from __future__ import annotations

from .pointcloud import PointCloud
from .transform import homogeneous_matrix_euler

import numpy as np
from numpy.typing import ArrayLike
import time


class ICP:
    def __init__(self: ICP) -> None:
        """
        Create a fresh ICP instance without any point clouds set.
        """
        # Point cloud references.
        self._ref = None
        self._qry = None

        # ICP parameters.
        self._axis_order = 'xyz'
        self._rotation = np.array([0., 0., 0.])
        self._translation = np.array([0., 0., 0.])
        self._correspondences = 1000
        self._iterations = 100
        self._neighbors = 10

        # Timing values.
        self._start_time = None
        self._finish_time = None

    def set_point_clouds(self: ICP,
                         reference: PointCloud,
                         query: PointCloud) -> None:
        """
        Assign two point clouds.

        Parameters:
            reference: The reference point cloud.
            query: The query point cloud (the one to find a transform for).
        """
        self._ref = reference
        self._qry = query

        print(
            f'ICP.set_point_clouds: reference={self._ref.num_points()} pts, query={self._qry.num_points()} pts')

    def set_axis_order(self: ICP, axis_order: str) -> None:
        """
        Set axis order for Euler rotations.
        """
        self._axis_order = axis_order
        if len(self._axis_order) != 3:
            raise ValueError('Axis order must be a string of length 3')

    def set_initial_rotation(self: ICP, rotation: ArrayLike) -> None:
        """
        Set the initial rotation in degrees (default: 0., 0., 0.).
        """
        self._rotation = np.array(rotation)
        if self._rotation.shape != (3,):
            raise ValueError('Initial rotation must have three elements')

    def set_initial_translation(self: ICP, translation: ArrayLike) -> None:
        """
        Set the initial translation (default: 0., 0., 0.).
        """
        self._translation = np.array(translation)
        if self._translation.shape != (3,):
            raise ValueError('Initial translation must have three elements')

    def set_correspondences(self: ICP, correspondences: int) -> None:
        """
        Set the number of correspondences selected in the reference cloud (default: 1000).
        """
        self._correspondences = correspondences

    def set_iterations(self: ICP, iterations: int) -> None:
        """
        Set the number of ICP iterations.
        """
        self._iterations = iterations

    def run(self: ICP) -> None:
        """
        Run the ICP algorithm.
        """
        assert self._ref is not None
        assert self._qry is not None

        self._start_time = time.time()

        # Convert the initial rotation to radians.
        self._rotation = np.radians(self._rotation)

        # Compute homogeneous matrix.
        H = homogeneous_matrix_euler(
            ypr=self._rotation, t=self._translation, axis_order=self._axis_order)

        # TODO: Check overlap.

        # Select reference points.
        print(
            f'Select {self._correspondences} points for correspondences in reference cloud')
        self._ref.select_n_points(n=self._correspondences)

        # Save the initial selection.
        initial_selection = self._ref['selected']

        # Estimate normals for the reference cloud.
        if not self._ref.has_normals():
            print('Estimate normals for the reference cloud')
            self._ref.estimate_normals(num_neighbors=self._neighbors)

        print('Start iterations')
        residuals = []
        for i in range(self._iterations):
            pass

        self._finish_time = time.time()
        duration = self._finish_time - self._start_time

        print(f'ICP duration={duration:.2f} seconds')
