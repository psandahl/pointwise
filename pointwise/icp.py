from __future__ import annotations

from .pointcloud import PointCloud

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
        self._initial_rotation = np.array([0., 0., 0.])
        self._initial_translation = np.array([0., 0., 0.])
        self._correspondences = 1000
        self._iterations = 100

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

    def set_initial_rotation(self: ICP, rotation: ArrayLike) -> None:
        """
        Set the initial rotation in degrees (default: 0., 0., 0.).
        """
        self._initial_rotation = np.array(rotation)

    def set_initial_translation(self: ICP, translation: ArrayLike) -> None:
        """
        Set the initial translation (default: 0., 0., 0.).
        """
        self._initial_translation = np.array(translation)

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
        self._start_time = time.time()
        self._finish_time = time.time()
        duration = self._finish_time - self._start_time

        print(f'ICP duration={duration:.2f} seconds')
