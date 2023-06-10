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
        self._ref = None
        self._qry = None

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

    def run(self: ICP,
            initial_rotation: ArrayLike = [0., 0., 0.],
            initial_translation: ArrayLike = [0., 0., 0.],
            correspondences: int = 1000,
            iterations: int = 100
            ) -> None:
        """
        Run the ICP algorithm.
        """
        self._start_time = time.time()
        self._finish_time = time.time()
        duration = self._finish_time - self._start_time

        print(f'ICP duration={duration:.2f} seconds')
