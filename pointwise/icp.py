from __future__ import annotations


from .pointcloud import PointCloud


class ICP:
    def __init__(self: ICP) -> None:
        self._ref = None
        self._qry = None

    def set_point_clouds(self: ICP,
                         reference: PointCloud,
                         query: PointCloud) -> None:
        self._ref = reference
        self._qry = query

    def run(self: ICP) -> None:
        pass
