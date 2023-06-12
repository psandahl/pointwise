from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def estimate_normal(data: NDArray) -> tuple[NDArray, float]:
    """
    Estimate surface normal and planarity from the given
    3D points.
    """
    assert data.shape[0] > 1
    assert data.shape[1] == 3

    cov = covariance_matrix(data)
    evals, evecs = np.linalg.eig(cov)

    # Sort indices to match larger to smaller values.
    indices = evals.argsort()[::-1]

    evals = evals[indices]
    evecs = evecs[:, indices]

    normal = evecs[:, 2]
    planarity = (evals[1] - evals[2]) / evals[0]

    return normal, planarity


def covariance_matrix(data: NDArray) -> NDArray:
    items = data.shape[0]
    centroid = np.sum(data, axis=0) / items
    mean_adjusted_data = data - centroid

    # Normalization to be compatible with np.cov(data.T, bias=False).
    return mean_adjusted_data.T @ mean_adjusted_data / (items - 1)
