import pytest

import numpy as np

from .neighbors import NumpyNeighbors, TorchNeighbors


@pytest.mark.parametrize("cls", [NumpyNeighbors, TorchNeighbors])
def test_nearest_neighbors(cls):
    source = np.random.normal(size=(200, 3))
    target = np.random.normal(size=(200, 3))
    source_neighbors, target_neighbors = cls().neighbors(source, target)
    for i, neighbor in enumerate(source_neighbors):
        actual = np.argmin(np.sum((source[i][None] - target) ** 2, axis=-1))
        assert neighbor == actual
    # Check target neighbors by making sure the operation
    # is symmetric.
    target_neighbors_2, source_neighbors_2 = cls().neighbors(target, source)
    assert (target_neighbors == target_neighbors_2).all()
    assert (source_neighbors == source_neighbors_2).all()
