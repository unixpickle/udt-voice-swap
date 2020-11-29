import numpy as np

from .ortho_udt import nearest_neighbors


def test_nearest_neighbors():
    source = np.random.normal(size=(10, 3))
    target = np.random.normal(size=(10, 3))
    neighbors = nearest_neighbors(source, target)
    for i, neighbor in enumerate(neighbors):
        actual = np.argmin(np.sum((source[i][None] - target) ** 2, axis=-1))
        assert neighbor == actual
