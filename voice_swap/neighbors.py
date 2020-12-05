from abc import ABC, abstractmethod

import numpy as np
from tqdm.auto import tqdm


class Neighbors(ABC):
    """
    A k-Nearest Neighbor calculator.
    """

    def __init__(self, verbose=False, batch_size=128):
        self.verbose = verbose
        self.batch_size = batch_size

    @staticmethod
    def create(verbose=False, batch_size=128):
        try:
            import torch

            if torch.cuda.is_available():
                return TorchNeighbors(verbose, batch_size)
        except ModuleNotFoundError:
            return NumpyNeighbors(verbose, batch_size)

    @abstractmethod
    def neighbors(self, a, b):
        """
        For each vector in arrays a and b, compute the nearest vector in the
        other array.

        :param a: a 2-D numpy array of vectors.
        :param b: a 2-D numpy array of vectors.
        :return: a tuple (a_neighbors, b_neighbors), where each element of
        a_neighbors is an index in b, and vice versa.
        """


class NumpyNeighbors(Neighbors):
    def neighbors(self, a, b):
        a_neighbors = np.zeros([len(a)], dtype=np.int)
        b_neighbors = []
        a_distances = np.inf * np.ones([len(a)], dtype=a.dtype)
        batches = range(0, len(b), self.batch_size)
        if self.verbose:
            batches = tqdm(batches)
        a_norms = np.sum(a ** 2, axis=-1)[:, None]
        for i in batches:
            batch = b[i : i + self.batch_size]
            batch_norms = np.sum(batch ** 2, axis=-1)[None]
            dots = a @ batch.T
            distance_mat = a_norms + batch_norms - 2 * dots
            min_indices = np.argmin(distance_mat, axis=-1)
            min_values = np.min(distance_mat, axis=-1)
            a_neighbors = np.where(
                min_values < a_distances, min_indices + i, a_neighbors
            )
            a_distances = np.minimum(a_distances, min_values)
            b_neighbors.append(np.argmin(distance_mat, axis=0))
        return a_neighbors, np.concatenate(b_neighbors, axis=0)


class TorchNeighbors(Neighbors):
    def neighbors(self, a, b):
        import torch

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        source = torch.from_numpy(a).to(torch.float32).to(dev)
        target = torch.from_numpy(b).to(source)
        indices = torch.zeros([len(source)], device=dev, dtype=torch.int64)
        distances = (torch.ones([len(source)]) * np.inf).to(source)
        target_indices = []
        batches = range(0, len(target), self.batch_size)
        if self.verbose:
            batches = tqdm(batches)
        source_norms = torch.sum(source ** 2, dim=-1)[:, None]
        for i in batches:
            batch = target[i : i + self.batch_size]
            target_norms = torch.sum(batch * batch, dim=-1)[None]
            dots = source @ batch.T
            distance_mat = source_norms + target_norms - 2 * dots
            min_indices = torch.argmin(distance_mat, dim=-1)
            min_values, _ = torch.min(distance_mat, dim=-1)
            indices = torch.where(min_values < distances, min_indices + i, indices)
            distances = torch.where(min_values < distances, min_values, distances)
            target_indices.append(torch.argmin(distance_mat, dim=0).cpu().numpy())
        return indices.cpu().numpy(), np.concatenate(target_indices, axis=0)
