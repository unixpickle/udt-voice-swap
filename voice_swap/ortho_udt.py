"""
Orthogonal unsupervised domain translation, as implemented by
https://arxiv.org/abs/2007.12568.
"""

import numpy as np
from tqdm.auto import tqdm


def ortho_udt(source, target, verbose=False, no_cycle_check=False, max_iters=None):
    """
    Compute an orthogonal matrix that translates from a source domain to a
    target domain.

    :param source: an [N x D] array of source vectors.
    :param target: an [N x D] array of target vectors.
    :param verbose: if True, log information during optimization.
    :param no_cycle_check: if True, don't enforce cycle consistency.
    :param max_iters: if specified, the maximum iteration count.
    :return: a [D x D] orthogonal matrix that takes vectors from the source
             and produces vectors from the target.
    """
    solution = np.eye(source.shape[1])
    num_iters = 0
    while True:
        new_source = source @ solution
        if verbose:
            print("Computing neighbors...")
        source_neighbors, target_neighbors = nearest_neighbors(
            new_source, target, verbose=verbose
        )

        use_sources = target_neighbors[source_neighbors] == np.arange(len(new_source))

        if no_cycle_check:
            source_vecs = np.concatenate([source, target], axis=0)
            target_vecs = np.concatenate(
                [target[source_neighbors], source[target_neighbors]], axis=0
            )
        else:
            source_vecs = source[use_sources]
            target_vecs = target[source_neighbors[use_sources]]

        u, _, vh = np.linalg.svd(source_vecs.T @ target_vecs)
        new_solution = u @ vh
        num_iters += 1
        if verbose:
            num_used = np.sum(use_sources)
            uniq_targets = len(set(source_neighbors))
            uniq_sources = len(set(target_neighbors))
            print(
                f"iter {num_iters}: used={num_used} uniq_targets={uniq_targets} uniq_sources={uniq_sources}"
            )
        if np.allclose(solution, new_solution) or num_iters == max_iters:
            break
        solution = new_solution
    return solution


def nearest_neighbors(source, target, batch_size=128, verbose=False):
    """
    For each source vector, compute the target vector that is nearest to the
    source vector, and vice versa.

    :return: a tuple (source_indices, target_indices), both 1-D arrays of
             integers, giving the index of a target or source vector for each
             source or target vector.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return _nearest_neighbors_torch(
                source, target, batch_size=batch_size, verbose=verbose
            )
    except ImportError:
        pass

    indices = np.zeros([len(source)], dtype=np.int)
    distances = np.inf * np.ones([len(source)], dtype=source.dtype)
    target_indices = []
    batches = range(0, len(target), batch_size)
    if verbose:
        batches = tqdm(batches)
    for i in batches:
        batch = target[i : i + batch_size]
        source_norms = np.sum(source * source, axis=-1)[:, None]
        target_norms = np.sum(batch * batch, axis=-1)[None]
        dots = source @ batch.T
        distance_mat = source_norms + target_norms - 2 * dots
        min_indices = np.argmin(distance_mat, axis=-1)
        min_values = np.min(distance_mat, axis=-1)
        indices = np.where(min_values < distances, min_indices + i, indices)
        distances = np.minimum(distances, min_values)
        target_indices.append(np.argmin(distance_mat, axis=0))
    return indices, np.concatenate(target_indices, axis=0)


def _nearest_neighbors_torch(source, target, batch_size=128, verbose=False):
    import torch

    dev = torch.device("cuda")
    source = torch.from_numpy(source).to(torch.float32).to(dev)
    target = torch.from_numpy(target).to(source)
    indices = torch.zeros([len(source)], device=dev, dtype=torch.int64)
    distances = (torch.ones([len(source)]) * np.inf).to(source)
    target_indices = []
    batches = range(0, len(target), batch_size)
    if verbose:
        batches = tqdm(batches)
    for i in batches:
        batch = target[i : i + batch_size]
        source_norms = torch.sum(source * source, dim=-1)[:, None]
        target_norms = torch.sum(batch * batch, dim=-1)[None]
        dots = source @ batch.T
        distance_mat = source_norms + target_norms - 2 * dots
        min_indices = torch.argmin(distance_mat, dim=-1)
        min_values, _ = torch.min(distance_mat, dim=-1)
        indices = torch.where(min_values < distances, min_indices + i, indices)
        distances, _ = torch.minimum(distances, min_values)
        target_indices.append(torch.argmin(distance_mat, dim=0).cpu().numpy())
    return indices.cpu().numpy(), np.concatenate(target_indices, axis=0)
