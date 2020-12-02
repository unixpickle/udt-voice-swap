"""
Orthogonal unsupervised domain translation, as implemented by
https://arxiv.org/abs/2007.12568.
"""

import numpy as np


def ortho_udt(source, target, verbose=False, no_cycle_check=False):
    """
    Compute an orthogonal matrix that translates from a source domain to a
    target domain.

    :param source: an [N x D] array of source vectors.
    :param target: an [N x D] array of target vectors.
    :param verbose: if True, log information during optimization.
    :param no_cycle_check: if True, don't enforce cycle consistency.
    :return: a [D x D] orthogonal matrix that takes vectors from the source
             and produces vectors from the target.
    """
    solution = np.eye(source.shape[1])
    num_iters = 0
    while True:
        new_source = source @ solution
        source_neighbors = nearest_neighbors(new_source, target)
        target_neighbors = nearest_neighbors(target, new_source)

        use_sources = target_neighbors[source_neighbors] == np.arange(len(new_source))

        if no_cycle_check:
            # TODO: go the other direction as well.
            source_vecs = source
            target_vecs = target[source_neighbors]
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
        if np.allclose(solution, new_solution):
            break
        solution = new_solution
    return solution


def nearest_neighbors(source, target, batch_size=128):
    """
    For each source vector, compute the target vector that is nearest to the
    source vector.

    :return: a 1-D array of integers, giving the index of a target vector for
             each source vector.
    """
    indices = np.zeros([len(source)], dtype=np.int)
    distances = np.inf * np.ones([len(source)], dtype=source.dtype)
    for i in range(0, len(target), batch_size):
        batch = target[i : i + batch_size]
        source_norms = np.sum(source * source, axis=-1)[:, None]
        target_norms = np.sum(batch * batch, axis=-1)[None]
        dots = source @ batch.T
        distance_mat = source_norms + target_norms - 2 * dots
        min_indices = np.argmin(distance_mat, axis=-1)
        min_values = np.min(distance_mat, axis=-1)
        indices = np.where(min_values < distances, min_indices + i, indices)
        distances = np.minimum(distances, min_values)
    return indices
